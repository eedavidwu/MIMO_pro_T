import logging
import math
import os
import numpy as np 

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from Models.SETR.transformer_model import Att_TransModel2d, TransConfig,TransModel2d,ReciverModel2d
from torch.autograd import Variable
import math 

class Channel(nn.Module):
    def __init__(self,MIMO_num):
        super(Channel, self).__init__()
        self.N_t=MIMO_num
    '''    
    def power_normalize(self,feature):        
        sig_pwr=torch.square(torch.abs(feature))
        ave_sig_pwr=sig_pwr.mean(dim=2).unsqueeze(dim=2)
        z_in_norm=feature/(torch.sqrt(ave_sig_pwr))
        return z_in_norm
    '''
    
    def power_normalize(self,feature):
        in_shape=feature.shape
        batch_size=in_shape[0]
        z_in=feature.reshape(batch_size,-1)
        sig_pwr=torch.square(torch.abs(z_in))
        ave_sig_pwr=sig_pwr.mean(dim=1).unsqueeze(dim=1)
        z_in_norm=z_in/(torch.sqrt(ave_sig_pwr))
        inputs_in_norm=z_in_norm.reshape(in_shape)
        return inputs_in_norm

    def forward(self, inputs,Uh,S_diag,snr):
        in_shape=inputs.shape
        batch_size=in_shape[0]
        z=inputs.view(batch_size,-1)
        
        complex_list=torch.split(z,(z.shape[1]//2),dim=1)
        z_in=torch.complex(complex_list[0],complex_list[1])
        #shape for MIMO:        
        Y=z_in.view(batch_size,self.N_t,-1)
        shape_each_antena=Y.shape[-1]
        #normalized_X=Y
        normalized_X=self.power_normalize(Y)

        ##awgn:
        #noise_stddev=(np.sqrt(10**(-snr/10))/np.sqrt(2)).reshape(-1,1,1)
        noise_stddev=torch.sqrt(10**(-snr/10)/2).unsqueeze(dim=1)
        #noise_stddev_board=torch.from_numpy(noise_stddev).repeat(1,2,shape_each_antena).float().cuda()
        noise_stddev_board=noise_stddev.repeat(1,self.N_t,shape_each_antena).float().cuda()
        mean=torch.zeros_like(noise_stddev_board).float().cuda()
        w_real=Variable(torch.normal(mean=mean,std=noise_stddev_board)).float()
        w_img=Variable(torch.normal(mean=mean,std=noise_stddev_board)).float()
        W=torch.complex(w_real,w_img)

        Y_out=normalized_X+torch.bmm(torch.inverse(S_diag),torch.bmm(Uh,W))

        Y_all_antena=Y_out.view(batch_size,-1)
        Y_head_=torch.cat((torch.real(Y_all_antena),torch.imag(Y_all_antena)),dim=1)
        channel_out=Y_head_.view(in_shape)
        return channel_out



class Encoder2D(nn.Module):
    def __init__(self, config: TransConfig, tcn,MIMO_num):
        super().__init__()
        self.config = config
        self.out_channels = config.out_channels
        self.bert_model = TransModel2d(config,MIMO_num)
        sample_rate = config.sample_rate
        sample_v = int(math.pow(2, sample_rate))
        #sample_rate=4,sample_v=16
        assert config.patch_size[0] * config.patch_size[1] * config.hidden_size % (sample_v**2) == 0, "不能除尽"
        self.final_dense = nn.Linear(config.hidden_size,  tcn)
        ##linear:x hidden-> 8*8*hidden/16/16
        self.patch_size = config.patch_size
        self.hh = self.patch_size[0] // sample_v
        self.ww = self.patch_size[1] // sample_v
        
    def forward(self, x,CSI_vector):
        ## x:(b, c, w, h)
        b, c, h, w = x.shape
        assert self.config.in_channels == c, "in_channels != 输入图像channel"
        p1 = self.patch_size[0]
        p2 = self.patch_size[1]
        if h % p1 != 0:
            print("请重新输入img size 参数 必须整除")
            os._exit(0)
        if w % p2 != 0:
            print("请重新输入img size 参数 必须整除")
            os._exit(0)
        hh = h // p1 
        ww = w // p2 

        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p1, p2 = p2,h = hh, w = ww)

        x_in=torch.cat((x,CSI_vector),dim=2)
        
        encode_x = self.bert_model(x_in)[-1] # 取出来最后一层

        x = self.final_dense(encode_x)
        #x_map = rearrange(x, "b (h w) (c) -> b c (h) (w)", h = hh, w = ww, c =self.tcn)
        return x 

class Decoder2D_trans(nn.Module):
    def __init__(self, config: TransConfig, tcn,MIMO_num):
        super().__init__()
        self.config = config
        self.out_channels = config.out_channels
        self.bert_model = ReciverModel2d(config,tcn+MIMO_num+1)
        #sample_rate=4,sample_v=16
        #self.final_dense = nn.Linear(config.hidden_size, 192)
        self.final_dense = nn.Linear(config.hidden_size, 48)
        ##linear:x hidden-> 8*8*hidden/16/16



    def forward(self, x):
        ## x:(b, path_num, c)
        encode_x = self.bert_model(x)[-1] # 取出来最后一层
        x = self.final_dense(encode_x)        
        #x_out = rearrange(x, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", p1 = 8, p2 = 8, h = 4, w = 4, c =3)
        x_out = rearrange(x, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", p1 = 4, p2 = 4, h = 8, w = 8, c =3)

        return x_out 



class SETRModel(nn.Module):
    def __init__(self, patch_size=(32, 32), 
                        in_channels=3, 
                        out_channels=1, 
                        hidden_size=1024, 
                        num_hidden_layers=8, 
                        num_attention_heads=16,
                        max_position_embeddings=64,
                        decode_features=[512, 256, 128, 64],
                        sample_rate=2,tcn=8,MIMO_num=2):
        super().__init__()
        config = TransConfig(patch_size=patch_size, 
                            in_channels=in_channels, 
                            out_channels=out_channels, 
                            sample_rate=sample_rate,
                            hidden_size=hidden_size, 
                            max_position_embeddings=max_position_embeddings,
                            num_hidden_layers=num_hidden_layers, 
                            num_attention_heads=num_attention_heads)
        
        self.encoder_2d = Encoder2D(config,tcn,MIMO_num)
        #self.decoder_2d = Decoder2D(in_channels=tcn, out_channels=config.out_channels, features=decode_features)
        #self.res_decoder=Decoder_Res(in_channel=tcn)
        self.decoder_tran=Decoder2D_trans(config,tcn,MIMO_num)
        self.channel=Channel(MIMO_num)
        self.SNR_max=20
        self.SNR_min=0
        self.MIMO_num=MIMO_num

    def forward(self, x,channel_snr):
        batch_size=x.shape[0]
        N_s=self.MIMO_num
        N_r=self.MIMO_num
        #reshape
        x_head_ave=torch.zeros_like(x).cuda()
        if channel_snr=='random':
            snr_attention=np.random.rand(batch_size,)*(self.SNR_max-self.SNR_min)+self.SNR_min
        else:
            snr_attention=np.full(batch_size,channel_snr)

        channel_snr_attention=torch.from_numpy(snr_attention).float().cuda().view(batch_size,-1)
        
        for transmit_times in range (3):
            #prepare H at each time for the attention and channel:
            h_stddev=torch.full((batch_size,N_r,N_s),1/np.sqrt(2)).float()
            h_mean=torch.zeros_like(h_stddev).float()
            h_real=Variable(torch.normal(mean=h_mean,std=h_stddev)).float()
            h_img=Variable(torch.normal(mean=h_mean,std=h_stddev)).float()
            H=torch.complex(h_real,h_img).cuda()
            
            #decompose:
            U, S, Vh = torch.linalg.svd(H, full_matrices=True)
            S_diag=torch.diag_embed(torch.complex(S,torch.zeros_like(S)))
            Uh=U.transpose(-2, -1).conj()

            #CSI attention
            h_attention=S
            CSI_attention_vector=torch.cat((h_attention,channel_snr_attention),dim=1).unsqueeze(dim=1).repeat(1,64,1)
            #CSI_attention_vector=h_attention.unsqueeze(dim=1).repeat(1,64,1)

            #encoder
            x_encoded = self.encoder_2d(x,CSI_attention_vector)
            #channel
            channel_out=self.channel(x_encoded,Uh,S_diag,channel_snr_attention)
            #decoder
            decoder_input=torch.cat((channel_out,CSI_attention_vector),dim=2)
            x_head= self.decoder_tran(decoder_input)  
            x_head_ave=x_head+x_head_ave
        x_head_ave=x_head_ave/3
        return x_head_ave

   
