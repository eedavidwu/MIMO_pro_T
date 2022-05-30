import torch 
from Models.SETR.transformer_seg import SETRModel
import torchvision
import torch
import torch.nn as nn 
from torchvision import datasets, transforms
import matplotlib.pyplot as plt 
import numpy as np
import os
import argparse
from torch.autograd import Variable

#nohup python train_SETR.py --snr 15 >double_trans_64_6_4_4_SNR_15.out&

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print("device is " + str(device))
def check_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def compute_AvePSNR(model,dataloader,snr):
    psnr_all_list = []
    model.eval()
    MSE_compute = nn.MSELoss(reduction='none')
    for batch_idx, (inputs, _) in enumerate(dataloader, 0):
        b,c,h,w=inputs.shape[0],inputs.shape[1],inputs.shape[2],inputs.shape[3]
        inputs = inputs.cuda()
        outputs = model(inputs,snr)
        MSE_each_image = (torch.sum(MSE_compute(outputs, inputs).view(b,-1),dim=1))/(c*h*w)
        PSNR_each_image = 10 * torch.log10(1 / MSE_each_image)
        one_batch_PSNR=PSNR_each_image.data.cpu().numpy()
        psnr_all_list.extend(one_batch_PSNR)
    Ave_PSNR=np.mean(psnr_all_list)
    return Ave_PSNR



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #Train:
    parser.add_argument("--best_ckpt_path", default='./ckpts/', type=str,help='best model path')
    parser.add_argument("--all_epoch", default='1200', type=int,help='Train_epoch')
    parser.add_argument("--best_choice", default='loss', type=str,help='select epoch [loss/PSNR]')
    parser.add_argument("--flag", default='train', type=str,help='train or eval for JSCC')

    # Model and Channel:
    parser.add_argument("--model", default='SETR', type=str,help='Model select: SETR/ADSETR/')
    parser.add_argument("--tcn", default=8, type=int,help='tansmit_channel_num for djscc')
    #parser.add_argument("--channel_type", default='awgn', type=str,help='awgn/slow fading/burst')
    parser.add_argument("--N_r", default=2, type=int,help='number of receiver')
    parser.add_argument("--N_s", default=2, type=int,help='number of sender')

    parser.add_argument("--snr", default=5, type=int,help='awgn/slow fading/')
    parser.add_argument("--trans_times", default=5, type=int,help='transmit_times')

    #parser.add_argument("--const_snr", default=True,help='SNR (db)')
    #parser.add_argument("--input_const_snr", default=1, type=float,help='SNR (db)')
    parser.add_argument("--input_snr_max", default=20, type=float,help='SNR (db)')
    parser.add_argument("--input_snr_min", default=0, type=int,help='SNR (db)')
    parser.add_argument("--N_t", default=5, type=int,help='SNR (db)')

    #parser.add_argument("--resume", default=False,type=bool, help='Load past model')
    args=parser.parse_args()

    #64*6 1/8 ->(4,4) tcn=6
    #16*24 1/8->(8,8) tcn=24
    #print("64*6 1/8 ->(4,4) tcn=8")
    #print("16*24 1/8->(8,8) tcn=24")
    print('rate:',args.tcn)
    print('MIMO number',args.N_t)
    
    model = SETRModel(patch_size=(4, 4), 
                    in_channels=3, 
                    out_channels=3, 
                    hidden_size=256, 
                    num_hidden_layers=4, 
                    num_attention_heads=4, 
                    tcn=args.tcn,MIMO_num=args.N_t)
    #channel_snr=args.snr
    channel_snr='random'

    GPU_ids = [0,1,2,3]
    print("############## Train model",args.model,",with SNR: ",channel_snr," ##############")
    print("this model size path 64, tcn 8")
    print(model)
    model = nn.DataParallel(model,device_ids = GPU_ids)
    model = model.cuda()
    #print(model)
    #model.to(device)
    #transform = transforms.Compose([transforms.ToTensor(),
    #                           transforms.Normalize(mean=[0.5],std=[0.5])])
    #
    # Load data
    transform = transforms.Compose(
        [transforms.ToTensor(), ])
    trainset = torchvision.datasets.CIFAR10(root='./data/cifar', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data/cifar', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                             shuffle=False, num_workers=2)


    optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
    loss_func = nn.MSELoss()
    best_psnr=0

    #for in_data, label in tqdm(data_loader_train, total=len(data_loader_train)):
    for epoch in range(0,args.all_epoch):
        step = 0
        report_loss = 0
        for in_data, label in trainloader:
            batch_size = len(in_data)
            in_data = in_data.to(device)
            #label = label.to(device)
            optimizer.zero_grad()
            step += 1
            out=model(in_data,channel_snr)            
            loss = loss_func(out, in_data)
            #print(loss)
            loss.backward()
            optimizer.step()
            report_loss += loss.item()
        print('Epoch:[',epoch,']',", loss : " ,str(report_loss/step))

        if ((epoch % 4 == 0) and (epoch>200)) or epoch==0:
            if args.model=='SETR':
                if channel_snr=='random':
                    PSNR_list=[]
                    for i in [1,5,10,15,19]:
                        validate_snr=i
                        val_ave_psnr=compute_AvePSNR(model,testloader,validate_snr)
                        PSNR_list.append(val_ave_psnr)
                    ave_PSNR=np.mean(PSNR_list)
                    if ave_PSNR > best_psnr:
                        best_psnr=ave_PSNR
                        print('Find one best model with best PSNR:',best_psnr,' under SNR: ',channel_snr)
                        checkpoint={
                            "model_name":args.model,
                            "net":model.state_dict(),
                            "op":optimizer.state_dict(),
                            "epoch":epoch,
                            "SNR":channel_snr,
                            "Best_PSNR":best_psnr
                        }
                        print(PSNR_list)
                        SNR_path='./checkpoints_'+str(args.tcn)+'/'
                        check_dir(SNR_path)      
                        save_path=os.path.join(SNR_path,'4_layer_MIMO_'+str(args.N_t)+'_SNR_'+str(channel_snr)+'.pth')
                        torch.save(checkpoint, save_path)
                        print('Saving Model at epoch',epoch,'at',save_path)       
                else:
                    validate_snr=channel_snr
                    val_ave_psnr=compute_AvePSNR(model,testloader,validate_snr)
                    if (val_ave_psnr > best_psnr):
                        best_psnr=val_ave_psnr
                        print('Find one best model with best PSNR:',best_psnr,' under SNR: ',validate_snr,'in epoch',epoch)
                        PSNR_list=[]
                        #for i in [1,4,10,16,19]:
                        for i in [1,5,10,15,19]:
                        #for i in [1]:    
                            ave_PSNR_test=compute_AvePSNR(model,testloader,i)
                            PSNR_list.append(ave_PSNR_test)
                        print(PSNR_list)
                        checkpoint={
                            "model_name":'SETR_layer_1',
                            "net":model.state_dict(),
                            "op":optimizer.state_dict(),
                            "epoch":epoch,
                            "SNR":channel_snr,
                            "Best_PSNR":best_psnr
                        }
                        SNR_path='./checkpoints_8/SNR_T_64_8_4layer_4head'
                        check_dir(SNR_path)      
                        save_path=os.path.join(SNR_path,'SETR_SNR_siam_'+str(channel_snr)+'.pth')
                        torch.save(checkpoint, save_path)
                        print('Saving Model at epoch',epoch,save_path)
            
