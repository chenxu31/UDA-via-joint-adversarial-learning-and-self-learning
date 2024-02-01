# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 17:14:55 2020

@author: 11627
"""
import os.path as osp
from networks.discriminator import get_done_entropy_discriminator,get_done_discriminator,get_exit_discriminator
import numpy as np
import numpy
from torch.utils.data import DataLoader
from torch.utils import data
from datasets import CT_liver,MR_liver
from networks.u_net import U_Net
from networks.student_model import S_Unet
from networks.att_u_net import AttU_Net
from networks.att_student_model import AttS_Net
from networks.final_model import AttSS_Net
from networks.temp_model import temp_Net
import os,sys
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from utils.loss import SoftDiceLoss,entropy_loss,bce_loss,EntropyLoss
import torch.nn.functional as F
from utils.metrics import diceCoeffv2
from utils.pamr import PAMR
import imageio
import platform
from datetime import datetime
import pdb
import argparse


if platform.system() == 'Windows':
    NUM_WORKERS = 0
    UTIL_DIR = r"E:\我的坚果云\sourcecode\python\util"
else:
    NUM_WORKERS = 4
    UTIL_DIR = r"/home/chenxu/我的坚果云/sourcecode/python/util"

sys.path.append(UTIL_DIR)
import common_metrics
import common_net_pt as common_net
import common_pelvic_pt as common_pelvic


#need two GPUs, 11GB respectively
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
seed = 2020
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

                


def pseudo_gtmask(output1,  exitlos):
    """Convert continuous mask into binary mask"""
    bs,c,h,w = output1.size()
    output1=output1.numpy()
    ave_output=np.zeros(shape=(1,c,h,w))
    for jj in range(bs):
        ave_output[0,:,:,:]=ave_output[0,:,:,:]+(1/bs)*output1[jj,:,:,:]
    
    
    pseudo_gt=np.zeros(shape=(bs,c,h,w))
    for j in range(bs):
        if exitlos[j]<0.5:
            pseudo_gt[j,:,:,:]=ave_output
        else:
            pseudo_gt[j,:,:,:]=output1[j,:,:,:]
            
    pseudo_gt=torch.from_numpy(np.array(pseudo_gt, dtype=np.float32))
    pseudo_gt = torch.sigmoid(pseudo_gt)
    pseudo_gt[pseudo_gt < 0.5] = 0
    pseudo_gt[pseudo_gt > 0.5] = 1

    return pseudo_gt



def main(device, args):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    if args.task == 'pelvic':
        num_classes = common_pelvic.NUM_CLASSES
        dataset_s = common_pelvic.Dataset(args.data_dir, "ct", n_slices=1, debug=args.debug)
        dataset_t = common_pelvic.Dataset(args.data_dir, "cbct", n_slices=1, debug=args.debug)
        _, val_data, _, val_label = common_pelvic.load_val_data(args.data_dir)
    else:
        raise NotImplementedError(config['Data_input']['dataset'])
    dataloader_s = torch.utils.data.DataLoader(dataset_s, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                               drop_last=True, num_workers=NUM_WORKERS)
    dataloader_t = torch.utils.data.DataLoader(dataset_t, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                               drop_last=True, num_workers=NUM_WORKERS)

    pred_model = torch.load(args.pretrained_ckpt)
    #model_dict = pred_model.state_dict()
    #teacher_model indicates U1
    #student_model indicates U2
    #d_d1 indicates D1
    #d_d1en indicates D2
    #final_model indicatets the desired model U3 
    #temp_model indicates U4
    teacher_model=AttU_Net(img_ch=1, num_classes=num_classes)
    teacher_model.load_state_dict(pred_model)
    student_model=AttS_Net(img_ch=1, num_classes=num_classes)
    student_model.load_state_dict(pred_model)
    
    final_model=AttSS_Net(img_ch=1, num_classes=num_classes)

    temp_model=temp_Net(img_ch=1, num_classes=num_classes)
    final_model.to(device)
    temp_model.to(device)
    # UDA TRAINING
    # Create the model and start the training.
    cudnn.benchmark = True
    cudnn.enabled = True
    torch.backends.cudnn.deterministic = True

    # freeze teacher network
    teacher_model.to(device)
    teacher_model.eval()

    
    student_model.to(device)
    student_model.train()


    # DISCRIMINATOR NETWORK
    # output discriminator
    d_d1 = get_done_discriminator(img_ch=num_classes)
    d_d1.train()
    d_d1.to(device)
    d_d1.apply(initialize_weights)

    d_d1en = get_done_entropy_discriminator(img_ch=num_classes)
    d_d1en.train()
    d_d1en.to(device)
    d_d1en.apply(initialize_weights)
    

    weight_cliping_limit = 0.01
    learning_rate1=0.0001
    learning_rate2=0.0002
    learning_rate3=0.0002
    learning_rate4=0.00012
    learning_rate5=0.00015

    # labels for adversarial training
    one = torch.FloatTensor([1]).to(device)
    mone = one * -1
    PAMR_KERNEL = [1, 2, 4, 8]
    PAMR_ITER = 10
    criterion = SoftDiceLoss(activation='sigmoid')
    pamr_aff = PAMR(PAMR_ITER, PAMR_KERNEL)
    entropyloss_map=EntropyLoss(reduction='none')
    entropyloss=EntropyLoss(reduction='mean')
    maxpool=nn.MaxPool2d(kernel_size=256, stride=1)
    file_handle=open('record_PAMR.txt',mode='w')

    pamr_aff.to(device)

    best_dsc = 0
    patch_shape = (1, val_data[0].shape[1], val_data[0].shape[2])
    for epoch in range(args.num_epoches):

        dices1=[]
        dices2=[]
        dices3=[]

        # mr is target, ct is source
        for data_s,data_t in zip(dataloader_s, dataloader_t):
            mr_img = data_t["image"].to(device)
            mr_img_trans = (torch.log(mr_img+0.00001))+3*mr_img
            mr_img_trans=(mr_img_trans-mr_img_trans.min())/(mr_img_trans.max()-mr_img_trans.min())
            mr_img_trans=mr_img_trans*mr_img_trans
            mr_img_trans = mr_img_trans.to(device)
            mr_img_trans1 = mr_img
            ct_img = data_s["image"].to(device)
            ct_mask = torch.nn.functional.one_hot(data_s["label"], num_classes).permute((0, 4, 1, 2, 3)).to(device)

            # OPTIMIZERS
        
            optimizer = torch.optim.RMSprop(student_model.parameters(), lr=learning_rate1, alpha=0.9)
            optimizer_d_d1 = torch.optim.RMSprop(d_d1.parameters(), lr=learning_rate2, alpha=0.9)   
            optimizer_d_d1en = torch.optim.RMSprop(d_d1en.parameters(), lr=learning_rate3, alpha=0.9) 
            optimizer_final = torch.optim.RMSprop(final_model.parameters(), lr=learning_rate4, alpha=0.9)
            optimizer_temp = torch.optim.RMSprop(temp_model.parameters(), lr=learning_rate5, alpha=0.9)

            # UDA Training
            # only train discriminators. Don't accumulate grads in student_model

            for param in teacher_model.parameters():
                param.requires_grad = False     
            for param in d_d1.parameters():
                param.requires_grad = True 
            for param in d_d1en.parameters():
                param.requires_grad = True                
            # reset optimizers
            optimizer_d_d1.zero_grad()
            optimizer_d_d1en.zero_grad()
            
            for param in d_d1.parameters():
                param.data.clamp_(-weight_cliping_limit, weight_cliping_limit)
            for param in d_d1en.parameters():
                param.data.clamp_(-weight_cliping_limit, weight_cliping_limit)
            # Train with ct images
            _, _, _, ct_d1 = teacher_model(ct_img)
            
            ct_d11=ct_d1.detach()
            ct_en=entropyloss_map(ct_d11)
            d_loss_d1_ct = d_d1(ct_d11)
            d_loss_d1_ct = d_loss_d1_ct.mean(0).view(1)
            d_loss_d1_ct=d_loss_d1_ct
            d_loss_d1_ct.backward(one)

            d_loss_d1en_ct = d_d1en(ct_en)
            d_loss_d1en_ct = d_loss_d1en_ct.mean(0).view(1)
            d_loss_d1en_ct=d_loss_d1en_ct/2
            d_loss_d1en_ct.backward(one)
    
            # Train with mr images
            _, _, _, mr_d1 = student_model(mr_img)
            mr_d11=mr_d1.detach()
            mr_en=entropyloss_map(mr_d11)
            d_loss_d1_mr = d_d1(mr_d11)
            d_loss_d1_mr = d_loss_d1_mr.mean(0).view(1)
            d_loss_d1_mr=d_loss_d1_mr/2
            d_loss_d1_mr.backward(mone)
            

            d_loss_d1en_mr = d_d1en(mr_en)
            d_loss_d1en_mr = d_loss_d1en_mr.mean(0).view(1)
            d_loss_d1en_mr=d_loss_d1en_mr/2
            d_loss_d1en_mr.backward(mone)

            # Train with mr_trans images
            _, _, _, mr_d1_final = final_model(mr_img_trans)
            mr_d11_final=mr_d1_final.detach()
            d_loss_d1_mr_final = d_d1(mr_d11_final)
            d_loss_d1_mr_final = d_loss_d1_mr_final.mean(0).view(1)
            d_loss_d1_mr_final=d_loss_d1_mr_final/2
            d_loss_d1_mr_final.backward(mone)


            
            optimizer_d_d1.step()
            optimizer_d_d1en.step()



#             only train student_model. Don't accumulate grads in discriminators

            _, _, _, ct_d12 = student_model(ct_img)
 
            for param in student_model.parameters():
                param.requires_grad = False   
            for param in student_model.Conv1.parameters():
                param.requires_grad = True  

            for param in d_d1.parameters():
                param.requires_grad = False 
            for param in d_d1en.parameters():
                param.requires_grad = False                 
            optimizer.zero_grad()  

            en=entropyloss_map(mr_d1)
            g_loss_d1 = d_d1(mr_d1)

            mr_d11=mr_d1.detach()
            g_loss_d1 = g_loss_d1.mean(0).view(1)
            g_loss_d1.backward(one,retain_graph=True)

            mr_d1_mask = torch.sigmoid(mr_d11)
            exit_loss=maxpool(mr_d1_mask)
            exit_loss=exit_loss[:,0,0,0].cpu().numpy()

            g_loss_d1en = d_d1en(en)
            g_loss_d1en = g_loss_d1en.mean(0).view(1)
            g_loss_d1en.backward(one)

            ct_loss = criterion(ct_d12, ct_mask)
            ct_loss.backward() 
            
            optimizer.step()

            final_model.train()    
            temp_model.train()
            for param in d_d1.parameters():
                param.requires_grad = False 
            for param in d_d1en.parameters():
                param.requires_grad = False 
                
            optimizer_final.zero_grad() 
            optimizer_temp.zero_grad() 
                
            _,_,_,mr_d1_final=final_model(mr_img_trans)

            _,_,_,mr_d1_temp=temp_model(mr_img_trans1)
            mr_d1_temp1=mr_d1_temp.detach()
            
            if epoch<50:
                pseudo_mask=pseudo_gtmask(mr_d11.cpu(),exit_loss)
                pseudo_mask_loss_final= criterion(mr_d1_final, pseudo_mask.detach().to(device))
                pseudo_mask_loss_final.backward(retain_graph=True)

                
            else:
                pseudo_mask1 = torch.sigmoid(mr_d1_temp1)
                pseudo_mask1[pseudo_mask1 < 0.5] = 0
                pseudo_mask1[pseudo_mask1 > 0.5] = 1
                pseudo_mask_loss_final1= criterion(mr_d1_final, pseudo_mask1.detach())
                pseudo_mask_loss_final1=pseudo_mask_loss_final1*5
                pseudo_mask_loss_final1.backward(retain_graph=True)


            g_loss_d1_final = d_d1(mr_d1_final)
            g_loss_d1_final = g_loss_d1_final.mean(0).view(1)
            g_loss_d1_final.backward(one,retain_graph=True)


            en_loss=5*entropyloss(mr_d1_final)
            en_loss.backward(retain_graph=True)


            pamr_masks = torch.sigmoid(mr_d1_final.detach())
            masks_dec = pamr_aff(mr_img_trans.detach(), pamr_masks.detach())
            masks_dec[masks_dec < 0.5] = 0
            masks_dec[masks_dec > 0.5] = 1
            pamr_mask_loss=criterion(mr_d1_final, masks_dec.detach())
            pamr_mask_loss.backward()
            file_handle.write(str(pamr_mask_loss.item()))
            file_handle.write('\t')





#########################temp_model


            pseudo_mask1 = torch.sigmoid(mr_d1_final.detach())
            pseudo_mask1[pseudo_mask1 < 0.5] = 0
            pseudo_mask1[pseudo_mask1 > 0.5] = 1
            temp_mask_loss=criterion(mr_d1_temp, pseudo_mask1.detach())
            temp_mask_loss.backward()
            


            optimizer_final.step()
            optimizer_temp.step()
            final_model.eval()
            temp_model.eval()
            

       
            """
            distance_d1=2*abs(d_loss_d1_mr.detach().cpu().item()-d_loss_d1_ct.detach().cpu().item())
            distance_d1en=2*abs(d_loss_d1en_mr.detach().cpu().item()-d_loss_d1en_ct.detach().cpu().item())

            distance_d1_final=2*abs(d_loss_d1_mr_final.detach().cpu().item()-d_loss_d1_ct.detach().cpu().item())



            mr_loss= criterion(mr_d1.detach().cpu(), mr_mask.detach().cpu())
            mr_loss_final= criterion(mr_d1_final.detach().cpu(), mr_mask.detach().cpu())
            mr_loss_temp= criterion(mr_d1_temp.detach().cpu(), mr_mask.detach().cpu())
            file_handle.write(str(mr_loss_final.item()))
            file_handle.write('\n')
            
            mr_d11=mr_d1.detach().cpu()
            mr_d11 = torch.sigmoid(mr_d11)
            mr_d11[mr_d11 < 0.5] = 0
            mr_d11[mr_d11 > 0.5] = 1
            Liver_dice1 = diceCoeffv2(mr_d11, mr_mask.detach().cpu(), activation=None).cpu().item()
            dices1.append(Liver_dice1)

            mr_d11_final=mr_d1_final.detach().cpu()
            mr_d11_final = torch.sigmoid(mr_d11_final)
            mr_d11_final[mr_d11_final < 0.5] = 0
            mr_d11_final[mr_d11_final > 0.5] = 1
            Liver_dice2 = diceCoeffv2(mr_d11_final.detach().cpu(), mr_mask.detach().cpu(), activation=None).cpu().item()
            dices2.append(Liver_dice2)
            
            mr_d11_temp=mr_d1_temp.detach().cpu()
            mr_d11_temp = torch.sigmoid(mr_d11_temp)
            mr_d11_temp[mr_d11_temp < 0.5] = 0
            mr_d11_temp[mr_d11_temp > 0.5] = 1
            Liver_dice3 = diceCoeffv2(mr_d11_temp.detach().cpu(), mr_mask.detach().cpu(), activation=None).cpu().item()
            dices3.append(Liver_dice3)
            string_print = "E=%d disd1=%.4f disd1f=%.4f disd1en=%.4f  pmasklosf=%.4f pamrlos=%.4f tplos=%.4f masklos=%.4f masklosf=%.4f masklost=%.4f"\
                           % (epoch, distance_d1,distance_d1_final,distance_d1en,pseudo_mask_loss_final.cpu().item(),pamr_mask_loss.cpu().item(),temp_mask_loss.cpu().item(),
                              mr_loss.cpu().item(),mr_loss_final.cpu().item(),mr_loss_temp.cpu().item())            
           
            
            print("\r"+string_print,end = "",flush=True)         
            """

        # evaluation
        dsc_list = numpy.zeros((len(val_data), num_classes - 1), numpy.float32)
        with torch.no_grad():
            for i in range(len(val_data)):
                pred = common_net.produce_results(device, lambda x: final_model(x)[3].softmax(1).unsqueeze(2),
                                                  [patch_shape, ], [val_data[i], ], data_shape=val_data[i].shape,
                                                  patch_shape=patch_shape, is_seg=True, num_classes=num_classes)
                pred = pred.argmax(0).astype(numpy.float32)
                dsc_list[i] = common_metrics.calc_multi_dice(pred, val_label[i], num_cls=num_classes)

        if dsc_list.mean() > best_dsc:
            best_dsc = dsc_list.mean()
            torch.save(final_model.state_dict(), os.path.join(args.checkpoint_dir, 'best.pth'))

        print("%s  Epoch:%d/%d  val_dsc:%f/%f  best_dsc:%f" %
              (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), epoch, args.num_epoches, dsc_list.mean(), dsc_list.std(), best_dsc))
        torch.save(final_model.state_dict(), os.path.join(args.checkpoint_dir, 'last.pth'))

        sys.stdout.flush()

        """
        print('\ntaking snapshot ...')
        print('exp =', 'model')
        
        torch.save(temp_model,
                   osp.join('./temp_model', f'temp_model_MR1_{epoch}.pth'))
        torch.save(final_model,
                   osp.join('./final_model', f'final_model_MR1_{epoch}.pth'))
        torch.save(student_model,
                   osp.join('./student_model', f'student_model_MR1_{epoch}.pth'))
        Liver_dice_average1 = np.mean(dices1)
        print('Train Liver Dice1 {:.4}'.format(Liver_dice_average1))
            
        Liver_dice_average2 = np.mean(dices2)
        print('Train Liver Dice2 {:.4}'.format(Liver_dice_average2))       
            
        Liver_dice_average3 = np.mean(dices3)
        print('Train Liver Dice3 {:.4}'.format(Liver_dice_average3))     
        """
    file_handle.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu', type=int, default=0, help="gpu device id")
    parser.add_argument('--data_dir', type=str, default='/home/chenxu/datasets/pelvic/h5_data', help='path of the dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='', help='checkpoint_dir')
    parser.add_argument('--pretrained_ckpt', type=str, default='', help='pretrained model file')
    parser.add_argument('--task', type=str, default='pelvic', choices=["pelvic", ], help='task')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--num_epoches', type=int, default=100, help='total epoches')
    parser.add_argument('--debug', type=int, default=0, help='debug flag')
    args = parser.parse_args()

    if args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = torch.device("cuda")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        device = torch.device("cpu")

    main(device, args)
