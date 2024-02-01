# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 13:34:34 2020

@author: 11627
"""
# train.py
import time
import os
import sys
import pdb
import numpy
import argparse
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

# from datasets import Liver
from utils.loss import SoftDiceLoss,entropy_loss
#from utils import tools
from utils.metrics import diceCoeffv2
import utils.joint_transforms as joint_transforms
import utils.transforms as extended_transforms
from networks.att_u_net import AttU_Net
#from datasets import liver
import torch.nn.functional as F
import torch
import platform
from datetime import datetime

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


#crop_size = 128
batch_size = 8
img_ch = 1
n_epoch = 100
model_name = 'AttSS_Net'
loss_name = 'dice_'
times = 'no1_'
extra_description = ''
n_class=1

def main(device, args):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    if args.task == 'pelvic':
        num_classes = common_pelvic.NUM_CLASSES
        dataset = common_pelvic.Dataset(args.data_dir, "ct", n_slices=img_ch, debug=args.debug)
        val_data, _, val_label, _ = common_pelvic.load_val_data(args.data_dir)
    else:
        raise NotImplementedError(config['Data_input']['dataset'])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                             drop_last=True, num_workers=NUM_WORKERS)

    net = AttU_Net(img_ch=img_ch, num_classes=num_classes)
    net.train()
    net.to(device)

    if loss_name == 'dice_':
        criterion = SoftDiceLoss(activation='sigmoid').cuda()
#    elif loss_name == 'bce_':
#        criterion = nn.BCEWithLogitsLoss().cuda()
#    elif loss_name == 'wbce_':
#        criterion = WeightedBCELossWithSigmoid().cuda()
#    elif loss_name == 'er_':
#        criterion = EdgeRefinementLoss().cuda()

    best_dsc = 0
    patch_shape = (img_ch, val_data[0].shape[1], val_data[0].shape[2])
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    loss_record=[]
    for epoch in range(1, args.num_epoches + 1):
        l_dice = 0.0
        d_len = 0

        for data in dataloader:
#            net.train()
            X = data["image"].to(device)
            y = torch.nn.functional.one_hot(data["label"], num_classes).permute((0, 4, 1, 2, 3)).to(device)
            optimizer.zero_grad()
            _,_,_,output = net(X)

            loss = criterion(output.unsqueeze(2), y)
            # CrossEntropyLoss
            # loss = criterion(output, torch.argmax(y, dim=1))
            output = torch.sigmoid(output)

            output[output < 0.5] = 0
            output[output > 0.5] = 1
            Liver_dice = diceCoeffv2(output, y, activation=None).cpu().item()
            d_len += 1
            l_dice += Liver_dice

            loss.backward()
            optimizer.step()
            loss_record.append(loss.item())

        #lrs=0.9*lrs
        l_dice = l_dice / d_len
        msg = '{} Epoch {}/{},Train Liver Dice {:.4}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), epoch, args.num_epoches, l_dice)

        net.eval()
        dsc_list = numpy.zeros((len(val_data), num_classes - 1), numpy.float32)
        with torch.no_grad():
            for i in range(len(val_data)):
                pred = common_net.produce_results(device, lambda x: net(x)[3].softmax(1).unsqueeze(2), [patch_shape, ],
                                                  [val_data[i], ], data_shape=val_data[i].shape, patch_shape=patch_shape,
                                                  is_seg=True, num_classes=num_classes)
                pred = pred.argmax(0).astype(numpy.float32)
                dsc_list[i] = common_metrics.calc_multi_dice(pred, val_label[i], num_cls=num_classes)

        net.train()
        if dsc_list.mean() > best_dsc:
            best_dsc = dsc_list.mean()
            torch.save(net.state_dict(), os.path.join(args.checkpoint_dir, 'best.pth'))

        msg += "  val_dsc:%f/%f  best_dsc:%f" % (dsc_list.mean(), dsc_list.std(), best_dsc)
        print(msg)
        torch.save(net.state_dict(), os.path.join(args.checkpoint_dir, 'last.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu', type=int, default=0, help="gpu device id")
    parser.add_argument('--data_dir', type=str, default='/home/chenxu/datasets/pelvic/h5_data', help='path of the dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='', help='checkpoint_dir')
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
