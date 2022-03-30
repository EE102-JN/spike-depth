from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import __datasets__
from models import __models__, model_loss
from utils import *
from torch.utils.data import DataLoader
import gc
from PIL import Image
from datasets.SpikeDataset import SpikeDataset
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from AWLoss import AutomaticWeightedLoss
from models.loss import OrdinalRegressionLoss

cudnn.benchmark = True
#os.environ["CUDA_VISIBLE_DEVICES"]='1,2,3,4,5,6,7'

parser = argparse.ArgumentParser(description='Cascade and Fused Cost Volume for Robust Stereo Matching(CFNet)')
parser.add_argument('--model', default='cfnet', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=64, help='maximum disparity')

parser.add_argument('--dataset', default='kitti', required=False, help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', default='/home/lijianing/kitti/', required=False, help='data path')
parser.add_argument('--trainlist', default='/home/lijianing/depth/CFNet-mod/filenames/kitti15_train.txt', required=False, help='training list')
parser.add_argument('--testlist', default='/home/lijianing/depth/CFNet-mod/filenames/kitti15_val.txt', required=False, help='testing list')

parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
parser.add_argument('--batch_size', type=int, default=4, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=2, help='testing batch size')
parser.add_argument('--epochs', type=int, default=150, required=False, help='number of epochs to train')
parser.add_argument('--lrepochs', type=str, default='35:5',required=False, help='the epochs to decay lr: the downscale rate')

parser.add_argument('--logdir', default='/home/lijianing/depth/MMlogs/256/cfnet/', required=False, help='the directory to save logs and checkpoints')
parser.add_argument('--loadckpt', help='load the weights from a specific checkpoint') 
parser.add_argument('--resume', action='store_true', help='continue training the model')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--summary_freq', type=int, default=20, help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=1, help='the frequency of saving checkpoint')

# parse arguments, set seeds
args = parser.parse_args() 
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
os.makedirs(args.logdir, exist_ok=True)

# create summary logger
print("creating new summary file")
logger = SummaryWriter(args.logdir)

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
train_dataset =  SpikeDataset(pathr = "/home/Datadisk/spikedata5622/spiking-2022/train/firsthright/", pathl = "/home/Datadisk/spikedata5622/spiking-2022/train/firsthleft/", mode = "training")   
#StereoDataset("/home/lijianing/kitti/", args.trainlist, True)
test_dataset = SpikeDataset(pathr = "/home/Datadisk/spikedata5622/spiking-2022/test/nrpz/", pathl = "/home/Datadisk/spikedata5622/spiking-2022/test/nlpz/", mode = "test")    
#StereoDataset("/home/lijianing/kitti/", args.testlist, False)
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=8, drop_last=True)
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=4, drop_last=False)

# model, optimizer
#model = __models__[args.model](args.maxdisp)
#model = __models__["stereonet"](args.batch_size, "concat", args.maxdisp)
from models.loss import gcnet_loss, gwc_loss, psm_loss, SL1Loss, unc_loss
from models.fusionet import fusionet, fusionet1, fusionet3, fusionet4
from models.monocular_way import UNet_KD
from models.stereonet import StereoNet
from models.psmnet import PSMNet
from models.gcnet import GCNet
from models.gwcnet_ import GwcNet
#from models.casmvsnet import CascadeMVSNet

# model, optimizer
#model = __models__[args.model](args.maxdisp)
model = fusionet(maxdisp=32, batch_size=2)#)
model = StereoNet(maxdisp=32, cost_volume_method = "subtract",batch_size=2)
model = PSMNet(max_disp = 160)
#model = GCNet(maxdisp=32, cost_volume_method = "subtract", batch_size=2)
model = GwcNet(maxdisp=160, use_concat_volume=True)
#model = CascadeMVSNet(n_depths=[8, 32, 48])
#model = UNet_KD(n_channels=32, n_classes=1, bilinear=True)

#model = fusionet1(maxdisp=32)
#model = fusionet3(maxdisp = 32)
#model = fusionet4(maxdisp=32, batch_size=2)
model = __models__[args.model](160)


optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
#optimzer = torch.optim.SGD(model.parameters(), lr=args.lr)
#state_dict = torch.load('/home/lijianing/depth/CFNet-mod/logs_base/checkpoint_max_ster.ckpt')
#model.load_state_dict(state_dict['model'])

device = torch.device("cuda:{}".format(5))

with torch.cuda.device([5,7]):
    model = nn.DataParallel(model, [5,7]) 

model.to(device)
# load parameters
start_epoch = 0
if args.resume:
    # find all checkpoints file and sort according to epoch id
    all_saved_ckpts = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    all_saved_ckpts = sorted(all_saved_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the latest checkpoint file
    loadckpt = os.path.join(args.logdir, all_saved_ckpts[-1])
    print("loading the lastest model in logdir: {}".format(loadckpt))
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    # load the checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])
print("start at epoch {}".format(start_epoch))


def train():
    bestepoch = 0
    error = 100
    for epoch_idx in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch_idx, args.lr, args.lrepochs)

        # training
        for batch_idx, sample in enumerate(TrainImgLoader):
            
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs = train_sample(epoch_idx, sample, compute_metrics=do_summary) #, image_outputs 
            if do_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)
                #save_images(logger, 'train', image_outputs, global_step)
            del scalar_outputs#, image_outputs
            print('Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, args.epochs,
                                                                                       batch_idx,
                                                                                       len(TrainImgLoader), loss,
                                                                                       time.time() - start_time))
        # saving checkpoints
        if (epoch_idx + 1) % args.save_freq == 0:
            checkpoint_data = {'epoch': epoch_idx, 'model': model.module.state_dict(), 'optimizer': optimizer.state_dict()} #module.
            torch.save(checkpoint_data, "{}/checkpoint_max_fus_unc_3.30.ckpt".format(args.logdir))
        gc.collect()

        # testing
        '''
        avg_test_scalars = AverageMeterDict()
        #bestepoch = 0
        #error = 100
        for batch_idx, sample in enumerate(TestImgLoader):
            global_step = len(TestImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs = test_sample(sample, compute_metrics=do_summary)
            if do_summary:
                save_scalars(logger, 'test', scalar_outputs, global_step)
               
            avg_test_scalars.update(scalar_outputs)
            del scalar_outputs
            print('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch_idx, args.epochs,
                                                                                     batch_idx,
                                                                                     len(TestImgLoader), loss,
                                                                                     time.time() - start_time))
        avg_test_scalars = avg_test_scalars.mean()
        nowerror = avg_test_scalars["D1"][0]
        if  nowerror < error :
            bestepoch = epoch_idx
            error = avg_test_scalars["D1"][0]
        save_scalars(logger, 'fulltest', avg_test_scalars, len(TrainImgLoader) * (epoch_idx + 1))
        print("avg_test_scalars", avg_test_scalars)
        print('MAX epoch %d total test error = %.5f' % (bestepoch, error))
        gc.collect()
        '''
    print('MAX epoch %d total test error = %.5f' % (bestepoch, error))


# train one sample
def train_sample(epoch, sample, compute_metrics=True):
    model.train()
    
    imgL, imgR, disp_gt, depth_gt = sample['left'], sample['right'], sample['disparity'], sample['depth']
    imgL = imgL.to(device)
    imgR = imgR.to(device)
    disp_gt = disp_gt.to(device)#*256.0
    depth_gt = depth_gt.to(device)

    stereo_gt = disp_gt[0].detach().cpu().squeeze(0)
        
    stereo_gt_np = np.array(stereo_gt, dtype = np.float32)
    stereo_gt_img = Image.fromarray(stereo_gt_np, 'L')
    stereo_gt_img.save('/home/lijianing/depth/CFNet-mod/gt{}.png'.format('1'))  
 

    optimizer.zero_grad()
    optimizer.zero_grad()
    
    ests = model(imgL, imgR)
    disp_ests = ests['stereo']
   
    depth_ests = ests['monocular']

    #uncert_ests = ests['monocular']
    #ord_prob, ord_label = ests['monocular']


    
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    #mask = mask.unsqueeze(1)

    #loss1 = gwc_loss(disp_ests, disp_gt, mask)
    #loss1 = psm_loss(disp_ests, disp_gt, mask)
    loss1 = model_loss(disp_ests, disp_gt, mask)  #(for cfnet)
    '''
    loss_casmv = SL1Loss()
    loss1 = loss_casmv(disp_ests, depth_gt)
    '''

    #loss1 = F.smooth_l1_loss(disp_ests[-1], disp_gt)

    loss2 = F.smooth_l1_loss(depth_ests, depth_gt)
    #loss2 = 0.5*unc_loss(depth_ests, depth_gt) + F.smooth_l1_loss(depth_ests["depth"], depth_gt)
    
    '''
    criterion = OrdinalRegressionLoss(ord_num=256, beta=80.0, discretization="SID")
    #prob = regression_layer(feat)
    loss2 = criterion(ord_prob, ord_target)
    '''        


    if epoch <= 250:
        loss = loss1 + loss2

    #loss =  2.0*loss2 + 1.0*loss1 + 1.0*loss_fusion
    scalar_outputs = {"loss": loss}
    image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}

    
    loss.backward()
    optimizer.step()

    return tensor2float(loss), tensor2float(scalar_outputs)#, image_outputs


# test one sample
@make_nograd_func
def test_sample(sample, compute_metrics=True):
    model.eval()

    imgL, imgR, disp_gt, depth_gt = sample['left'], sample['right'], sample['disparity'], sample['depth']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()
    depth_gt = depth_gt.cuda()
    
    ests = model(imgL, imgR)
    
    disp_ests = ests['stereo'][-1]
    depth_ests,_,_,_,_ = ests['monocular']
    

    
    train_fusion = False
    train_mono = True
    train_ster = False#True

    if train_fusion:
    
        disp_depth = 1 / disp_ests
        depth_ests = 128.0 * (depth_ests + 1.0)
        fusion = (disp_depth + depth_ests) / 2.0

        
        mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
       
        loss = F.smooth_l1_loss(fusion, depth_gt)
        
        scalar_outputs = {"loss": loss}
        
    elif train_mono:
        
        disp_depth = None
        fusion = 128.0 * (depth_ests + 1.0)
  
        mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
       
        loss = F.smooth_l1_loss(fusion, depth_gt)#model_loss(disp_ests, disp_gt, mask)
        
        scalar_outputs = {"loss": loss}
                    
    elif train_ster:
    
        fusion = 1 / disp_ests
        depth_ests = None

        mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
       
        loss = F.smooth_l1_loss(fusion, depth_gt)
        
        scalar_outputs = {"loss": loss}    
        
        

    
    scalar_outputs["D1"] = [D1_metric(fusion, depth_gt, mask) for disp_est in disp_ests]

    
    return tensor2float(loss), tensor2float(scalar_outputs) 


if __name__ == '__main__':
    train()
