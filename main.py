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
from models import __models__, model_loss#, get_smooth_loss
from utils import *
from torch.utils.data import DataLoader
import gc
from PIL import Image
from datasets.SpikeDataset import SpikeDataset
from datasets.Spikeset import JYSpikeDataset
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from AWLoss import AutomaticWeightedLoss
#from models.loss import OrdinalRegressionLoss
from models.dorn import OrdinalRegressionLoss

cudnn.benchmark = True
#os.environ["CUDA_VISIBLE_DEVICES"]='1,2,3,4,5,6,7'

parser = argparse.ArgumentParser(description='Cascade and Fused Cost Volume for Robust Stereo Matching(CFNet)')
parser.add_argument('--model', default='cfnet', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=160, help='maximum disparity')

parser.add_argument('--dataset', default='kitti', required=False, help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', default='/home/lijianing/kitti/', required=False, help='data path')
parser.add_argument('--trainlist', default='/home/lijianing/depth/CFNet-mod/filenames/kitti15_train.txt', required=False, help='training list')
parser.add_argument('--testlist', default='/home/lijianing/depth/CFNet-mod/filenames/kitti15_val.txt', required=False, help='testing list')

parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=2, help='testing batch size')
parser.add_argument('--epochs', type=int, default=150, required=False, help='number of epochs to train')
parser.add_argument('--lrepochs', type=str, default='35:3',required=False, help='the epochs to decay lr: the downscale rate')

parser.add_argument('--logdir', default='/home/lijianing/depth/MMlogs/256/cfnet', required=False, help='the directory to save logs and checkpoints')
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
 

StereoDataset = __datasets__[args.dataset]
train_dataset =  SpikeDataset(pathr = "/home/Datadisk/spikedata5622/spiking-2022/train/firsthright/", pathl = "/home/Datadisk/spikedata5622/spiking-2022/train/firsthleft/", mode = "training")   
#train_dataset =  JYSpikeDataset(path="/home/Datadisk/spikedata5622/spiking-2022/JYsplit/train/")


test_dataset = SpikeDataset(pathr = "/home/Datadisk/spikedata5622/spiking-2022/test/nrpz/", pathl = "/home/Datadisk/spikedata5622/spiking-2022/test/nlpz/", mode = "test")

TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=2, drop_last=True)
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=4, drop_last=False)


from models.fusionet import fusionet, fusionet1, fusionet3
from models.monocular_way import UNet_KDD
# model, optimizer
#model = __models__[args.model](args.maxdisp)
#model = fusionet(maxdisp=32, batchsize=2)#)
#model = fusionet1(maxdisp=32)
model = __models__[args.model](args.maxdisp)

device = torch.device("cuda:{}".format(2))

model = model.to(device)
'''
if args.n_gpu > 1:
    model = torch.nn.DataParallel(model, device_ids=[4,5,6,7])
'''
with torch.cuda.device([2,3]):
    model = nn.DataParallel(model, [2,3]) 
#model.cuda()

'''
optimizer = optim.Adam([
                {'params': model.parameters()},
                {'params': awl.parameters(), 'weight_decay': 0}
            ],lr=args.lr, betas=(0.9, 0.999))
'''
            
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
#state_dict = torch.load('/home/lijianing/depth/CFNet-mod/logs3/checkpoint_max_JY_3.ckpt')
#model.load_state_dict(state_dict['model'])

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
            checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(checkpoint_data, "{}/checkpoint_max_cfnetres_uncertain.ckpt".format(args.logdir))
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
                #save_images(logger, 'test', image_outputs, global_step)
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
def train_sample(epoch, sample, compute_metrics=False):
    model.train()

    imgL, imgR, disp_gt, depth_gt, sid_gt, depth_dorn = sample['left'], sample['right'], sample['disparity'], sample['depth'], sample['SID'], sample['depth_dorn']
    imgL = imgL.to(device)
    imgR = imgR.to(device)
    disp_gt = disp_gt.to(device)
    depth_gt = depth_gt.to(device)
    sid_gt = sid_gt.to(device)

    stereo_gt = disp_gt[0].detach().cpu().squeeze(0)
        
    stereo_gt_np = np.array(stereo_gt, dtype = np.float32)
    stereo_gt_img = Image.fromarray(stereo_gt_np, 'L')
    stereo_gt_img.save('/home/lijianing/depth/CFNet-mod/gt{}.png'.format('1'))  
 

    optimizer.zero_grad()

    ests = model(imgL, imgR)
    disp_ests = ests['stereo']
    ####ord_prob, ord_label = ests['monocular']
    depth_ests = ests['monocular']
    #print(depth_ests.size())
    #fusion_ests = ests['fusion']
    
    #stereo_depth_ests = 1 / (disp_ests[-1])  
    
    
    #stereo_depth_ests = 1 / (disp_ests[-1]) 
    #mono_depth4sim = (depth_ests + 1.0) * 128.0#.unsqueeze(1)
    
    #unc = torch.mean(torch.abs(stereo_depth_ests - mono_depth4sim)/255)
    
 
    #stereo_depth4sim = stereo_depth_ests.unsqueeze(1)
    
    #mono_depth4sim = depth_ests.unsqueeze(1)
    
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    ######loss1 = model_loss(disp_ests, disp_gt, mask)

    #disp_ests = disp_ests.unsqueeze(1)
    
    #print(disp_ests.size())
    #depth_ests = depth_ests.unsqueeze(1)
    
    #disp_ests = F.upsample(disp_ests, [256,512], mode='bilinear', align_corners=True).squeeze(1)
    #depth_ests = F.upsample(depth_ests, [256,512], mode='bilinear', align_corners=True).squeeze(1)
    #print(disp_ests.size())
    
    loss1 = model_loss(disp_ests, disp_gt, mask)
    
    #loss_ORD = OrdinalRegressionLoss(ord_num=256, beta=80.0, discretization="SID")
    #loss_ord = loss_ORD(ord_prob, sid_gt)
    
    loss2 = F.smooth_l1_loss(depth_ests, depth_gt)
    '''
    ORD_NUM = 256
    BETA = 256
    ord_loss = OrdinalRegressionLoss(ord_num=ORD_NUM, beta=BETA)
    loss_cls = nn.CrossEntropyLoss()
    loss2 = ord_loss(ord_prob, depth_dorn)
    '''
    #F.cross_entropy(depth_ests.long(), sid_gt.long()) 
    #loss2 = loss_cls(depth_ests, sid_gt.long())
    #loss2 = F.mse_loss(depth_ests, depth_gt) + F.smooth_l1_loss(depth_ests, depth_gt)
    

    
    
    if epoch <= 200:
        loss = loss1 + loss2
        
        #loss_ord
        #+ loss2 + loss_ord
    #else:
    #    loss = (loss1 + loss2)



    scalar_outputs = {"loss": loss}
    image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}
    '''
    if compute_metrics:
        with torch.no_grad():
            #image_outputs["errormap"] = [disp_error_image_func()(disp_est, disp_gt) for disp_est in disp_ests]
            scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
            scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
            scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
            scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
            scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]
          
    '''    
    loss.backward()
    optimizer.step()

    return tensor2float(loss), tensor2float(scalar_outputs)#, image_outputs


def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + gra

# test one sample
@make_nograd_func
def test_sample(sample, compute_metrics=True): 
    model.eval()

    imgL, imgR, disp_gt, depth_gt = sample['left'], sample['right'], sample['disparity'], sample['depth']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()
    depth_gt = depth_gt.cuda()
    '''
    ests = model(imgL, imgR)
    disp_ests = ests['stereo']
    depth_ests = ests['monocular']
    '''
    #disp_ests, pred_s3, pred_s4, pred_depth = model(imgL, imgR)
    pred_depth,_,_,_,_ = model(imgL, imgR)["monocular"]
    
    #pred_depth = F.upsample(pred_depth, [256,512], mode='bilinear', align_corners=True).squeeze(1)
    
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    #loss = F.smooth_l1_loss(pred_depth[-1], depth_gt)#model_loss(disp_ests, disp_gt, mask)
    loss = F.smooth_l1_loss(pred_depth, depth_gt)
    scalar_outputs = {"loss": loss}
    
    #image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}
   
    scalar_outputs["D1"] = [D1_metric(100.0*(pred_depth + 1.0) , 100.0*(depth_gt + 1.0), mask)] #for disp_est in disp_ests]
    # scalar_outputs["D1_pred_s2"] = [D1_metric(pred, disp_gt, mask) for pred in pred_s2]
    #scalar_outputs["D1_pred_s3"] = [D1_metric(pred, disp_gt, mask) for pred in pred_s3]
    #scalar_outputs["D1_pred_s4"] = [D1_metric(pred, disp_gt, mask) for pred in pred_s4]
    #scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    #scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
    #scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
    #scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]

    #if compute_metrics:
        #image_outputs["errormap"] = [disp_error_image_func()(disp_est, disp_gt) for disp_est in disp_ests]
    
    return tensor2float(loss), tensor2float(scalar_outputs)#, image_outputs


if __name__ == '__main__':
    train()
