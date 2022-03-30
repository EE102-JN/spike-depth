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
from tqdm import tqdm
import cv2

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Cascade and Fused Cost Volume for Robust Stereo Matching(CFNet)')
parser.add_argument('--model', default='cfnet', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')

parser.add_argument('--dataset', default='kitti', required=False, help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', default='/home/lijianing/kitti/', required=False, help='data path')
parser.add_argument('--trainlist', default='/home/lijianing/depth/CFNet-mod/filenames/kitti15_train.txt', required=False, help='training list')
parser.add_argument('--testlist', default='/home/lijianing/depth/CFNet-mod/filenames/kitti15_val.txt', required=False, help='testing list')

parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--epochs', type=int, default=150, required=False, help='number of epochs to train')
parser.add_argument('--lrepochs', type=str, default='50:5',required=False, help='the epochs to decay lr: the downscale rate')

parser.add_argument('--logdir', default='/home/lijianing/depth/MMlogs/256/cfnet', required=False, help='the directory to save logs and checkpoints')
parser.add_argument('--loadckpt', help='load the weights from a specific checkpoint')
parser.add_argument('--resume', action='store_true', help='continue training the model')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--summary_freq', type=int, default=1, help='the frequency of saving summary')
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
from datasets.SpikeDataset import SpikeDataset, SpikeRGBDataset
from datasets.Spikeset import JYSpikeDataset
#test_dataset = JYSpikeDataset(path="/home/Datadisk/spikedata5622/spiking-2022/JYsplit/train/") 
test_dataset = SpikeDataset(pathr = "/home/Datadisk/spikedata5622/spiking-2022/test/nrpz/", pathl = "/home/Datadisk/spikedata5622/spiking-2022/test/nlpz/", mode = "test")   
#test_dataset =  SpikeRGBDataset(path_rgb = "/home/lijianing/depth/left_rgb/", path_spike = "/home/Datadisk/spikedata5622/spiking-2022/train/firsthleft/", mode = "training")
#test_dataset = SpikeDataset(pathr = "/home/Datadisk/spikedata5622/spiking-2022/train/firsthright/", pathl = "/home/Datadisk/spikedata5622/spiking-2022/train/firsthleft/", mode = "training")    
train_dataset = StereoDataset("/home/lijianing/kitti/", args.trainlist, True)
#test_dataset = StereoDataset("/home/lijianing/kitti/", args.testlist, False)
TrainImgLoader = DataLoader(train_dataset, 1, shuffle=True, num_workers=2, drop_last=True)
TestImgLoader = DataLoader(test_dataset, 1, shuffle=False, num_workers=2, drop_last=False)



from models.stereonet import StereoNet
from models.psmnet import PSMNet
from models.monocular_way import UNet, UNet_P, UNet_KD, UNet_KD1
from models.dorn import DORN


#model = __models__[args.model](args.maxdisp)
'''
model = StereoNet(1, "subtract", 32)
model = PSMNet(256)
'''
#model = nn.DataParallel(model)

model = UNet(n_channels=32, n_classes=1, bilinear=True)
model = UNet_KD(n_channels=32, n_classes=1, bilinear=True)
model = DORN()

device = torch.device("cuda:{}".format(6))

model.to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))



def test():
    bestepoch = 0
    error = 100
    avg_test_scalars = AverageMeterDict()
    #avg_test_scalars = AverageMeterDict()
    for epoch_idx in range(0,1):
        #bestepoch = 0
        #error = 100
        for batch_idx, sample in enumerate(TestImgLoader):
            global_step = len(TestImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = test_sample(sample, compute_metrics=do_summary)
            if do_summary:
                save_scalars(logger, 'test', scalar_outputs, global_step)
                save_images(logger, 'test', image_outputs, global_step)
            avg_test_scalars.update(scalar_outputs)
            
            #transfer to images                                                                             
            stereo_result = image_outputs["disp_est"][0].detach().cpu().squeeze(0)
            stereo_np = np.array(255.0*np.array(stereo_result, dtype = np.float32), dtype = np.uint8)
            stereo_img = Image.fromarray(stereo_np, 'L')
            stereo_img.save('/home/lijianing/depth/CFNet-mod/results/{}.png'.format(batch_idx))
            
            stereo_gt = image_outputs["disp_gt"].detach().cpu().squeeze(0)
            print(stereo_gt.size())
            stereo_gt_np = np.array(256*stereo_gt, dtype = np.uint8)
            stereo_gt_img = Image.fromarray(stereo_gt_np, 'L')
            stereo_gt_img.save('/home/lijianing/depth/CFNet-mod/results/gt{}.png'.format(batch_idx))   
                     
            del scalar_outputs, image_outputs
            print('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch_idx, args.epochs,
                                                                                     batch_idx,
                                                                                     len(TestImgLoader), loss,
                                                                                     time.time() - start_time))
                                                                             
        avg_test_scalars = avg_test_scalars.mean()
        nowerror = avg_test_scalars["D1"][0]
        nowdeptherror = avg_test_scalars["Depth_D1"][0]
        if  nowerror < error :
            bestepoch = epoch_idx
            error = avg_test_scalars["D1"][0]
            depth_error = avg_test_scalars["Depth_D1"][0]
        save_scalars(logger, 'fulltest', avg_test_scalars, len(TrainImgLoader) * (epoch_idx + 1))
        print("avg_test_scalars", avg_test_scalars)
        print("avg_test_scalars_depth", depth_error)
        print('MAX epoch %d total test error = %.5f' % (bestepoch, error))
        gc.collect()
    print('MAX epoch %d total test error = %.5f' % (bestepoch, error))


        



# test one sample
@make_nograd_func
def test_sample(sample, compute_metrics=True):
    model.eval()

    imgL, imgR, disp_gt, depth_gt = sample['left'], sample['right'], sample['disparity'], sample['depth']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()
    depth_gt = depth_gt.cuda()
    #depth_gt = torch.unsqueeze(1)
    
    #print(disp_gt.size(), depth_gt.size())
    
    disp_ests, pred_s3, pred_s4, pred_depth = model(imgL, imgR)
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    loss = model_loss(disp_ests, disp_gt, mask)
    #print(torch.max(disp_gt))
    scalar_outputs = {"loss": loss}
    image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR, "depth_est": pred_depth}
    
    scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    #scalar_outputs["Depth_D1"] = [D1_metric(pred_depth, depth_gt, mask)]
    # scalar_outputs["D1_pred_s2"] = [D1_metric(pred, disp_gt, mask) for pred in pred_s2]
    scalar_outputs["D1_pred_s3"] = [D1_metric(pred, disp_gt, mask) for pred in pred_s3]
    scalar_outputs["D1_pred_s4"] = [D1_metric(pred, disp_gt, mask) for pred in pred_s4]
    scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
    scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
    scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]
    
    #compute depth
    DE = torch.abs(depth_gt - pred_depth[0])
    DE = torch.mean(DE)
    scalar_outputs["Depth_D1"] = [DE]
    
    #if compute_metrics:
        #image_outputs["errormap"] = [disp_error_image_func()(disp_est, disp_gt) for disp_est in disp_ests]

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


def test_spike():
    state_dict = torch.load('/home/lijianing/depth/CFNet-mod/logs_distill/single_spike/69checkpoint_max_student_single_unet.ckpt')
    model.load_state_dict(state_dict['model'])
    model.eval()
    #model.eval()
    errors = {"abs_rel_":0, "sq_rel_":0, "rmse_":0, "rmse_log_":0, "a1_":0, "a2_":0,
    "abs_rel":0, "sq_rel":0, "rmse":0, "rmse_log":0, "a1":0, "a2":0}
    n = 0
    length = len(test_dataset)
    for sample in tqdm(TestImgLoader):

        imgL, imgR, disp_gt, depth_gt = sample['left'], sample['right'], sample['disparity'], sample['depth']

        imgL = imgL.cuda()
        imgR = imgR.cuda()
        disp_gt = disp_gt.cuda()
        depth_gt = depth_gt.cuda()
        
        #disp_ests, pred_s3, pred_s4, pred_depth, depth_fusion = model(imgL, imgR)
        #pred = model(imgL, imgR)#["fusion"]
        pred_depth = model(imgL)
        #disp_ests, pred_s3, pred_s4, pred_depth = model(imgL, imgR)
        #disp1, disp2, disp3 = model(imgL, imgR)
        #######disp_ests = model(imgL, imgR)
        #print(disp_ests)
        #pred_depth = disp3######disp_ests["stereo"]
        #disp_ests = disp3#########disp_ests["stereo"]
       
        #pred_depth = pred["fusion"]
        #disp_ests = pred["stereo"]
        #pred_depth = pred["monocular"]
        #pred_depth = disp_ests
        #print(disp_ests, disp_gt)
        pred_depth = (1/(256.0*disp_ests))
        
        #####disp_ests = np.array(disp_ests[-1].detach().cpu(), dtype = np.float32).squeeze(0)
        disp_ests = np.array(disp_ests.detach().cpu(), dtype = np.float32).squeeze(0)
        pred_depth_ = np.array(pred_depth.detach().cpu(), dtype = np.float32).squeeze(0)
        depth_gt_ = np.array(depth_gt.detach().cpu(), dtype = np.float32).squeeze(0)
        disp_gt = np.array(disp_gt.detach().cpu(), dtype = np.float32).squeeze(0)
        '''
        disp_ests = np.array(disp_ests.cpu().detach(), dtype = np.float32).squeeze(0)
        pred_depth_ = np.array(pred_depth.cpu().detach(), dtype = np.float32).squeeze(0)
        depth_gt_ = np.array(depth_gt.cpu().detach(), dtype = np.float32).squeeze(0)
        disp_gt = np.array(disp_gt.cpu().detach(), dtype = np.float32).squeeze(0)
        '''
        #print(pred_depth)
        #print(depth_gt)
        #print(disp_ests)
        #print(disp_gt)
        
        #abs_rel_, sq_rel_, rmse_, rmse_log_, a1_, a2_ = compute_errors(1/(256*disp_gt), 1/(256*disp_ests))
        abs_rel_, sq_rel_, rmse_, rmse_log_, a1_, a2_ = compute_errors(disp_gt, disp_ests)
        abs_rel, sq_rel, rmse, rmse_log, a1, a2 = compute_errors(depth_gt_, pred_depth_)           
        
        errors["abs_rel"] = errors["abs_rel"] + abs_rel
        errors["abs_rel_"] = errors["abs_rel_"] + abs_rel_
        errors["rmse"] = errors["rmse"] + rmse
        errors["rmse_"] = errors["rmse_"] + rmse_
        
        errors["sq_rel"] = errors["sq_rel"] + sq_rel
        errors["sq_rel_"] = errors["sq_rel_"] + sq_rel_
        errors["rmse_log"] = errors["rmse_log"] + rmse_log
        errors["rmse_log_"] = errors["rmse_log_"] + rmse_log_
        errors["a1"] = errors["a1"] + a1
        errors["a1_"] = errors["a1_"] + a1_
        errors["a2"] = errors["a2"] + a2
        errors["a2_"] = errors["a2_"] + a2_
        
    errors["abs_rel"] = errors["abs_rel"] / length
    errors["abs_rel_"] = errors["abs_rel_"] / length
    errors["rmse"] = errors["rmse"] / length
    errors["rmse_"] = errors["rmse_"] / length
        
    errors["sq_rel"] = errors["sq_rel"] / length
    errors["sq_rel_"] = errors["sq_rel_"] / length
    errors["rmse_log"] = errors["rmse_log"] / length
    errors["rmse_log_"] = errors["rmse_log_"] / length
    errors["a1"] = errors["a1"] / length
    errors["a1_"] = errors["a1_"] / length
    errors["a2"] = errors["a2"] / length
    errors["a2_"] = errors["a2_"] / length
    #errors["a3"] = errors["a3"] / length     
    #errors["a3_"] = errors["a3_"] / length        
    print("errors evaluate disparity:\n abs_rel: {}, rmse: {}, sq_rel: {}, rmse_log: {}, a1: {}, a2: {}\n errors evaluate depth:\n abs_rel: {}, rmse: {}, sq_rel: {}, rmse_log: {}, a1:{}, a2:{}".format(
          abs_rel_, rmse_, sq_rel_, rmse_log_, a1_, a2_, abs_rel, rmse, sq_rel, rmse_log, a1, a2))        
        
        

def test_distill():
    state_dict = torch.load('/home/lijianing/depth/MMlogs/256/unet/0checkpoint_max_student_single_unet.ckpt')
    model.load_state_dict(state_dict['model'])
    model.eval()
    #model.eval()
    errors = {"abs_rel_":0, "sq_rel_":0, "rmse_":0, "rmse_log_":0, "a1_":0, "a2_":0,
    "abs_rel":0, "sq_rel":0, "rmse":0, "rmse_log":0, "a1":0, "a2":0}
    n = 0
    length = len(test_dataset)
    for sample in tqdm(TestImgLoader): 

        imgL, imgR, disp_gt, depth_gt = sample['left'].cuda(), sample['right'].cuda(), sample['disparity'].cuda(), sample['depth'].cuda()
        #img_spike, img_rgb, disp_gt, depth_gt = sample['spike'].cuda(), sample['rgb'].cuda(), sample['disparity'].cuda(), sample['depth'].cuda()

   
        #disp_ests, pred_s3, pred_s4, pred_depth, depth_fusion = model(imgL, imgR)
        pred = model(imgL)#["fusion"]
        
        
        '''
        #disp_ests, pred_s3, pred_s4, pred_depth = model(imgL, imgR)
        disp1, disp2, disp3 = model(imgL, imgR)
        #######disp_ests = model(imgL, imgR)
        #print(disp_ests)
        pred_depth = disp3######disp_ests["stereo"]
        disp_ests = disp3#########disp_ests["stereo"]
        '''
        #pred_depth = pred["fusion"]
        #disp_ests = pred["stereo"]
        #pred_depth = pred["monocular"]
        #pred_depth = disp_ests
        #print(disp_ests, disp_gt)
        #pred_depth = 128.0 * (pred + 1.0)  #(1/(256.0*disp_ests))
        
        #####disp_ests = np.array(disp_ests[-1].detach().cpu(), dtype = np.float32).squeeze(0)
        pred_depth = pred
        pred_depth_ = np.array(pred_depth.detach().cpu(), dtype = np.float32)#.squeeze(0)
        disp_ests = pred_depth#np.array(disp_ests.detach().cpu(), dtype = np.float32).squeeze(0)
        depth_gt_ = np.array(depth_gt.detach().cpu(), dtype = np.float32).squeeze(0)
        disp_gt = np.array(disp_gt.detach().cpu(), dtype = np.float32).squeeze(0)
        '''
        disp_ests = np.array(disp_ests.cpu().detach(), dtype = np.float32).squeeze(0)
        pred_depth_ = np.array(pred_depth.cpu().detach(), dtype = np.float32).squeeze(0)
        depth_gt_ = np.array(depth_gt.cpu().detach(), dtype = np.float32).squeeze(0)
        disp_gt = np.array(disp_gt.cpu().detach(), dtype = np.float32).squeeze(0)
        '''
        #print(pred_depth)
        #print(depth_gt)
        #print(disp_ests)
        #print(disp_gt)
        
        #abs_rel_, sq_rel_, rmse_, rmse_log_, a1_, a2_ = compute_errors(1/(256*disp_gt), 1/(256*disp_ests))
        
        abs_rel, sq_rel, rmse, rmse_log, a1, a2 = compute_errors(depth_gt_, pred_depth_)           
        
        errors["abs_rel"] = errors["abs_rel"] + abs_rel

        errors["rmse"] = errors["rmse"] + rmse
 
        
        errors["sq_rel"] = errors["sq_rel"] + sq_rel

        errors["rmse_log"] = errors["rmse_log"] + rmse_log

        errors["a1"] = errors["a1"] + a1

        errors["a2"] = errors["a2"] + a2
   
        
    errors["abs_rel"] = errors["abs_rel"] / length

    errors["rmse"] = errors["rmse"] / length

        
    errors["sq_rel"] = errors["sq_rel"] / length

    errors["rmse_log"] = errors["rmse_log"] / length

    errors["a1"] = errors["a1"] / length

    errors["a2"] = errors["a2"] / length
 
    #errors["a3"] = errors["a3"] / length    
    #errors["a3_"] = errors["a3_"] / length        
    print("errors evaluate depth:\n abs_rel: {}, rmse: {}, sq_rel: {}, rmse_log: {}, a1:{}, a2:{}".format(
          abs_rel, rmse, sq_rel, rmse_log, a1, a2))        
        
            
def test_dorn():
    state_dict = torch.load('/home/lijianing/depth/MMlogs/256/dorn/checkpoint_max_student_single_unet.ckpt')
    model.load_state_dict(state_dict['model'])
    model.eval()
    #model.eval()
    errors = {"abs_rel_":0, "sq_rel_":0, "rmse_":0, "rmse_log_":0, "a1_":0, "a2_":0,
    "abs_rel":0, "sq_rel":0, "rmse":0, "rmse_log":0, "a1":0, "a2":0}
    n = 0
    length = len(test_dataset)
    for sample in tqdm(TestImgLoader): 
    
        imgL, imgR, disp_gt, depth_gt, depth_dorn, seq_ldorn = sample['left'].to(device), sample['right'].to(device), sample['disparity'].to(device), sample['depth'].to(device), sample['depth_dorn'].to(device), sample['seq_ldorn'].to(device)
   
        #disp_ests, pred_s3, pred_s4, pred_depth, depth_fusion = model(imgL, imgR)
        prob, label = model(imgL)#["fusion"]
        print(prob.size(), label.size())
        


        ord_num = 256
        beta = 256        
        gamma = 1.0
        discretization = "SID"
        
        if discretization == "SID":
            t0 = torch.exp(np.log(beta) * label / ord_num)
            t1 = torch.exp(np.log(beta) * (label + 1) / ord_num)
        else:
            t0 = 1.0 + (beta - 1.0) * label / ord_num
            t1 = 1.0 + (beta - 1.0) * (label + 1) / ord_num
        depth = (t0 + t1) / 2 - gamma

        depth = np.array(depth.detach().cpu(), dtype = np.float32).squeeze(0)
        depth_dorn = np.array(depth_dorn.detach().cpu(), dtype = np.float32).squeeze(0)
        #t1 = np.array(t1.detach().cpu(), dtype = np.float32).squeeze(0)
        
        
        abs_rel, sq_rel, rmse, rmse_log, a1, a2 = compute_errors_dorn(depth, depth_dorn)           
        
        errors["abs_rel"] = errors["abs_rel"] + abs_rel

        errors["rmse"] = errors["rmse"] + rmse
 
        
        errors["sq_rel"] = errors["sq_rel"] + sq_rel

        errors["rmse_log"] = errors["rmse_log"] + rmse_log

        errors["a1"] = errors["a1"] + a1

        errors["a2"] = errors["a2"] + a2
   
        
    errors["abs_rel"] = errors["abs_rel"] / length

    errors["rmse"] = errors["rmse"] / length

        
    errors["sq_rel"] = errors["sq_rel"] / length

    errors["rmse_log"] = errors["rmse_log"] / length

    errors["a1"] = errors["a1"] / length

    errors["a2"] = errors["a2"] / length
 
    #errors["a3"] = errors["a3"] / length    
    #errors["a3_"] = errors["a3_"] / length        
    print("errors evaluate depth:\n abs_rel: {}, rmse: {}, sq_rel: {}, rmse_log: {}, a1:{}, a2:{}".format(
          abs_rel, rmse, sq_rel, rmse_log, a1, a2))                

        
def compute_errors_dorn(gt, pred):
   
     
    #gt = (gt +1 ) *128
    #pred = (pred +1 )* 128
    #gt = (gt + 1.0) * 500.0
    #pred = (pred + 1.0) * 500.0
    #pred = (pred +1) * 128.0
    #gt = (gt+1) * 128.0
    print(pred)
    print(gt)
    pred[ pred> 255.0] = 255.0
    pred[ pred< 2.0] = 2.0
    
    gt[gt>255.0] = 255.0
    gt[gt<2.0] = 2.0
    
    #gt[ gt >=100 ] = 0
    #pred[ pred >=100 ] = 0
    
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2,         
        
def compute_errors(gt, pred):
   
     
    gt = (gt +1 ) *128
    pred = (pred +1 )* 128

    #pred = (pred +1) * 128.0
    #gt = (gt+1) * 128.0
    print(pred)
    print(gt)
    pred[ pred> 255.0] = 255.0
    pred[ pred< 2.0] = 2.0
    
    gt[gt>255.0] = 255.0
    gt[gt<2.0] = 2.0
    
    #gt[ gt >=100 ] = 0
    #pred[ pred >=100 ] = 0
    
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2,       
            
def to_video():
    #device = "cuda:6" 
    #model = __models__['cfnet'](256)
    model = StereoNet(1, "subtract", 32)
    estimator = model.cuda()
    
    #estimator = torch.nn.DataParallel(estimator)
    estimator.load_state_dict(torch.load('/home/lijianing/depth/CFNet-mod/logs_base/checkpoint_max_sternew.ckpt')["model"])
    
    dataset = SpikeDataset(pathr = "/home/Datadisk/spikedata5622/spiking-2022/test/nrpz/", pathl = "/home/Datadisk/spikedata5622/spiking-2022/test/nlpz/", mode = "train")
    dataloader = DataLoader(dataset, 1, False, pin_memory=True)
    print(dataset.filesr)
    
    fps=20
    fourcc=cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter('base.mp4', cv2.VideoWriter_fourcc(*"mp4v"), fps, (512,256))
    
    for data in tqdm(dataloader):
        x_l, x_r, real_y, real_d = data["left"].cuda(), data["right"].cuda(), data["disparity"].cuda(), data["depth"].cuda()
        
        fake_y = estimator(x_l, x_r)["stereo"].squeeze(0)#["stereo"][-1]
        #print(fake_y.size())
        #y_map = fake_y[-1].detach().cpu()
        y_map = fake_y.detach().cpu()            

        #y_map = 1/y_map #128*np.array((y_map+1), dtype = np.float32)
        y_map = 1/np.array(y_map, dtype = np.float32)
        
        y_map[ y_map> 255] = 255
        y_map[ y_map< 0.3] = 0.3

        y_map = y_map.squeeze(0)      
        
        
        fig = Image.fromarray(y_map).convert('L')
        fig.save("fig.png")

        
        fig = cv2.cvtColor(np.asarray(fig),cv2.COLOR_GRAY2RGB) 
        fig = cv2.applyColorMap(fig, cv2.COLORMAP_MAGMA)
        video.write(fig)


def addvideo():
    
    INPUT_FILE1 = 'vfusion.mp4'
    INPUT_FILE2 = 'gt.mp4'
    OUTPUT_FILE = 'train_fusion.avi'

    reader1 = cv2.VideoCapture(INPUT_FILE1)
    reader2 = cv2.VideoCapture(INPUT_FILE2)
    width = int(reader1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(reader1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(OUTPUT_FILE,
                  cv2.VideoWriter_fourcc('I', '4', '2', '0'), # (*"mp4v") for mp4 output
                  20, # fps
                  (width, height//2)) # resolution
     
    print(reader1.isOpened())
    print(reader2.isOpened())
    have_more_frame = True
    c = 0
    while have_more_frame:
        have_more_frame, frame1 = reader1.read()
        _, frame2 = reader2.read()
        frame1 = cv2.resize(frame1, (width//2, height//2))
        frame2 = cv2.resize(frame2, (width//2, height//2))
        img = np.hstack((frame1, frame2))
        cv2.waitKey(1)
        writer.write(img)
        c += 1
        print(str(c) + ' is ok')
     
     
    writer.release()
    reader1.release()
    reader2.release()
    cv2.destroyAllWindows()


        
def gt_video():
    device = "cuda:0" 
    dataset = SpikeDataset(pathr = "/home/Datadisk/spikedata5622/spiking-2022/train/firsthright/", pathl = "/home/Datadisk/spikedata5622/spiking-2022/train/firsthleft/", mode = "train")
    dataloader = DataLoader(dataset, 1, False, pin_memory=True)
    
    fps=20
    fourcc=cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter('gt.mp4', cv2.VideoWriter_fourcc(*"mp4v"), fps, (512,256))
    for data in tqdm(dataloader):
        x_l, x_r, real_y, real_d = data["left"].to(device), data["right"].to(device), data["disparity"].to(device), data["depth"].to(device)
        y_map = real_d.detach().cpu()
        y_map = y_map.squeeze(0)
        y_map = 256*(np.array(y_map, dtype = np.float32))
        y_map[ y_map> 255] = 255

        fig = Image.fromarray(y_map).convert('L')
        #fig.save("fig.png")
  
        fig = cv2.cvtColor(np.asarray(fig),cv2.COLOR_GRAY2RGB) 
        fig = cv2.applyColorMap(fig, cv2.COLORMAP_MAGMA)
        video.write(fig)    
        


if __name__ == '__main__':
    #test()
    #test_spike()
    #test_distill()
    test_dorn()
    #to_video()
    #gt_video()
    #addvideo()


