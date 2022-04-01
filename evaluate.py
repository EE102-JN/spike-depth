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
parser.add_argument('--maxdisp', type=int, default=160, help='maximum disparity')

parser.add_argument('--dataset', default='kitti', required=False, help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', default='/home/lijianing/kitti/', required=False, help='data path')
parser.add_argument('--trainlist', default='/home/lijianing/depth/CFNet-mod/filenames/kitti15_train.txt', required=False, help='training list')
parser.add_argument('--testlist', default='/home/lijianing/depth/CFNet-mod/filenames/kitti15_val.txt', required=False, help='testing list')

parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--epochs', type=int, default=150, required=False, help='number of epochs to train')
parser.add_argument('--lrepochs', type=str, default='50:5',required=False, help='the epochs to decay lr: the downscale rate')

parser.add_argument('--logdir', default='/home/lijianing/depth/CFNet-mod/logs', required=False, help='the directory to save logs and checkpoints')
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
from datasets.SpikeDataset import SpikeDataset
from datasets.Spikeset import JYSpikeDataset
#test_dataset = JYSpikeDataset(path="/home/Datadisk/spikedata5622/spiking-2022/JYsplit/validation/") 
test_dataset = SpikeDataset(pathr = "/home/Datadisk/spikedata5622/spiking-2022/test/nrpz/", pathl = "/home/Datadisk/spikedata5622/spiking-2022/test/nlpz/", mode = "test")    
#test_dataset = SpikeDataset(pathr = "/home/Datadisk/spikedata5622/spiking-2022/train/firsthright/", pathl = "/home/Datadisk/spikedata5622/spiking-2022/train/firsthleft/", mode = "training")    
train_dataset = StereoDataset("/home/lijianing/kitti/", args.trainlist, True)
#test_dataset = StereoDataset("/home/lijianing/kitti/", args.testlist, False)
TrainImgLoader = DataLoader(train_dataset, 1, shuffle=True, num_workers=1, drop_last=True)
TestImgLoader = DataLoader(test_dataset, 1, shuffle=False, num_workers=1, drop_last=False)



from models.uncertfusionet import SpikeFusionet
# model, optimizer

model = SpikeFusionet(max_disp=160)
device = torch.device("cuda:{}".format(2,7))
#model = nn.DataParallel(model)

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))


model.to(device)


from collections import OrderedDict

def test_spike():
    
    state_dict = torch.load('/home/lijianing/depth/MMlogs/256/ours/checkpoint_max_3.31.ckpt', map_location='cuda:2')
    '''
    new_state_dict = OrderedDict()
    for k, v in state_dict["model"].items():
        name = k[7:] 
        new_state_dict[name] = v

    
    '''
    model.load_state_dict(state_dict['model'])
    model.eval()
    #model.eval()
    errors = {"abs_rel_":0, "sq_rel_":0, "rmse_":0, "rmse_log_":0, "a1_":0, "a2_":0,
    "abs_rel":0, "sq_rel":0, "rmse":0, "rmse_log":0, "a1":0, "a2":0}
    n = 0
    length = len(test_dataset)
    for sample in tqdm(TestImgLoader):

        imgL, imgR, disp_gt, depth_gt = sample['left'], sample['right'], sample['disparity'], sample['depth']

        imgL = imgL.to(device)
        imgR = imgR.to(device)
        disp_gt = disp_gt.to(device)
        depth_gt = depth_gt.to(device)
        
        #disp_ests, pred_s3, pred_s4, pred_depth, depth_fusion = model(imgL, imgR)
        pred = model(imgL, imgR)#["fusion"]
        
        #disp_ests, pred_s3, pred_s4, pred_depth = model(imgL, imgR)
        
        disp_ests = model(imgL, imgR)["stereo"]
        depth_ests = model(imgL, imgR)["monocular"]#["depth"]
        #uncertainty_ests_mono = model(imgL, imgR)["monocular"]["uncertainty"]
        uncertainty_ests_ster = model(imgL, imgR)["stereo_uncertainty"]
        #print(uncertainty_ests_ster)
        #mask = np.zeros((256, 512), dtype = np.uint8)
        
        
        uncertainty_ests_mono = depth_ests["uncertainty"]
        pred_depth = depth_ests["depth"]
        disp_ests = disp_ests     

        uncertainty_ests_mono = np.array(uncertainty_ests_mono.detach().cpu(), dtype = np.float32).squeeze(0)
        disp_ests = np.array(1/disp_ests[-1].detach().cpu(), dtype = np.float32).squeeze(0)
        pred_depth_ = np.array(pred_depth.detach().cpu(), dtype = np.float32).squeeze(0)
        depth_gt_ = np.array(depth_gt.detach().cpu(), dtype = np.float32).squeeze(0)
        disp_gt = np.array(disp_gt.detach().cpu(), dtype = np.float32).squeeze(0)
     
        mask = np.zeros((256, 512), dtype=np.uint8)
        mask[ uncertainty_ests_mono > 0.005 ] =  0
        mask[ uncertainty_ests_mono <=0.005] = 1
        

        
        fusion =  mask* (pred_depth_ + 1.0)*128.0 + (1-mask) * disp_ests
        fusion = (fusion / 128.0) - 1.0
        pred_depth_ = fusion
       
        #pred_depth_ = 0.5*disp_ests + 0.5*(1+pred_depth_)*128.0    
        #pred_depth_ = (pred_depth_ / 128.0) - 1.0
        
        ##pred_depth_ = (1-mask)*disp_ests + mask*(1+pred_depth_)*128.0    
        
        #abs_rel_, sq_rel_, rmse_, rmse_log_, a1_, a2_ = compute_errors(1/(256*disp_gt), 1/(256*disp_ests))
        abs_rel_, sq_rel_, rmse_, rmse_log_, a1_, a2_ = compute_errors(depth_gt_, disp_ests)
        abs_rel, sq_rel, rmse, rmse_log, a1, a2 = compute_errors_(depth_gt_, pred_depth_)           
        
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
    
    abs_rel = errors["abs_rel"]
    sq_rel = errors["sq_rel"]
    rmse = errors["rmse"]
    rmse_log = errors["rmse_log"]
    a1 = errors["a1"]
    a2 = errors["a2"]
    
    abs_rel_ = errors["abs_rel_"]
    sq_rel_ = errors["sq_rel_"]
    rmse_ = errors["rmse_"]
    rmse_log_ = errors["rmse_log_"]
    a1_ = errors["a1_"]
    a2_ = errors["a2_"]    
    
       
    print("errors evaluate disparity:\n abs_rel: {}, rmse: {}, sq_rel: {}, rmse_log: {}, a1: {}, a2: {}\n errors evaluate depth:\n abs_rel: {}, rmse: {}, sq_rel: {}, rmse_log: {}, a1:{}, a2:{}".format(
          abs_rel_, rmse_, sq_rel_, rmse_log_, a1_, a2_, abs_rel, rmse, sq_rel, rmse_log, a1, a2))        
        
        
        
        
        
def compute_errors(gt, pred): # for disparity
   
     
    gt = (gt +1.0 ) *128.0

    
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
        
def compute_errors_(gt, pred): # for depth
   
     
    gt = (gt +1.0 ) * 128.0
    pred = (pred +1.0 )* 128.0
    
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
    
    state_dict = torch.load('/home/lijianing/depth/MMlogs/256/psmnet/checkpoint_max.ckpt')
    
    new_state_dict = OrderedDict()
    for k, v in state_dict["model"].items():
        name = k[7:] 
        new_state_dict[name] = v

    
    
    model.load_state_dict(new_state_dict)#['model'])
    model.eval()
    dataset = SpikeDataset(pathr = "/home/Datadisk/spikedata5622/spiking-2022/train/firsthright/", pathl = "/home/Datadisk/spikedata5622/spiking-2022/train/firsthleft/", mode = "training")
    #dataset = SpikeDataset(pathr = "/home/Datadisk/spikedata5622/spiking-2022/test/nrpz/", pathl = "/home/Datadisk/spikedata5622/spiking-2022/test/nlpz/", mode = "train")
    #dataset = JYSpikeDataset(path="/home/Datadisk/spikedata5622/spiking-2022/JYsplit/train/") 
    dataloader = DataLoader(dataset, 1, False, pin_memory=True)
    #print(dataset.filesr)
    
    fps=20
    fourcc=cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter('unc.mp4', cv2.VideoWriter_fourcc(*"mp4v"), fps, (512,256))
    
    for data in tqdm(dataloader):
        x_l, x_r, real_y, real_d = data["left"].to(device), data["right"].to(device), data["disparity"].to(device), data["depth"].to(device)
        
        #fake_y = model(x_l, x_r)["monocular"]["depth"]#["uncertainty"]#["stereo"][-1]
        fake_y = model(x_l, x_r)["stereo"][-1]
        
        #y_map = fake_y[-1].detach().cpu()
        y_map = fake_y.detach().cpu()            
        #print(y_map.max(), y_map.min())
        

        #y_map = 128.0*np.array((y_map+1), dtype = np.float32)
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
    #dataset = SpikeDataset(pathr = "/home/Datadisk/spikedata5622/spiking-2022/train/firsthright/", pathl = "/home/Datadisk/spikedata5622/spiking-2022/train/firsthleft/", mode = "train")
    
    dataset = JYSpikeDataset(path="/home/Datadisk/spikedata5622/spiking-2022/JYsplit/train/") 
    dataloader = DataLoader(dataset, 1, False, pin_memory=True)
    
    fps=50
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
        
def validate_spike(model, dataloader):
    '''
    state_dict = torch.load('/home/lijianing/depth/MMlogs/256/psmnet/checkpoint_max.ckpt')
    
    new_state_dict = OrderedDict()
    for k, v in state_dict["model"].items():
        name = k[7:] 
        new_state_dict[name] = v

    
    
    model.load_state_dict(new_state_dict)#['model'])
    '''
    model.eval()
    
    TestImgLoader = dataloader
    errors = {"abs_rel_":0, "sq_rel_":0, "rmse_":0, "rmse_log_":0, "a1_":0, "a2_":0,
    "abs_rel":0, "sq_rel":0, "rmse":0, "rmse_log":0, "a1":0, "a2":0}
    n = 0
    length = len(test_dataset)
    for sample in tqdm(TestImgLoader):

        imgL, imgR, disp_gt, depth_gt = sample['left'], sample['right'], sample['disparity'], sample['depth']

        imgL = imgL.to(device)
        imgR = imgR.to(device)
        disp_gt = disp_gt.to(device)
        depth_gt = depth_gt.to(device)
        

        pred = model(imgL, imgR)#["fusion"]
        
        disp_ests = model(imgL, imgR)["stereo"]
        depth_ests = model(imgL, imgR)["monocular"]["depth"]#["fusion"]["depth"]#
        uncertainty_ests = model(imgL, imgR)["monocular"]["uncertainty"]
        
        
        pred_depth = depth_ests
        disp_ests = disp_ests     

        uncertainty_ests = np.array(uncertainty_ests.detach().cpu(), dtype = np.float32)#.squeeze(0)
        disp_ests = np.array(1/disp_ests[-1].detach().cpu(), dtype = np.float32)#.squeeze(0)
        pred_depth_ = np.array(pred_depth.detach().cpu(), dtype = np.float32)#.squeeze(0)
        depth_gt_ = np.array(depth_gt.detach().cpu(), dtype = np.float32)#.squeeze(0)
        disp_gt = np.array(disp_gt.detach().cpu(), dtype = np.float32)#.squeeze(0)
        
        #mask = np.zeros((2,256, 512), dtype=np.uint8)
        '''
        uncertainty_ests[uncertainty_ests > 0.5] = 0
        uncertainty_ests[uncertainty_ests <= 0.5] = 1
        
        mask = uncertainty_ests
        '''
        #pred_depth_ = (1-mask)*disp_ests + mask*(1+pred_depth_)*128.0    
        
        pred_depth_ = (pred_depth_ + 1.0) * 128.0  
        
        abs_rel_, sq_rel_, rmse_, rmse_log_, a1_, a2_ = validate_errors(depth_gt_, disp_ests)
        abs_rel, sq_rel, rmse, rmse_log, a1, a2 = validate_errors(depth_gt_, pred_depth_) 
                  
         
        
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
    
    abs_rel = errors["abs_rel"]
    sq_rel = errors["sq_rel"]
    rmse = errors["rmse"]
    rmse_log = errors["rmse_log"]
    a1 = errors["a1"]
    a2 = errors["a2"]
    
    abs_rel_ = errors["abs_rel_"]
    sq_rel_ = errors["sq_rel_"]
    rmse_ = errors["rmse_"]
    rmse_log_ = errors["rmse_log_"]
    a1_ = errors["a1_"]
    a2_ = errors["a2_"]
         
    print("errors evaluate disparity:\n abs_rel: {}, rmse: {}, sq_rel: {}, rmse_log: {}, a1: {}, a2: {}\n errors evaluate depth:\n abs_rel: {}, rmse: {}, sq_rel: {}, rmse_log: {}, a1:{}, a2:{}".format(
          abs_rel_, rmse_, sq_rel_, rmse_log_, a1_, a2_, abs_rel, rmse, sq_rel, rmse_log, a1, a2))        
        
    return abs_rel         



def validate_errors(gt, pred): # for disparity
   
     
    gt = (gt +1.0 ) *128.0

    
    pred[ pred> 255.0] = 255.0
    pred[ pred< 0.3] = 0.3
    
    gt[gt>255.0] = 255.0
    gt[gt<0.3] = 0.3


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
        
def validate_errors_(gt, pred): # for fusion depth
   
     
    gt = (gt +1.0 ) * 128.0
    pred = (pred +1.0 )* 128.0
    
    pred[ pred> 255.0] = 255.0
    pred[ pred< 0.3] = 0.3
    
    gt[gt>255.0] = 255.0
    gt[gt<0.3] = 0.3

    
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


if __name__ == '__main__':
    #test()
    test_spike()
    #to_video()
    #gt_video()
    #addvideo()


