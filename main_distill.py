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
from datasets.SpikeDataset import SpikeDataset, SpikeRGBDataset
from datasets.Spikeset import JYSpikeDataset
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from AWLoss import AutomaticWeightedLoss


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
parser.add_argument('--batch_size', type=int, default=2, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=2, help='testing batch size')
parser.add_argument('--epochs', type=int, default=100, required=False, help='number of epochs to train')
parser.add_argument('--lrepochs', type=str, default='35:3',required=False, help='the epochs to decay lr: the downscale rate')

parser.add_argument('--logdir', default='/home/lijianing/depth/MMlogs/256/dorn', required=False, help='the directory to save logs and checkpoints')
parser.add_argument('--loadckpt', help='load the weights from a specific checkpoint')
parser.add_argument('--resume', action='store_true', help='continue training the model')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--summary_freq', type=int, default=5, help='the frequency of saving summary')
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

#train_dataset =  SpikeRGBDataset(path_rgb = "/home/lijianing/depth/left_rgb/", path_spike = "/home/Datadisk/spikedata5622/spiking-2022/train/firsthleft/", mode = "training")

#train_dataset =  JYSpikeDataset(path="/home/Datadisk/spikedata5622/spiking-2022/JYsplit/train/")
#StereoDataset("/home/lijianing/kitti/", args.trainlist, True)
#test_dataset = JYSpikeDataset(path="/home/Datadisk/spikedata5622/spiking-2022/JYsplit/validation/")#   
test_dataset = SpikeDataset(pathr = "/home/Datadisk/spikedata5622/spiking-2022/test/nrpz/", pathl = "/home/Datadisk/spikedata5622/spiking-2022/test/nlpz/", mode = "test")
#StereoDataset("/home/lijianing/kitti/", args.testlist, False)
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=2, drop_last=True)
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=4, drop_last=False)


from models.fusionet import fusionet, fusionet1
from models.monocular_way import UNet, UNet_KD, UNet_KD1, KD_bottle
from models.dorn import DORN
from models.dorn import OrdinalRegressionLoss
#from models.
# model, optimizer
bottleneck = KD_bottle().cuda()
bottleneck_t = KD_bottle().cuda()

model_student = UNet_KD(n_channels=32, n_classes=1, bilinear=True)#__models__[args.model](args.maxdisp)

model_student_spike = UNet(n_channels=32, n_classes=1, bilinear=True)
model_teacher = UNet_KD1(n_channels=3, n_classes=1, bilinear=True)

model_student_learn = __models__[args.model](args.maxdisp)#UNet_KD(n_channels=32, n_classes=1, bilinear=True)

#model = fusionet(maxdisp=32, batchsize=2)#)
#model = fusionet1(maxdisp=32)
#model = nn.DataParallel(model, [0,4,5,6])

model_t = model_teacher

model = model_student_spike

model = model_student
model = DORN()

device = torch.device("cuda:{}".format(5))

model = model.to(device)
'''
if args.n_gpu > 1:
    model = torch.nn.DataParallel(model, device_ids=[4,5,6,7])
'''

#model = model_student_learn

#model_student = __models__[args.model](args.maxdisp)
#model.cuda()
#awl = AutomaticWeightedLoss(3)


            
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
#state_dict = torch.load('/home/lijianing/depth/MMlogs/256/dorn/checkpoint_max_student_single_unet.ckpt')
#model.load_state_dict(state_dict['model'])

# load parameters
start_epoch = 0
if args.resume:

    all_saved_ckpts = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    all_saved_ckpts = sorted(all_saved_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))

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
            del scalar_outputs
            print('Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, args.epochs,
                                                                                       batch_idx,
                                                                                       len(TrainImgLoader), loss,
                                                                                       time.time() - start_time))
        # saving checkpoints
        if (epoch_idx + 1) % args.save_freq == 0:
            checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(checkpoint_data, "{}/checkpoint_max_student_single_unet.ckpt".format(args.logdir))
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
    
    model_t.cuda()
    model_t.load_state_dict(torch.load("/home/lijianing/depth/CFNet-mod/logs_distill/checkpoint_max_student_single_rgb.ckpt")["model"])
    with torch.no_grad():
        model_t.eval()
    

    #imgL, imgR, disp_gt, depth_gt = sample['left'], sample['right'], sample['disparity'], sample['depth']
    imgL, imgR, disp_gt, depth_gt, depth_dorn, seq_ldorn = sample['left'].to(device), sample['right'].to(device), sample['disparity'].to(device), sample['depth'].to(device), sample['depth_dorn'].to(device), sample['seq_ldorn'].to(device)
    '''
    img_spike, img_rgb, disp_gt, depth_gt, depth_dorn, seq_ldorn = sample['spike'].to(device), sample['rgb'].to(device), sample['disparity'].to(device), sample['depth'].to(device), sample['depth_dorn'].to(device), 
    sample['seq_ldorn'].to(device)
    '''
    ###############################
    '''
    knowledge = model_t(img_rgb)
    knowledge = F.upsample(knowledge, [256,512], mode='bilinear', align_corners=True).squeeze(1)
    '''
    ###############################
    '''
    img_spike = img_spike.to(device)
    img_rgb = img_rgb.to(device)
    disp_gt = disp_gt.to(device)
    depth_gt = depth_gt.to(device)

    stereo_gt = disp_gt[0].detach().cpu().squeeze(0)
       
    stereo_gt_np = np.array(stereo_gt, dtype = np.float32)
    stereo_gt_img = Image.fromarray(stereo_gt_np, 'L')
    stereo_gt_img.save('/home/lijianing/depth/CFNet-mod/gt{}.png'.format('1'))  
    '''

    optimizer.zero_grad()
    
    ord_prob, ord_label = model(imgL)
    ####ests = model(img_rgb)
    #ests,y1,y2,y3,y4,b = model(img_spike)
    
    
    '''
    disp_ests = ests['stereo']
    depth_ests, y1,y2,y3,y4 = ests['monocular']
    '''

    #fusion_ests = ests['fusion']
    #stereo_depth_ests = 1 / (disp_ests[-1])  
    
    
    #####stereo_depth_ests = 1 / (disp_ests[-1])    
    #stereo_depth4sim = stereo_depth_ests.unsqueeze(1)
    
    #mono_depth4sim = depth_ests.unsqueeze(1)
    
    #mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    ######loss1 = model_loss(disp_ests, disp_gt, mask)
    #print(disp_ests[-1].size())
    
    #####disp_ests = disp_ests[-1].unsqueeze(1)
    
    #depth_ests = depth_ests.unsqueeze(1) #fusion on
    #print(depth_ests.size())
    
    ##################disp_ests = F.upsample(disp_ests, [256,512], mode='bilinear', align_corners=True).squeeze(1)
    #depth_ests = depth_ests.squeeze(1)
    #print(depth_ests.size())
    #depth_ests = F.upsample(depth_ests, [256,512], mode='bilinear', align_corners=True).squeeze(1)
    #depth_ests = depth_ests.squeeze(1)
    
    #loss1 = F.smooth_l1_loss(disp_ests, disp_gt)
    #print(t1.size(), z1.size())
    ###loss_kd = F.mse_loss(tb, fb) #+ 0.2*F.mse_loss(zb, fb) + 0.2*F.mse_loss(z1, f1) + 0.2*F.mse_loss(z2, f2) + 0.2*F.mse_loss(z3, f3) + 0.2*F.mse_loss(z4, f4)#smooth_l1_loss(t1,z1)+F.smooth_l1_loss(t2,z2)+F.smooth_l1_loss(t3,z3)+F.smooth_l1_loss(t4,z4)
    
    ORD_NUM = 256
    BETA = 256
    ord_loss = OrdinalRegressionLoss(ord_num=ORD_NUM, beta=BETA)
    loss_cls = nn.CrossEntropyLoss()
    loss2 = ord_loss(ord_prob, depth_dorn)
    
    #loss2 = F.smooth_l1_loss(depth_ests, depth_gt) #+ 0.02*get_smooth_loss(depth_ests, imgR)
    #loss2 = F.mse_loss(depth_ests, depth_gt) + F.smooth_l1_loss(depth_ests, depth_gt)
    #loss_fusion = F.smooth_l1_loss(fusion_ests, depth_gt)
 
    #loss = 10.0*loss2 
    
    if epoch <= 100:
        loss = loss2 #+ 0.5*loss_kd 
    '''
    elif epoch > 25 and epoch <=100:
        loss = loss1 + F.smooth_l1_loss(stereo_depth4sim, mono_depth4sim)#loss3#loss2 + loss_fusion 
    elif epoch > 100 and epoch <=200:
        loss = loss2 + F.smooth_l1_loss(stereo_depth4sim, mono_depth4sim)#loss_fusion
    '''
    #awl(loss1, loss2, loss_fusion)
    #loss =  2.0*loss2 + 1.0*loss1 + 1.0*loss_fusion
    scalar_outputs = {"loss": loss}
    ##############image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}
    '''
    if compute_metrics:
        with torch.no_grad():
            #image_outputs["errormap"] = [disp_error_image_func()(disp_est, disp_gt) for disp_est in disp_ests]
            #scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
            scalar_outputs["D1"] = [D1_metric(depth_ests, depth_gt, mask) for disp_est in disp_ests]
            #scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
            #scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
            #scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]
          
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
    pred_depth,_,_,_,_,_ = model(imgL)#["monocular"]
    #pred_depth = model(imgL, imgR)["monocular"]
    
    pred_depth = F.upsample(pred_depth, [256,512], mode='bilinear', align_corners=True).squeeze(1)
    
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    #loss = F.smooth_l1_loss(pred_depth[-1], depth_gt)#model_loss(disp_ests, disp_gt, mask)
    loss = F.smooth_l1_loss(pred_depth, depth_gt)
    scalar_outputs = {"loss": loss}
    
    #image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}
   
    scalar_outputs["D1"] = [D1_metric(1000.0*(pred_depth), 1000.0*(depth_gt), mask)] #for disp_est in disp_ests]
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
