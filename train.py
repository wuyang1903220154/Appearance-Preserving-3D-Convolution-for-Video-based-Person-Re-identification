# encoding: utf-8
from __future__ import print_function, absolute_import
import os
import gc
import sys
import time
import math
import h5py
import scipy
import datetime
import argparse
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import lr_scheduler

import models
import transforms.spatial_transforms as ST
import transforms.temporal_transforms as TT
import tools.data_manager as data_manager
from tools.video_loader import VideoDataset
from tools.losses import TripletLoss
from tools.utils import AverageMeter, Logger, save_checkpoint
from tools.eval_metrics import evaluate
from tools.samplers import RandomIdentitySampler
from IPython import embed


parser = argparse.ArgumentParser(description='Train AP3D')  ## 这里是来对预配置命令的使用的说明的。
# Datasets
parser.add_argument('--root', type=str, default='E:/datasets')  ## 然后是来对所使用的数据根路径的
parser.add_argument('-d', '--dataset', type=str, default='mars',
                    choices=data_manager.get_names())
parser.add_argument('-j', '--workers', default=4, type=int)  ## 最开始的时候是4，这里是来工作线程数的。
parser.add_argument('--height', type=int, default=256)  ## 这里最开始所对应的图片的高是256
parser.add_argument('--width', type=int, default=128)  ## 图片所对应的宽是 128
# Augment
parser.add_argument('--seq_len', type=int, default=4,   ## 这里是每个视频序列采样4帧的图像的。
                    help="number of images to sample in a tracklet")
parser.add_argument('--sample_stride', type=int, default=8,   ## 这里是来对8帧进行采样一次。这里是来采样的步长的
                    help="stride of images to sample in a tracklet")
# Optimization options
parser.add_argument('--max_epoch', default=240, type=int)  ## 最开始的最大epoch =240的
parser.add_argument('--start_epoch', default=0, type=int)  ## 对应的每个epoch开始的时候是从0开始的。
parser.add_argument('--train_batch', default=32, type=int)  ## 最开始的值是32  其中每个批次是包含8个行人的，每个行人是包含4个视频帧的
parser.add_argument('--test_batch', default=32, type=int)
parser.add_argument('--lr', default=0.0003, type=float)  ## 这里是来对应的最开始时候的，所对应的基础的学习率的。
parser.add_argument('--stepsize', default=[60, 120, 180], nargs='+', type=int,
                    help="stepsize to decay learning rate")   ## 其中每到一定的步骤的，学习率是会发生衰减的。
parser.add_argument('--gamma', default=0.1, type=float,
                    help="learning rate decay")  ## 其中这里是对应的衰减的比列的。
parser.add_argument('--weight_decay', default=5e-04, type=float)   ## 在最开始的时候，所对应的权重的衰减的损失的
parser.add_argument('--margin', type=float, default=0.3, 
                    help="margin for triplet loss")  ## 这里是对应的三元组损失所对应的边框的。
parser.add_argument('--distance', type=str, default='cosine', 
                    help="euclidean or cosine")  ## 我们采用的距离的计算公式是cosine的计算的公式的。
parser.add_argument('--num_instances', type=int, default=4, 
                    help="number of instances per identity")  ## 这里是每个id所对应的实例数的。
# Architecture
parser.add_argument('-a', '--arch', type=str, default='ap3dres50', 
                    help="ap3dres50, ap3dnlres50")   ## 这里是来采用网络结构的。
# Miscs
parser.add_argument('--seed', type=int, default=1)  ## 这里是来定义的随机数的种子的。这样是可以来稳定训练的过程的。
parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument('--eval_step', type=int, default=10)  ## 其中是每10步开始来进行一次测试的步骤的。每10个epcoch来进行测试一次
parser.add_argument('--start_eval', type=int, default=0, 
                    help="start to evaluate after specific epoch")  ## 这里是来开始进行相应的评估的过程的。
parser.add_argument('--save_dir', type=str, default='E:/codes/datasets/AP3D-master/log-mars-ap3d')  ## 这里是来对训练的存储的路径的。
parser.add_argument('--use_cpu', action='store_true', help="use cpu")  ## 是否是来使用的gpu来进行计算的过程的。
parser.add_argument('--gpu', default='0, 1', type=str, 
                    help='gpu device ids for CUDA_VISIBLE_DEVICES') ##  这里是来使用的那几块gpu来进行计算的过程的。

args = parser.parse_args()  ## 这里是来使用的args是可以来调用这些命令的。

def main():
    torch.manual_seed(args.seed)  ## 为cpu设置用于生成随机数的种子，这样是可以使得随机数的种子是确定的。是能够让训练是更加稳定的
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu  ## 这里是来调用的gpu来进行相应的计算的过程的。
    use_gpu = torch.cuda.is_available()  ## 这里是来测量gpu是否是合理的。
    if args.use_cpu: use_gpu = False  # 如果是使用cpu的话，那么就是把gpu置False的

    sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))  ## 然后是把训练日志输出在这个日志文件的下面的log-mars-ap3d/log_train.txt
    print("==========\nArgs:{}\n==========".format(args))  ## 这里是来全部打印出参数信息的。

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu))  ## 这里是来打印出GPU所对应的信息的。
        torch.cuda.manual_seed_all(args.seed)  ## 为gpu来设置用于生成随机数的种子，这样是可以来稳定训练的过程的。
    else:
        print("Currently using CPU (GPU is highly recommended)") ## 否则是打印输出现在使用的cpu,其中gpu是有更加高的需求的。

    print("Initializing dataset {}".format(args.dataset))   ##打印出的对mars数据集进行初始化的过程的。这里是来打印输出相应的名称后
    dataset = data_manager.init_dataset(name=args.dataset, root=args.root) ## D:/Pratice Code/DataSet/视频数据集/mars  来对数据集进行初始化的过程的。

    # Data augmentation  ## 这里是对数据增强的过程的。在训练集中进行相应的空间的转换训练的过程的。这里是在空间上，进行转换的过程，是来对图像进行数据增强的。
    spatial_transform_train = ST.Compose([
                ST.Scale((args.height, args.width), interpolation=3),  ## 将输入的像素重新定义到，预定义的尺寸的。
                ST.RandomHorizontalFlip(),  ## 然后是对图像进行随机的水平翻转的过程的。
                ST.ToTensor(),  ##  这里是来把图像转换为张量所对应的形式的。
                ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  ## 然后我们是来进行正则化的过程
            ])
    ## 这里是在时间上进行转换的过程的。seq_len = 4  sample_stride = 8 我们是来对8帧进行采样的一次的，
    temporal_transform_train = TT.TemporalRandomCrop(size=args.seq_len, stride=args.sample_stride) ## 其中对于每个序列采样长度4，所对应的步长是8

    ## 这里是来对空间测试进行采样的过程的。
    spatial_transform_test = ST.Compose([
                ST.Scale((args.height, args.width), interpolation=3),  ## 首先是来改变相应的图像的大小的，并且采样的方式是interpolation = 3的方法的
                ST.ToTensor(),  ## 然后是来转换为张量所对应的形式的
                ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  ## 然后我们这里是来对应的正则化的效果就行了。
            ])
    ## 这里是来对应时间的测试转换的过程中的。这里是来提取相应的帧的过程的。
    temporal_transform_test = TT.TemporalBeginCrop()

    ## use_gpu如果是为真的就是使用gpu 否则不是使用gpu来进行计算的。
    pin_memory = True if use_gpu else False

    ## 这里是如果是数据集不是mar数据集的，那么我们来对相应的数据集进行加载的过程的。这里是不执行的
    if args.dataset != 'mars':
        trainloader = DataLoader(
            VideoDataset(dataset.train_dense, spatial_transform=spatial_transform_train, temporal_transform=temporal_transform_train),
            sampler=RandomIdentitySampler(dataset.train_dense, num_instances=args.num_instances),
            batch_size=args.train_batch, num_workers=args.workers,  ## num_workers是来表示相应的工作的线程数的
            pin_memory=pin_memory, drop_last=True)  ##  最后是不满足一个批次的然后就是的被丢弃掉的。

    ## 表示的这是mars数据集所要做的一些事情的。这里是来把数据集的训练集给加载进来的。
    else:
        trainloader = DataLoader(
            # 这里是来返回经过变换后的clip图像的，pid,camid的过程的。
            VideoDataset(dataset.train, spatial_transform=spatial_transform_train, temporal_transform=temporal_transform_train),
            sampler=RandomIdentitySampler(dataset.train, num_instances=args.num_instances), ## 这里是来对应的采样器，总共是采样N个身份，然后是每个身份采用实例数是4的
            batch_size=args.train_batch, num_workers=args.workers,
            pin_memory=pin_memory, drop_last=True)

    ## 这里是来对查询数据集的加载的过程的。这里是来对query训练数据集的处理的过程的。
    queryloader = DataLoader(
        VideoDataset(dataset.query, spatial_transform=spatial_transform_test, temporal_transform=temporal_transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=0,
        pin_memory=pin_memory, drop_last=False)

    ## 这里是来对图库数据集的加载的过程的。
    galleryloader = DataLoader(
        VideoDataset(dataset.gallery, spatial_transform=spatial_transform_test, temporal_transform=temporal_transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=0,
        pin_memory=pin_memory, drop_last=False)

    print("Initializing model: {}".format(args.arch))  ## 这里是来打印输出模型的所对应的结构的。这里是来对应的输出相应的网络的结构的
    ## 这里是来输入模型的网络结构，并且是要输入训练id数目的。这里就是可以来得到训练的模型的。
    model = models.init_model(name=args.arch, num_classes=dataset.num_train_pids)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))  ## 这里是来对应的总共输出模型所对应的参数的过程的

    criterion_xent = nn.CrossEntropyLoss()  ## 这里是来调动的相应的交叉熵的损失的过程的。
    criterion_htri = TripletLoss(margin=args.margin, distance=args.distance)  ## 这里是来对应的三元组损失的过程的。

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  ## 首先我们是来选用Adam来对里面的参数进行更新的
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.stepsize, gamma=args.gamma)  ## 这里是来我们对学习率的更新的步骤的。
    start_epoch = args.start_epoch  ## 对于start_epoch = 0 是来对应的所对应开始的步骤的过程的。

    ## 是来对应的这个路径中数据进行加载的过程的。
    if args.resume:
        print("Loading checkpoint from '{}'".format(args.resume))
        checkpoint = torch.load(args.resume) # 是对应的这个路径下的加载的过程的。
        model.load_state_dict(checkpoint['state_dict'])  ## 这里是来加载模型所对应的状态字典的。
        start_epoch = checkpoint['epoch']  # 这里是对应的通过epoch进行加载的过程的。

    if use_gpu:
        model = nn.DataParallel(model).cuda()  ## 如果是存在gpu的情况下，那么我们使用分布式的计算方式的。
        # model = model.cuda()

    start_time = time.time()  ## 这里是对应的模型的训练的开始的时间的。
    train_time = 0 # 开始的时候训练时间为0
    best_rank1 = -np.inf  ## 在开始的时候的，best_rank1是对应的负无穷大的数的。
    best_epoch = 0  # 最开始的时候获得最好epoch是为0的。
    print("==> Start training")  ## 我们从这里就是所对应的数据开始来进行相应的训练的过程的。

    ## 其中所对应的epoch开始的时候，是从0开始来进行训练的。其中的最大epoch是240个
    for epoch in range(start_epoch, args.max_epoch):
        scheduler.step()  ## 在开始的时候的，所对应的学习率的步骤的更新的过程的。在开始的时候学习率是按照scheduler步骤进行更新的

        start_train_time = time.time()  ##  这里是对应的训练开始的时间的。
        ## 然后我们是来开始进行相应的训练的过程的，在开始训练的过程中，我们是传入epoch
        ## 和所需要使用的model
        ## 和所应的交叉熵的损失
        ## criterion_htri是和所对应的三元组的损失的过程的。
        ## optimizer是来把相应的参数优化器，进行传入的
        ## trainloader是来对数据集进行加载的。
        ## use_gpu是否是要来使用相应的gpu来进行计算的过程的。
        train(epoch, model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu) # 这里是训练一个epoch
        train_time += round(time.time() - start_train_time) ## 全部epoch时间的总和的。


        ## 如果我们是满足下面的这个条件，那么我们就是要来开始进行测试的。
        ## args.start_eval = 0
        # args.eval_step = 10
        if (epoch+1) >= args.start_eval and args.eval_step > 0 and (epoch+1) % args.eval_step == 0 or (epoch+1) == args.max_epoch:
            print("==> Test")  ## 每10次，然后我们就是来开始进行测试一次的。
            with torch.no_grad():  ## 在测试的时候，是没有进行梯度更新的。
                # test using 4 frames  ## 测试的时候我们使用4帧的图像的。
                ## queryloader的数据集来进行加载
                ## galleryloader这里是对应图库数据集的加载的过程的
                rank1 = test(model, queryloader, galleryloader, use_gpu)  ## 这里是来测试返回rank-所对应的值的
            is_best = rank1 > best_rank1  ## 如果是满足rank1的值是大于best_rank1的值，那么我们就是对最好的值来进行更新的过程的。
            if is_best: 
                best_rank1 = rank1  ## 这里首先还是的rank1的值来进行更新的。
                best_epoch = epoch + 1  ##

            if use_gpu:
                state_dict = model.module.state_dict()  ## 在gpu上面来进行计算的
            else:
                state_dict = model.state_dict()  ## 这里是在cpu上面来进行计算的。
            ## 然后是来把这些状态值给存储起来的。
            save_checkpoint({
                'state_dict': state_dict,
                'rank1': rank1,            ## 首先是来把这些参数是给存储在这个文件当中的。
                'epoch': epoch,
            }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch+1) + '.pth.tar'))

    ## 然后是来打印输出最好的best_rank1的值，best_epoch所对应的值的。
    print("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))

    elapsed = round(time.time() - start_time)  ## 这里是计算出总共花费的时间的。elapsed是来考虑全部所需要的时间的。
    elapsed = str(datetime.timedelta(seconds=elapsed))  ## 把时间转换字符串所对应的seconds的形式的。
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))

## 传入相应的参数来进行训练的。这里是对应的一个epoch的训练的过程的。我们所对应的epoch是有240个的。
def train(epoch, model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu):
    batch_xent_loss = AverageMeter()  ## 批的交叉熵的损失的过程的。
    batch_htri_loss = AverageMeter()  ## 批所对应的三元组损失的。
    batch_loss = AverageMeter()  ## 批损失
    batch_corrects = AverageMeter()  ## 批所对应的正确率的
    batch_time = AverageMeter()  ## 批所对应的时间的。
    data_time = AverageMeter()  ## 是来对应的数据集所对应的时间的。

    model.train()   ## 是对应的模型开始来进行训练，然后是对应的模型的训练完成以后，我们能够所得到的解的过程的

    end = time.time()   ## 这里是对应的训练的开始的时间的
    ## 是对应的batch_idx,
    for batch_idx, (vids, pids, _) in enumerate(trainloader): ## 其中vids可能是对应的相应的视频id的过程的。
        if (pids-pids[0]).sum() == 0:
            # can't compute triplet loss  ## 这里是不能够来计算出相应的三元组的损失的过程的。
            continue

        if use_gpu:
            vids, pids = vids.cuda(), pids.cuda()  ## 放在相应的gpu上面来进行计算的过程的。

        # measure data loading time ## 测试的是加载数据所需要的时间的。
        data_time.update(time.time() - end) # 25.320008277893066

        # zero the parameter gradients
        optimizer.zero_grad()   ## 首先是来对应的全部的梯度是的为0的。


        # forward features = torch.Size([8, 2048]) outputs = torch.Size([8, 625]) 对应的这个
        outputs, features = model(vids)   ##vid =  torch.Size([8, 3, 4, 256, 128]) 得到特征和预测出的标签的


        # combine hard triplet loss with cross entropy loss  结合硬的三元组损失的和相应的交叉熵的损失的过程的。
        # xent_loss = 6.4367第0个批次所对应的损失的
        # htri_loss = 0.2577第0个批次所对应的损失的
        # loss = 6.6944 是两个损失相加的过程的 是对应的第0个批次所对应的损失的。
        xent_loss = criterion_xent(outputs, pids)  ## 这里是来对应的交叉熵的损失的，是对应的预测标签与真实标签之间的差距的
        htri_loss = criterion_htri(features, pids)   ## 对应三元组损失的过程的是，这样是可以得到特征与相识度之间的距离，计算出损失

        loss = xent_loss + htri_loss  ## 两个损失相加

        # backward + optimize
        loss.backward()  ## 是通过loss来反向传播，对相应的梯度来对参数进行更新的过程的
        optimizer.step()  ## 优化器，也是要来对相应的参数进行更新的过程的。

        # statistics
        _, preds = torch.max(outputs.data, 1)  ## 找出其中最大值的，所对应的标签的。
        batch_corrects.update(torch.sum(preds == pids.data).float()/pids.size(0), pids.size(0))  ## 得到相应的patch的准确率的。
        
        batch_xent_loss.update(xent_loss.item(), pids.size(0))
        batch_htri_loss.update(htri_loss.item(), pids.size(0))
        batch_loss.update(loss.item(), pids.size(0))



        # measure elapsed time
        batch_time.update(time.time() - end)  ## 这里是训练这一批数据所需要的时间的。
        end = time.time()  # 开始进行下一次循环的过程的。

    print('Epoch{0} '
          'Time:{batch_time.sum:.1f}s '  ## 这里是来训练一个批次所花费总的时间的。
          'Data:{data_time.sum:.1f}s '   ## 这里是来测量数据加载所运用的一些时间的。
          'Loss:{loss.avg:.4f} '  ## 总的损失
          'Xent:{xent.avg:.4f} '  ## 交叉熵所对应的损失的
          'Htri:{htri.avg:.4f} '  ## 三元组所对应的损失的
          'Acc:{acc.avg:.2%} '.format(   ## 得到相应的平均的准确率的。
          epoch+1, batch_time=batch_time,
          data_time=data_time, loss=batch_loss,
          xent=batch_xent_loss, htri=batch_htri_loss,
          acc=batch_corrects))  ## 这里是来打印出训练损失的过程的。
    
## 这里是来对应的测试的模型的。
## 首先是把model进行输入
## queryloader 是来加载所需要查询的图像的
## galleryloader 图库中所对应的图像的
## 是来对应的所需要查询的准确率的。rank-1,rank-5,rank-10,rank-20所对应的值的
# vids = torch.Size([16, 3, 4, 256, 128]) 测试的时候批次是16通道数是3批次是4
# pids = [ 2,  2,  2,  2,  4,  4,  4,  6,  6,  8,  8,  8, 10, 10, 16, 16]
# camids = [0, 1, 2, 5, 0, 1, 4, 0, 1, 0, 2, 5, 0, 4, 0, 1]
def test(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20]):
    # test using 4 frames
    since = time.time()  ## 开始时间
    model.eval()  ## 开始测试
    qf, q_pids, q_camids = [], [], []  ## 得到查询后qf,q_pids,q_camid
    for batch_idx, (vids, pids, camids) in enumerate(queryloader):
        if use_gpu:
            vids = vids.cuda()  ## 如果是存在gpu话，那么就是来使用gpu进行计算的。
        feat = model(vids)  ## 这里首先是来得到相应的特征的。torch.Size([16, 4, 2048])
        feat = feat.mean(1)  ## 这里是来压缩列，然后是来对每一行求取相应的平均值的。
        feat = model.module.bn(feat)  ## 这里的特征是来经过正则化的过程的。
        feat = feat.data.cpu() # torch.Size([16, 2048])

        qf.append(feat)   ## 把相应的到特征给添加列表里面
        q_pids.extend(pids)  ## 这里是来把行人id给添加到列表里面 tensor(2, dtype=torch.int32)
        q_camids.extend(camids)  ## 然后是把相机id给添加到列表里面 tensor(0, dtype=torch.int32)


    qf = torch.cat(qf, 0)  ## 然后是来对查询的特征的进行连接的过程的。torch.Size([1980, 2048]) 测试的时候是有1980序列，每个特征2048
    q_pids = np.asarray(q_pids) ## 把查询的行人id转换为数组所对应的形式的 array([   2,    2,    2, ..., 1496, 1500, 1500]) 长度是为1980
    q_camids = np.asarray(q_camids)  ## 把查询的相机id转换为数组所对应的形式的。 array([0, 1, 2, ..., 3, 0, 4]) 长度是1980
    ## [1980,2048]维度的向量的1980表示的是query是有1980个序列的，其中每个序列所的到的特征维度是2048的
    print("Extracted features for query set, obtained {} matrix".format(qf.shape))

    ## 这里是来对应的图库数据集的。gf的特征 g_pids的行人id， g_camids所对应的相机id
    # vids = torch.Size([16, 3, 4, 256, 128])
    gf, g_pids, g_camids = [], [], []
    for batch_idx, (vids, pids, camids) in enumerate(galleryloader):
        if use_gpu:
            vids = vids.cuda()  ## 把这个视频帧放到gpu上面来进行相应的计算的过程的。
        feat = model(vids)  ## gallery的特征的
        feat = feat.mean(1)  ## 是对应的平均池化的过程，是来对每一行求取一个平均值，这样就是来把列进行压缩的过程的。
        feat = model.module.bn(feat)  ## 也是来对特征的进行批归一化的处理的。
        feat = feat.data.cpu()  ## 把特征数据转换到cpu()上面来进行计算的过程的。

        gf.append(feat)  ## 把相应的特征的给添加到列表当中的。
        g_pids.extend(pids)  ## 把图库中的pids给添加到列表当中的
        g_camids.extend(camids) ##  把g_camids是来给添加到里面的。

    gf = torch.cat(gf, 0)   ## 然后是要把查询的特征的给全部的添加到里面的。
    g_pids = np.asarray(g_pids)  ## g_pids是来转化为相应的数组的
    g_camids = np.asarray(g_camids)  ## 也是来转换为相应的数组的。

    ## 如果是数据集mars数据集，那么就是要来进行相应操作的过程的。
    if args.dataset == 'mars':
        # gallery set must contain query set, otherwise 140 query imgs will not have ground truth.
        gf = torch.cat((qf, gf), 0)  ## qf和对应的gf进行连接的过程的。torch.Size([11310, 2048]) 是来把这两个维度进行连接的
        g_pids = np.append(q_pids, g_pids) # len(g_pids) 11310 array([   2,    2,    2, ..., 1496, 1496, 1500])
        g_camids = np.append(q_camids, g_camids) # array([0, 1, 2, ..., 3, 3, 0]) len =  11310

    ## 这里是对应的11310，2048 对于这里 11310是包含query和gallery这两个序列的。
    print("Extracted features for gallery set, obtained {} matrix".format(gf.shape))

    time_elapsed = time.time() - since  ## 开始的时间-减去结束的时间的。这里query和gallery特征的萃取的时间的。
    ## 完成特征萃取的时间的分钟和s所对应的时间的。
    print('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print("Computing distance matrix")  ## 然后是来计算出相应的距离矩阵的。
    m, n = qf.size(0), gf.size(0) ## [1980,11310]
    distmat = torch.zeros((m,n))  ## 创建的一个为0 维度为[1980,11310]的距离方正的

    ## 通过相应的欧式距离来进行计算的。用来计算相应的距离方正(qf-gf)^2 = qf^2 + gf^2 - 2qfgf qf = [1980,2048] gf = [11310,2048]
    if args.distance == 'euclidean':
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        for i in range(m):
            distmat[i:i+1].addmm_(1, -2, qf[i:i+1], gf.t())  # 这里是来求出相应的距离方正
    else:
        q_norm = torch.norm(qf, p=2, dim=1, keepdim=True) #  这里的距离是余弦距离所对应的情况的。这里是来求范数在1维度上来求范数的，也就是来返回多少行torch.Size([1980, 1])
        g_norm = torch.norm(gf, p=2, dim=1, keepdim=True) #  在1维度上来求出2范数也就是返回的是多少行的 torch.Size([11310, 1])
        qf = qf.div(q_norm.expand_as(qf)) # torch.Size([1980, 2048]) 这里是对尺寸进行扩展的过程的。div()使得各行的数据进行标准化后
        gf = gf.div(g_norm.expand_as(gf)) # torch.Size([11310, 2048])
        # 其中m是等于的1980的
        for i in range(m):
            distmat[i] = - torch.mm(qf[i:i+1], gf.t())  # 这个负号是对数组的里面的值的全部取负号的过程的。这里是对应的每查询一个序列所对应序列方正的。
    distmat = distmat.numpy()  ## 然后是来计算出相应的距离转换为numpy所对应的形式的。torch.Size([1980, 11310])这里是对应每个query在图库中查询矩阵的

    print("Computing CMC and mAP")  ## 计算出相应的CMC也就是对应的rank-1曲线和mAP的曲线的。
    ## 然后我们是来传入相应的距离矩阵，q_pids的值 ， g_pids所对应值的，g_camids相机号 q_camids是对应的查询图像所对应的相机id
    # array([   2,    2,    2, ..., 1496, 1500, 1500]) 长度是1980
    # array([   2,    2,    2, ..., 1496, 1496, 1500]) 长度是11310
    # array([0, 1, 2, ..., 3, 0, 4]) 这里是1980其中q_camids也是同样的。
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print("Results ----------")  ## 这里是来答应输出的rank-1,rank-5,rank-10所对应的值的。mAP值
    print('top1:{:.1%} top5:{:.1%} top10:{:.1%} mAP:{:.1%}'.format(cmc[0],cmc[4],cmc[9],mAP))
    print("------------------")

    return cmc[0] ## 其中是要对rank-1的值进行返回的过程的。

if __name__ == '__main__':
    main()
