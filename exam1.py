# encoding: utf-8
# 学校:西安邮电大学
# 学院:计算机学院
# 姓名:吴洋
# 目的:python学习
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
# 1.sample_people过程及源码解析
# 从数据集中进行抽样图片，参数为训练数据集，每一个batch抽样多少人，每个人抽样多少张
def sample_people(dataset, people_per_batch, images_per_person):
    # 总共应该抽样多少张 默认：people_per_batch：45  images_per_person：40
    nrof_images = people_per_batch * images_per_person  ## 这里表示的是总共要抽取的人数和所对应的图片的数量的。

    # 数据集中一共有多少人的图像
    nrof_classes = len(dataset)
    # 每个人的索引
    class_indices = np.arange(nrof_classes)
    # 随机打乱一下，是来随机的把这些数据通过索引来进行相应的打乱的过程的。
    np.random.shuffle(class_indices)

    i = 0
    # 保存抽样出来的图像的路径
    image_paths = []
    # 抽样的样本是属于哪一个人的，作为label
    num_per_class = []
    sampled_class_indices = []
    # Sample images from these classes until we have enough
    # 不断抽样直到达到指定数量，是不断的来进行抽取后，要达到我们所指定的数量的。
    ## 这里是对应的抽样图像的所对应的数量是小于我们需要的抽取的数量的时候的。
    while len(image_paths) < nrof_images:
        # 从第i个人开始抽样，这里是我们来得到每个人所对应的索引的时候。
        class_index = class_indices[i]
        # 第i个人有多少张图片
        nrof_images_in_class = len(dataset[class_index])  ## 这里是来对应的这个类标签是对应的有多少张图片的。
        # 这些图片的索引
        image_indices = np.arange(nrof_images_in_class)  ## 得到这些图片所对应索引，然后是来对这些图片进行打乱的操作的过程的。
        np.random.shuffle(image_indices)
        # 从第i个人中抽样的图片数量  nrof_images-len(image_paths) 这里是总共需要抽取图片的数量-已经抽取到图片所对应的数量的。
        nrof_images_from_class = min(nrof_images_in_class, images_per_person, nrof_images - len(image_paths))
        idx = image_indices[0:nrof_images_from_class]  ## 这里是来从第i人中抽取的图片的数量然后是来得到相应的索引的过程的。
        # 抽样出来的人的路径
        image_paths_for_class = [dataset[class_index].image_paths[j] for j in idx]
        # 图片的label
        sampled_class_indices += [class_index] * nrof_images_from_class
        image_paths += image_paths_for_class
        # 第i个人抽样了多少张
        num_per_class.append(nrof_images_from_class)
        i += 1

    return image_paths, num_per_class


# 3. 调用select_triplets()得到（A,P,N）三元组
## 首先是来传入相应的特征后，nrof_images_per_class其中是每个类所对应的图像的数目的。
## image_paths是对应的图像所对应的路径的。
## 其中是来对应每个批次所对应的图像的数目的，alpha是我们的所对应的权值的通道的过程的。
def select_triplets(embeddings, nrof_images_per_class, image_paths, people_per_batch, alpha):
    """ Select the triplets for training
    """
    trip_idx = 0
    # 某个人的图片的embedding在emb_arr中的开始的索引
    emb_start_idx = 0
    num_trips = 0
    triplets = []

    # VGG Face: Choosing good triplets is crucial and should strike a balance between
    #  selecting informative (i.e. challenging) examples and swamping training with examples that
    #  are too hard. This is achieve by extending each pair (a, p) to a triplet (a, p, n) by sampling
    #  the image n at random, but only between the ones that violate the triplet loss margin. The
    #  latter is a form of hard-negative mining, but it is not as aggressive (and much cheaper) than
    #  choosing the maximally violating example, as often done in structured output learning.
    # 遍历每一个人
    for i in xrange(people_per_batch):
        # 这个人有多少张图片
        nrof_images = int(nrof_images_per_class[i])
        # 遍历第i个人的所有图片
        for j in xrange(1, nrof_images):
            # 第j张图的embedding在emb_arr 中的位置
            a_idx = emb_start_idx + j - 1
            # 第j张图跟其他所有图片的欧氏距离
            neg_dists_sqr = np.sum(np.square(embeddings[a_idx] - embeddings), 1)
            # 遍历每一对可能的(anchor,postive)图片，记为(a,p)吧
            for pair in xrange(j, nrof_images):  # For every possible positive pair.
                # 第p张图片在emb_arr中的位置
                p_idx = emb_start_idx + pair
                # (a,p)之前的欧式距离
                pos_dist_sqr = np.sum(np.square(embeddings[a_idx] - embeddings[p_idx]))
                # 同一个人的图片不作为negative，所以将距离设为无穷大
                neg_dists_sqr[emb_start_idx:emb_start_idx + nrof_images] = np.NaN
                # all_neg = np.where(np.logical_and(neg_dists_sqr-pos_dist_sqr<alpha, pos_dist_sqr<neg_dists_sqr))[0]  # FaceNet selection
                # 其他人的图片中有哪些图片与a之间的距离-p与a之间的距离小于alpha的
                all_neg = np.where(neg_dists_sqr - pos_dist_sqr < alpha)[0]  # VGG Face selecction
                # 所有可能的negative
                nrof_random_negs = all_neg.shape[0]
                # 如果有满足条件的negative
                if nrof_random_negs > 0:
                    # 从中随机选取一个作为n
                    rnd_idx = np.random.randint(nrof_random_negs)
                    n_idx = all_neg[rnd_idx]
                    # 选到(a,p,n)作为三元组
                    triplets.append((image_paths[a_idx], image_paths[p_idx], image_paths[n_idx]))
                    # print('Triplet %d: (%d, %d, %d), pos_dist=%2.6f, neg_dist=%2.6f (%d, %d, %d, %d, %d)' %
                    #    (trip_idx, a_idx, p_idx, n_idx, pos_dist_sqr, neg_dists_sqr[n_idx], nrof_random_negs, rnd_idx, i, j, emb_start_idx))
                    trip_idx += 1

                num_trips += 1

        emb_start_idx += nrof_images

    np.random.shuffle(triplets)
    return triplets, num_trips, len(triplets)


