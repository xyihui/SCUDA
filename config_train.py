# -*- coding: utf-8 -*-
"""
Created on Tue June 12 14:03:52 2024

@author: Lang Chen
"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class DefaultConfig(object):
    checkpoint_dir = r'code/results/checkpoint/'
    dataset_path = r"D:\pytorch-CycleGAN-and-pix2pix-master\datasets\pingsao_2_jingmai_f1_3c"
    # checkpoint_dir = r'code/results/checkpoint'
    # dataset_path = r"code/adaption_datasets/f1"
    num_epoch = 200
    s1_num_epoch = 100
    fold = 1
    mode = 'train'
    kernel_size = 3
    checkpoint_step = 5
    validation_step = 1
    crop_height = 256
    crop_width = 256
    batchSize = 2
    input_nc = 3
    output_nc = 3
    feature_channel = 256
    feature_size_height = 64
    feature_size_width = 64
    epoch_save_list = [s1_num_epoch-15*4, s1_num_epoch-15*3, s1_num_epoch-15*2, s1_num_epoch-15, s1_num_epoch]
    ngf = 64
    ndf = 64
    gpu_ids = 0
    gan_mode = 'lsgan'
    beta1 = 0.5
    size = 256
    name = 'pancreases'
    net_work = 'UNet'
    lr = 0.01
    lr_mode = 'poly'
    momentum = 0.9
    weight_decay = 1e-4
    num_workers = 2
    num_classes = 2
    cuda = '0'
    use_gpu = True
