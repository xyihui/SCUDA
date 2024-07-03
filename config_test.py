# -*- coding: utf-8 -*-
"""
Created on Tue June 12 14:03:52 2024

@author: Lang Chen
"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class DefaultConfig_test(object):
    test_result_path = r'code/results'
    coarse_model = r'f1.pth.tar'
    dataset_path = r"adaption_datasets/f1"
    patients = [62, 65]     # Number of patients divided
    fold = 1
    mode = 'test'
    checkpoint_step = 1
    is_label = True
    validation_step = 1
    crop_height = 256
    crop_width = 256
    batchSize = 1
    input_nc = 3
    output_nc = 3
    ngf = 64
    ndf = 64
    gpu_ids = 0
    gan_mode = 'lsgan'
    beta1 = 0.5
    feature_channel = 256
    name = 'pancreases'
    net_work = 'UNet'
    lr = 0.01
    lr_mode = 'poly'
    momentum = 0.9
    weight_decay = 1e-4
    num_workers = 0
    num_classes = 2
    cuda = '0'
    use_gpu = True
