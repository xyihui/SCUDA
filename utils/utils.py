# import torch.nn as nn
import random
from torch.autograd import Variable

import torch
from torch.nn import functional as F
# from PIL import Image
import numpy as np
import pandas as pd
# import os
import os.path as osp
import shutil


# import math

def save_checkpoint(state, best_pred, epoch, is_best, checkpoint_path, filename='./checkpoint/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename,
                        osp.join(checkpoint_path, 'model_{:03d}_{:.4f}.pth.tar'.format((epoch + 1), best_pred)))


def save_checkpoint_2(state, best_convlstm_pred, best_pred, epoch, is_best, checkpoint_path,
                      filename='./checkpoint/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, osp.join(checkpoint_path,
                                           'model_{:03d}_dice1_{:.4f}_dice2_{:.4f}.pth.tar'.format((epoch + 1),
                                                                                                   best_pred,
                                                                                                   best_convlstm_pred)))


def save_checkpoint_3(state, best_convlstm_pred, Dice1, Dice2, epoch, is_best, checkpoint_path,
                      filename='./checkpoint/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, osp.join(checkpoint_path,
                                           'model_{:03d}_dice1_{:.4f}_dice2_{:.4f}_dice3_{:.4f}.pth.tar'.format(
                                               (epoch + 1), Dice1, Dice2, best_convlstm_pred)))


def save_checkpoint_4(state, best_convlstm_pred, Dice1, Dice2, epoch, is_best, checkpoint_path,
                      filename='./checkpoint/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, osp.join(checkpoint_path,
                                           'model_{:03d}_dice4_{:.4f}_dice5_{:.4f}_dice6_{:.4f}.pth.tar'.format(
                                               (epoch + 1), Dice1, Dice2, best_convlstm_pred)))


def save_checkpoint_5(state, best_pred, epoch, is_best, checkpoint_path, filename='./checkpoint/checkpoint.pth.tar'):
    torch.save(state, filename)  # state是训练过程中产生的一些信息的整合构成的字典
    # if is_best:
    shutil.copyfile(filename, osp.join(checkpoint_path, 'model_{:03d}_{:.4f}.pth.tar'.format((epoch + 1), best_pred)))


def save_checkpoint_6(state, best_convlstm_pred, Dice1, Dice2, epoch, is_best, checkpoint_path,
                      filename='./checkpoint/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, osp.join(checkpoint_path,
                                           'model_{:03d}_dice1_{:.4f}_dice4_{:.4f}_dice5_{:.4f}.pth.tar'.format(
                                               (epoch + 1), Dice1, Dice2, best_convlstm_pred)))


# def save_checkpoint(state,best_pred, epoch,is_best,checkpoint_path,filename='./checkpoint/checkpoint.pth.tar'):
#     torch.save(state, filename) # state是训练过程中产生的一些信息的整合构成的字典
#     # if is_best:
#     shutil.copyfile(filename, osp.join(checkpoint_path,'model_{:03d}_{:.4f}.pth.tar'.format((epoch + 1),best_pred)))

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


def adjust_learning_rate(opt, optimizer, epoch):
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs(step = 30)
    """
    if opt.lr_mode == 'step':
        lr = opt.lr * (0.1 ** (epoch // opt.step))
    elif opt.lr_mode == 'poly':
        lr = opt.lr * (1 - epoch / opt.num_epochs) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(opt.lr_mode))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_learning_rate_model(opt, optimizer, epoch):
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs(step = 30)
    """
    if opt.lr_mode == 'step':
        lr = opt.lr_model * (0.1 ** (epoch // opt.step))
    elif opt.lr_mode == 'poly':
        lr = opt.lr_model * (1 - epoch / opt.num_epochs) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(opt.lr_mode))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_learning_rate_lstm(opt, optimizer, epoch):
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs(step = 30)
    """
    if opt.lr_mode == 'step':
        lr = opt.lr_lstm * (0.1 ** (epoch // opt.step))
    elif opt.lr_mode == 'poly':
        lr = opt.lr_lstm * (1 - epoch / opt.num_epochs) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(opt.lr_mode))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def one_hot_it(label, label_info):
    # return semantic_map -> [H, W, num_classes]
    semantic_map = []
    for info in label_info:
        color = label_info[info]
        # colour_map = np.full((label.shape[0], label.shape[1], label.shape[2]), colour, dtype=int)
        equality = np.equal(label, color)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
    return semantic_map


def compute_score_multi(predict, target, forground=1, smooth=1):
    score = 0
    count = 0
    target[target != forground] = 0
    predict[predict != forground] = 0
    assert (predict.shape == target.shape)
    overlap = ((predict == forground) * (target == forground)).sum()  # TP
    union = (predict == forground).sum() + (target == forground).sum() - overlap  # FP+FN+TP
    FP = (predict == forground).sum() - overlap  # FP
    FN = (target == forground).sum() - overlap  # FN
    TN = target.shape[0] * target.shape[1] - union  # TN

    # print('overlap:',overlap)
    dice = (2 * overlap + smooth) / (union + overlap + smooth)

    precsion = ((predict == target).sum() + smooth) / (target.shape[0] * target.shape[1] + smooth)

    jaccard = (overlap + smooth) / (union + smooth)

    Sensitivity = (overlap + smooth) / ((target == forground).sum() + smooth)

    Specificity = (TN + smooth) / (FP + TN + smooth)

    return dice, precsion, jaccard, Sensitivity, Specificity


def eval_multi_seg(predict, target, forground=1):
    pred_seg = torch.argmax(torch.exp(predict), dim=1).int()
    pred_seg = pred_seg.data.cpu().numpy()
    label_seg = target.data.cpu().numpy().astype(dtype=np.int)
    assert (pred_seg.shape == label_seg.shape)

    Dice = []
    Precsion = []
    Jaccard = []
    Sensitivity = []
    Specificity = []

    n = pred_seg.shape[0]

    for i in range(n):
        dice, precsion, jaccard, sensitivity, specificity = compute_score_multi(pred_seg[i], label_seg[i])
        Dice.append(dice)
        Precsion.append(precsion)
        Jaccard.append(jaccard)
        Sensitivity.append(sensitivity)
        Specificity.append(specificity)

    return Dice, Precsion, Jaccard, Sensitivity, Specificity


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


# AadIN
def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def compute_per_dice(pred, label, classes):
    pred = pred.flatten()
    label = label.flatten()

    I = np.zeros(classes)
    U = np.zeros(classes)
    eps = 1e-6
    per_dice = []
    for index in range(classes):
        pred_i = pred == index
        label_i = label == index
        if label_i.sum() == 0:
            per_dice.append(1)
        else:
            I[index] = float(np.sum(np.logical_and(label_i, pred_i)))  # TP
            U[index] = float(np.sum(np.logical_or(label_i, pred_i)))  # TP + FN + FP
            per_dice.append((2 * I[index] + eps) / (U[index] + I[index] + eps))

    return per_dice


def compute_accuracy(pred, label):
    eps = 1e-6
    valid = (label >= 0)
    acc_sum = (valid * (pred == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum + eps) / (valid_sum + eps)
    return acc


def compute_evaluation(pred, label, classes):
    print(pred.shape, label.shape)
    pred = pred.flatten()
    label = label.flatten()
    eps = 1e-6
    per_dice = []
    per_jaccard = []
    per_Precision = []
    per_Sensitivity = []
    accuracy = compute_accuracy(pred, label)
    for index in range(classes):
        pred_i = pred == index
        label_i = label == index
        if label_i.sum() == 0:
            per_dice.append(1)
            per_jaccard.append(1)
            per_Precision.append(1)
            per_Sensitivity.append(1)
        else:
            I = float(np.sum(np.logical_and(label_i, pred_i)))  # TP
            U = float(np.sum(np.logical_or(label_i, pred_i)))  # TP + FN + FP
            FP = pred_i.sum() - I
            FN = label_i.sum() - I
            dice = (2 * I + eps) / (U + I + eps)
            jaccard = I / (U + eps)
            Precision = I / (I + FP + eps)
            Sensitivity = I / (I + FN + eps)
            per_dice.append(dice)
            per_jaccard.append(jaccard)
            per_Precision.append(Precision)
            per_Sensitivity.append(Sensitivity)

    return per_dice, per_jaccard, per_Precision, per_Sensitivity, accuracy


def compute_score_single(predict, target, forground=1, smooth=1):
    score = 0
    count = 0
    target[target != forground] = 0
    predict[predict != forground] = 0
    assert (predict.shape == target.shape)
    overlap = ((predict == forground) * (target == forground)).sum()  # TP
    union = (predict == forground).sum() + (target == forground).sum() - overlap  # FP+FN+TP
    FP = (predict == forground).sum() - overlap  # FP
    FN = (target == forground).sum() - overlap  # FN
    TN = target.shape[0] * target.shape[1] * target.shape[2] - union  # TN

    accuracy = compute_accuracy(predict, target)
    # print('overlap:',overlap)
    dice = (2 * overlap + smooth) / (union + overlap + smooth)

    precsion = ((predict == target).sum() + smooth) / (target.shape[0] * target.shape[1] * target.shape[2] + smooth)

    jaccard = (overlap + smooth) / (union + smooth)

    Sensitivity = (overlap + smooth) / ((target == forground).sum() + smooth)

    Specificity = (TN + smooth) / (FP + TN + smooth)

    return dice, precsion, jaccard, Sensitivity, Specificity, accuracy


def eval_single_seg(predict, target):
    pred_seg = torch.round(torch.sigmoid(predict)).int()
    pred_seg = pred_seg.data.cpu().numpy()
    label_seg = target.data.cpu().numpy().astype(dtype=np.int)
    assert (pred_seg.shape == label_seg.shape)

    Dice = []
    ACCu = []
    Precsion = []
    Jaccard = []
    Sensitivity = []
    Specificity = []

    n = pred_seg.shape[0]

    for i in range(n):
        dice, precsion, jaccard, sensitivity, specificity, acc = compute_score_single(pred_seg[i], label_seg[i])
        Dice.append(dice)
        Precsion.append(precsion)
        Jaccard.append(jaccard)
        Sensitivity.append(sensitivity)
        Specificity.append(specificity)
        ACCu.append(acc)

    return Dice, Precsion, Jaccard, Sensitivity, Specificity, ACCu


def eval_single_seg_lstm(predict, target, forground=1):
    pred_seg = torch.round(predict).int()
    pred_seg = pred_seg.data.cpu().numpy()
    label_seg = target.data.cpu().numpy().astype(dtype=np.int)
    assert (pred_seg.shape == label_seg.shape)

    Dice = []
    Precsion = []
    Jaccard = []
    Sensitivity = []
    Specificity = []

    n = pred_seg.shape[0]

    for i in range(n):
        dice, precsion, jaccard, sensitivity, specificity = compute_score_single(pred_seg[i], label_seg[i])
        Dice.append(dice)
        Precsion.append(precsion)
        Jaccard.append(jaccard)
        Sensitivity.append(sensitivity)
        Specificity.append(specificity)

    return Dice, Precsion, Jaccard, Sensitivity, Specificity


def batch_pix_accuracy(pred, label, nclass=1):
    if nclass == 1:
        pred = torch.round(torch.sigmoid(pred)).int()
        pred = pred.cpu().numpy()
    else:
        pred = torch.max(pred, dim=1)
        pred = pred.cpu().numpy()
    label = label.cpu().numpy()
    pixel_labeled = np.sum(label >= 0)
    pixel_correct = np.sum(pred == label)

    assert pixel_correct <= pixel_labeled, \
        "Correct area should be smaller than Labeled"

    return pixel_correct, pixel_labeled


def batch_intersection_union(predict, target, nclass):
    """Batch Intersection of Union
    Args:
        predict: input 4D tensor
        target: label 3D tensor
        nclass: number of categories (int),note: not include background
    """
    if nclass == 1:
        pred = torch.round(torch.sigmoid(predict)).int()
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()
        area_inter = np.sum(pred * target)
        area_union = np.sum(pred) + np.sum(target) - area_inter

        return area_inter, area_union

    if nclass > 1:
        _, predict = torch.max(predict, 1)
        mini = 1
        maxi = nclass
        nbins = nclass
        predict = predict.cpu().numpy() + 1
        target = target.cpu().numpy() + 1
        # target = target + 1

        predict = predict * (target > 0).astype(predict.dtype)
        intersection = predict * (predict == target)
        # areas of intersection and union
        area_inter, _ = np.histogram(intersection, bins=nbins - 1, range=(mini + 1, maxi))
        area_pred, _ = np.histogram(predict, bins=nbins - 1, range=(mini + 1, maxi))
        area_lab, _ = np.histogram(target, bins=nbins - 1, range=(mini + 1, maxi))
        area_union = area_pred + area_lab - area_inter
        assert (area_inter <= area_union).all(), \
            "Intersection area should be smaller than Union area"
        return area_inter, area_union
import torch.fft as fft

def sam(feature_map):
    # 对特征图进行傅里叶变换
    fourier_transform = fft.fftn(feature_map, dim=(-2, -1))

    # 计算幅度谱和相位谱
    amplitude_spectrum = torch.abs(fourier_transform)
    phase_spectrum = torch.angle(fourier_transform)

    feature_map = torch.zeros(fourier_transform.size()).cuda()
    feature_map[:, :, 128 - 32:128 + 32, 128 - 32:128 + 32] = 1
    amplitude_spectrum = amplitude_spectrum * feature_map

    # 通过极坐标转换计算复数表示
    fourier_transform = amplitude_spectrum * torch.exp(1j * phase_spectrum)

    # 进行反傅里叶变换
    reconstructed_image = fft.ifft2(fourier_transform).real
    return reconstructed_image

def bem(feature_map):
    # 对特征图进行傅里叶变换
    fourier_transform = fft.fftn(feature_map, dim=(-2, -1))

    # 计算幅度谱和相位谱
    phase_spectrum = torch.angle(fourier_transform)

    fourier_transform = torch.exp(1j * phase_spectrum)

    # 进行反傅里叶变换
    reconstructed_image = torch.fft.ifft2(fourier_transform).real

    # 进行反傅里叶变换
    return reconstructed_image + feature_map
def pixel_accuracy(im_pred, im_lab):
    im_pred = np.asarray(im_pred)
    im_lab = np.asarray(im_lab)

    # Remove classes from unlabeled pixels in gt image. 
    # We should not penalize detections in unlabeled portions of the image.
    pixel_labeled = np.sum(im_lab > 0)
    pixel_correct = np.sum((im_pred == im_lab) * (im_lab > 0))
    # pixel_accuracy = 1.0 * pixel_correct / pixel_labeled
    return pixel_correct, pixel_labeled


def reverse_one_hot(image):
    """
	Transform a 2D array in one-hot format (depth is num_classes),
	to a 2D array with only 1 channel, where each pixel value is
	the classified class key.

	# Arguments
		image: The one-hot format image

	# Returns
		A 2D array with the same width and height as the input, but
		with a depth size of 1, where each pixel value is the classified
		class key.
	"""
    # w = image.shape[0]
    # h = image.shape[1]
    # x = np.zeros([w,h,1])

    # for i in range(0, w):
    #     for j in range(0, h):
    #         index, value = max(enumerate(image[i, j, :]), key=operator.itemgetter(1))
    #         x[i, j] = index
    image = image.permute(1, 2, 0)
    x = torch.argmax(image, dim=-1)
    return x


def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.

    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """

    label_values = [label_values[key] for key in label_values]
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x
