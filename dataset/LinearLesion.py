from copy import deepcopy

import torch
import glob
import os
from torchvision import transforms
# import cv2
from PIL import Image
# import pandas as pd
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia
# from utils import get_label_info, one_hot_it
import random

from dataset.transform import blur


def augmentation():
    # augment images with spatial transformation: Flip, Affine, Rotation, etc...
    # see https://github.com/aleju/imgaug for more details
    pass


def augmentation_pixel():
    # augment images with pixel intensity transformation: GaussianBlur, Multiply, etc...
    pass


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir) or os.path.islink(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


class LinearLesion(torch.utils.data.Dataset):
    def __init__(self, dataset_path, scale, mode='train', args=None):
        super().__init__()
        self.mode = mode
        self.args = args
        self.dir_A = os.path.join(dataset_path, 'trainA')
        self.dir_B = os.path.join(dataset_path, 'trainB')
        self.A_paths = sorted(make_dataset(self.dir_A, float("inf")))  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, float("inf")))  # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        self.resize_img = transforms.Resize((256, 256), Image.BILINEAR)
        self.resize_label = transforms.Resize((256, 256), Image.NEAREST)
        self.to_tensor = transforms.ToTensor()
        self.flip = iaa.SomeOf((2, 4),
                               [iaa.Fliplr(0.5),
                                iaa.Flipud(0.5),
                                iaa.Affine(rotate=(-30, 30)),
                                iaa.AdditiveGaussianNoise(scale=(0.0, 0.08 * 255))], random_order=True)
        self.flip_weak = iaa.SomeOf((2, 2),
                                    [iaa.Fliplr(0.5),
                                     iaa.Flipud(0.5),
                                     ], random_order=True)
        self.to_tensor = transforms.ToTensor()
        self.Normalize = transforms.Normalize((0.5), (0.5))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        A_mask_path = A_path.replace('trainA', 'trainA_label')  # make sure index is within then range
        A_uncertainty_mask_path = A_path.replace('trainA', 'trainA_uncertainty_map')  # make sure index is within then range

        index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        B_mask_path = B_path.replace('trainB', 'trainB_label')  # make sure index is within then range
        B_uncertainty_mask_path = B_path.replace('trainB', 'trainB_uncertainty_map')
        A_img = Image.open(A_path).convert('RGB')
        A_img = self.resize_img(A_img)
        A_img = np.array(A_img)

        # B_img = Image.open(B_path).convert('L')
        B_img = Image.open(B_path).convert('RGB')
        B_img = self.resize_img(B_img)
        B_img_s1, B_img_s2 = deepcopy(B_img), deepcopy(B_img)
        B_img = np.array(B_img)
        # 标签读取
        A_uncertainty_mask = Image.open(A_uncertainty_mask_path).convert('L')
        A_uncertainty_mask = self.resize_label(A_uncertainty_mask)
        A_uncertainty_mask = (np.array(A_uncertainty_mask) / 51).astype(np.uint8)
        A_uncertainty_mask[A_uncertainty_mask == 5] = 0
        A_uncertainty_mask[A_uncertainty_mask == 1] = 2
        A_uncertainty_mask[A_uncertainty_mask == 2] = 3
        A_uncertainty_mask[A_uncertainty_mask == 3] = 3
        A_uncertainty_mask[A_uncertainty_mask == 4] = 2

        B_uncertainty_mask = Image.open(B_uncertainty_mask_path).convert('L')
        B_uncertainty_mask = self.resize_label(B_uncertainty_mask)
        B_uncertainty_mask = (np.array(B_uncertainty_mask) / 51).astype(np.uint8)
        B_uncertainty_mask[B_uncertainty_mask != 5] = 0
        B_uncertainty_mask[B_uncertainty_mask == 5] = 1

        # 数据增强
        if random.random() < 0.8:
            B_img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(B_img_s1)
        B_img_s1 = blur(B_img_s1, p=0.5)
        if random.random() < 0.8:
            B_img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(B_img_s2)
        B_img_s2 = blur(B_img_s2, p=0.5)
        B_img_s1 = np.array(B_img_s1)
        B_img_s2 = np.array(B_img_s2)

        A_label = Image.open(A_mask_path).convert('L')
        A_label = self.resize_label(A_label)
        A_label = np.array(A_label)
        A_label[A_label != 255] = 0
        A_label[A_label == 255] = 1
        B_label = Image.open(B_mask_path).convert('L')
        B_label = self.resize_label(B_label)
        B_label = np.array(B_label)
        B_label[B_label != 255] = 0
        B_label[B_label == 255] = 1
        if self.mode == 'train':
            seq_det = self.flip.to_deterministic()  # 确定一个数据增强的序列
            A_img = seq_det.augment_image(A_img)  # 将方法应用在原图像上
            B_img = seq_det.augment_image(B_img)  # 将方法应用在原图像上
            B_img_s1 = seq_det.augment_image(B_img_s1)  # 将方法应用在原图像上
            B_img_s2 = seq_det.augment_image(B_img_s2)  # 将方法应用在原图像上
            segmap_A = ia.SegmentationMapsOnImage(A_label, shape=A_label.shape)  # #将图片转换为SegmentationMapOnImage类型，方便后面可视化
            segmap_B = ia.SegmentationMapsOnImage(B_label, shape=B_label.shape)  # #将图片转换为SegmentationMapOnImage类型，方便后面可视化
            segmap_A_uncertainty = ia.SegmentationMapsOnImage(A_uncertainty_mask, shape=A_uncertainty_mask.shape)  # #将图片转换为SegmentationMapOnImage类型，方便后面可视化
            segmap_B_uncertainty = ia.SegmentationMapsOnImage(B_uncertainty_mask, shape=B_uncertainty_mask.shape)  # #将图片转换为SegmentationMapOnImage类型，方便后面可视化

            B_label = seq_det.augment_segmentation_maps([segmap_B])[0].get_arr().astype(np.uint8)  # 将方法应用在分割标签上，并且转换成np类型
            A_label = seq_det.augment_segmentation_maps([segmap_A])[0].get_arr().astype(np.uint8)  # 将方法应用在分割标签上，并且转换成np类型
            A_uncertainty_mask = seq_det.augment_segmentation_maps([segmap_A_uncertainty])[0].get_arr().astype(np.uint8)  # 将方法应用在分割标签上，并且转换成np类型
            B_uncertainty_mask = seq_det.augment_segmentation_maps([segmap_B_uncertainty])[0].get_arr().astype(np.uint8)  # 将方法应用在分割标签上，并且转换成np类型

        A_label = torch.from_numpy(A_label.copy()).float()
        B_label = torch.from_numpy(B_label.copy()).float()
        A_uncertainty_mask = torch.from_numpy(A_uncertainty_mask.copy()).float()
        B_uncertainty_mask = torch.from_numpy(B_uncertainty_mask.copy()).float()
        A = self.to_tensor(A_img.copy()).float()
        B = self.to_tensor(B_img.copy()).float()
        B_img_s1 = self.to_tensor(B_img_s1.copy()).float()
        B_img_s2 = self.to_tensor(B_img_s2.copy()).float()
        return {'S': A, 'T': B, 'S_paths': A_path, 'T_paths': B_path, 'S_label': A_label, 'T_label': B_label,
                'S_uncertainty_mask': A_uncertainty_mask, 'T_uncertainty_mask': B_uncertainty_mask,
                'T_img_s1': B_img_s1, 'T_img_s2': B_img_s2}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
class LinearLesion_noseg(torch.utils.data.Dataset):
    def __init__(self, dataset_path, scale, mode='train', args=None):
        super().__init__()
        self.mode = mode
        self.args = args
        self.dir_A = os.path.join(dataset_path, 'trainA')
        self.dir_B = os.path.join(dataset_path, 'trainB')
        self.A_paths = sorted(make_dataset(self.dir_A, float("inf")))  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, float("inf")))  # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        self.resize_img = transforms.Resize((256, 256), Image.BILINEAR)
        self.resize_label = transforms.Resize((256, 256), Image.NEAREST)
        self.to_tensor = transforms.ToTensor()
        self.flip = iaa.SomeOf((2, 4),
                               [iaa.Fliplr(0.5),
                                iaa.Flipud(0.5),
                                iaa.Affine(rotate=(-30, 30)),
                                iaa.AdditiveGaussianNoise(scale=(0.0, 0.08 * 255))], random_order=True)
        self.to_tensor = transforms.ToTensor()
        self.Normalize = transforms.Normalize((0.5), (0.5))

    def __getitem__(self, index):

        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        A_mask_path = A_path.replace('trainA', 'trainA_label')  # make sure index is within then range

        index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        B_mask_path = B_path.replace('trainB', 'trainB_label')  # make sure index is within then range
        A_img = Image.open(A_path).convert('RGB')
        A_img = self.resize_img(A_img)
        A_img = np.array(A_img)

        # B_img = Image.open(B_path).convert('L')
        B_img = Image.open(B_path).convert('RGB')
        B_img = self.resize_img(B_img)
        B_img = np.array(B_img)


        A_label = Image.open(A_mask_path).convert('L')
        A_label = self.resize_label(A_label)
        A_label = np.array(A_label)
        A_label[A_label != 255] = 0
        A_label[A_label == 255] = 1
        B_label = Image.open(B_mask_path).convert('L')
        B_label = self.resize_label(B_label)
        B_label = np.array(B_label)
        B_label[B_label != 255] = 0
        B_label[B_label == 255] = 1
        if self.mode == 'train':
            seq_det = self.flip.to_deterministic()  # 确定一个数据增强的序列
            A_img = seq_det.augment_image(A_img)  # 将方法应用在原图像上
            B_img = seq_det.augment_image(B_img)  # 将方法应用在原图像上
            segmap_A = ia.SegmentationMapsOnImage(A_label, shape=A_label.shape)  # #将图片转换为SegmentationMapOnImage类型，方便后面可视化
            segmap_B = ia.SegmentationMapsOnImage(B_label, shape=B_label.shape)  # #将图片转换为SegmentationMapOnImage类型，方便后面可视化

            B_label = seq_det.augment_segmentation_maps([segmap_B])[0].get_arr().astype(np.uint8)  # 将方法应用在分割标签上，并且转换成np类型
            A_label = seq_det.augment_segmentation_maps([segmap_A])[0].get_arr().astype(np.uint8)  # 将方法应用在分割标签上，并且转换成np类型

        A_label = torch.from_numpy(A_label.copy()).float()
        B_label = torch.from_numpy(B_label.copy()).float()
        A = self.to_tensor(A_img.copy()).float()
        B = self.to_tensor(B_img.copy()).float()
        return {'S': A, 'T': B, 'S_paths': A_path, 'T_paths': B_path, 'S_label': A_label, 'T_label': B_label}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
class LinearLesion_val(torch.utils.data.Dataset):
    def __init__(self, dataset_path, scale, mode='val', args=None):
        super().__init__()
        self.mode = mode
        self.args = args
        self.dir_B = os.path.join(dataset_path, 'testB')
        self.B_paths = sorted(make_dataset(self.dir_B, float("inf")))  # load images from '/path/to/data/trainB'
        self.B_size = len(self.B_paths)  # get the size of dataset B
        self.resize_img = transforms.Resize((256, 256), Image.BILINEAR)
        self.resize_label = transforms.Resize((256, 256), Image.NEAREST)
        self.to_tensor = transforms.ToTensor()
        self.flip = iaa.SomeOf((2, 4), [
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Affine(rotate=(-30, 30)),
            iaa.AdditiveGaussianNoise(scale=(0.0, 0.08 * 255))], random_order=True)
        self.to_tensor = transforms.ToTensor()
        self.Normalize = transforms.Normalize((0.5), (0.5))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        B_path = self.B_paths[index]
        B_mask_path = B_path.replace('testB', 'testB_label')  # make sure index is within then range
        B_img = Image.open(B_path).convert('RGB')
        B_img = self.resize_img(B_img)
        B_img = np.array(B_img)
        # 标签读取
        B_label = Image.open(B_mask_path).convert('L')
        B_label = self.resize_label(B_label)
        B_label = np.array(B_label)
        B_label[B_label != 255] = 0
        B_label[B_label == 255] = 1
        # B_label = np.reshape(B_label, (1,) + B_label.shape)
        B_label = torch.from_numpy(B_label.copy()).float()
        B = self.to_tensor(B_img.copy()).float()
        return B, B_label

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return self.B_size
class LinearLesion_fake_image(torch.utils.data.Dataset):
    def __init__(self, dataset_path, scale, mode='train', args=None):
        super().__init__()
        self.mode = mode
        self.args = args
        self.dir_A = os.path.join(dataset_path, 'trainA')
        self.dir_B = os.path.join(dataset_path, 'trainB')
        self.A_paths = sorted(make_dataset(self.dir_A, float("inf")))  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, float("inf")))  # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        self.resize_img = transforms.Resize((256, 256), Image.BILINEAR)
        self.resize_label = transforms.Resize((256, 256), Image.NEAREST)
        self.to_tensor = transforms.ToTensor()

        self.to_tensor = transforms.ToTensor()
        self.Normalize = transforms.Normalize((0.5), (0.5))

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        A_mask_path = A_path.replace('trainA', 'trainA_label')  # make sure index is within then range
        index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        A_img = self.resize_img(A_img)
        A_img = np.array(A_img)
        A_label = Image.open(A_mask_path).convert('L')
        A_label = self.resize_label(A_label)
        A_label = np.array(A_label)
        A_label[A_label != 255] = 0
        A_label[A_label == 255] = 1
        B_img = Image.open(B_path).convert('RGB')
        B_img = self.resize_img(B_img)
        B_img = np.array(B_img)

        A_label = torch.from_numpy(A_label.copy()).float()
        A = self.to_tensor(A_img.copy()).float()
        B = self.to_tensor(B_img.copy()).float()
        return {'S': A, 'T': B, 'S_paths': A_path, 'T_paths': B_path, 'S_label': A_label}


    def __len__(self):
            """Return the total number of images in the dataset.

            As we have two datasets with potentially different number of images,
            we take a maximum of
            """
            return self.A_size

class LinearLesion_test(torch.utils.data.Dataset):
    def __init__(self, args, j, dataset_path):
        super().__init__()
        self.j = j
        self.img_path = os.path.join(dataset_path, 'testB')
        self.image_lists, self.label_lists = self.read_list(self.j, self.img_path)
        self.to_tensor = transforms.ToTensor()
        self.resize_label = transforms.Resize((256, 256), Image.NEAREST)
        self.resize_img = transforms.Resize((256, 256), Image.BILINEAR)
        self.Normalize = transforms.Normalize((0.5), (0.5))

    def __getitem__(self, index):
        B_path = self.image_lists[index]
        B_mask_path = B_path.replace('testB', 'testB_label')  # make sure index is within then range
        B_img = Image.open(B_path).convert('L')
        B_img = self.resize_img(B_img)
        B_img = np.array(B_img)
        # 标签读取
        B_label = Image.open(B_mask_path).convert('L')
        B_label = self.resize_label(B_label)
        B_label = np.array(B_label)
        B_label[B_label != 255] = 0
        B_label[B_label == 255] = 1
        B_label = np.reshape(B_label, (1,) + B_label.shape)
        B_label = torch.from_numpy(B_label.copy()).float()
        B = self.Normalize(self.to_tensor(B_img.copy())).float()
        return B, B_label, B_mask_path

    def __len__(self):
        return len(self.image_lists)

    # 将所有需要训练的image和mask的图片都放进img_list, label_list里
    def read_list(self, j, image_path):
        global label_list
        img_list = []
        png_path = os.listdir(image_path)  # f1下面的所有病人路径
        path = png_path[j]
        path = os.path.join(image_path, path)
        img_list += glob.glob(path + '/*.png')
        label_list = [x.replace('testB', 'testB_label').split('.')[0] + '.png' for x in img_list]

        return img_list, label_list
