# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import numpy as np
import torch
from torch import nn

# from imaginaire.generators.funit import (MLP, ContentEncoder, Decoder,
#                                          StyleEncoder)
from models.imaginaire.generators.funit import StyleEncoder, ContentEncoder, MLP, Decoder
from torch.nn import functional as F

class Generator(nn.Module):
    r"""COCO-FUNIT Generator.
    """

    def __init__(self, gen_cfg=None, data_cfg=None):
        r"""COCO-FUNIT Generator constructor.

        Args:
            gen_cfg (obj): Generator definition part of the yaml config file.
            data_cfg (obj): Data definition part of the yaml config file.
        """
        super().__init__()
        self.generator = COCOFUNITTranslator()

    def forward(self, content, style, is_IDT=False):
        r"""In the FUNIT's forward pass, it generates a content embedding and
        a style code from the content image, and a style code from the style
        image. By mixing the content code and the style code from the content
        image, we reconstruct the input image. By mixing the content code and
        the style code from the style image, we have a translation output.

        Args:
            data (dict): Training data at the current iteration.
        """
        if not is_IDT:
            content_a = self.generator.content_encoder(content)
            content_zero = self.generator.content_encoder(style)
            # style_a = self.generator.style_encoder(content)
            style_b = self.generator.style_encoder(style)
            style_zero = self.generator.style_encoder(content)
            images_trans = self.generator.decode(content_a, style_b)
            # images_recon = self.generator.decode(content_a, style_a)

            # net_G_output = dict(images_trans=images_trans,
            #                     images_recon=images_recon)
            # return images_trans, images_recon
            return images_trans, content_a, style_b, content_zero, style_zero
        if is_IDT:
            content_a = self.generator.content_encoder(content)
            # content_zero = self.generator.content_encoder(style)
            # style_a = self.generator.style_encoder(content)
            style_b = self.generator.style_encoder(style)
            # style_zero = self.generator.style_encoder(content)
            images_trans = self.generator.decode(content_a, style_b)
            # images_recon = self.generator.decode(content_a, style_a)

            # net_G_output = dict(images_trans=images_trans,
            #                     images_recon=images_recon)
            # return images_trans, images_recon
            return images_trans

    def inference(self, data, keep_original_size=True):
        r"""COCO-FUNIT inference.

        Args:
            data (dict): Training data at the current iteration.
              - images_content (tensor): Content images.
              - images_style (tensor): Style images.
            a2b (bool): If ``True``, translates images from domain A to B,
                otherwise from B to A.
            keep_original_size (bool): If ``True``, output image is resized
            to the input content image size.
        """
        content_a = self.generator.content_encoder(data['images_content'])
        style_b = self.generator.style_encoder(data['images_style'])
        output_images = self.generator.decode(content_a, style_b)
        if keep_original_size:
            height = data['original_h_w'][0][0]
            width = data['original_h_w'][0][1]
            # print('( H, W) = ( %d, %d)' % (height, width))
            output_images = torch.nn.functional.interpolate(
                output_images, size=[height, width])
        file_names = data['key']['images_content'][0]
        return output_images, file_names


class COCOFUNITTranslator(nn.Module):
    r"""COCO-FUNIT Generator architecture.

    Args:
        num_filters (int): Base filter numbers.
        num_filters_mlp (int): Base filter number in the MLP module.
        style_dims (int): Dimension of the style code.
        usb_dims (int): Dimension of the universal style bias code.
        num_res_blocks (int): Number of residual blocks at the end of the
            content encoder.
        num_mlp_blocks (int): Number of layers in the MLP module.
        num_downsamples_content (int): Number of times we reduce
            resolution by 2x2 for the content image.
        num_downsamples_style (int): Number of times we reduce
            resolution by 2x2 for the style image.
        num_image_channels (int): Number of input image channels.
        weight_norm_type (str): Type of weight normalization.
            ``'none'``, ``'spectral'``, or ``'weight'``.
    """

    def __init__(self,
                 num_filters=64,
                 num_filters_mlp=256,
                 style_dims=64,
                 usb_dims=64,
                 num_res_blocks=2,
                 num_mlp_blocks=3,
                 num_downsamples_style=4,
                 num_downsamples_content=2,
                 num_image_channels=3,
                 weight_norm_type='',
                 **kwargs):
        super().__init__()
        self.conv_avg_style = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                       nn.Conv2d(num_filters_mlp, style_dims, 1, 1, 0),
                                       nn.BatchNorm2d(style_dims))
        self.conv_max_style = nn.Sequential(nn.AdaptiveMaxPool2d(1),
                                      nn.Conv2d(num_filters_mlp, style_dims, 1, 1, 0),
                                      nn.BatchNorm2d(style_dims))
        self.conv_avg_content = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                            nn.Conv2d(num_filters_mlp, style_dims, 1, 1, 0),
                                            nn.BatchNorm2d(style_dims))
        self.conv_max_content = nn.Sequential(nn.AdaptiveMaxPool2d(1),
                                            nn.Conv2d(num_filters_mlp, style_dims, 1, 1, 0),
                                            nn.BatchNorm2d(style_dims))
        self.conv = nn.Sequential(nn.Conv2d(num_filters_mlp + 3, num_filters_mlp, 3, 1, 1),
                                  nn.BatchNorm2d(num_filters_mlp),
                                  nn.ReLU(inplace=True))
        self.conv_style = nn.Sequential(nn.Conv2d(num_filters_mlp + 3, num_filters_mlp, 3, 1, 1),
                                        nn.BatchNorm2d(num_filters_mlp),
                                        nn.ReLU(inplace=True))
        self.style_encoder = StyleEncoder(num_downsamples_style,
                                          num_image_channels,
                                          num_filters,
                                          style_dims,
                                          'reflect',
                                          'batch',
                                          weight_norm_type,
                                          'relu')

        self.content_encoder = ContentEncoder(num_downsamples_content,
                                              num_res_blocks,
                                              num_image_channels,
                                              num_filters,
                                              'reflect',
                                              'batch',
                                              weight_norm_type,
                                              'relu')

        self.decoder = Decoder(self.content_encoder.output_dim,
                               num_filters_mlp,
                               num_image_channels,
                               num_downsamples_content,
                               'reflect',
                               weight_norm_type,
                               'relu')

        self.usb = torch.nn.Parameter(torch.randn(1, usb_dims))

        self.mlp = MLP(style_dims,
                       num_filters_mlp,
                       num_filters_mlp,
                       num_mlp_blocks,
                       'none',
                       'relu')

        num_content_mlp_blocks = 2
        num_style_mlp_blocks = 2
        self.mlp_content = MLP(style_dims*2,
                               style_dims,
                               num_filters_mlp,
                               num_content_mlp_blocks,
                               'none',
                               'relu')

        self.mlp_style = MLP(style_dims*2 + usb_dims,
                             style_dims,
                             num_filters_mlp,
                             num_style_mlp_blocks,
                             'none',
                             'relu')

    def forward(self, images):
        r"""Reconstruct the input image by combining the computer content and
        style code.

        Args:
            images (tensor): Input image tensor.
        """
        # reconstruct an image
        content, style = self.encode(images)
        images_recon = self.decode(content, style)
        return images_recon

    def encode(self, images):
        r"""Encoder images to get their content and style codes.

        Args:
            images (tensor): Input image tensor.
        """
        style = self.style_encoder(images)
        content = self.content_encoder(images)
        return content, style

    def parse_dynamic_one_params(self, params, channels, model=None):
        global weight_splits
        assert params.dim() == 3
        if model == 'v':
            weight_splits = params.reshape(channels, -1, 1, 3)
        if model == 'h':
            weight_splits = params.reshape(channels, -1, 3, 1)
        if model == 's':
            weight_splits = params.reshape(channels, -1, 3, 3)
        return weight_splits

    def parse_dynamic_params(self, params, channels, weight_nums, bias_nums):
        assert params.dim() == 2
        assert len(weight_nums) == len(bias_nums)
        assert params.size(1) == sum(weight_nums) + sum(bias_nums)

        num_insts = params.size(0)
        num_layers = len(weight_nums)

        params_splits = list(torch.split_with_sizes(
            params, weight_nums + bias_nums, dim=1
        ))

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l < num_layers - 1:
                weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
            else:
                weight_splits[l] = weight_splits[l].reshape(num_insts * 3, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * 3)

        return weight_splits, bias_splits

    def decode(self, content, style, real_kv=None, real_kh=None, real_ks=None, real_B_kv=None, real_B_kh=None, real_B_ks=None):
        batch_size = content.size(0)
        if (real_kv != None) and (real_kh != None) and (real_ks != None):
            self.fusion_label_content = torch.cat([F.conv2d(content.view(1, -1, content.size(2), content.size(3)), weight=self.parse_dynamic_one_params(real_kv, 2, 'v'),
                                                            stride=1, padding=(0, 1), groups=batch_size).reshape(batch_size, -1,  content.size(2), content.size(3)),
                                                   F.conv2d(content.view(1, -1, content.size(2), content.size(3)), weight=self.parse_dynamic_one_params(real_kh, 2, 'h'),
                                                            stride=1, padding=(1, 0), groups=batch_size).reshape(batch_size, -1,  content.size(2), content.size(3)),
                                                   F.conv2d(content.view(1, -1, content.size(2), content.size(3)), weight=self.parse_dynamic_one_params(real_ks, 2, 's'),
                                                            stride=1, padding=(1, 1), groups=batch_size).reshape(batch_size, -1,  content.size(2), content.size(3)),
                                                   content], dim=1)
            content = self.conv(self.fusion_label_content)
        if (real_B_kv != None) and (real_B_kh != None) and (real_B_ks != None):
            self.fusion_label_style = torch.cat([F.conv2d(style.view(1, -1, style.size(2), style.size(3)), weight=self.parse_dynamic_one_params(real_B_kv, 2, 'v'),
                                                          stride=1, padding=(0, 1), groups=batch_size).reshape(batch_size, -1,  style.size(2), style.size(3)),
                                                 F.conv2d(style.view(1, -1, style.size(2), style.size(3)), weight=self.parse_dynamic_one_params(real_B_kh, 2, 'h'),
                                                          stride=1, padding=(1, 0), groups=batch_size).reshape(batch_size, -1,  style.size(2), style.size(3)),
                                                 F.conv2d(style.view(1, -1, style.size(2), style.size(3)), weight=self.parse_dynamic_one_params(real_B_ks, 2, 's'),
                                                          stride=1, padding=(1, 1), groups=batch_size).reshape(batch_size, -1,  style.size(2), style.size(3)),
                                                          style], dim=1)
            style = self.conv_style(self.fusion_label_style)
        style_code = torch.cat([self.conv_avg_style(style), self.conv_max_style(style)], 1).squeeze()
        content_code = torch.cat([self.conv_avg_style(content), self.conv_max_style(content)], 1).squeeze()
        content_code = self.mlp_content(content_code)
        batch_size = style_code.size(0)
        usb = self.usb.repeat(batch_size, 1)
        style_code = style_code.view(batch_size, -1)
        style_in = self.mlp_style(torch.cat([style_code, usb], 1))  # 风格向量
        coco_style = style_in * content_code

        coco_style = self.mlp(coco_style)
        images = self.decoder(content, coco_style)
        return images


if __name__ == '__main__':
    model = Generator()
    data = {}
    data['images_content'] = torch.randn(2, 1, 256, 256)
    data['images_style'] = torch.randn(2, 1, 256, 256)
    out = model(data)
    print(model)
    # 定义总参数量、可训练参数量及非可训练参数量变量
    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0

    # 遍历model.parameters()返回的全局参数列表
    for param in model.parameters():
        mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
        Total_params += mulValue  # 总参数量
        if param.requires_grad:
            Trainable_params += mulValue  # 可训练参数量
        else:
            NonTrainable_params += mulValue  # 非可训练参数量

    print(f'Total params: {Total_params}')
    print(f'Trainable params: {Trainable_params}')
    print(f'Non-trainable params: {NonTrainable_params}')
