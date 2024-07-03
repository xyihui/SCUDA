# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import cv2
import numpy as np
import torch
from torch import nn

# from imaginaire.generators.funit import (MLP, ContentEncoder, Decoder,
#                                          StyleEncoder)
from models.imaginaire.generators.funit_org import StyleEncoder, ContentEncoder, MLP, Decoder


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
        # self.generator = COCOFUNITTranslator(**vars(gen_cfg))
        self.generator = COCOFUNITTranslator()

    def forward(self, content, style):
        r"""In the FUNIT's forward pass, it generates a content embedding and
        a style code from the content image, and a style code from the style
        image. By mixing the content code and the style code from the content
        image, we reconstruct the input image. By mixing the content code and
        the style code from the style image, we have a translation output.

        Args:
            data (dict): Training data at the current iteration.
        """
        content_a = self.generator.content_encoder(content)
        content_b = self.generator.content_encoder(style)
        style_b = self.generator.style_encoder(style)
        fft = self.fft_coco2(content_a, content_b)
        images_trans = self.generator.decode(content_a, style_b, fft)
        # images_recon = self.generator.decode(content_a, style_a)

        # net_G_output = dict(images_trans=images_trans,
        #                     images_recon=images_recon)
        # return images_trans, images_recon
        return images_trans

    def Normalize_cl(self, data):
        # m = np.mean(data)
        mx = data.max()
        mn = data.min()
        # return [(float(i) - m) / (mx - mn) for i in data]
        return (data - mn) / (mx - mn)

    def fft_coco1(self, content, style):
        B, C, H, W = content.size()
        content = content.cpu().detach().numpy()
        style = style.cpu().detach().numpy()
        ifft = []
        for b in range(B):
            ifft1 = []
            for c in range(C):
                # dft1 = cv2.dft(np.float32(content[b][c]), flags=cv2.DFT_COMPLEX_OUTPUT)
                # dft_shift1 = np.fft.fftshift(dft1)
                content_fft = np.fft.fft2(np.float32(content[b][c]))
                style_fft = np.fft.fft2(np.float32(style[b][c]))
                P1 = np.abs(style_fft) * np.exp2(np.angle(content_fft) * 1j)
                p1 = np.fft.ifft2(P1)
                # 将低频区域转移到中间位置
                # f_ishift1 = np.fft.ifftshift(fshift1)
                # img_back1 = cv2.idft(f_ishift1)
                # 使用cv2.magnitude将实部和虚部投影到空间域，将实部和虚部转换为实部
                # img_back1 = cv2.magnitude(img_back1[:, :, 0], img_back1[:, :, 1])
                ifft1.append(p1)
            ifft.append(np.stack(ifft1))
        return torch.tensor(self.Normalize_cl(np.stack(ifft))).cuda()

    def extract_ampl_phase(self, fft_im):
        # fft_im: size should be bx3xhxwx2
        fft_amp = fft_im[:, :, :, :, 0] ** 2 + fft_im[:, :, :, :, 1] ** 2
        fft_amp = torch.sqrt(fft_amp + 1e-20)
        fft_pha = torch.atan2(fft_im[:, :, :, :, 1], fft_im[:, :, :, :, 0])
        return fft_amp, fft_pha

    def fft_coco2(self, content, style):
        content = torch.fft.fft2(content, dim=(-2, -1))  # fft tranform
        content = torch.stack((content.real, content.imag), -1)
        style = torch.fft.fft2(style, dim=(-2, -1))  # fft tranform
        style = torch.stack((style.real, style.imag), -1)
        # amp1, pha1 = self.extract_ampl_phase(fft1)
        amp_content, pha_content = self.extract_ampl_phase(content)
        amp_style, pha_style = self.extract_ampl_phase(style)
        fft_ = torch.zeros(content.size(), dtype=torch.float)
        fft_[:, :, :, :, 0] = torch.cos(pha_content) * amp_style
        fft_[:, :, :, :, 1] = torch.sin(pha_content) * amp_style
        fft_ = fft_[:, :, :, :, 0] + 1j * fft_[:, :, :, :, 1]
        # fft_ = self.batch_fftshift2d(fft_)
        out = torch.fft.ifft2(fft_)
        return out.cuda()

    def roll_n(self, X, axis, n):
        f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
        b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
        front = X[f_idx]
        back = X[b_idx]
        return torch.cat([back, front], axis)

    def batch_fftshift2d(self, x):
        real, imag = torch.unbind(x, -1)
        for dim in range(1, len(real.size())):
            n_shift = real.size(dim) // 2
            if real.size(dim) % 2 != 0:
                n_shift += 1  # for odd-sized images
            real = self.roll_n(real, axis=dim, n=n_shift)
            imag = self.roll_n(imag, axis=dim, n=n_shift)
        return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

    def batch_ifftshift2d(self, x):
        real, imag = torch.unbind(x, -1)
        for dim in range(len(real.size()) - 1, 0, -1):
            real = self.roll_n(real, axis=dim, n=real.size(dim) // 2)
            imag = self.roll_n(imag, axis=dim, n=imag.size(dim) // 2)
        return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

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
                 usb_dims=1024,
                 num_res_blocks=2,
                 num_mlp_blocks=3,
                 num_downsamples_style=4,
                 num_downsamples_content=2,
                 num_image_channels=3,
                 weight_norm_type='',
                 **kwargs):
        super().__init__()

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

        self.decoder = Decoder(2 * self.content_encoder.output_dim,
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
        self.mlp_content = MLP(self.content_encoder.output_dim,
                               style_dims,
                               num_filters_mlp,
                               num_content_mlp_blocks,
                               'none',
                               'relu')

        self.mlp_style = MLP(style_dims + usb_dims,
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

    def decode(self, content, style, fft):
        r"""Generate images by combining their content and style codes.

        Args:
            content (tensor): Content code tensor.
            style (tensor): Style code tensor.
        """
        content_style_code = content.mean(3).mean(2)
        content_style_code = self.mlp_content(content_style_code)
        batch_size = style.size(0)
        usb = self.usb.repeat(batch_size, 1)
        style = style.view(batch_size, -1)
        style_in = self.mlp_style(torch.cat([style, usb], 1))
        coco_style = style_in * content_style_code
        coco_style = self.mlp(coco_style)
        images = self.decoder(torch.cat([content, fft.float()], dim=1), coco_style)
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
