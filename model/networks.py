# -*- coding: utf-8 -*-
"""
Created on Tue June 12 14:03:52 2024

@author: Lang Chen
"""
import os
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from model import layers


###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the networkss
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize networkss weights.

    Parameters:
        net (networkss)   -- networkss to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize networkss with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a networkss: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the networkss weights
    Parameters:
        net (networkss)      -- the networkss to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the networkss runs on: e.g., 0,1,2

    Return an initialized networkss.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02,
             gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the networkss: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the networkss runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def feature_extractor(input_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02,
                      gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = Resnet_feature_extractor(input_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=4)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def FusionGenerator(input_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02,
                      gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = Fusion_feature_extractor(input_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=4)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)

def style_extractor(input_nc, ngf, netG, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = Resnet_style_extractor(input_nc, ngf, norm_layer=norm_layer, n_blocks=3)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def feature_decoder(output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02,
                    gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = Resnet_feature_decoder(output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def feature_decoder_output(output_nc, ngf, netG, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if netG == 'resnet_9blocks':
        net = Resnet_feature_decoder_output(output_nc, ngf)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD='basic', n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the networkss.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the networkss runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':  # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=0.9, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.GAP = torch.nn.AdaptiveAvgPool2d((1, 1))
        model_decoder = []
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model_encoder = [nn.ReflectionPad2d(3),
                         nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                         norm_layer(ngf),
                         nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model_encoder += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                              norm_layer(ngf * mult * 2),
                              nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(4):  # add ResNet blocks

            model_encoder += [
                ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias)]
        for i in range(n_blocks - 4):  # add ResNet blocks

            model_decoder += [
                ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias)]
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model_decoder += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                                 kernel_size=3, stride=2,
                                                 padding=1, output_padding=1,
                                                 bias=use_bias),
                              norm_layer(int(ngf * mult / 2)),
                              nn.ReLU(True)]
        model_decoder += [nn.ReflectionPad2d(3)]
        model_decoder += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model_decoder += [nn.Tanh()]
        # model += [nn.Sigmoid()]

        self.model_encoder = nn.Sequential(*model_encoder)
        self.model_decoder = nn.Sequential(*model_decoder)

    def forward(self, input):
        """Standard forward"""
        mid_feature = self.model_encoder(input)
        return self.model_decoder(mid_feature)

class ResnetGenerator_decoder(nn.Module):
    def __init__(self, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):

        assert (n_blocks >= 0)
        super(ResnetGenerator_decoder, self).__init__()
        self.GAP = torch.nn.AdaptiveAvgPool2d((1, 1))
        model_decoder = []
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        n_downsampling = 2
        mult = 2 ** n_downsampling
        for i in range(n_blocks - 4):  # add ResNet blocks

            model_decoder += [
                ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias)]
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model_decoder += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                                 kernel_size=3, stride=2,
                                                 padding=1, output_padding=1,
                                                 bias=use_bias),
                              norm_layer(int(ngf * mult / 2)),
                              nn.ReLU(True)]
        model_decoder += [nn.ReflectionPad2d(3)]
        model_decoder += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model_decoder += [nn.Tanh()]

        self.model_decoder = nn.Sequential(*model_decoder)

    def forward(self, input):
        """Standard forward"""
        return self.model_decoder(input)
class KernelGenerator(nn.Module):
    def __init__(self):
        super(KernelGenerator, self).__init__()
        self.pool9 = torch.nn.AdaptiveMaxPool1d(9)
        self.model1 = torch.nn.Conv1d(3, 3, 1)
        self.model2 = torch.nn.Conv1d(3, 3, 1)
        self.model3 = torch.nn.Conv1d(9, 9, 1)

    def forward(self, input):
        """Standard forward"""
        ps2 = self.pool9(input)
        input = input.permute(0, 2, 1)
        ps2 = ps2.permute(0, 2, 1)
        out1 = self.model1(input)
        out2 = self.model2(input)
        out3 = self.model3(ps2)
        return out1, out2, out3
class ResnetGenerator_encoder(nn.Module):
    def __init__(self, input_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(ResnetGenerator_encoder, self).__init__()
        self.GAP = torch.nn.AdaptiveAvgPool2d((1, 1))
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model_encoder = [nn.ReflectionPad2d(3),
                         nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                         norm_layer(ngf),
                         nn.ReLU(True)]
        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model_encoder += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                              norm_layer(ngf * mult * 2),
                              nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(4):  # add ResNet blocks
            model_encoder += [
                ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias)]
        self.model_encoder = nn.Sequential(*model_encoder)

    def forward(self, input):
        """Standard forward"""
        return self.model_encoder(input)

class Resnet_feature_extractor(nn.Module):

    def __init__(self, input_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=2,
                 padding_type='reflect'):

        assert (n_blocks >= 0)
        super(Resnet_feature_extractor, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
        mult = 2 ** n_downsampling
        for i in range(4):  # add ResNet blocks
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]
        self.model = nn.Sequential(*model)
        self.GAP = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.GMP = torch.nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, input):
        """Standard forward"""
        x = self.model(input)
        return x, self.GAP(x) + self.GMP(x)

class Fusion_feature_extractor(nn.Module):

    def __init__(self, input_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=2,
                 padding_type='reflect'):

        assert (n_blocks >= 0)
        super(Fusion_feature_extractor, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
        mult = 2 ** n_downsampling
        for i in range(4):  # add ResNet blocks
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]
        self.model = nn.Sequential(*model)
        self.GAP = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.GMP = torch.nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, input):
        """Standard forward"""
        x = self.model(input)
        return x, self.GAP(x) + self.GMP(x)


class Resnet_style_extractor(nn.Module):

    def __init__(self, input_nc, ngf=64, norm_layer=nn.BatchNorm2d, n_blocks=2):

        assert (n_blocks >= 0)
        super(Resnet_style_extractor, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        model1 = [nn.ReflectionPad2d(3),
                  nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                  norm_layer(ngf),
                  nn.ReLU(True)]
        model2 = [nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                  norm_layer(ngf * 2),
                  nn.ReLU(True)]
        model3 = [nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=use_bias),
                  norm_layer(ngf * 4),
                  nn.ReLU(True)]

        self.GAP = torch.nn.AdaptiveAvgPool2d((1, 1))
        # self.GMP = torch.nn.AdaptiveMaxPool2d((1, 1))
        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(448, 63)
        )

    def forward(self, input):
        """Standard forward"""
        x1 = self.model1(input)
        x2 = self.model2(x1)
        x3 = self.model3(x2)
        # final = self.fc(torch.cat([self.GAP(x1), self.GAP(x2), self.GAP(x3), self.GAP(x4)], dim=1))
        # final = F.log_softmax(final, dim=1)
        return x3, self.GAP(x3).squeeze_(2).squeeze_(2)


class Resnet_feature_decoder(nn.Module):

    def __init__(self, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(Resnet_feature_decoder, self).__init__()
        self.GAP = torch.nn.AdaptiveAvgPool2d((1, 1))
        model_decoder = []
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        n_downsampling = 2

        mult = 2 ** n_downsampling

        for i in range(n_blocks - 4):  # add ResNet blocks

            model_decoder += [
                ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias)]
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model_decoder += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                                 kernel_size=3, stride=2,
                                                 padding=1, output_padding=1,
                                                 bias=use_bias),
                              norm_layer(int(ngf * mult / 2)),
                              nn.ReLU(True)]
        model_decoder += [nn.ReflectionPad2d(3)]
        model_decoder += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        # model_decoder += [nn.ReLU()]
        model_decoder += [nn.Tanh()]
        self.model_decoder = nn.Sequential(*model_decoder)

    def forward(self, input):
        """Standard forward"""
        mid_feature = self.model_decoder(input)
        return mid_feature


class Resnet_feature_decoder_output(nn.Module):

    def __init__(self, output_nc, ngf=64):
        super(Resnet_feature_decoder_output, self).__init__()
        self.GAP = torch.nn.AdaptiveAvgPool2d((1, 1))
        model_decoder = []

        model_decoder += [nn.ReflectionPad2d(3)]
        model_decoder += [nn.Conv2d(3, output_nc, kernel_size=7, padding=0)]
        model_decoder += [nn.Tanh()]
        self.model_decoder = nn.Sequential(*model_decoder)

    def forward(self, input):
        """Standard forward"""
        mid_feature = self.model_decoder(input)
        return mid_feature


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:  # add skip connections
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)

import torch
import torch.nn as nn
import torch.nn.functional as F


class build_resnet_block(nn.Module):
    def __init__(self, in_c, out_c, drop_rate=0.25, normtype='batch'):
        super(build_resnet_block, self).__init__()
        conv_block = [  nn.ReflectionPad2d(1),  #镜像填充
                        layers.general_conv2d(in_c, out_c, k_s=3, drop=drop_rate, norm_type=normtype),
                        nn.ReflectionPad2d(1),
                        layers.general_conv2d(out_c, out_c, k_s=3, do_relu=False, drop=drop_rate, norm_type=normtype)]

        self.conv_block = nn.Sequential(*conv_block)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(x + self.conv_block(x))

class build_resnet_block_ds(nn.Module):
    def __init__(self, in_c, out_c, drop_rate=0.25, normtype='batch'):
        super(build_resnet_block_ds, self).__init__()
        conv_block = [  nn.ReflectionPad2d(1),  #镜像填充
                        layers.general_conv2d(in_c, out_c, k_s=3, drop=drop_rate, norm_type=normtype),
                        nn.ReflectionPad2d(1),
                        layers.general_conv2d(out_c, out_c, k_s=3, do_relu=False, drop=drop_rate, norm_type=normtype)]

        self.conv_block = nn.Sequential(*conv_block)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x1 = x.repeat([1, 2, 1, 1])
        return self.relu(x1 + self.conv_block(x))

class build_drn_block(nn.Module):
    def __init__(self, in_c, out_c, drop_rate=0.25, normtype='batch'):
        super(build_drn_block, self).__init__()

        conv_block = [  nn.ReflectionPad2d(2),  #镜像填充
                        layers.dilate_conv2d(in_c, out_c, k_s=3, drop=drop_rate, norm_type=normtype),
                        nn.ReflectionPad2d(2),
                        layers.dilate_conv2d(out_c, out_c, k_s=3, do_relu=False, drop=drop_rate, norm_type=normtype)]

        self.conv_block = nn.Sequential(*conv_block)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(x + self.conv_block(x))

class build_drn_block_ds(nn.Module):
    def __init__(self, in_c, out_c, drop_rate=0.25, normtype='batch'):
        super(build_drn_block_ds, self).__init__()

        conv_block = [  nn.ReflectionPad2d(2),  #镜像填充
                        layers.dilate_conv2d(in_c, out_c, k_s=3, drop=drop_rate, norm_type=normtype),
                        nn.ReflectionPad2d(2),
                        layers.dilate_conv2d(out_c, out_c, k_s=3, do_relu=False, drop=drop_rate, norm_type=normtype)]

        self.conv_block = nn.Sequential(*conv_block)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x1 = x.repeat([1, 2, 1, 1])
        return self.relu(x1 + self.conv_block(x))


class build_encoderc(nn.Module):
    def __init__(self, in_c, drop_rate=0.25):
        fb = 16
        super(build_encoderc, self).__init__()
        self.o_c1 = layers.general_conv2d(in_c, fb, k_s=7, stride=1, norm_type="batch", drop=drop_rate)
        self.o_r1 = build_resnet_block(fb, fb, normtype='batch')
        self.maxpool_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.o_r2 = build_resnet_block_ds(fb, fb*2, normtype='batch')
        self.maxpool_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.o_r3 = build_resnet_block_ds(fb*2, fb * 4, normtype='batch')
        self.o_r4 = build_resnet_block(fb * 4, fb * 4, normtype='batch')
        self.maxpool_3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.o_r5 = build_resnet_block_ds(fb * 4, fb * 8, normtype='batch')
        self.o_r6 = build_resnet_block(fb * 8, fb * 8, normtype='batch')
        self.o_r7 = build_resnet_block_ds(fb * 8, fb * 16, normtype='batch')
        self.o_r8 = build_resnet_block(fb * 16, fb * 16, normtype='batch')
        self.o_r9 = build_resnet_block(fb * 16, fb * 16, normtype='batch')
        self.o_r10 = build_resnet_block(fb * 16, fb * 16, normtype='batch')
        self.o_r11 = build_resnet_block_ds(fb * 16, fb * 32, normtype='batch')
        self.o_r12 = build_resnet_block(fb * 32, fb * 32, normtype='batch')
    def forward(self, x):
        x = self.o_c1(x)
        x = self.o_r1(x)
        x = self.maxpool_1(x)
        x = self.o_r2(x)
        x = self.maxpool_2(x)
        x = self.o_r3(x)
        x = self.o_r4(x)
        x = self.maxpool_3(x)
        x = self.o_r5(x)
        x = self.o_r6(x)
        x = self.o_r7(x)
        x = self.o_r8(x)
        x = self.o_r9(x)
        x = self.o_r10(x)
        x = self.o_r11(x)
        x = self.o_r12(x)
        return x

class build_encoders(nn.Module):
    def __init__(self, in_c, drop_rate=0.25):
        fb = 16
        super(build_encoders, self).__init__()
        self.o_d1 = build_drn_block(in_c, fb*32, normtype='batch')
        self.o_d2 = build_drn_block(fb * 32, fb * 32, normtype='batch')
    def forward(self, x):
        x = self.o_d1(x)
        x = self.o_d2(x)
        return x

class build_encodert(nn.Module):
    def __init__(self, in_c, drop_rate=0.25):
        fb = 16
        super(build_encodert, self).__init__()
        self.o_d1 = build_drn_block(in_c, fb*32, normtype='batch')
        self.o_d2 = build_drn_block(fb * 32, fb * 32, normtype='batch')
    def forward(self, x):
        x = self.o_d1(x)
        x = self.o_d2(x)
        return x

class build_encoderdiffa(nn.Module):
    def __init__(self, in_c, drop_rate=0.25):
        fb = 8
        k1 = 3
        super(build_encoderdiffa, self).__init__()
        model = [ layers.general_conv2d(in_c, fb, k_s=7, stride=1, norm_type='batch', drop=drop_rate, pd=3),
                  build_resnet_block(fb, fb, normtype='batch', drop_rate=drop_rate),
                  nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                  build_resnet_block_ds(fb, fb*2, normtype='batch', drop_rate=drop_rate),
                  nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                  build_resnet_block_ds(fb*2, fb*4, normtype='batch', drop_rate=drop_rate),
                  build_resnet_block(fb*4, fb*4, normtype='batch', drop_rate=drop_rate),
                  nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                  layers.general_conv2d(fb*4, 32, k_s=k1, stride=1, norm_type='batch', drop=drop_rate, pd=1),
                  layers.general_conv2d(32, 32, k_s=k1, stride=1, norm_type='batch', drop=drop_rate, pd=1),]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)


#输出是8,32,32
class build_encoderdiffa1(nn.Module):
    def __init__(self, in_c, drop_rate=0.25):
        fb = 8
        k1 = 3
        super(build_encoderdiffa1, self).__init__()
        model = [ layers.general_conv2d(in_c, fb, k_s=7, stride=1, norm_type='batch', drop=drop_rate, pd=3),
                  build_resnet_block(fb, fb, normtype='batch', drop_rate=drop_rate),
                  nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                  build_resnet_block_ds(fb, fb*2, normtype='batch', drop_rate=drop_rate),
                  nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                  build_resnet_block_ds(fb*2, fb*4, normtype='batch', drop_rate=drop_rate),
                  build_resnet_block(fb*4, fb*4, normtype='batch', drop_rate=drop_rate),
                  nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                  layers.general_conv2d(fb*4, 32, k_s=k1, stride=1, norm_type='batch', drop=drop_rate, pd=1),
                  layers.general_conv2d(32, 8, k_s=k1, stride=1, norm_type='batch', drop=drop_rate, pd=1),]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)

class build_encoderdiffa2(nn.Module):
    def __init__(self, in_c, drop_rate=0.25):
        fb = 8
        k1 = 3
        super(build_encoderdiffa2, self).__init__()
        model = [ layers.general_conv2d(in_c, fb, k_s=7, stride=1, norm_type='batch', drop=drop_rate, pd=3),
                  build_resnet_block(fb, fb, normtype='batch', drop_rate=drop_rate),
                  nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                  build_resnet_block_ds(fb, fb*2, normtype='batch', drop_rate=drop_rate),
                  nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                  build_resnet_block_ds(fb*2, fb*4, normtype='batch', drop_rate=drop_rate),
                  build_resnet_block(fb*4, fb*4, normtype='batch', drop_rate=drop_rate)]
        self.model = nn.Sequential(*model)
        self.fc = nn.Linear(32, 8)
    def forward(self, x):
        x = self.model(x)
        x = F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
        x = self.fc(x)
        x = x.unsqueeze(1).unsqueeze(2)
        x = x.permute(0, 3, 1, 2)
        x = x.repeat([1, 1, 32, 32])
        return x


class build_encoderdiffb(nn.Module):
    def __init__(self, in_c, drop_rate=0.25):
        fb = 8
        k1 = 3
        super(build_encoderdiffb, self).__init__()
        model = [ layers.general_conv2d(in_c, fb, k_s=7, stride=1, norm_type='batch', drop=drop_rate, pd=3),
                  build_resnet_block(fb, fb, normtype='batch', drop_rate=drop_rate),
                  nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                  build_resnet_block_ds(fb, fb*2, normtype='batch', drop_rate=drop_rate),
                  nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                  build_resnet_block_ds(fb*2, fb*4, normtype='batch', drop_rate=drop_rate),
                  build_resnet_block(fb*4, fb*4, normtype='batch', drop_rate=drop_rate),
                  nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                  layers.general_conv2d(fb*4, 32, k_s=k1, stride=1, norm_type='batch', drop=drop_rate, pd=1),
                  layers.general_conv2d(32, 32, k_s=k1, stride=1, norm_type='batch', drop=drop_rate, pd=1),]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)

class build_decodera(nn.Module):
    def __init__(self, in_c, drop_rate=0.25):
        f = 7
        ks = 3
        ngf = 32
        super(build_decodera, self).__init__()
        model = [layers.general_conv2d(in_c, ngf*4, k_s=ks, stride=1, norm_type='ins', pd=1),
                 build_resnet_block(ngf*4, ngf*4, normtype='ins'),
                 build_resnet_block(ngf * 4, ngf * 4, normtype='ins'),
                 build_resnet_block(ngf * 4, ngf * 4, normtype='ins'),
                 build_resnet_block(ngf * 4, ngf * 4, normtype='ins'),
                 layers.general_deconv2d(ngf*4, ngf*2, k_s=ks, stride=2, norm_type='ins', pd=1),
                 layers.general_deconv2d(ngf * 2, ngf * 2, k_s=ks, stride=2, norm_type='ins'),
                 layers.general_deconv2d(ngf * 2, ngf, k_s=ks, stride=2, norm_type='ins'),
                 layers.general_conv2d(ngf, 1, k_s=f, stride=1, do_norm=False, do_relu=False),
                 torch.nn.Tanh()]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        x = self.model(x)
        return x

class build_decoderb(nn.Module):
    def __init__(self, in_c, drop_rate=0.25):
        f = 7
        ks = 3
        ngf = 32
        super(build_decoderb, self).__init__()
        model = [layers.general_conv2d(in_c, ngf*4, k_s=ks, stride=1, norm_type='ins', pd=1),
                 build_resnet_block(ngf*4, ngf*4, normtype='ins'),
                 build_resnet_block(ngf * 4, ngf * 4, normtype='ins'),
                 build_resnet_block(ngf * 4, ngf * 4, normtype='ins'),
                 build_resnet_block(ngf * 4, ngf * 4, normtype='ins'),
                 layers.general_deconv2d(ngf*4, ngf*2, k_s=ks, stride=2, norm_type='ins', pd=1),
                 layers.general_deconv2d(ngf * 2, ngf * 2, k_s=ks, stride=2, norm_type='ins'),
                 layers.general_deconv2d(ngf * 2, ngf, k_s=ks, stride=2, norm_type='ins'),
                 layers.general_conv2d(ngf, 1, k_s=f, stride=1, do_norm=False, do_relu=False),
                 torch.nn.Tanh()]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)

class build_decoderc(nn.Module):
    def __init__(self, in_c, drop_rate=0.25):
        f = 7
        ks = 3
        ngf = 32
        super(build_decoderc, self).__init__()
        self.oc1 = layers.general_conv2d(in_c, ngf * 4, k_s=ks, stride=1, pd=1, norm_type='ins')
        self.or1 = build_resnet_block(ngf * 4, ngf * 4, normtype='ins')
        self.or2 = build_resnet_block(ngf * 4, ngf * 4, normtype='ins')
        self.or3 = build_resnet_block(ngf * 4, ngf * 4, normtype='ins')
        self.or4 = build_resnet_block(ngf * 4, ngf * 4, normtype='ins')
    def forward(self, x):
        x = self.oc1(x)
        x = self.or1(x)
        x = self.or2(x)
        x = self.or3(x)
        x = self.or4(x)
        return x

class build_decodernewa(nn.Module):
    def __init__(self, in_c, drop_rate=0.25):
        f = 7
        ks = 3
        ngf = 32
        super(build_decodernewa, self).__init__()
        self.oc3 = layers.general_deconv2d(in_c, ngf*2, k_s=ks, stride=2, norm_type='ins', pd=1)
        self.or4 = layers.general_deconv2d(ngf*2, ngf*2, k_s=ks, stride=2, norm_type='ins', pd=1)
        self.or5 = layers.general_deconv2d(ngf*2, ngf, k_s=ks, stride=2, norm_type='ins', pd=1)
        self.or6 = layers.general_conv2d(ngf, 1, k_s=f, stride=1, do_norm=False, do_relu=False, pd=3)
    def forward(self, x):
        x = self.oc3(x)
        x = self.or4(x)
        x = self.or5(x)
        x = self.or6(x)
        return x

class build_decodernewb(nn.Module):
    def __init__(self, in_c, drop_rate=0.25):
        f = 7
        ks = 3
        ngf = 32
        super(build_decodernewb, self).__init__()
        self.oc3 = layers.general_deconv2d(in_c, ngf*2, k_s=ks, stride=2, norm_type='ins', pd=1)
        self.or4 = layers.general_deconv2d(ngf*2, ngf*2, k_s=ks, stride=2, norm_type='ins', pd=1)
        self.or5 = layers.general_deconv2d(ngf*2, ngf, k_s=ks, stride=2, norm_type='ins', pd=1)
        self.or6 = layers.general_conv2d(ngf, 1, k_s=f, stride=1, do_norm=False, do_relu=False, pd=3)
    def forward(self, x):
        x = self.oc3(x)
        x = self.or4(x)
        x = self.or5(x)
        x = self.or6(x)
        return x

class build_segmenternew(nn.Module):
    def __init__(self, in_c, out_c, drop_rate=0.25):
        f = 7
        ks = 3
        ngf = 32
        super(build_segmenternew, self).__init__()
        self.oc1 = layers.general_conv2d(in_c, ngf*4, k_s=ks, stride=1, norm_type='ins', pd=1, drop=drop_rate)
        self.or1 = build_resnet_block(ngf*4, ngf*4, normtype='ins')
        self.or2 = build_resnet_block(ngf * 4, ngf * 4, normtype='ins')
        self.or3 = build_resnet_block(ngf * 4, ngf * 4, normtype='ins')
        # self.or4 = build_resnet_block(ngf * 4, ngf * 4, normtype='ins')
        self.oc3 = layers.general_deconv2d(ngf * 4, ngf * 2, k_s=ks, stride=2, norm_type='ins', pd=1)
        # self.oc4 = layers.general_deconv2d(ngf * 2, ngf * 2, k_s=ks, stride=2, norm_type='ins', pd=1)
        self.oc5 = layers.general_deconv2d(ngf * 2, ngf, k_s=ks, stride=2, norm_type='ins', pd=1)
        self.oc6 = layers.general_conv2d(ngf, out_c, k_s=f, stride=1, do_norm=False, do_relu=False, pd=3)
    def forward(self, x):
        x = self.oc1(x)
        x = self.or1(x)
        x = self.or2(x)
        x = self.or3(x)
        # x = self.or4(x)
        x = self.oc3(x)
        # x = self.oc4(x)
        x = self.oc5(x)
        x = self.oc6(x)
        return x

class build_encoder_whole(nn.Module):
    def __init__(self, in_c, drop_rate=0.25):
        super(build_encoder_whole, self).__init__()
        self.en1 = build_encoderc(in_c)
        self.en2 = build_encoders(512)
    def forward(self, x):
        x = self.en1(x)
        x = self.en2(x)
        return x

class build_decoder_whole(nn.Module):
    def __init__(self, in_c, drop_rate=0.25):
        super(build_decoder_whole, self).__init__()
        self.en1 = build_decoderc(in_c)
        self.en2 = build_decodernewa(128)
    def forward(self, x):
        x = self.en1(x)
        x = self.en2(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

#AadIN
def adain(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model_G = ResnetGenerator(3, 2)
    model = Resnet_feature_extractor(3)
    model_d = define_D(3, 64)
    input = torch.rand(1, 3, 256, 256)
    out = model(input)
    out_d = model_d(input)
    feature, feature_gap, out_G = model_G(input)
    proto_out = out.unsqueeze_(2).unsqueeze_(2)
    x_cond = torch.cat([feature_gap, proto_out], 1)

    # self.controller = nn.Conv2d(256 + 7, 162, kernel_size=1, stride=1, padding=0)
    print(out.shape)
    print(out_d.shape)
    print(out_G.shape)
    print(feature.shape)
    print(x_cond.shape)
