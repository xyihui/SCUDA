import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class general_conv2d(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, k_s=7, pd=0, stride=1, do_norm=True,
                 norm_type='batch', do_relu=True, drop=None):
        super().__init__()
        model = [nn.Conv2d(in_channels, out_channels, kernel_size=k_s, padding=pd, stride=stride)]
        if do_norm:
            if norm_type=='batch':
                model+=[nn.BatchNorm2d(out_channels)]
            elif norm_type=='ins':
                model += [nn.InstanceNorm2d(out_channels)]
        if do_relu:
            model += [nn.ReLU(inplace=True)]
        if not drop is None:
            #这里的drop是丢弃的概率
            model += [nn.Dropout2d(p=drop)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class dilate_conv2d(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, k_s=7, pd=0, stride=1, do_norm=True,
                 norm_type='batch', do_relu=True, drop=None):
        super().__init__()
        model = [nn.Conv2d(in_channels, out_channels, kernel_size=k_s, padding=pd, stride=stride, dilation=2)]
        if do_norm:
            if norm_type=='batch':
                model+=[nn.BatchNorm2d(out_channels)]
            elif norm_type=='ins':
                model += [nn.InstanceNorm2d(out_channels)]
        if do_relu:
            model += [nn.ReLU(inplace=True)]
        if not drop is None:
            model += [nn.Dropout2d(p=drop)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

#这个待定
class general_deconv2d(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, k_s=7, pd=0, stride=1, do_norm=True,
                 norm_type='batch', do_relu=True, drop=None, outputpd=1):
        super().__init__()
        model = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=k_s, padding=pd, stride=stride, output_padding=outputpd)]
        if do_norm:
            if norm_type=='batch':
                model+=[nn.BatchNorm2d(out_channels)]
            elif norm_type=='ins':
                model += [nn.InstanceNorm2d(out_channels)]
        if do_relu:
            model += [nn.ReLU(inplace=True)]
        if not drop is None:
            model += [nn.Dropout2d(p=drop)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)



if __name__ == '__main__':
    #net1 = general_conv2d(in_channels=1, out_channels=64).cuda()
    net1 = dilate_conv2d(in_channels=1, out_channels=64).cuda()
    x = torch.rand((1, 1, 256, 256)).cuda()
    y = net1(x)

    # x1 = torch.rand((1, 4)).cuda()
    # y = net1(x, x1)


    print(y.shape)


    Trainable_params = 0
    NonTrainable_params = 0
    Total_params = 0
    for param in net1.parameters():
        mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
        Total_params += mulValue  # 总参数量
        if param.requires_grad:
            Trainable_params += mulValue  # 可训练参数量
        else:
            NonTrainable_params += mulValue  # 非可训练参数量

    print(f'Total params: {Total_params}')
    print(f'Trainable params: {Trainable_params}')
    print(f'Non-trainable params: {NonTrainable_params}')