from dataset.LinearLesion import *
from imaginaire.generators.coco_funit_zerol_cycle_new import COCOFUNITTranslator
from model.networks import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datetime import datetime
import itertools
import os
import torch
import tqdm
import torch.nn as nn
from utils.utils import ReplayBuffer
import numpy as np
import utils.utils as u
import utils.loss_1 as LS
from config_train import DefaultConfig
import torch.backends.cudnn as cudnn

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_in')
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
    elif isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_in')

def val(encoder_content, seg, dataloader):
    print('start test!')
    with torch.no_grad():
        encoder_content.eval()
        seg.eval()
        tbar = tqdm.tqdm(dataloader, desc='\r')
        dice = {}
        for n in range(1, args.num_classes):
            dice[n] = []

        for i, (data, label) in enumerate(tbar):
            image = data.cuda()
            label = label.cuda()
            content_s = encoder_content(image)
            predict = seg(content_s)
            predict = predict.data.cpu().numpy()
            label = label.data.cpu().numpy()
            predict = np.squeeze(predict)
            label = np.squeeze(label)
            predict = np.argmax(predict, axis=0)
            per_dice = u.compute_per_dice(predict, label, args.num_classes)

            for n in range(1, args.num_classes):
                dice[n].append(per_dice[n])

        all_mean = 0
        for key in dice:
            mean = (np.sum(dice[key]) - dice[key].count(1)) / (len(dice[key]) - dice[key].count(1))
            print('DICE %d : %f' % (key, mean))
            all_mean += mean
        all_mean /= len(dice)
        print('DICE_MEAN = %f' % all_mean)
        return dice, all_mean


def train(args, encoder_content, encoder_s, encoder_t, decoder_t, netG_A_content_kernel, netG_A_style_kernel, seg, Ds,
          Dt, D_pc, D_seg, optimizer, criterion, dataloader_train, dataloader_val, k_fold):
    best_pred = 0.0
    fake_t_buffer = ReplayBuffer()
    fake_s_buffer = ReplayBuffer()
    pre_seg_s_buffer = ReplayBuffer()
    pre_seg_t_buffer = ReplayBuffer()
    pc_s_buffer = ReplayBuffer()
    pc_t_buffer = ReplayBuffer()
    encoder_content.train()
    encoder_s.train()
    encoder_t.train()
    decoder_t.train()
    netG_A_content_kernel.train()
    netG_A_style_kernel.train()
    seg.train()
    Ds.train()
    Dt.train()
    D_pc.train()
    y_real = torch.ones(1)
    y_fake = torch.zeros(1)
    y_real, y_fake = Variable(y_real.cuda()), Variable(y_fake.cuda())
    l1 = torch.Tensor([0])  # l1表示s域
    l11 = l1.repeat(args.batchSize, args.feature_channel, args.feature_size_height, args.feature_size_width)
    l11 = Variable(l11.cuda())
    resize_label64 = transforms.Resize((args.feature_size_height, args.feature_size_width), Image.NEAREST)
    pool3 = torch.nn.AdaptiveMaxPool1d(args.kernel_size).cuda()
    model_save_list = []
    for epoch in range(args.num_epoch):
        tq = tqdm.tqdm(total=len(dataloader_train) * args.batchSize)
        tq.set_description('fold %d,epoch %d' % (int(k_fold), epoch))
        train_loss = 0.0
        # s1 stage
        if epoch < args.s1_num_epoch:
            for i, batch in enumerate(dataloader_train):
                t_image = batch['T'].cuda()
                s_image = batch['S'].cuda()
                s_mask = batch['S_label'].cuda()

                # train Generator
                optimizer[0].zero_grad()
                optimizer[1].zero_grad()
                z_s_style = encoder_s(s_image)
                z_t_style = encoder_t(t_image)
                loss0_1 = criterion[1](z_s_style, l11)
                loss0_2 = criterion[1](z_t_style, l11)
                z_s_content = encoder_content(s_image)
                z_t_content = encoder_content(t_image)
                # LPSF module, before PSF
                real_S_label_feature1 = resize_label64(s_mask).unsqueeze(1)[0].bool().repeat(args.feature_channel, 1, 1)
                real_S_label_feature2 = resize_label64(s_mask).unsqueeze(1)[1].bool().repeat(args.feature_channel, 1, 1)
                real_S_label_fuse_content1 = z_s_content[0][real_S_label_feature1].reshape(args.feature_channel, -1)
                real_S_label_fuse_content2 = z_s_content[1][real_S_label_feature2].reshape(args.feature_channel, -1)
                real_S_label_fuse_style1 = z_s_style[0][real_S_label_feature1].reshape(args.feature_channel, -1)
                real_S_label_fuse_style2 = z_s_style[1][real_S_label_feature2].reshape(args.feature_channel, -1)
                # dynamic convolution kernel of pancreatic contents of S
                if (real_S_label_fuse_content1.size(1) >= args.kernel_size) and (
                        real_S_label_fuse_content2.size(1) >= args.kernel_size):
                    # 2*256*3
                    real_S_label_fuse_content_pool = torch.cat([pool3(real_S_label_fuse_content1.unsqueeze(0)),
                                                                pool3(real_S_label_fuse_content2.unsqueeze(0))], dim=0)
                    real_S_content_kv, real_S_content_kh, real_S_content_ks = netG_A_content_kernel(
                        real_S_label_fuse_content_pool)
                else:
                    real_S_content_kv, real_S_content_kh, real_S_content_ks = None, None, None
                # dynamic convolution kernel of pancreatic style of S
                if (real_S_label_fuse_style1.size(1) >= args.kernel_size) and (
                        real_S_label_fuse_style2.size(1) >= args.kernel_size):
                    # 2*256*3
                    real_S_label_fuse_style_pool = torch.cat([pool3(real_S_label_fuse_style1.unsqueeze(0)),
                                                              pool3(real_S_label_fuse_style2.unsqueeze(0))], dim=0)
                    real_S_style_kv, real_S_style_kh, real_S_style_ks = netG_A_style_kernel(
                        real_S_label_fuse_style_pool)
                else:
                    real_S_style_kv, real_S_style_kh, real_S_style_ks = None, None, None
                # LPSF option
                same_s, _ = decoder_t.decode(z_s_content, z_s_style, real_S_content_kv, real_S_content_kh,
                                             real_S_content_ks, real_S_style_kv, real_S_style_kh, real_S_style_ks)
                same_t, _ = decoder_t.decode(z_t_content, z_t_style)
                fake_t, z_s_pc = decoder_t.decode(z_s_content, z_t_style, real_S_content_kv, real_S_content_kh,
                                                  real_S_content_ks)
                fake_s, z_t_pc = decoder_t.decode(z_t_content, z_s_style, None, None, None, real_S_style_kv,
                                                  real_S_style_kh, real_S_style_ks)
                z_s2t_content = encoder_content(fake_t)
                loss_11 = criterion[1](same_s, s_image)
                loss_22 = criterion[1](same_t, t_image)
                predict_z_t_content_seg = seg(z_t_content)
                predict_z_s_content_seg = seg(z_s_content)
                predict_z_s2t_content_seg = seg(z_s2t_content)

                loss_seg1 = criterion[-2](predict_z_s_content_seg, s_mask.long()) + criterion[-1](
                    predict_z_s_content_seg, s_mask)
                loss_seg2 = criterion[-2](predict_z_s2t_content_seg, s_mask.long()) + criterion[-1](
                    predict_z_s2t_content_seg, s_mask)

                fake_t_1 = fake_t
                pred_fake_t = Dt(fake_t_1)
                loss_1 = criterion[4](pred_fake_t, y_real.unsqueeze(1).expand(args.batchSize, -1))

                fake_s_1 = fake_s
                pred_fake_s = Ds(fake_s_1)
                loss_2 = criterion[4](pred_fake_s, y_real.unsqueeze(1).expand(args.batchSize, -1))

                predict_z_t_content_seg_1 = predict_z_t_content_seg
                predict_z_t_content_seg_1 = D_seg(predict_z_t_content_seg_1)
                loss_3 = criterion[4](predict_z_t_content_seg_1, y_real.unsqueeze(1).expand(args.batchSize, -1))
                # Phase Discriminator
                z_t_pc_1 = z_t_pc
                z_t_pc_1 = D_pc(z_t_pc_1)
                loss_pc = criterion[4](z_t_pc_1, y_real.unsqueeze(1).expand(args.batchSize, -1))

                loss_G_total = (loss_1 + loss_2 + loss_11 + loss_22 + loss_seg1 +
                                loss_seg2 + (loss_3 + loss_pc) +  (loss0_1 + loss0_2))
                loss_G_total.backward()
                optimizer[0].step()
                optimizer[1].step()
                ###############################################################################################################

                # train Discriminator
                optimizer[2].zero_grad()
                pred_real = Dt(t_image)
                lo_1 = criterion[4](pred_real, y_real.unsqueeze(1).expand(args.batchSize, -1))

                fake_t = fake_t_buffer.push_and_pop(fake_t)  # 
                pred_fake = Dt(fake_t.detach())
                lo_2 = criterion[4](pred_fake, y_fake.unsqueeze(1).expand(args.batchSize, -1))

                pred_real_1 = Ds(s_image)
                lo_3 = criterion[4](pred_real_1, y_real.unsqueeze(1).expand(args.batchSize, -1))

                fake_s = fake_s_buffer.push_and_pop(fake_s)  # 
                pred_fake_1 = Ds(fake_s.detach())
                lo_4 = criterion[4](pred_fake_1, y_fake.unsqueeze(1).expand(args.batchSize, -1))

                predict_z_t_content_seg = pre_seg_s_buffer.push_and_pop(predict_z_t_content_seg)
                predict_z_s_content_seg = pre_seg_t_buffer.push_and_pop(predict_z_s_content_seg)
                predict_z_t_content_seg_real = D_seg(predict_z_t_content_seg.detach())
                predict_z_s_content_seg_fake = D_seg(predict_z_s_content_seg.detach())
                seg_real_loss = criterion[4](predict_z_t_content_seg_real,
                                             y_real.unsqueeze(1).expand(args.batchSize, -1))
                seg_fake_loss = criterion[4](predict_z_s_content_seg_fake,
                                             y_fake.unsqueeze(1).expand(args.batchSize, -1))
                z_s_pc = pc_s_buffer.push_and_pop(z_s_pc)
                z_t_pc = pc_t_buffer.push_and_pop(z_t_pc)
                z_s_pc_real = D_pc(z_s_pc.detach())
                z_t_pc_fake = D_pc(z_t_pc.detach())
                z_pc_real_loss = criterion[4](z_s_pc_real, y_real.unsqueeze(1).expand(args.batchSize, -1))
                z_pc_fake_loss = criterion[4](z_t_pc_fake, y_real.unsqueeze(1).expand(args.batchSize, -1))

                loss_D_total = z_pc_real_loss + z_pc_fake_loss + seg_real_loss + seg_fake_loss + lo_1 + lo_2 + lo_3 + lo_4

                loss_D_total.backward()
                optimizer[2].step()

                tq.update(args.batchSize)
                train_loss += loss_G_total.item()
                tq.set_postfix(loss='%.3f' % (train_loss / (i + 1)))
            tq.close()

            if epoch % args.validation_step == 0:
                dice_list, Dice = val(encoder_content, seg, dataloader_val)

                is_best = Dice > best_pred
                best_pred = max(best_pred, Dice)

                checkpoint_dir_root = args.checkpoint_dir
                checkpoint_dir = os.path.join(checkpoint_dir_root, str(k_fold))
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                checkpoint_latest = os.path.join(checkpoint_dir, 'checkpoint_latest.pth.tar')
                u.save_checkpoint({
                    'dice_list': dice_list,
                    'state_dict': encoder_content.state_dict(),
                    'state_dict1': encoder_s.state_dict(),
                    'state_dict2': encoder_t.state_dict(),
                    'state_dict3': decoder_t.state_dict(),
                    'state_dict41': netG_A_content_kernel.state_dict(),
                    'state_dict42': netG_A_style_kernel.state_dict(),
                    'state_dict5': seg.state_dict(),
                    'state_dict6': Ds.state_dict(),
                    'state_dict7': Dt.state_dict(),
                    'state_dict8': D_pc.state_dict(),
                    'state_dict9': D_seg.state_dict(),
                }, best_pred, epoch, is_best, checkpoint_dir, filename=checkpoint_latest)
                # Synthesized target domain images in different training epochs
            if epoch in args.epoch_save_list:
                model_save_list.append([encoder_content, seg])
        # s2 stage
        else:
            for i, batch in enumerate(dataloader_train):
                t_image = batch['T'].cuda()
                s_image = batch['S'].cuda()
                s_mask = batch['S_label'].cuda()
                print('start MESC culculate!')
                predict_s_list = []
                predict_t_list = []
                for model in model_save_list:
                    encoder_content, seg = model[0], model[1]
                    predict_s = seg(encoder_content(s_image))
                    predict_t = seg(encoder_content(t_image))
                    predict_s_list.append(predict_s)
                    predict_t_list.append(predict_t)
                    weight_s = torch.var(torch.stack(predict_s_list), dim=0)
                    weight_t = torch.var(torch.stack(predict_t_list), dim=0)
                # train Generator
                optimizer[0].zero_grad()
                optimizer[1].zero_grad()
                z_s_style = encoder_s(s_image)
                z_t_style = encoder_t(t_image)
                loss0_1 = criterion[1](z_s_style, l11)
                loss0_2 = criterion[1](z_t_style, l11)
                z_s_content = encoder_content(s_image)
                z_t_content = encoder_content(t_image)
                # LPSF module, before PSF
                real_S_label_feature1 = resize_label64(s_mask).unsqueeze(1)[0].bool().repeat(args.feature_channel,
                                                                                             1, 1)  # S Label features
                real_S_label_feature2 = resize_label64(s_mask).unsqueeze(1)[1].bool().repeat(args.feature_channel,
                                                                                             1, 1)  # S Label features
                real_S_label_fuse_content1 = z_s_content[0][real_S_label_feature1].reshape(args.feature_channel,
                                                                                           -1)  # S Pancreatic content features
                real_S_label_fuse_content2 = z_s_content[1][real_S_label_feature2].reshape(args.feature_channel,
                                                                                           -1)  # S Pancreatic content features
                real_S_label_fuse_style1 = z_s_style[0][real_S_label_feature1].reshape(args.feature_channel,
                                                                                       -1)  # S Pancreatic style features
                real_S_label_fuse_style2 = z_s_style[1][real_S_label_feature2].reshape(args.feature_channel,
                                                                                       -1)  # S Pancreatic style features
                # dynamic convolution kernel of pancreatic contents of S
                if (real_S_label_fuse_content1.size(1) >= args.kernel_size) and (
                        real_S_label_fuse_content2.size(1) >= args.kernel_size):
                    # 2*256*3
                    real_S_label_fuse_content_pool = torch.cat([pool3(real_S_label_fuse_content1.unsqueeze(0)),
                                                                pool3(real_S_label_fuse_content2.unsqueeze(0))],
                                                               dim=0)
                    real_S_content_kv, real_S_content_kh, real_S_content_ks = netG_A_content_kernel(
                        real_S_label_fuse_content_pool)
                else:
                    real_S_content_kv, real_S_content_kh, real_S_content_ks = None, None, None
                # dynamic convolution kernel of pancreatic style of S
                if (real_S_label_fuse_style1.size(1) >= args.kernel_size) and (
                        real_S_label_fuse_style2.size(1) >= args.kernel_size):
                    # 2*256*3
                    real_S_label_fuse_style_pool = torch.cat([pool3(real_S_label_fuse_style1.unsqueeze(0)),
                                                              pool3(real_S_label_fuse_style2.unsqueeze(0))], dim=0)
                    real_S_style_kv, real_S_style_kh, real_S_style_ks = netG_A_style_kernel(
                        real_S_label_fuse_style_pool)
                else:
                    real_S_style_kv, real_S_style_kh, real_S_style_ks = None, None, None
                same_s, _ = decoder_t.decode(z_s_content, z_s_style, real_S_content_kv, real_S_content_kh,
                                             real_S_content_ks, real_S_style_kv, real_S_style_kh, real_S_style_ks)
                same_t, _ = decoder_t.decode(z_t_content, z_t_style)
                fake_t, z_s_pc = decoder_t.decode(z_s_content, z_t_style, real_S_content_kv, real_S_content_kh,
                                                  real_S_content_ks)
                fake_s, z_t_pc = decoder_t.decode(z_t_content, z_s_style, None, None, None, real_S_style_kv,
                                                  real_S_style_kh, real_S_style_ks)
                z_s2t_content = encoder_content(fake_t)
                loss_11 = criterion[1](same_s, s_image)
                loss_22 = criterion[1](same_t, t_image)
                predict_z_t_content_seg = seg(z_t_content)
                predict_z_s_content_seg = seg(z_s_content)
                predict_z_s2t_content_seg = seg(z_s2t_content)

                loss_seg1 = criterion[-2](predict_z_s_content_seg, s_mask.long()) + criterion[-1](
                    predict_z_s_content_seg, s_mask)
                loss_seg2 = criterion[-2](predict_z_s2t_content_seg, s_mask.long()) + criterion[-1](
                    predict_z_s2t_content_seg, s_mask)
                loss_mesc = F.binary_cross_entropy_with_logits(
                    torch.softmax(predict_z_s_content_seg, dim=1)[:, 1, :, :],
                    s_mask, weight_s)
                fake_t_1 = fake_t
                pred_fake_t = Dt(fake_t_1)
                loss_1 = criterion[4](pred_fake_t, y_real.unsqueeze(1).expand(args.batchSize, -1))

                fake_s_1 = fake_s
                pred_fake_s = Ds(fake_s_1)
                loss_2 = criterion[4](pred_fake_s, y_real.unsqueeze(1).expand(args.batchSize, -1))

                predict_z_t_content_seg_1 = predict_z_t_content_seg
                predict_z_t_content_seg_1 = D_seg(predict_z_t_content_seg_1)
                loss_3 = criterion[4](predict_z_t_content_seg_1, y_real.unsqueeze(1).expand(args.batchSize, -1))
                # Phase Discriminator
                z_t_pc_1 = z_t_pc
                z_t_pc_1 = D_pc(z_t_pc_1)
                loss_pc = criterion[4](z_t_pc_1, y_real.unsqueeze(1).expand(args.batchSize, -1))
                loss_sce = torch.sum(-predict_z_t_content_seg_1 * torch.log2(predict_z_t_content_seg_1 + 1e-8)) * weight_t

                loss_G_total = (loss_1 + loss_2 + loss_11 + loss_22 + loss_seg1 + loss_sce +
                                loss_seg2 + loss_mesc + (loss_3 + loss_pc) + (loss0_1 + loss0_2))
                loss_G_total.backward()
                optimizer[0].step()
                optimizer[1].step()
                ###############################################################################################################

                # train Discriminator
                optimizer[2].zero_grad()
                pred_real = Dt(t_image)
                lo_1 = criterion[4](pred_real, y_real.unsqueeze(1).expand(args.batchSize, -1))

                fake_t = fake_t_buffer.push_and_pop(fake_t)  # 这边不能像上面那样直接用net,不然会报错
                pred_fake = Dt(fake_t.detach())
                lo_2 = criterion[4](pred_fake, y_fake.unsqueeze(1).expand(args.batchSize, -1))

                pred_real_1 = Ds(s_image)
                lo_3 = criterion[4](pred_real_1, y_real.unsqueeze(1).expand(args.batchSize, -1))

                fake_s = fake_s_buffer.push_and_pop(fake_s)  # 这边不能像上面那样直接用net,不然会报错
                pred_fake_1 = Ds(fake_s.detach())
                lo_4 = criterion[4](pred_fake_1, y_fake.unsqueeze(1).expand(args.batchSize, -1))

                predict_z_t_content_seg = pre_seg_s_buffer.push_and_pop(predict_z_t_content_seg)
                predict_z_s_content_seg = pre_seg_t_buffer.push_and_pop(predict_z_s_content_seg)
                predict_z_t_content_seg_real = D_seg(predict_z_t_content_seg.detach())
                predict_z_s_content_seg_fake = D_seg(predict_z_s_content_seg.detach())
                seg_real_loss = criterion[4](predict_z_t_content_seg_real,
                                             y_real.unsqueeze(1).expand(args.batchSize, -1))
                seg_fake_loss = criterion[4](predict_z_s_content_seg_fake,
                                             y_fake.unsqueeze(1).expand(args.batchSize, -1))
                z_s_pc = pc_s_buffer.push_and_pop(z_s_pc)
                z_t_pc = pc_t_buffer.push_and_pop(z_t_pc)
                z_s_pc_real = D_pc(z_s_pc.detach())
                z_t_pc_fake = D_pc(z_t_pc.detach())
                z_pc_real_loss = criterion[4](z_s_pc_real, y_real.unsqueeze(1).expand(args.batchSize, -1))
                z_pc_fake_loss = criterion[4](z_t_pc_fake, y_real.unsqueeze(1).expand(args.batchSize, -1))

                loss_D_total = z_pc_real_loss + z_pc_fake_loss + seg_real_loss + seg_fake_loss + lo_1 + lo_2 + lo_3 + lo_4

                loss_D_total.backward()
                optimizer[2].step()

                tq.update(args.batchSize)
                train_loss += loss_G_total.item()
                tq.set_postfix(loss='%.3f' % (train_loss / (i + 1)))
            tq.close()

            if epoch % args.validation_step == 0:
                dice_list, Dice = val(encoder_content, seg, dataloader_val)

                is_best = Dice > best_pred
                best_pred = max(best_pred, Dice)

                checkpoint_dir_root = args.checkpoint_dir
                checkpoint_dir = os.path.join(checkpoint_dir_root, str(k_fold))
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                checkpoint_latest = os.path.join(checkpoint_dir, 'checkpoint_latest.pth.tar')
                u.save_checkpoint({
                    'dice_list': dice_list,
                    'state_dict': encoder_content.state_dict(),
                    'state_dict1': encoder_s.state_dict(),
                    'state_dict2': encoder_t.state_dict(),
                    'state_dict3': decoder_t.state_dict(),
                    'state_dict41': netG_A_content_kernel.state_dict(),
                    'state_dict42': netG_A_style_kernel.state_dict(),
                    'state_dict5': seg.state_dict(),
                    'state_dict6': Ds.state_dict(),
                    'state_dict7': Dt.state_dict(),
                    'state_dict8': D_pc.state_dict(),
                    'state_dict9': D_seg.state_dict(),
                }, best_pred, epoch, is_best, checkpoint_dir, filename=checkpoint_latest)
            if (epoch - args.s1_num_epoch) % args.save_freq == 0:
                model_save_list.pop(0)
                model_save_list.append([encoder_content, seg])

def main(args=None, k_fold=1):
    dataroot_train = args.dataset_path
    dataset_train = LinearLesion(dataroot_train, scale=(args.crop_height, args.crop_width), mode='train')
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batchSize,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )

    dataset_val = LinearLesion_val(dataroot_train, scale=(args.crop_height, args.crop_width), mode='val')
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )

    # build model
    encoder_a = ResnetGenerator_encoder(args.input_nc).cuda()
    encoder_s = ResnetGenerator_encoder(args.input_nc).cuda()
    encoder_t = ResnetGenerator_encoder(args.input_nc).cuda()
    decoder_t = COCOFUNITTranslator().cuda()
    netG_A_content_kernel = KernelGenerator().cuda()
    netG_A_style_kernel = KernelGenerator().cuda()

    seg = build_segmenternew(args.feature_channel, args.num_classes).cuda()
    Ds = Discriminator(args.input_nc).cuda()
    Dt = Discriminator(args.input_nc).cuda()
    D_be = Discriminator(args.feature_channel).cuda()
    D_seg = Discriminator(args.num_classes).cuda()
    # init model
    encoder_a.apply(weight_init)
    encoder_s.apply(weight_init)
    encoder_t.apply(weight_init)
    netG_A_content_kernel.apply(weight_init)
    netG_A_style_kernel.apply(weight_init)
    seg.apply(weight_init)
    Ds.apply(weight_init)
    Dt.apply(weight_init)
    D_be.apply(weight_init)
    D_seg.apply(weight_init)

    cudnn.benchmark = True

    optimizer_G = torch.optim.Adam(
        itertools.chain(encoder_a.parameters(), encoder_s.parameters(), encoder_t.parameters(), decoder_t.parameters(),
                        netG_A_content_kernel.parameters(), netG_A_style_kernel.parameters()),
        lr=0.0002,
        betas=(0.5, 0.999))
    optimizer_seg = torch.optim.Adam(seg.parameters(), lr=0.001, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(
        itertools.chain(Ds.parameters(), Dt.parameters(), D_be.parameters(), D_seg.parameters()), lr=0.0001,
        betas=(0.5, 0.999))

    criterion_cro = torch.nn.CrossEntropyLoss(weight=None)
    criterion_aux = LS.Multi_DiceLoss(class_num=args.num_classes)
    criterion_GAN = torch.nn.MSELoss()
    criterion_identity = torch.nn.L1Loss()
    criterion_1 = nn.BCELoss()
    criterion_2 = nn.BCEWithLogitsLoss(weight=None)
    criterion_main = LS.DiceLoss()
    criterion = [criterion_2, criterion_identity, criterion_1, criterion_main, criterion_GAN, criterion_cro,
                 criterion_aux]
    optimizer = [optimizer_G, optimizer_seg, optimizer_D]
    train(args, encoder_a, encoder_s, encoder_t, decoder_t, netG_A_content_kernel, netG_A_style_kernel, seg, Ds, Dt,
          D_be, D_seg, optimizer, criterion, dataloader_train, dataloader_val, k_fold)


if __name__ == '__main__':
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    args = DefaultConfig()
    modes = args.mode

    comments = os.getcwd().split('/')[-1]
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    main(args=args, k_fold=int(args.fold))
