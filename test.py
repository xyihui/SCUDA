import csv
from config_test import DefaultConfig_test
from dataset.LinearLesion import *
from model.networks import *
from torch.utils.data import DataLoader
from datetime import datetime
import os
import torch
import numpy as np
import torch.backends.cudnn as cudnn

def compute_accuracy(pred, label):
    eps = 1e-6
    valid = (label >= 0)
    acc_sum = (valid * (pred == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum + eps) / (valid_sum + eps)
    return acc

def compute_evaluation(pred, label, classes):
    
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
            U = float(np.sum(np.logical_or(label_i, pred_i)))   #TP + FN + FP
            FP = pred_i.sum()-I
            FN = label_i.sum()-I
            dice = (2*I + eps) / (U + I + eps)
            jaccard = I / (U + eps)
            Precision = I / (I + FP + eps)
            Sensitivity = I / (I + FN + eps)
            per_dice.append(dice)
            per_jaccard.append(jaccard)            
            per_Precision.append(Precision)
            per_Sensitivity.append(Sensitivity)
            
    return per_dice, per_jaccard, per_Precision, per_Sensitivity, accuracy
    
def eval(encoder_content, seg, dataloader):
    print('start test!')
    with torch.no_grad():
        encoder_content.eval()
        seg.eval()
        with open("%s/%s_%s_test_per_seg_dice.csv" % (args.test_result_path, args.net_work, args.fold), 'w',
                  newline='', encoding='utf-8') as csv_file:
            csv_write = csv.writer(csv_file)
            csv_write.writerow(['image_name', 'seg_dice', 'seg_jaccard', 'seg_Precision', 'seg_Sensitivity', 'seg_accuracy'])

            for indexs, datal in enumerate(dataloader):

                seg_dice = []
                seg_jaccard = []
                seg_Precision = []
                seg_Sensitivity = []
                seg_accuracy = []
                for i, (data, label, label_path) in enumerate(datal):
                    print(label_path)
                    if torch.cuda.is_available() and args.use_gpu:
                        data = data.cuda()
                        label = label.cuda()
                    content_s = encoder_content(data)
                    predict1 = seg(content_s)
                    predict1 = predict1.data.cpu().numpy()
                    predict1 = np.squeeze(predict1)
                    predict1 = np.argmax(predict1, axis=0)
                    label = label.data.cpu().numpy()
                    label = np.squeeze(label)
                    seg_pd, seg_pj, seg_pp, seg_ps, seg_ac = compute_evaluation(predict1, label, args.num_classes)
                    seg_dice.append(seg_pd[1])
                    seg_jaccard.append(seg_pj[1])
                    seg_Precision.append(seg_pp[1])
                    seg_Sensitivity.append(seg_ps[1])
                    seg_accuracy.append(seg_ac)

                seg_dice_mean = (np.sum(seg_dice) - seg_dice.count(1)) / (len(seg_dice) - seg_dice.count(1))
                seg_jaccard_mean = (np.sum(seg_jaccard) - seg_jaccard.count(1)) / (
                            len(seg_jaccard) - seg_jaccard.count(1))
                seg_Precision_mean = (np.sum(seg_Precision) - seg_Precision.count(1)) / (
                        len(seg_Precision) - seg_Precision.count(1))
                seg_Sensitivity_mean = (np.sum(seg_Sensitivity) - seg_Sensitivity.count(1)) / (
                        len(seg_Sensitivity) - seg_Sensitivity.count(1))
                seg_acc_mean = np.mean(seg_accuracy)
                csv_write.writerow(
                    [(indexs + 1), seg_dice_mean, seg_jaccard_mean, seg_Precision_mean, seg_Sensitivity_mean,
                     float(seg_acc_mean)])
            csv_write.writerow([args.coarse_model])
        csv_file.close()
class LinearLesion_test(torch.utils.data.Dataset):
    def __init__(self, j, dataset_path):
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
        B_img = Image.open(B_path).convert('RGB')
        B_img = self.resize_img(B_img)
        B_img = np.array(B_img)
        B_label = Image.open(B_mask_path).convert('L')
        B_label = self.resize_label(B_label)
        B_label = np.array(B_label)
        B_label[B_label != 255] = 0
        B_label[B_label == 255] = 1
        B_label = torch.from_numpy(B_label.copy()).float()
        B = self.to_tensor(B_img.copy()).float()
        return B, B_label, B_mask_path

    def __len__(self):
        return len(self.image_lists)

    def read_list(self, j, image_path):
        img_list = []
        png_path = os.listdir(image_path)
        path = png_path[j]
        path = os.path.join(image_path, path)
        img_list += glob.glob(path + '/*.png')
        label_list = [x.replace('testB', 'testB_label').split('.')[0] + '.png' for x in img_list]

        return img_list, label_list

def main(args=None, k_fold=1):
    dataroot_train = args.dataset_path

    # build model
    encoder_a = ResnetGenerator_encoder(args.input_nc).cuda()
    seg = model.build_segmenternew(args.feature_channel, args.num_classes).cuda()
    cudnn.benchmark = True
    dataset_tests = []
    dataloader_tests = []
    for j in range(args.patients[k_fold - 1]):
        dataset_test = LinearLesion_test(args, j, dataroot_train)
        dataloader_test = DataLoader(
            dataset_test,
            batch_size=1,  # the default is 1(the number of gpu), you can set it to what you want
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )
        dataset_tests.append(dataset_test)
        dataloader_tests.append(dataloader_test)
    state_dict = torch.load(args.coarse_model)
    encoder_a.load_state_dict({k.replace('module.', ''): v for k, v in state_dict['state_dict'].items()})
    seg.load_state_dict({k.replace('module.', ''): v for k, v in state_dict['state_dict5'].items()})
    eval(encoder_a, seg, dataloader_tests)

if __name__ == '__main__':
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    args = DefaultConfig_test()  # args参数在这里设置
    modes = args.mode
    comments = os.getcwd().split('/')[-1]
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    main(args=args, k_fold=int(args.fold))
