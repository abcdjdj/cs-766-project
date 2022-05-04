from euler_double_u_net import DoubleUNet as DoubleUNet
import euler_utils
import pickle as pkl
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import cv2
import glob
from tqdm import tqdm as tqdm
from torchgeometry.losses.dice import DiceLoss
import pickle as pkl
import os
from matplotlib import pyplot as plt
import shutil
import euler_metrics

torch.manual_seed(0)
#torch.use_deterministic_algorithms(True)

# Use GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Change directory to /srv/home/kanbur/double-u-net
os.chdir('/srv/home/kanbur/double-u-net')

tmp = torch.ones(2, 3, 288, 384)
double_u_net = DoubleUNet(tmp).eval().to(device)
double_u_net.load_state_dict(torch.load("trained_models/double_unet_cvc-clinic.pt", map_location=torch.device('cpu')))


def load_split_sets():
    #print("Loading same train-val-test split as Double U-Net..")
    with open("train_set.pkl", 'rb') as f:
        train_set = pkl.load(f)
    with open("val_set.pkl", 'rb') as f:
        val_set = pkl.load(f)
    with open("test_set.pkl", 'rb') as f:
        test_set = pkl.load(f)
    
    return train_set, val_set, test_set

def infer(input_filenames, number):
    #print('Files = ' + str(input_filenames))
    x = [euler_utils.img_to_tensor(euler_utils.read_img(ele[0])) for ele in input_filenames]
    x = torch.cat(x, dim = 0)
    x = x.float()
    x = x.to(device)
    #print(f'X Shape = {x.shape}')

    with torch.no_grad():
        y_pred = double_u_net(x)
        y_pred = y_pred[0]
        y_pred1 = y_pred[0]
        y_pred2 = y_pred[1]

    y_truth = torch.from_numpy(euler_utils.read_mask(input_filenames[0][1]))

    dice_loss = euler_metrics.dice_loss(y_truth, y_pred2)
    iou = euler_metrics.iou(y_truth, y_pred2)
    print(f'Dice Loss = {dice_loss}')
    print(f'IOU = {iou}')

    mask_inference = y_pred2.detach().cpu().numpy()*(255)
    cv2.imwrite(f'comparable_results/infer_{number}_double.png', mask_inference)
    shutil.copyfile(input_filenames[0][1], f'comparable_results/infer_exp_{number}.png')

    return dice_loss, iou

#number = 1368
# train_set, val_set, test_set = load_split_sets()

with open("train_set_new.pkl", 'rb') as f:
    train_set = pkl.load(f)

with open("test_set_new.pkl", 'rb') as f:
    test_set = pkl.load(f)


#print(test_set[number])

indices = [0, 1200, 1203, 1216, 1235, 1248, 1272, 1275]

for idx in tqdm(indices):
    infer([train_set[idx] if idx!=0 else test_set[idx], train_set[idx] if idx!=0 else test_set[idx]], idx)

# dice_avg = 0
# iou_avg = 0
# for img in tqdm(train_set):
#     dice_loss, iou = infer([img, img])
#     dice_avg += dice_loss
#     iou_avg += iou

# dice_avg = dice_avg / len(train_set)
# iou_avg = iou_avg / len(train_set)
# print(f'Avg Dice Loss = {dice_avg}')
# print(f'Avg IOU Loss = {iou_avg}')