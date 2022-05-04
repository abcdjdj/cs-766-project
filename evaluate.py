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
from tqdm import tqdm as tqdm
from ignite.metrics import Precision, Recall

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

    mask_inference = y_pred2.detach().cpu().numpy()*(255)
    cv2.imwrite(f'good_results_double/infer{number}.png', mask_inference)
    shutil.copyfile(input_filenames[0][1], f'good_results_double/infer_exp{number}.png')
    shutil.copyfile(input_filenames[0][0], f'comparable_results/infer_exp_{number}.png')
    
    y_truth = (y_truth >= 0.5).int()

    y_pred2 = (y_pred2 >= 0.5).int()

    precision = Precision()

    precision.update((y_pred2, y_truth))

    prec = precision.compute()

    recall = Recall()

    recall.update((y_pred2, y_truth))

    rec = recall.compute()

    print(f'Dice Loss = {dice_loss}')
    print(f'IOU = {iou}')
    print(f'Precision = {prec}')
    print(f'Recall = {rec}')
    
    return dice_loss, iou, prec, rec

torch.manual_seed(0)
#torch.use_deterministic_algorithms(True)

# Use GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Change directory to /srv/home/kanbur/double-u-net
os.chdir('/srv/home/kanbur/double-u-net')

tmp = torch.ones(2, 3, 288, 384)
double_u_net = DoubleUNet(tmp).eval().to(device)
double_u_net.load_state_dict(torch.load("trained_models/double_unet_cvc-clinic.pt", map_location=torch.device('cpu')))

indices = [1300, 1299, 1296, 1289, 1280, 1283, 1310, 1302, 1305, 1347, 1355, 1379, 1567, 1711]

train_set, _, _  = load_split_sets()

avg_dice = 0
avg_iou = 0
avg_precision = 0
avg_recall = 0

for idx in tqdm(indices):
    dice, iou, prec, rec = infer([train_set[idx], train_set[idx]], idx)
    avg_dice += dice
    avg_iou += iou
    avg_precision += prec
    avg_recall += rec

print(f"Average Dice loss - {avg_dice/len(indices)}")
print(f"Average IOU - {avg_iou/len(indices)}")
print(f"Average Precision -{avg_precision/len(indices)}")
print(f"Average Recall - {avg_recall/len(indices)}")