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

smooth = 1e-15

def dice_coef(y_true, y_pred):
    y_true = torch.flatten(y_true)
    y_pred = torch.flatten(y_pred)
    intersection = torch.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (torch.sum(y_true) + torch.sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def iou(y_true, y_pred):
    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred) - intersection
    x = (intersection + smooth) / (union + smooth)
    return x

def bce_dice_loss(y_true, y_pred):
    binary_crossentropy = nn.BCELoss()
    dice_loss = DiceLoss()
    return binary_crossentropy(y_pred, y_true) + dice_loss(y_pred, y_true)

