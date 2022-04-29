import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torchgeometry.losses.dice import DiceLoss
from torchvision.models.resnet import BasicBlock
import cv2
import glob
from tqdm import tqdm as tqdm
import pickle as pkl
import os

'''
Reads the image specified by 'path' and returns it
param : path - path of image file
return : image as a numpy array
'''
def read_img(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = np.clip(image - np.median(image)+127, 0, 255)
    image = image/255.0
    image = image.astype(np.float32)
    return image

def read_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = mask/255.0
    mask = mask.astype(np.float32)
    #mask = np.expand_dims(mask, axis=-1)
    return mask

'''
Converts numpy img to tensor
param : img - numpy arr containing image data
return : t - torch tensor of shape [1, 3, H, W]
'''
def img_to_tensor(img):
    t = torch.from_numpy(img)
    t = t.view(-1, 3, t.shape[0], t.shape[1])
    return t

def mask_to_tensor(mask):
    t = torch.from_numpy(mask)
    t = t.view(-1, t.shape[0], t.shape[1])
    return t

'''
t - tensor of shape [H, W]
'''
def tensor_to_mask(t):
    t = t.view(t.shape[0], t.shape[1])
    return t.numpy()

'''
Converts tensor back to numpy img
param : t - torch tensor of shape [1, 3, H, W]
return : img - numpy arr containing image data
'''
def tensor_to_img(t):
    t = t.view(t.shape[2], t.shape[3], 3)
    return t.numpy()
