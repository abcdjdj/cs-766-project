import torch
import cv2

'''
Reads the image specified by 'path' and returns it
param : path - path of image file
return : image as a numpy array
'''
def read_img(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    return image

'''
Converts numpy img to tensor
param : img - numpy arr containing image data
return : t - torch tensor of shape [1, 3, H, W]
'''
def img_to_tensor(img):
    t = torch.from_numpy(img)
    t = t.view(-1, 3, t.shape[0], t.shape[1])
    return t

'''
Converts tensor back to numpy img
param : t - torch tensor of shape [1, 3, H, W]
return : img - numpy arr containing image data
'''
def tensor_to_img(t):
    t = t.view(t.shape[2], t.shape[3], 3)
    return t.numpy()

if __name__ == "__main__":
    img = cv2.imread('out/image/1_1.png')
    t = img_to_tensor(img)
    i = tensor_to_img(t)
    cv2.imwrite('out.png', i)
    print(i.shape)