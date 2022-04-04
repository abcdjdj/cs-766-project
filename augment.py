import os
import glob
import cv2
from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,
    RandomGamma,
    HueSaturationValue,
    RGBShift,
    RandomBrightness,
    RandomContrast,
    MotionBlur,
    MedianBlur,
    GaussianBlur,
    GaussNoise,
    ChannelShuffle,
    CoarseDropout
)

'''
Reads the image specified by 'path' and returns it
param : path - path of image file
return : image as a numpy array
'''
def read_img(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    return image

'''
Generates 25 additional augmented images per image for the list of (images, masks)
passed
param : list of image paths, list of mask paths
return : void
'''
def augment(img_list, mask_list, out_path):
    crop_size = (192-32, 256-32)
    size = (256, 192)

    os.makedirs(os.path.join(out_path, 'image/'))
    os.makedirs(os.path.join(out_path, 'mask/'))

    for image_path, mask_path in zip(img_list, mask_list):
        img_name = image_path[image_path.index(os.sep) + 1:]
        mask_name = mask_path[mask_path.index(os.sep) + 1:]
        print(img_name, mask_name)

        x = read_img(image_path)
        y = read_img(mask_path)

        ## Center Crop
        augmented = CenterCrop(p=1, height=crop_size[0], width=crop_size[1])(image=x, mask=y)
        x1 = augmented['image']
        y1 = augmented['mask']

        ## Crop
        x_min = 0
        y_min = 0
        x_max = x_min + size[0]
        y_max = y_min + size[1]
        augmented = Crop(p=1, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)(image=x, mask=y)
        x2 = augmented['image']
        y2 = augmented['mask']

        ## Random Rotate 90 degree
        augmented = RandomRotate90(p=1)(image=x, mask=y)
        x3 = augmented['image']
        y3 = augmented['mask']

        ## Transpose
        augmented = Transpose(p=1)(image=x, mask=y)
        x4 = augmented['image']
        y4 = augmented['mask']

        ## ElasticTransform
        augmented = ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)(image=x, mask=y)
        x5 = augmented['image']
        y5 = augmented['mask']

        ## Grid Distortion
        augmented = GridDistortion(p=1)(image=x, mask=y)
        x6 = augmented['image']
        y6 = augmented['mask']

        ## Optical Distortion
        augmented = OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)(image=x, mask=y)
        x7 = augmented['image']
        y7 = augmented['mask']

        ## Vertical Flip
        augmented = HorizontalFlip(p=1)(image=x, mask=y)
        x8 = augmented['image']
        y8 = augmented['mask']

        ## Horizontal Flip
        augmented = HorizontalFlip(p=1)(image=x, mask=y)
        x9 = augmented['image']
        y9 = augmented['mask']

        ## Grayscale
        x10 = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
        y10 = y

        ## Grayscale Vertical Flip
        augmented = VerticalFlip(p=1)(image=x10, mask=y10)
        x11 = augmented['image']
        y11 = augmented['mask']

        ## Grayscale Horizontal Flip
        augmented = HorizontalFlip(p=1)(image=x10, mask=y10)
        x12 = augmented['image']
        y12 = augmented['mask']

        ## Grayscale Center Crop
        augmented = CenterCrop(p=1, height=crop_size[0], width=crop_size[1])(image=x10, mask=y10)
        x13 = augmented['image']
        y13 = augmented['mask']

        augmented = RandomBrightnessContrast(p=1)(image=x, mask=y)
        x14 = augmented['image']
        y14 = augmented['mask']

        augmented = RandomGamma(p=1)(image=x, mask=y)
        x15 = augmented['image']
        y15 = augmented['mask']

        augmented = HueSaturationValue(p=1)(image=x, mask=y)
        x16 = augmented['image']
        y16 = augmented['mask']

        augmented = RGBShift(p=1)(image=x, mask=y)
        x17 = augmented['image']
        y17 = augmented['mask']

        augmented = RandomBrightness(p=1)(image=x, mask=y)
        x18 = augmented['image']
        y18 = augmented['mask']

        augmented = RandomContrast(p=1)(image=x, mask=y)
        x19 = augmented['image']
        y19 = augmented['mask']

        augmented = MotionBlur(p=1, blur_limit=7)(image=x, mask=y)
        x20 = augmented['image']
        y20 = augmented['mask']

        augmented = MedianBlur(p=1, blur_limit=9)(image=x, mask=y)
        x21 = augmented['image']
        y21 = augmented['mask']

        augmented = GaussianBlur(p=1, blur_limit=9)(image=x, mask=y)
        x22 = augmented['image']
        y22 = augmented['mask']

        augmented = GaussNoise(p=1)(image=x, mask=y)
        x23 = augmented['image']
        y23 = augmented['mask']

        augmented = ChannelShuffle(p=1)(image=x, mask=y)
        x24 = augmented['image']
        y24 = augmented['mask']

        augmented = CoarseDropout(p=1, max_holes=8, max_height=32, max_width=32)(image=x, mask=y)
        x25 = augmented['image']
        y25 = augmented['mask']

        images = [x, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18,
            x19, x20, x21, x22, x23, x24, x25]
        masks = [y, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18,
            y19, y20, y21, y22, y23, y24, y25]

        counter = 1
        img_name = img_name[0:img_name.index('.')] # Remove the extension
        for image, mask in zip(images, masks):
            image_path = os.path.join(out_path, "image/", f'{img_name}_{counter}.png')
            mask_path  = os.path.join(out_path, "mask/", f'{img_name}_{counter}.png')
            cv2.imwrite(image_path, image)
            cv2.imwrite(mask_path, mask)
            counter = counter + 1
        
        break

    return 0

def main():
    cvc_db_img_list = sorted(glob.glob("cvc-db/PNG/Original/*"))
    cvc_db_mask_list = sorted(glob.glob("cvc-db/PNG/Ground Truth/*"))

    augment(cvc_db_img_list, cvc_db_mask_list, out_path = 'out/' )

if __name__ == "__main__":
    main()