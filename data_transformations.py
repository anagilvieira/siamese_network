import numpy as np
import scipy.ndimage as ndi
from skimage import morphology
import math
import cv2
import torch


class Normalize(object):  # MinMax Scaler
    def __init__(self):
        pass

    def __call__(self, sample):
        sample = (sample - sample.min()) / (sample.max() - sample.min())
        return sample

    
class Window(object):
    def __init__(self):
        pass
    
    def __call__(self, image, window_level, window_width):
        # WC: Window level - midpoint of the range of the CT numbers
        # WW: Window width - range of the CT numbers
        img_min = window_level - window_width // 2
        img_max = window_level + window_width // 2
        window_image = image.copy()
        window_image[window_image > img_max] = img_max
        window_image[window_image < img_min] = img_min
        return window_image


class Remove_Noise(object):
    def __init__(self):
        self.masked_image = []
        self.center = []
    
    def __call__(self, volume, idx):
        segmentation = morphology.dilation(volume[idx, :, :], np.ones((1, 1)))
        labels, label_nb = ndi.label(segmentation)
        label_count = np.bincount(labels.ravel().astype(np.int64))
        label_count[0] = 0
        mask = labels == label_count.argmax()
        mask = morphology.dilation(mask, np.ones((1, 1)))
        mask = ndi.morphology.binary_fill_holes(mask)
        mask = morphology.dilation(mask, np.ones((3, 3)))
        masked_img = mask * volume[idx, :, :]
        self.masked_image.append(masked_img)
        return self.masked_image


class Tilt_Correction(object):
    def __init__(self):
        self.rotated_image = []

    def __call__(self, volume, idx):
        img = np.uint8(volume[idx, :, :])
        contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # find the biggest contour (c) by the area
        c = max(contours, key=cv2.contourArea)
        (x, y), (Ma, ma), angle = cv2.fitEllipse(c)
        # cv2.ellipse(img, ((x, y), (Ma, ma), angle), color=(0, 255, 0), thickness=2)
        rmajor = max(Ma, ma) / 2
        if angle > 90:
            angle -= 90
        else:
            angle += 90
        xtop = x + math.cos(math.radians(angle)) * rmajor
        ytop = y + math.sin(math.radians(angle)) * rmajor
        xbot = x + math.cos(math.radians(angle + 180)) * rmajor
        ybot = y + math.sin(math.radians(angle + 180)) * rmajor
        # cv2.line(img, (int(xtop), int(ytop)), (int(xbot), int(ybot)), (0, 255, 0), 3)
        M = cv2.getRotationMatrix2D((x, y), angle - 90, 1)  # transformation matrix
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), cv2.INTER_CUBIC)
        self.rotated_image.append(img)
        return self.rotated_image


class Crop_Image(object):
    def __init__(self):
        self.croped_image = []

    def __call__(self, image, idx):
        # Create a mask with the background pixels
        mask = image[idx, :, :] == 0
        # Find the brain area
        coords = np.array(np.nonzero(~mask))
        top_left = np.min(coords, axis=1)
        bottom_right = np.max(coords, axis=1)
        # Remove the background
        croped_aux = image[idx, top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
        self.croped_image.append(croped_aux)
        return self.croped_image


class Add_Pad(object):
    def __init__(self):
        self.final_image = []

    def __call__(self, image, idx, new_height=512, new_width=512):
        # image is a list of arrays
        height, width = image[idx].shape[0], image[idx].shape[1]
        final_aux = np.zeros((new_height, new_width))
        pad_left = int((new_width - width) // 2)
        pad_top = int((new_height - height) // 2)
        # Replace the pixels with the image's pixels
        final_aux[pad_top:pad_top + height, pad_left:pad_left + width] = image[idx]
        self.final_image.append(final_aux)
        return self.final_image


def Half_Brain(image):
    croped_left = []
    croped_right = []
    image = image.cpu().numpy()
    if len(image)>1:
        image = np.squeeze(image)
        for i in range(len(image)):
            mask = image[i, :, :] == 0
            coords = np.array(np.nonzero(~mask))
            center = ndi.center_of_mass(image[i, :, :])
            top_left = np.min(coords, axis=1)
            bottom_right = np.max(coords, axis=1)
            # Remove the background
            croped_aux1 = image[i, top_left[0]:bottom_right[0], top_left[1]:int(center[1])]
            croped_left.append(croped_aux1)
            croped_aux2 = image[i, top_left[0]:bottom_right[0], int(center[1]):bottom_right[1]]
            croped_right.append(croped_aux2)
    else:
        image = np.squeeze(image)
        mask = image == 0
        coords = np.array(np.nonzero(~mask))
        center = ndi.center_of_mass(image)
        top_left = np.min(coords, axis=1)
        bottom_right = np.max(coords, axis=1)
        # Remove the background
        croped_left = image[top_left[0]:bottom_right[0], top_left[1]:int(center[1])]
        croped_right = image[top_left[0]:bottom_right[0], int(center[1]):bottom_right[1]]

    return croped_left, croped_right


def Padding_Half(image1, image2, new_height=512, new_width=512):
    image_left = []
    image_right = []
    if len(image1)<20:
        for i in range(len(image1)):
            height1, width1 = image1[i].shape[0], image1[i].shape[1]
            height2, width2 = image2[i].shape[0], image2[i].shape[1]
            final_aux1 = np.zeros((new_height, new_width))
            final_aux2 = np.zeros((new_height, new_width))
            pad_left1 = int((new_width - width1) // 2)
            pad_top1 = int((new_height - height1) // 2)
            pad_left2 = int((new_width - width2) // 2)
            pad_top2 = int((new_height - height2) // 2)
            # Replace the pixels with the image's pixels
            final_aux1[pad_top1:pad_top1 + height1, pad_left1:pad_left1 + width1] = image1[i]
            image_left.append(final_aux1)
            final_aux2[pad_top2:pad_top2 + height2, pad_left2:pad_left2 + width2] = image2[i]
            image_right.append(final_aux2)
    else:
        image1 = np.squeeze(image1)
        image2 = np.squeeze(image2)
        height1, width1 = image1.shape[0], image1.shape[1]
        height2, width2 = image2.shape[0], image2.shape[1]
        final_aux1 = np.zeros((new_height, new_width))
        final_aux2 = np.zeros((new_height, new_width))
        pad_left1 = int((new_width - width1) // 2)
        pad_top1 = int((new_height - height1) // 2)
        pad_left2 = int((new_width - width2) // 2)
        pad_top2 = int((new_height - height2) // 2)
        # Replace the pixels with the image's pixels
        final_aux1[pad_top1:pad_top1 + height1, pad_left1:pad_left1 + width1] = image1
        image_left.append(final_aux1)
        final_aux2[pad_top2:pad_top2 + height2, pad_left2:pad_left2 + width2] = image2
        image_right.append(final_aux2)

    return image_left, image_right


def Flip_Brain(image):
    flip = []
    image = image.cpu().detach().numpy()
    if len(image)>1:
        image = np.squeeze(image)
        for i in range(len(image)):
            flip_aux = np.fliplr(image[i,:,:])
            flip.append(flip_aux)

    else:
        image = np.squeeze(image)
        flip_aux = np.fliplr(image)
        flip.append(flip_aux)
        
    return np.array(flip)