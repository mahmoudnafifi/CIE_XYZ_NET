"""
 Copyright 2020 Mahmoud Afifi.
 Released under the MIT License.
 If you use this code, please cite the following paper:
 Mahmoud Afifi, Abdelrahman Abdelhamed, Abdullah Abuolaim, Abhijith
 Punnappurath, and Michael S Brown.
 CIE XYZ Net: Unprocessing Images for Low-Level Computer Vision Tasks.
 arXiv preprint, 2020.
"""

__author__ = "Mahmoud Afifi"
__credits__ = ["Mahmoud Afifi"]

import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2

def outOfGamutClipping(I):
    """ Clips out-of-gamut pixels. """
    I[I > 1] = 1  # any pixel is higher than 1, clip it to 1
    I[I < 0] = 0  # any pixel is below 0, clip it to 0
    return I


def compute_loss(input, target_xyz, rec_xyz, rendered):
    loss = torch.sum(torch.abs(input - rendered) + (
        1.5 * torch.abs(target_xyz - rec_xyz)))/input.size(0)
    return loss

def from_tensor_to_image(tensor, device='cuda'):
    """ converts tensor to image """
    tensor = torch.squeeze(tensor, dim=0)
    if device == 'cpu':
        image = tensor.data.numpy()
    else:
        image = tensor.cpu().data.numpy()
    # CHW to HWC
    image = image.transpose((1, 2, 0))
    image = from_rgb2bgr(image)
    return image

def from_image_to_tensor(image):
    image = from_bgr2rgb(image)
    image = im2double(image)  # convert to double
    image = np.array(image)
    assert len(image.shape) == 3, ('Input image should be 3 channels colored '
                                   'images')
    # HWC to CHW
    image = image.transpose((2, 0, 1))
    return torch.unsqueeze(torch.from_numpy(image), dim=0)


def from_bgr2rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert from BGR to RGB

def from_rgb2bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # convert from BGR to RGB


def imshow(img, xyz_out=None, srgb_out=None, task=None):
    """ displays images """

    if task.lower() == 'srgb-2-xyz-2-srgb':
        if xyz_out is None:
            raise Exception('XYZ image is not given')
        if srgb_out is None:
            raise Exception('sRGB re-rendered image is not given')

        fig, ax = plt.subplots(1, 3)
        ax[0].set_title('input')
        ax[0].imshow(from_bgr2rgb(img))
        ax[0].axis('off')
        ax[1].set_title('rec. XYZ')
        ax[1].imshow(from_bgr2rgb(xyz_out))
        ax[1].axis('off')
        ax[2].set_title('re-rendered')
        ax[2].imshow(from_bgr2rgb(srgb_out))
        ax[2].axis('off')

    if task.lower() == 'srgb-2-xyz':
        if xyz_out is None:
            raise Exception('XYZ image is not given')

        fig, ax = plt.subplots(1, 2)
        ax[0].set_title('input')
        ax[0].imshow(from_bgr2rgb(img))
        ax[0].axis('off')
        ax[1].set_title('rec. XYZ')
        ax[1].imshow(from_bgr2rgb(xyz_out))
        ax[1].axis('off')

    if task.lower() == 'xyz-2-srgb':
        if srgb_out is None:
            raise Exception('sRGB re-rendered image is not given')

        fig, ax = plt.subplots(1, 2)
        ax[0].set_title('input')
        ax[0].imshow(from_bgr2rgb(img))
        ax[0].axis('off')
        ax[1].set_title('re-rendered')
        ax[1].imshow(from_bgr2rgb(srgb_out))
        ax[1].axis('off')

    if task.lower() == 'pp':
        if srgb_out is None:
            raise Exception('sRGB re-rendered image is not given')

        fig, ax = plt.subplots(1, 2)
        ax[0].set_title('input')
        ax[0].imshow(from_bgr2rgb(img))
        ax[0].axis('off')
        ax[1].set_title('result')
        ax[1].imshow(from_bgr2rgb(srgb_out))
        ax[1].axis('off')

    plt.xticks([]), plt.yticks([])
    plt.show()


def im2double(im):
    """ Returns a double image [0,1] of the uint im. """
    return cv2.normalize(im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
