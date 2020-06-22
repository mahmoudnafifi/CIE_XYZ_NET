"""
 If you use this code, please cite the following paper:
 Mahmoud Afifi, Abdelrahman Abdelhamed, Abdullah Abuolaim, Abhijith Punnappurath, and Michael S Brown.
 CIE XYZ Net: Unprocessing Images for Low-Level Computer Vision Tasks. arXiv preprint, 2020.
"""

__author__ = "Mahmoud Afifi"
__credits__ = ["Mahmoud Afifi"]

from os.path import join
from os import listdir
from os import path
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
import cv2
from src import utils


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, xyz_dir, patch_size=256):
        self.imgs_dir = imgs_dir
        self.xyz_dir = xyz_dir
        self.patch_size = patch_size
        logging.info('Loading training images information...')
        self.imgfiles = [join(imgs_dir, file) for file in listdir(imgs_dir) if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.imgfiles)} examples')

    def __len__(self):
        return len(self.imgfiles)

    @classmethod
    def preprocess(cls, img, patch_size, w, h, patch_coords, aug_op, scale=1):
        if aug_op is 1:
            img = cv2.flip(img, 0)
        elif aug_op is 2:
            img = cv2.flip(img, 1)
        elif aug_op is 3:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        img_nd = np.array(img)
        assert len(img_nd.shape) == 3, 'Training/validation images should be 3 channels colored images'
        img_nd = img_nd[patch_coords[1]:patch_coords[1]+patch_size, patch_coords[0]:patch_coords[0]+patch_size, :]
        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        return img_trans

    def __getitem__(self, i):
        img_file = self.imgfiles[i]
        in_img = cv2.imread(img_file)
        in_img = utils.from_bgr2rgb(in_img)  # convert from BGR to RGB
        in_img = utils.im2double(in_img)  # convert to double
        # get image size
        h, w, _ = in_img.shape
        # get ground truth images
        in_dir, filename = path.split(img_file)
        name, _ = path.splitext(filename)
        gt_name = join(self.xyz_dir, name + '.png')
        xyz_img = cv2.imread(gt_name, -1)
        xyz_img = utils.from_bgr2rgb(xyz_img)  # convert from BGR to RGB
        xyz_img = utils.im2double(xyz_img)  # convert to double
        # get augmentation option
        aug_op = np.random.randint(4)
        if aug_op == 3:
            scale = np.random.uniform(low=1.0, high=1.2)
        else:
            scale = 1
        # get random patch coord
        patch_x = np.random.randint(0, high=w - self.patch_size)
        patch_y = np.random.randint(0, high=h - self.patch_size)
        in_img_patch = self.preprocess(in_img, self.patch_size, w, h, (patch_x, patch_y), aug_op, scale=scale)
        xyz_patch = self.preprocess(xyz_img, self.patch_size, w, h, (patch_x, patch_y), aug_op, scale=scale)

        return {'image': torch.from_numpy(in_img_patch), 'gt_xyz': torch.from_numpy(xyz_patch)}
