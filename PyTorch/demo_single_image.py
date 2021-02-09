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

import argparse
import logging
import os
import torch
import cv2
import numpy as np
from src import sRGB2XYZ
from src import utils


def get_args():
    parser = argparse.ArgumentParser(
        description='Converting from sRGB to CIE XYZ and back.')
    parser.add_argument('--model_dir', '-m', default='./models',
                        help="Specify the directory of the trained model.",
                        dest='model_dir')
    parser.add_argument('--input', '-i', help='Input image filename',
                        dest='input', default='../images/a0280-IMG_0854.JPG')
    parser.add_argument('--task', '-t', default='srgb-2-xyz-2-srgb',
                        help="Specify the required task: 'srgb-2-xyz-2-srgb', "
                             "'srgb-2-xyz', or 'xyz-2-srgb'.", dest='task')
    parser.add_argument('--show', '-v', action='store_true', default=True,
                        help="Visualize the input and output images",
                        dest='show')
    parser.add_argument('--save', '-s', action='store_true',
                        help="Save the output images",
                        default=True, dest='save')
    parser.add_argument('--device', '-d', default='cuda',
                        help="Device: cuda or cpu.", dest='device')

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    if args.device.lower() == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    filename = args.input
    save_output = args.save
    task = args.task


    assert task.lower() == 'srgb-2-xyz-2-srgb' or task.lower(
        ) == 'srgb-2-xyz' or task.lower() == 'xyz-2-srgb', (
        "The task should be one of the following: 'srgb-2-xyz-2-srgb', "
        "'srgb-2-xyz', or 'xyz-2-srgb', but the given one is %s" % task)

    logging.info(f'Using device {device}')

    if save_output:
        out_dir = {"xyz-rec": '../reconstructed_xyz', "re-rendered":
            '../re-rendered_srgb'}
        if not os.path.exists(out_dir['xyz-rec']):
            os.mkdir(out_dir['xyz-rec'])
        if not os.path.exists(out_dir['re-rendered']):
            os.mkdir(out_dir['re-rendered'])

    if os.path.exists(os.path.join(args.model_dir, 'model_sRGB-XYZ-sRGB.pth')):
        ciexyzNet = sRGB2XYZ.CIEXYZNet(device=device)
        logging.info("Loading model {}".format(os.path.join(
            args.model_dir, 'model_sRGB-XYZ-sRGB.pth')))
        ciexyzNet.to(device=device)
        ciexyzNet.load_state_dict(
            torch.load(os.path.join(args.model_dir, 'model_sRGB-XYZ-sRGB.pth'),
                       map_location=device))

    else:
        raise Exception('Model not found!')

    ciexyzNet.eval()

    in_img = cv2.imread(filename)
    if in_img is None:
        raise Exception('Image not found!')

    logging.info(f'Processing image {filename}')

    in_img_tensor = utils.from_image_to_tensor(in_img).to(device=device,
                                                          dtype=torch.float32)

    if task.lower() == 'srgb-2-xyz-2-srgb':
        with torch.no_grad():
            output_XYZ, output_sRGB = ciexyzNet(in_img_tensor)
        output_XYZ = utils.from_tensor_to_image(output_XYZ, device=device)
        output_sRGB = utils.from_tensor_to_image(output_sRGB, device=device)
        output_XYZ = utils.outOfGamutClipping(output_XYZ)
        output_sRGB = utils.outOfGamutClipping(output_sRGB)

        if args.show:
            logging.info("Visualizing results for image:"
                         " {}, close to continue ...".format(filename))
            utils.imshow(in_img, xyz_out=output_XYZ, srgb_out=output_sRGB,
                         task=task)

        if save_output:
            in_dir, fn = os.path.split(filename)
            name, _ = os.path.splitext(fn)
            outxyz_name = os.path.join(out_dir['xyz-rec'], name +
                                       '_XYZ_reconstructed.png')
            outsrgb_name = os.path.join(out_dir['re-rendered'], name +
                                        '_sRGB_re-rendered.png')
            output_XYZ = output_XYZ * 65535
            output_sRGB = output_sRGB * 255
            cv2.imwrite(outxyz_name, output_XYZ.astype(np.uint16))
            cv2.imwrite(outsrgb_name, output_sRGB.astype(np.uint8))

    elif task.lower() == 'srgb-2-xyz':
        with torch.no_grad():
            output_XYZ = ciexyzNet.forward_srgb2xyz(in_img_tensor)
        output_XYZ = utils.from_tensor_to_image(output_XYZ, device=device)
        output_XYZ = utils.outOfGamutClipping(output_XYZ)

        if args.show:
            logging.info("Visualizing results for image:"
                         " {}, close to continue ...".format(filename))
            utils.imshow(in_img, xyz_out=output_XYZ, task=task)

        if save_output:
            in_dir, fn = os.path.split(filename)
            name, _ = os.path.splitext(fn)
            outxyz_name = os.path.join(out_dir['xyz-rec'], name +
                                       '_XYZ_reconstructed.png')
            output_XYZ = output_XYZ * 65535
            cv2.imwrite(outxyz_name, output_XYZ.astype(np.uint16))

    else:
        with torch.no_grad():
            output_sRGB = ciexyzNet.forward_xyz2srgb(in_img_tensor)
        output_sRGB = utils.from_tensor_to_image(output_sRGB, device=device)
        output_sRGB = utils.outOfGamutClipping(output_sRGB)

        if args.show:
            logging.info("Visualizing results for image:"
                         " {}, close to continue ...".format(filename))
            utils.imshow(in_img, srgb_out=output_sRGB, task=task)

        if save_output:
            in_dir, fn = os.path.split(filename)
            name, _ = os.path.splitext(fn)
            outsrgb_name = os.path.join(out_dir['re-rendered'], name +
                                        '_sRGB_re-rendered.png')
            output_sRGB = output_sRGB * 255
            cv2.imwrite(outsrgb_name, output_sRGB.astype(np.uint8))


