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

####################################################################
# Please adjust pp_code/postprocessing.py with your custom functions
####################################################################


import argparse
import logging
import os
import torch
import cv2
import numpy as np
from src import sRGB2XYZ
from src import utils
from pp_code import postprocessing as pp


def get_args():
    parser = argparse.ArgumentParser(
        description='Converting from sRGB to CIE XYZ and back.')
    parser.add_argument('--model_dir', '-m', default='./models',
                        help="Specify the directory of the trained model.",
                        dest='model_dir')
    parser.add_argument('--input', '-i', help='Input image filename',
                        dest='input',
                        default='../images/a0280-IMG_0854.JPG')
    parser.add_argument('--pp_method', '-t',
                        default='none|none|denoise+deblur|none|none',
                        help="Post-processing methods; denoise, deblur, dehaze,"
                             " editdetails, exposure-fusion, transfer-colors,"
                             " chrom-adapt, super-res. The order is: "
                             "localLayer|sRGB-localLayer|CIE XYZ|sRGB|localLayer",
                        dest='task')
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
    tasks = task.split('|')

    logging.info(f'Using device {device}')

    if save_output:
        out_dir = '../results/'
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

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

    with torch.no_grad():

        local_to_xyz = ciexyzNet.forward_local(in_img_tensor, 'xyz')

        if tasks[0] != 'none':
            local_to_xyz = pp.postprocessing(local_to_xyz, tasks[0]).to(
                device=device, dtype=torch.float32)

        unprocessed_l = in_img_tensor - local_to_xyz

        if tasks[1] != 'none':
            unprocessed_l = pp.postprocessing(unprocessed_l, tasks[1]).to(
                device=device, dtype=torch.float32)

        xyz = ciexyzNet.forward_global(unprocessed_l, target='xyz')

        if tasks[2] != 'none':
            xyz = pp.postprocessing(xyz, tasks[2]).to(device=device,
                                                      dtype=torch.float32)

        srgb = ciexyzNet.forward_global(xyz, target='srgb')

        if tasks[3] != 'none':
            srgb = pp.postprocessing(srgb, tasks[3]).to(device=device,
                                                        dtype=torch.float32)

        local_t_srgb = ciexyzNet.forward_local(srgb, target='srgb')

        if tasks[4] != 'none':
            local_t_srgb = pp.postprocessing(local_t_srgb, tasks[4]).to(
                device=device, dtype=torch.float32)

        result = utils.outOfGamutClipping(utils.from_tensor_to_image(
            srgb + local_t_srgb))

    if args.show:
        logging.info("Visualizing results for image:"
                     " {}, close to continue ...".format(filename))
        utils.imshow(in_img, srgb_out=result, task='pp')

    if save_output:
        in_dir, fn = os.path.split(filename)
        name, _ = os.path.splitext(fn)
        out_filename = os.path.join(out_dir, name + '_result.png')
        result = result * 255
        cv2.imwrite(out_filename, result.astype(np.uint8))

