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

########################################################
# Please adjust this function with your custom functions
########################################################


import logging


def postprocessing(input_img, PP, opt=None):
    """ Post-processing function; adjust it with your custom pp functions. """
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    if '+' not in PP:
        PP = [PP]
    else:
        PP = PP.split('+')

    output = input_img

    for pp in PP:
        if pp == "denoise":
            logging.info('denoising...')
            # denoising code goes here
        elif pp == "chrom-adapt":
            logging.info('chrom adapting...')
            # chromatic adaptation code goes here
        elif pp == "deblur":
            logging.info('deblurring...')
            # deblurring code goes here
        elif pp == "dehaze":
            logging.info('dehazing...')
            # dehazing code goes here
        elif pp == "editdetails":
            logging.info('editing local details...')
            # local detail enhancement code goes here
        elif pp == "exposure-fusion":
            logging.info('exposure fusion...')
            # exposure fusion code goes here
        elif pp == "transfer-colors":
            logging.info('color transfering...')
            # color transfer code goes here
            # you may need to use the 'opt' variable here
        elif pp == "super-res":
            logging.info('super-resolution processing...')
            # super resolution code goes here
        else:
            logging.info('wrong post-processing task!')

    return output
