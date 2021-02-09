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
import sys
import numpy as np
import torch
from torch import optim
from tqdm import tqdm
from src import sRGB2XYZ
import src.utils as utls

try:
    from torch.utils.tensorboard import SummaryWriter
    use_tb = True
except ImportError:
    use_tb = False

from src.dataset import BasicDataset
from torch.utils.data import DataLoader

def train_net(net, device, dir_img, dir_gt, val_dir, val_dir_gt, epochs=300,
              batch_size=4, lr=0.0001, lrdf=0.5, lrdp=75, l2reg=0.001,
              chkpointperiod=1, patchsz=256, validationFrequency=10,
              save_cp=True):

    dir_checkpoint = 'checkpoints/'

    train = BasicDataset(dir_img, dir_gt, patch_size=patchsz)
    val = BasicDataset(val_dir, val_dir_gt, patch_size=patchsz)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                              num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False,
                            num_workers=8, pin_memory=True, drop_last=True)
    if use_tb:
        writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs} epochs
        Batch size:      {batch_size}
        Patch size:      {patchsz} x {patchsz}
        Learning rate:   {lr}
        Training size:   {len(train)}
        Validation size: {len(val)}
        Validation Frq.: {validationFrequency}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        TensorBoard:     {use_tb}
    ''')

    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999),
                           eps=1e-08, weight_decay=l2reg)
    scheduler = optim.lr_scheduler.StepLR(optimizer, lrdp, gamma=lrdf,
                                          last_epoch=-1)

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=len(train), desc=f'Epoch {epoch + 1}/{epochs}',
                  unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                xyz_gt = batch['gt_xyz']
                assert imgs.shape[1] == 3, (
                    f'Network has been defined with 3 input channels, '
                    f'but loaded training images have {imgs.shape[1]} channels.'
                    f' Please check that the images are loaded correctly.')

                assert xyz_gt.shape[1] == 3, (
                    f'Network has been defined with 3 input channels, '
                    f'but loaded XYZ images have {xyz_gt.shape[1]} channels. '
                    f'Please check that the images are loaded correctly.')

                imgs = imgs.to(device=device, dtype=torch.float32)
                xyz_gt = xyz_gt.to(device=device, dtype=torch.float32)

                rec_imgs, rendered_imgs = net(imgs)
                loss = utls.compute_loss(imgs, xyz_gt, rec_imgs, rendered_imgs)

                epoch_loss += loss.item()

                if use_tb:
                    writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.update(np.ceil(imgs.shape[0]))
                global_step += 1

        if (epoch + 1) % validationFrequency == 0:
            val_score = vald_net(net, val_loader, device)
            logging.info('Validation loss: {}'.format(val_score))
            if use_tb:
                writer.add_scalar('learning_rate',
                                  optimizer.param_groups[0]['lr'], global_step)
                writer.add_scalar('Loss/test', val_score, global_step)
                writer.add_images('images', imgs, global_step)
                writer.add_images('rendered-imgs', rendered_imgs, global_step)
                writer.add_images('rec-xyz', rec_imgs, global_step)
                writer.add_images('gt-xyz', xyz_gt, global_step)

        scheduler.step()

        if save_cp and (epoch + 1) % chkpointperiod == 0:
            if not os.path.exists(dir_checkpoint):
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')

            torch.save(net.state_dict(), dir_checkpoint +
                       f'ciexyznet{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved!')

    if not os.path.exists('models'):
        os.mkdir('models')
        logging.info('Created trained models directory')
    torch.save(net.state_dict(), 'models/' + 'model_sRGB-XYZ-sRGB.pth')
    logging.info('Saved trained model!')
    if use_tb:
        writer.close()
    logging.info('End of training')


def vald_net(net, loader, device):
    """Evaluation using MAE"""
    net.eval()
    n_val = len(loader) + 1
    mae = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch',
              leave=False) as pbar:
        for batch in loader:
            imgs = batch['image']
            xyz_gt = batch['gt_xyz']
            assert imgs.shape[1] == 3, (
                f'Network has been defined with 3 input channels, but loaded '
                f'training images have {imgs.shape[1]} channels. Please check '
                f'that the images are loaded correctly.')

            assert xyz_gt.shape[1] == 3, (
                f'Network has been defined with 3 input channels, but loaded '
                f'XYZ images have {xyz_gt.shape[1]} channels. Please check '
                f'that the images are loaded correctly.')

            imgs = imgs.to(device=device, dtype=torch.float32)
            xyz_gt = xyz_gt.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                rec_imgs, rendered_imgs = net(imgs)
                loss = utls.compute_loss(imgs, xyz_gt, rec_imgs, rendered_imgs)
                mae = mae + loss

            pbar.update(np.ceil(imgs.shape[0]))

    net.train()
    return mae / n_val

def get_args():
    parser = argparse.ArgumentParser(description='Train CIE XYZ Net.')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=300,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?',
                        default=4, help='Batch size', dest='batchsize')
    parser.add_argument('-lr', '--learning-rate', metavar='LR', type=float,
                        nargs='?', default=0.0001, help='Learning rate',
                        dest='lr')
    parser.add_argument('-l2r', '--l2reg', metavar='L2Reg', type=float,
                        nargs='?', default=0.001, help='L2 Regularization '
                                                       'factor', dest='l2r')
    parser.add_argument('-l', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-vf', '--validation-frequency', dest='val_frq',
                        type=int, default=10, help='Validation frequency.')
    parser.add_argument('-s', '--patch-size', dest='patchsz', type=int,
                        default=256, help='Size of training patch')
    parser.add_argument('-c', '--checkpoint-period', dest='chkpointperiod',
                        type=int, default=10,
                        help='Number of epochs to save a checkpoint')
    parser.add_argument('-ldf', '--learning-rate-drop-factor', dest='lrdf',
                        type=float, default=0.5,
                        help='Learning rate drop factor')
    parser.add_argument('-ldp', '--learning-rate-drop-period', dest='lrdp',
                        type=int, default=75, help='Learning rate drop period')
    parser.add_argument('-ntrd', '--training_dir_in', dest='in_trdir',
                        default='E:/sRGB-XYZ-dataset/sRGB_training/',
                        help='Input training image directory')
    parser.add_argument('-gtrd', '--training_dir_gt', dest='gt_trdir',
                        default='E:/sRGB-XYZ-dataset/XYZ_training/',
                        help='Ground truth training image directory')
    parser.add_argument('-nvld', '--validation_dir_in', dest='in_vldir',
                        default='E:/sRGB-XYZ-dataset/sRGB_validation/',
                        help='Input validation image directory')
    parser.add_argument('-gvld', '--validation_dir_gt', dest='gt_vldir',
                        default='E:/sRGB-XYZ-dataset/XYZ_validation/',
                        help='Ground truth validation image directory')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info('Training of CIE XYZ Net')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = sRGB2XYZ.CIEXYZNet(device=device)
    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)




    try:
        train_net(net=net, device=device, dir_img=args.in_trdir,
                  dir_gt=args.gt_trdir, val_dir=args.in_vldir,
                  val_dir_gt=args.gt_vldir, epochs=args.epochs,
                  batch_size=args.batchsize, lr=args.lr, lrdf=args.lrdf,
                  lrdp=args.lrdp, chkpointperiod=args.chkpointperiod,
                  validationFrequency=args.val_frq, patchsz=args.patchsz)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'intrrupted_check_point.pth')
        logging.info('Saved interrupt checkpoint backup')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
