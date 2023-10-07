##############################################################################
####                             Yurong Chen                              ####
####                      chenyurong1998 AT outlook.com                   ####
####                           Hunan University                           ####
####                       Happy Coding Happy Ending                      ####
##############################################################################

import os
import numpy as np
import scipy.io as sio
from model.utils import MSIQA
import warnings
warnings.filterwarnings('ignore')

def eval(data_name):
    # --------------------- Load Clean Image --------------------------- #
    dataset_dir = '/media/D/Data/CASSI_Data/Kaist/kaist_data/'
    matfile = dataset_dir + '/' + data_name + '.mat'
    X_ori = sio.loadmat(matfile)['img'].transpose(2, 0, 1)


    # --------------------- Load Recon Image --------------------------- #
    res_dir = './Results/' + data_name + '/'
    resfile = os.listdir(res_dir)
    resfile = res_dir + resfile[0]
    x_rec = sio.loadmat(resfile)['img'].transpose(2, 0, 1)


    # --------------------- Caculate Results --------------------------- #
    psnr_x, ssim_x, _ = MSIQA(X_ori, x_rec)
    print('\n')
    print('{} | PSNR = {:.3f}dB | SSIM = {:.3f}'.format(data_name, psnr_x, ssim_x))



if __name__ == '__main__':
    data_list = ['scene01']
    for data_name in data_list:
        eval(data_name)
