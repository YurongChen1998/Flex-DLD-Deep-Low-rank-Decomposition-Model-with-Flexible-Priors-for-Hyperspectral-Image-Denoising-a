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
    dataset_dir = '../Data/CAVE_Noisy_data/'
    matfile = dataset_dir + '/' + data_name + '.mat'
    X_ori = sio.loadmat(matfile)['img'].transpose(2, 0, 1)


    # --------------------- Load Recon Image --------------------------- #
    res_dir = './Results/3_CAVE_Results/' + data_name + '/'
    resfile = os.listdir(res_dir)
    resfile = res_dir + resfile[0]
    x_rec = sio.loadmat(resfile)['x_rec'].transpose(2, 0, 1)


    # --------------------- Caculate Results --------------------------- #
    psnr_x, ssim_x, _ = MSIQA(X_ori, x_rec)
    print('\n')
    print('{} | PSNR = {:.3f}dB | SSIM = {:.3f}'.format(data_name, psnr_x, ssim_x))



if __name__ == '__main__':
    data_list = ['Case0_15', 'Case0_35', 'Case0_55', 'Case1', 'Case2', 'Case3', 'Case4', 'Case5', 'Case6']
    for data_name in data_list:
        eval(data_name)
