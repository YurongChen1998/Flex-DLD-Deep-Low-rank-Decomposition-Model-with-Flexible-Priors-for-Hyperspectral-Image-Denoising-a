##############################################################################
####                             Yurong Chen                              ####
####                      chenyurong1998 AT outlook.com                   ####
####                           Hunan University                           ####
####                       Happy Coding Happy Ending                      ####
##############################################################################

import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import cv2
import torch
import argparse
import numpy as np
from func import *
import scipy.io as sio
import matplotlib.pyplot as plt
from optimization import ADMM_Iter
import warnings

warnings.filterwarnings('ignore')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
random.seed(5)

#-----------------------Opti. Configuration -----------------------#
parser = argparse.ArgumentParser()
parser.add_argument('--iter_num', default = 50,             help="Maximum number of iterations")
parser.add_argument('--lambda_',  default = 0.03,           help="Facotr of the deep Low-rank regularization")
parser.add_argument('--LR_iter',  default = 3000,           help="Training epochs of deep Low-rank networks")
parser.add_argument('--R_iter',   default = 700,            help="Reduced Training epochs of deep Low-rank networks")
parser.add_argument('--lambda_R', default = 0.03,           help="Factor of TV/SSTV regularization in DLD")
parser.add_argument('--rank',     default = 13,             help="Rank of hyperspectral images")
args = parser.parse_args()


#----------------------- Data Configuration -----------------------#
h, w, nC, step = 256, 256, 191, 1
dataset_dir = '/home/user/Desktop/Working_Yurong/Working_HyperDenoise/DLD_PnP_Ours/Alation_Study/Class_after_denoising/dataset'
data_name = 'Indian_pines_noisy_band220'

results_dir = './Results/' + data_name + '/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
matfile = dataset_dir + '/' + data_name + '.mat'

data_truth = sio.loadmat(matfile)['indian_pines']
data_truth = torch.from_numpy(data_truth / np.max(data_truth))
noisy_data = data_truth
noisy_data[noisy_data > 1] = 1
noisy_data[noisy_data < 0] = 0
print('Noise Image PSNR:', calculate_psnr(data_truth, noisy_data), 'Noise Image SSIM:', calculate_ssim(data_truth, noisy_data))


#-------------------------- Optimization --------------------------#
x_rec = ADMM_Iter(noisy_data.to(device), data_truth.to(device), args, index = int(data_name[-1:]), save_path = results_dir)
