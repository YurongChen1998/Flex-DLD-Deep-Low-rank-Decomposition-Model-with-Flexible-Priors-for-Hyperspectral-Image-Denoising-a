##############################################################################
####                             Yurong Chen                              ####
####                      chenyurong1998 AT outlook.com                   ####
####                           Hunan University                           ####
####                       Happy Coding Happy Ending                      ####
##############################################################################

import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import torch
import time
import argparse
import numpy as np
from func import *
from numpy import *
import scipy.io as sio
import matplotlib.pyplot as plt
from optimziation import ADMM_Iter
import warnings

warnings.filterwarnings('ignore')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
random.seed(1234)

#-----------------------Opti. Configuration -----------------------#
parser = argparse.ArgumentParser()
parser.add_argument('--iter_num', default = 400,            help="Maximum number of iterations")
parser.add_argument('--lambda_',  default = 0.001,          help="Facotr of the deep Low-rank regularization")
parser.add_argument('--LR_iter',  default = 6000,           help="Training epochs of deep Low-rank networks")
parser.add_argument('--R_iter',   default = 700,            help="Reduced Training epochs of deep Low-rank networks")
parser.add_argument('--lambda_R', default = 0.01,           help="Factor of TV/SSTV regularization in DLD")
parser.add_argument('--rank',     default = 12,             help="Rank of hyperspectral images")
parser.add_argument('--scene',    default = 'scene01',      help="Scene01-10")
args = parser.parse_args()


#----------------------- Data Configuration -----------------------#
h, w, nC, step = 256, 256, 28, 1
dataset_dir = '../../Data/KAIST_Dataset/Orig_data/'
data_name =  args.scene

results_dir = './Results/' + data_name + '/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    
matfile = dataset_dir + '/' + data_name + '.mat'
data_truth = torch.from_numpy(sio.loadmat(matfile)['img']).to(device)
data_truth_shift = torch.zeros((h, w + step*(nC - 1), nC)).to(device)
for i in range(nC):
    data_truth_shift[:, i*step:i*step+w, i] = data_truth[:, :, i]

mask = torch.zeros((h, w + step*(nC - 1)))
mask_3d = torch.unsqueeze(mask, 2).repeat(1, 1, nC)
mask_256 = torch.from_numpy(sio.loadmat('../../Data/KAIST_Dataset/mask256.mat')['mask'])
for i in range(nC):
    mask_3d[:, i*step:i*step+w, i] = mask_256
Phi = mask_3d.to(device)

meas = torch.sum(Phi * data_truth_shift, 2).to(device)


#-------------------------- Optimization --------------------------#
recon = ADMM_Iter(meas, Phi, data_truth, args, results_dir)

