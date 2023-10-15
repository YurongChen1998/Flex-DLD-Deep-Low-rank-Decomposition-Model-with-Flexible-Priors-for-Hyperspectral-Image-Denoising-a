##############################################################################
####                             Yurong Chen                              ####
####                      chenyurong1998 AT outlook.com                   ####
####                           Hunan University                           ####
####                       Happy Coding Happy Ending                      ####
##############################################################################

import os
import copy
import torch
torch.cuda.current_device()
import argparse
import numpy as np
from func import *
import scipy.io as sio
from model.utils import MSIQA
from model.model_loader import low_rank_model_load
import warnings

warnings.filterwarnings('ignore')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def ADMM_Iter(noisy_data, X_ori, args, index = None, save_path = None):
    # -------------------- Initialization -------------------- #
    l = noisy_data.to(device)
    u1 = torch.zeros_like(l).to(device)
    s = torch.zeros_like(l).to(device)
    ssim_all = []
    psnr_all = []
    loss_y_min = 30
    lambda_, lambda_R, rank = args.lambda_, args.lambda_R, args.rank
    iter_num, train_iter, R_iter = args.iter_num, args.LR_iter, args.R_iter
    
    truth_tensor = X_ori.permute(2, 0, 1).to(device)
    noisy_data_tensor = noisy_data.permute(2, 0, 1).unsqueeze(0).to(device)

    # ----------------------- Iteration ---------------------- #
    for it in range(iter_num):
        # -------- Updata x and s
        l, u1, noisy_data = l.to(device), u1.to(device), noisy_data.to(device)
        x = (noisy_data - s + lambda_*(l + u1)) / (1 + lambda_)
        s = shrink(noisy_data - x, 1000)
        
        # -------- Updata l
        temp_l = x - u1
        temp_l = temp_l.permute(2, 0, 1).unsqueeze(0)
        im_net, spec_net = low_rank_model_load(rank)
        train_iter, lambda_R = int(train_iter*1.01), lambda_R*0.96
        out, loss_y_iter = Low_Rank_Decomposition_PnP(truth_tensor, temp_l, im_net, spec_net, train_iter, R_iter, lambda_R, rank)
        l = out.squeeze(0).permute(1, 2, 0).to(device)
        
        # -------- Updata Dual Variable u1
        u1 = u1 - lambda_*(l - x)
        
        # -------- Evaluation
        x_rec = l
        psnr_x, ssim_x, _ = MSIQA(X_ori.permute(2, 0, 1).cpu().numpy(), x_rec.permute(2, 0, 1).cpu().numpy())
        print('Iter {} | loss = {:.3f} | PSNR = {:.2f}dB | SSIM = {:.2f}'.format( it+1, loss_y_iter, psnr_x, ssim_x))
        
        if loss_y_min < psnr_x and it > 10:
            loss_y_min = psnr_x
            sio.savemat(save_path + 'scene0{}_{}_{:.2f}_{:.3f}.mat'.format(index, it+1, psnr_x, ssim_x),{'x_rec': x_rec.detach().cpu().numpy()})
    return x_rec
    
  
def Low_Rank_Decomposition_PnP(truth_tensor, temp_l, im_net, spec_net, iter_num, R_iter, lambda_R, rank):   
    reg_noise_std = 1.0/30.0
    loss_array = np.zeros(iter_num)
    best_loss = float('inf')
    loss_fn = torch.nn.L1Loss().to(device)
    loss_l2 = torch.nn.MSELoss().to(device)
    Band, H, W = truth_tensor.shape

    im_input = get_input([1, rank, H, W]).to(device)
    spec_input = get_input([1, rank, Band]).to(device)
        
    if os.path.exists('Results/model_weights.pth'):
        im_net[0].load_state_dict(torch.load('Results/model_weights.pth')['im_net'])
        spec_net[0].load_state_dict(torch.load('Results/model_weights.pth')['spec_net'])
        iter_num = R_iter
        print('----------------------- Load model weights -----------------------')


    im_net[0].train()
    spec_net[0].train()
    net_params = list(im_net[0].parameters()) + list(spec_net[0].parameters())

    input_params = [im_input] + [spec_input]
    im_input_temp = im_input.detach().clone()
    spec_input_temp = spec_input.detach().clone()

    params = net_params + input_params
    optimizer = torch.optim.Adam(lr=1e-3, params=params)
    
    for idx in range(iter_num):
        im_input_perturbed = im_input + im_input_temp.normal_()*reg_noise_std
        spec_input_perturbed = spec_input + spec_input_temp.normal_()*reg_noise_std

        U_img0 = im_net[0](im_input_perturbed)
        V0 = spec_net[0](spec_input_perturbed)
        U0 = U_img0.reshape(-1, rank, H*W).permute(0, 2, 1)
        model_out0 = torch.bmm(U0, V0)
        model_out0 = model_out0.reshape(H, W, Band).to(device)
        model_out0 = model_out0.permute(2, 0, 1).unsqueeze(0)
                
        loss_1 = tv_loss(model_out0)       
        loss_ = loss_fn(temp_l, model_out0)
        loss = loss_ + lambda_R*(loss_1)
        loss_array[idx] = loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss_.item() < best_loss:
            best_loss = loss_.item()
            best_epoch = idx
            best_hs_recon = model_out0.detach()
            torch.save({'im_net': im_net[0].state_dict(), 'spec_net': spec_net[0].state_dict()}, 'Results/model_weights.pth')
                
        if (idx+1)%100==0:
            PSNR0 = calculate_psnr(truth_tensor, model_out0.squeeze(0))
            print('Lowrank--Iter {}, x_loss:{:.4f}, 1_loss:{:.4f}, PSNR:{:.2f}'.format(idx+1, loss_.detach().cpu().numpy(), loss_1.detach().cpu().numpy(), PSNR0))
        
        if False:
            im_idx = idx%191
            with torch.no_grad():
                U_img_ = truth_tensor[im_idx, :, :].squeeze().detach().cpu().numpy()
                img = model_out0[0, im_idx, :, :].squeeze().detach().cpu().numpy()
                noise_img = temp_l[0, im_idx, :, :].squeeze().detach().cpu().numpy()
                
                cv2.imshow('Band', np.sqrt(abs(np.hstack((U_img_, img, noise_img)))))
                cv2.waitKey(1)
                
    print('----------------------------------------------')
    return best_hs_recon, best_loss
