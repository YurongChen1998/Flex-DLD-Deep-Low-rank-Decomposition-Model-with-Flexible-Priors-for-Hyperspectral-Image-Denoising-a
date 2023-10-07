##############################################################################
####                             Yurong Chen                              ####
####                      chenyurong1998 AT outlook.com                   ####
####                          Hunan University                            ####
####                       Happy Coding Happy Ending                      ####
##############################################################################

import torch
from func import *
from os.path import exists
from model.model_loader import *
import scipy.io as sio
from model.utils import MSIQA
import warnings

warnings.filterwarnings('ignore')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def ADMM_Iter(meas, Phi, data_truth, args, result_dir):
    #-------------- Initialization --------------#
    x0 = At(meas, Phi)
    Phi_sum = torch.sum(Phi, 2)
    Phi_sum[Phi_sum==0] = 1
    theta = x0.to(device)
    b = torch.zeros_like(x0).to(device)
    best_PSNR = 0
    lambda_, lambda_R, rank = args.lambda_, args.lambda_R, args.rank
    iter_num, LR_iter, R_iter = args.iter_num, args.LR_iter,args.R_iter

    # ---------------- Iteration ----------------#
    for iter in range(iter_num):
        c = theta.to(device) + b.to(device)
        meas_b = A(c, Phi)
        x = c + At((meas - meas_b)/(Phi_sum + lambda_), Phi)
        x1 = shift_back(x-b, 1)
        
        if iter < 300:
            theta = TV_minimization(x1, 90, 10)
        else:
            theta = Low_Rank_Decomposition_PnP(data_truth, x1, meas, Phi, LR_iter, R_iter, lambda_R, rank, result_dir)
            LR_iter, lambda_R = int(LR_iter * 1.05), lambda_R * 0.96
   
        # --------------- Evaluation ---------------#
        psnr_x, ssim_x, _ = MSIQA(data_truth.permute(2, 0, 1).cpu().numpy(), theta.permute(2, 0, 1).cpu().numpy())
        print('Iter {} | PSNR = {:.2f}dB | SSIM = {:.2f}'.format( iter, psnr_x, ssim_x))
        
        if psnr_x>best_PSNR and iter>300:
            sio.savemat(result_dir + '/Iter-{}-PSNR-{:2.2f}.mat'.format(iter, psnr_x), {'img':theta.cpu().numpy()})
            best_PSNR = psnr_x
        
        theta = shift(theta, 1)
        b = b - (x.to(device) - theta.to(device))
    return theta, psnr_all
 
    
def Low_Rank_Decomposition_PnP(data_truth, temp_theta, meas, Phi, LR_iter, R_iter, lambda_R, rank, result_dir):
    best_loss = float('inf')
    reg_noise_std = 1.0/30.0
    loss_array = np.zeros(LR_iter)
    H, W, B = data_truth.shape
    loss_fn = torch.nn.L1Loss().to(device)
    loss_l2 = torch.nn.MSELoss().to(device)
    im_net, spec_net = low_rank_model_load(rank)
    
    if os.path.exists('Results/model_weights.pth'):
        im_net[0].load_state_dict(torch.load('Results/model_weights.pth')['im_net'])
        spec_net[0].load_state_dict(torch.load('Results/model_weights.pth')['spec_net'])
        LR_iter = R_iter
        print('----------------------- Load model weights -----------------------')
    
    truth_tensor = data_truth.permute(2, 0, 1).unsqueeze(0).to(device)
    temp_theta = temp_theta.permute(2, 0, 1).unsqueeze(0).to(device) 
    im_input = get_input([1, rank, 256, 256]).to(device)
    spec_input = get_input([1, rank, 28]).to(device)

    im_net[0].train()
    spec_net[0].train()
    net_params = list(im_net[0].parameters()) + list(spec_net[0].parameters())

    input_params = [im_input] + [spec_input]
    im_input_temp = im_input.detach().clone()
    spec_input_temp = spec_input.detach().clone()

    params = net_params + input_params
    optimizer = torch.optim.Adam(params=params)
    
    for idx in range(LR_iter):
        im_input_perturbed = im_input + im_input_temp.normal_()*reg_noise_std
        spec_input_perturbed = spec_input + spec_input_temp.normal_()*reg_noise_std

        U_img0 = im_net[0](im_input_perturbed)
        V0 = spec_net[0](spec_input_perturbed)
        U0 = U_img0.reshape(-1, rank, 256*256).permute(0, 2, 1)
        model_out0 = torch.bmm(U0, V0)
        model_out0 = model_out0.reshape(256, 256, 28).to(device)
        model_out = model_out0.permute(2, 0, 1).unsqueeze(0)
        
        loss = loss_fn(meas, A(shift(model_out.squeeze(0).permute(1, 2, 0), 1).to(device), Phi.to(device)))
        loss += loss_l2(temp_theta.float().to(device), model_out.float().to(device))
        tv_loss_ = tv_loss(model_out)
        loss += lambda_R*tv_loss_
        
        optimizer.zero_grad()   
        loss.backward()
        optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_epoch = idx
            best_hs_recon = model_out.detach()
            torch.save({'im_net': im_net[0].state_dict(), 'spec_net': spec_net[0].state_dict()}, 'Results/model_weights.pth')
        
        if (idx+1)%100==0:
            PSNR = calculate_psnr(truth_tensor, model_out.squeeze(0))
            print('Iter {}, x_loss:{:.4f}, tv_loss:{:.4f}, PSNR:{:.2f}'.format(idx+1, loss.detach().cpu().numpy(), tv_loss_.detach().cpu().numpy(), PSNR))
        
        if False:
            im_idx = idx%B
            with torch.no_grad():
                diff = abs(model_out.squeeze() - truth_tensor).mean(1).squeeze().detach().cpu().numpy()
                img = model_out[0, im_idx, :, :].squeeze().detach().cpu().numpy()
                noise_img = temp_theta[0, im_idx, :, :].squeeze().detach().cpu().numpy()
                data_img = truth_tensor[0, im_idx, :, :].detach().cpu().numpy()
                
                cv2.imshow('Band',
                        np.sqrt(abs(np.hstack((diff, img, noise_img, data_img)))))
                cv2.waitKey(1)
                
    return best_hs_recon.squeeze(0).permute(1, 2, 0)
