##############################################################################
####                             Yurong Chen                              ####
####                      chenyurong1998 AT outlook.com                   ####
####                           Hunan University                           ####
####                       Happy Coding Happy Ending                      ####
##############################################################################

import cv2
import math
import torch
import numpy as np
from pytorch_msssim import ssim
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def A(data, Phi):
    return torch.sum(data * Phi, 2)


def A_torch(x, Phi):
    temp = x*Phi
    y = torch.sum(temp, 1)
    return y


def At(meas, Phi):
    meas = torch.unsqueeze(meas, 2).repeat(1, 1, Phi.shape[2])
    return meas * Phi


def shift(inputs, step):
    [h, w, nC] = inputs.shape
    output = torch.zeros((h, w+(nC - 1)*step, nC))
    for i in range(nC):
        output[:, i*step : i*step + w, i] = inputs[:, :, i]
    return output


def shift_torch(inputs, step):
    [bs, nC, row, col] = inputs.shape
    output = torch.zeros(bs, nC, row, col+(nC-1)*step)
    for i in range(nC):
        output[:, i, :, i*step:i*step+col] = inputs[:, i,:,:]
    return output.to(device)

def shift_back(inputs, step):
    [h, w, nC] = inputs.shape
    for i in range(nC):
        inputs[:, :, i] = torch.roll(inputs[:, :, i], (-1)*step*i, dims=1)
    output = inputs[:, 0 : w - step*(nC - 1), :]
    return output


def ssim_(data, recon):
    C1 = (0.01 * 1) ** 2
    C2 = (0.03 * 1) ** 2
    data = data.astype(np.float64)
    recon = recon.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(data, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(recon, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(data ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(recon ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(data * recon, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(data, recon, border=0):
    if not data.shape == recon.shape:
        raise ValueError('Data size must have the same dimensions!')
    if not data.dtype == recon.dtype:
        data, recon = data.float(), recon.float()
        
    h, w = data.shape[:2]
    data = data[border:h - border, border:w - border]
    recon = recon[border:h - border, border:w - border]
    if data.ndim == 2:
        return ssim_(data, recon)
    elif data.ndim == 3:
        return ssim(torch.unsqueeze(data, 0).permute(3, 0, 1, 2), torch.unsqueeze(recon, 0).permute(3, 0, 1, 2), data_range=1).data
     

def calculate_psnr(data, recon):
    mse = torch.mean((recon - data)**2)
    if mse == 0:
        return 100
    Pixel_max = 1.
    return 20 * torch.log10(Pixel_max / torch.sqrt(mse))
   
    
def tv_loss(input_t):
    input_t = input_t.squeeze(0).permute(1, 2, 0)
    temp1 = torch.cat((input_t[1:, :, :], input_t[-1, :, :].unsqueeze(0)), 0)
    temp2 = torch.cat((input_t[:, 1:, :], input_t[:, -1, :].unsqueeze(1)), 1)
    temp1_, temp2_ = temp1 - input_t, temp2 - input_t
    tv = torch.abs(temp1_) + torch.abs(temp2_)
    return tv.mean()
    
      
def ss_tv_loss(input_t):
    input_t = input_t.squeeze(0).permute(1, 2, 0)
    temp1 = torch.cat((input_t[1:, :, :], input_t[-1, :, :].unsqueeze(0)), 0)
    temp2 = torch.cat((input_t[:, 1:, :], input_t[:, -1, :].unsqueeze(1)), 1)
    temp3 = torch.cat((input_t[:, :, 1:], input_t[:, :, -1].unsqueeze(2)), 2)
    temp1_, temp2_, temp3_ = temp1 - input_t, temp2 - input_t, temp3 - input_t
    tv = torch.abs(temp1_) + torch.abs(temp2_) + torch.abs(temp3_)
    tv2 = (temp1_)**2 + (temp2_)**2 + (temp3_)**2

    if False:
        imgx = torch.abs(temp1_)
        imgx = imgx[:, :, 20].detach().cpu().numpy()
        imgy = torch.abs(temp2_)
        imgy = imgy[:, :, 20].detach().cpu().numpy()
        cv2.imshow('S', np.hstack((imgx*100, imgy*100)))
        cv2.waitKey(1)   
    return tv.mean() - 0.5*tv2.mean()


def shrink(x, _lambda):
    u = torch.sign(x) * torch.clamp(x - 1/_lambda, 0)
    return u


def TV_denoiser(x, _lambda, n_iter_max):
    dt = 0.24
    N = x.shape
    idx = torch.arange(1, N[0]+1)
    idx[-1] = N[0]-1
    iux = torch.arange(-1, N[0]-1)
    iux[0] = 0
    ir = torch.arange(1, N[1]+1)
    ir[-1] = N[1]-1
    il = torch.arange(-1, N[1]-1)
    il[0] = 0
    p1 = torch.zeros_like(x)
    p2 = torch.zeros_like(x)
    divp = torch.zeros_like(x)

    for i in range(n_iter_max):
        z = divp - x*_lambda
        z1 = z[:,ir] - z
        z2 = z[idx,:] - z
        denom_2d = 1 + dt*torch.sum(z1**2 + z2**2)
        p1 = (p1+dt*z1)/denom_2d
        p2 = (p2+dt*z2)/denom_2d
        divp = p1-p1[:,il] + p2 - p2[iux,:]
    u = x - divp/_lambda
    return u

def get_input(tensize, const=10.0):
    inp = torch.rand(tensize)/const
    inp = torch.autograd.Variable(inp, requires_grad=True).to(device)
    inp = torch.nn.Parameter(inp)
    return inp


def add_noise(data):
    case = 1
    noiselevel = 0.1
    H, W, B = data.shape
    noisy_data = torch.zeros_like(data)
    if case == 1:
        for i in range(B):
            noisy_data[:,:,i] = data[:,:,i] + noiselevel*torch.randn(H, W)
    return noisy_data
