import torch
import os
from collections import OrderedDict
from model.LRNet import Deep_Image_Prior_Network

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def low_rank_model_load(rank):
    im_net = Deep_Image_Prior_Network(rank, 'skip', 'reflection',
                            upsample_mode= 'bilinear',
                            skip_n33d=64,
                            skip_n33u=64,
                            skip_n11=16,
                            num_scales=3,
                            n_channels=rank).to(device)   
                                           
    spec_net0 = Deep_Image_Prior_Network(rank, 'skip1d', 'reflection',
                            upsample_mode='linear',
                            skip_n33d=36,
                            skip_n33u=36,
                            skip_n11=16,
                            num_scales=2,
                            n_channels=rank).to(device)

    return [im_net], [spec_net0]
      
def get_input(tensize, const=10.0):
    inp = torch.rand(tensize)/const
    inp = torch.autograd.Variable(inp, requires_grad=True).to(device)
    inp = torch.nn.Parameter(inp)
    return inp
