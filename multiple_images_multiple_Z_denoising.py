# Code for "Blind restoration of a JPEG-compressed image" and "Blind image denoising" figures. Select `fname` below to switch between the two.
# To see overfitting set `num_iter` to a large value.


########## Import libs ##########
from __future__ import print_function
import matplotlib.pyplot as plt

import os
flag_use_gpu = False
if flag_use_gpu:
    # select GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import numpy as np
from models import *

import torch
import torch.optim
from torch.utils.tensorboard import SummaryWriter
#from tensorboardX import SummaryWriter
from skimage.measure import compare_psnr
from utils.denoising_utils import *

import timeit

if flag_use_gpu:
    # use GPU
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    dtype = torch.cuda.FloatTensor
else:       
    # use CPU
    #os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.backends.cudnn.enabled = False  
    torch.backends.cudnn.benchmark = False 
    dtype = torch.FloatTensor 


imsize =-1
PLOT = False
sigma = 25
sigma_ = sigma/255.

LOG_IMAGES = False
noisy_imgs_num = 4
add_gaussian_noise = True
img_enhancements = False
img_enhancements_params = [0.03, 0.03, 0]

# deJPEG 
# fname = 'data/denoising/snail.jpg'

## denoising
fname = 'data/F16/F16_GT.png'


########## Load image ##########
if fname == 'data/denoising/snail.jpg':
    img_noisy_pil = crop_image(get_image(fname, imsize)[0], d=32)
    img_noisy_np = pil_to_np(img_noisy_pil)
    
    # As we don't have ground truth
    img_pil = img_noisy_pil
    img_np = img_noisy_np
    
    if PLOT:
        plot_image_grid([img_np], 4, 5)
        
elif fname == 'data/F16/F16_GT.png':
    # Add synthetic noise
    img_pil = crop_image(get_image(fname, imsize)[0], d=32)
    img_np = pil_to_np(img_pil)
    
    
    imgs_noisy_pil_list, imgs_noisy_np_list = get_noisy_images_list(img_np, sigma_,noisy_imgs_num,
                                                add_gaussian_noise,img_enhancements,img_enhancements_params)
    writer = SummaryWriter(comment= '_' + str(noisy_imgs_num)+"_noisy_images") # TB-log

    if PLOT:
        plot_image_grid([img_np] + imgs_noisy_np_list, 4, 6)
    if LOG_IMAGES:
        fig = plot_image_grid([img_np] + imgs_noisy_np_list, 4, 6)
        writer.add_image("clean and noisy images",fig,0)
        plt.close('all')
        

else:
    assert False




########## Setup ##########
INPUT = 'noise' # 'meshgrid'
pad = 'reflection'
OPT_OVER = 'net' # 'net,input'

reg_noise_std = 1./30. # set to 1./20. for sigma=50
LR = 0.01

OPTIMIZER='adam' # 'LBFGS'
show_every = 200
exp_weight=0.99

if fname == 'data/denoising/snail.jpg':
    num_iter = 2400
    input_depth = 3
    figsize = 5 
    
    net = skip(
                input_depth, 3, 
                num_channels_down = [8, 16, 32, 64, 128], 
                num_channels_up   = [8, 16, 32, 64, 128],
                num_channels_skip = [0, 0, 0, 4, 4], 
                upsample_mode='bilinear',
                need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

    net = net.type(dtype)

elif fname == 'data/F16/F16_GT.png':
    num_iter = 3000
    input_depth = 32 
    figsize = 4 
    
    
    net = get_net(input_depth, 'skip', pad,
                  skip_n33d=128, 
                  skip_n33u=128, 
                  skip_n11=4, 
                  num_scales=5,
                  upsample_mode='bilinear').type(dtype)

else:
    assert False
    
net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype)
net_input = torch.cat([net_input]*noisy_imgs_num)
for j in range(noisy_imgs_num):
    net_input[j,:,:,:] += get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).squeeze()


# Compute number of parameters
s  = sum([np.prod(list(p.size())) for p in net.parameters()]); 
print ('Number of params: %d' % s)

# Loss
mse = torch.nn.MSELoss().type(dtype)

imgs_noisy_np = np.array(imgs_noisy_np_list)
imgs_noisy_torch = np_to_torch(imgs_noisy_np).type(dtype)


########## Optimize ##########
net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()
out_avg = None
last_net = None
psnr_noisy_last = 0

i = 0
def closure():
    
    global i, out_avg, psnr_noisy_last, last_net, net_input
    
    if reg_noise_std > 0:
        for j in range(noisy_imgs_num):
            net_input[j,:,:,:] += torch.squeeze(noise.normal_() * reg_noise_std)
    
    foward_tic = timeit.default_timer()
    out = net(net_input)
    foward_toc = timeit.default_timer()
    writer.add_scalar('Foward time',foward_toc - foward_tic, i)
    # DEBUG - expanding out tensor according to number of dirty images
    # test_total_loss1 = mse(out, imgs_noisy_torch)
    # out = out.unsqueeze(1).repeat(1,6,1,1,1)
    # test_total_loss2 = mse(out, imgs_noisy_torch)

    # Smoothing
    if out_avg is None:
        out_avg = out.detach()
    else:
        out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)

    backward_tic = timeit.default_timer()         
    total_loss = mse(out, imgs_noisy_torch)
    writer.add_scalar('Loss', total_loss, i) # TB-log
    total_loss.backward()
    backward_toc = timeit.default_timer() 
    writer.add_scalar('Backward time',backward_toc - backward_tic, i)

    
    #psnr_noisy = compare_psnr(img_noisy_np, out.detach().cpu().numpy()[0])
    psnr_noisy = [compare_psnr(img_noisy_np, out.detach().cpu().numpy()[0]) for img_noisy_np in imgs_noisy_np_list] 
    min_psnr_noisy, max_psnr_noisy = min(psnr_noisy), max(psnr_noisy)
    psnr_gt    = compare_psnr(img_np, out.detach().cpu().numpy()[0]) 
    psnr_gt_sm = compare_psnr(img_np, out_avg.detach().cpu().numpy()[0])
    # TB-log
    writer.add_scalar('net_out vs noisy images - minimal [PSNR]', min_psnr_noisy, i)
    writer.add_scalar('net_out vs noisy images - maximal [PSNR]', max_psnr_noisy, i)
    writer.add_scalar('net_out vs ground truth image [PSNR]', psnr_gt, i)
    writer.add_scalar('smoothened net_out vs ground truth image [PSNR]', psnr_gt_sm, i) 

    
    # Note that we do not have GT for the "snail" example
    # So 'PSRN_gt', 'PSNR_gt_sm' make no sense
    #print ('Iteration %05d    Loss %f   PSNR_noisy: %f   PSRN_gt: %f PSNR_gt_sm: %f' % (i, total_loss.item(), psnr_noisy, psnr_gt, psnr_gt_sm), '\r', end='')
    print ('Iteration %05d    Loss %f   MIN_PSNR_noisy: %f   MAX_PSNR_noisy: %f   PSRN_gt: %f PSNR_gt_sm: %f' % (i, total_loss.item(), min_psnr_noisy, max_psnr_noisy, psnr_gt, psnr_gt_sm), '\r', end='')
    if  PLOT and i % show_every == 0:
        out_np = torch_to_np(out)
        plot_image_grid([np.clip(out_np, 0, 1), 
                         np.clip(torch_to_np(out_avg), 0, 1)], factor=figsize, nrow=1)

    if LOG_IMAGES and i % show_every == 0:
        out_np = torch_to_np(out)
        fig = plot_image_grid([np.clip(out_np, 0, 1), 
                         np.clip(torch_to_np(out_avg), 0, 1)], factor=figsize, nrow=1)
        writer.add_image("net out and net out averaged",fig,i)
        plt.close('all')
        
        
    
    # Backtracking
    if i % show_every:
        if min_psnr_noisy - psnr_noisy_last < -5: 
            print('Falling back to previous checkpoint.')

            for new_param, net_param in zip(last_net, net.parameters()):
                net_param.data.copy_(new_param.cuda())

            return total_loss*0
        else:
            last_net = [x.detach().cpu() for x in net.parameters()]
            psnr_noisy_last = min_psnr_noisy
            
    i += 1

    return total_loss

p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, LR, num_iter)

# result
out_np = torch_to_np(net(net_input))
q = plot_image_grid([np.clip(out_np, 0, 1), img_np], factor=13)
writer.add_image("final image - out",q,i)
plt.close('all')

out_avg_np = torch_to_np(out_avg) 
q = plot_image_grid([np.clip(out_avg_np, 0, 1), img_np], factor=13)
writer.add_image("final image - out averged",q,i)
plt.close('all')
writer.close() # TB-log


a = 1
















































