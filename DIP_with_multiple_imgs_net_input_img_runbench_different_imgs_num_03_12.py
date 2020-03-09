# Code for "Blind restoration of a JPEG-compressed image" and "Blind image denoising" figures. Select `fname` below to switch between the two.
# To see overfitting set `num_iter` to a large value.


########## Import libs ##########
from __future__ import print_function
import matplotlib.pyplot as plt
import os
import numpy as np
from models import *
import torch
import torch.optim
from torch.utils.tensorboard import SummaryWriter
from skimage.measure import compare_psnr
from utils.denoising_utils import *
import timeit

########## Run Settings ##########
# settings.a - gpu vs cpu 
flag_use_gpu = True
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# settings.b - image/images settings
single_img_seq = True # img sequence of length 1
img_seq_name_list = ['data/Benchmarks/Baboon','data/Benchmarks/F16',
                     'data/Benchmarks/House','data/Benchmarks/Lena',
                     'data/Benchmarks/Peppers'] 
 

    
imsize = -1  # use original size image
# settings.c - image/images synthetic noise settings
noisy_imgs_num = 2 # number of noisy images 
add_homographies = False
homographies_params = [1,0,4,0,1,4,0,0]
add_patch_movement = False
add_gaussian_noise = True
add_img_enhancements = False # changes of color,brightness, and sharpness
img_enhancements_params = [0.03, 0.03, 0]
imgs_noise_sigma = 25 
imgs_noise_sigma_ = imgs_noise_sigma/255.
# settings.d - user interface settings
PLOT = False
LOG_IMAGES = True
show_every = 100
figsize = 4
# settings.e - net inputs params
Z3_var = 1./50.      

# settings.e - run params
num_iter = 5000



if flag_use_gpu:
    # use GPU
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    dtype = torch.cuda.FloatTensor
else:       
    # use CPU
    #os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.backends.cudnn.enabled = False  
    torch.backends.cudnn.benchmark = False 
    dtype = torch.FloatTensor 

for img_seq_name in img_seq_name_list:
    if single_img_seq:
        fname = img_seq_name + '.png'
        ########## Load image ##########
        img_pil = crop_image(get_image(fname, imsize)[0], d=32)
        img_np = pil_to_np(img_pil)
        ########## create synthetic noisy images ##########
        
        # nadav_wp_noise2noise
        imgs_noisy_pil_list, imgs_noisy_np_list,imgs_np_list = get_noisy_images_list(img_np, imgs_noise_sigma_,noisy_imgs_num,
                                                            add_gaussian_noise,add_img_enhancements,img_enhancements_params,
                                                            add_homographies,homographies_params,add_patch_movement)

        # imgs_noisy_pil_list, imgs_noisy_np_list,imgs_np_list = get_noisy_images_list(
        #                                                     np.clip(img_np + np.random.normal(scale=25./255, size=img_np.shape), 0, 1).astype(np.float32), 
        #                                                     imgs_noise_sigma_,noisy_imgs_num,
        #                                                     add_gaussian_noise,add_img_enhancements,img_enhancements_params,
        #                                                     add_homographies,homographies_params,add_patch_movement)    
        # nadav_wp_noise2noise


        if add_patch_movement:
            img_np = img_np[:,16:-16,16:-16]
            img_pil = np_to_pil(img_np)
        if PLOT:
            plot_image_grid([img_np] + imgs_noisy_np_list, 4, 6, PLOT=True)

        # writer = SummaryWriter(comment= '_' + str(noisy_imgs_num)+"_noisy_images") # TB-log
        writer = SummaryWriter() # TB-log

        if LOG_IMAGES:
            fig = plot_image_grid([img_np] + imgs_noisy_np_list, 4, 6)
            writer.add_image("clean and noisy images",fig,0)
            plt.close('all')


    else: # img sequence with number of images>1
        imgs_np_list = []
        imgs_noisy_np_list = [] 
        
        for j in range(noisy_imgs_num):
            fname = img_seq_name + ('_%d.png' %j) #('_%d.JPG' %j)
            ########## Load image ##########
            img_pil = crop_image(get_image(fname, imsize)[0], d=32)
            img_np = pil_to_np(img_pil)
            imgs_np_list.append(img_np)
            _ , img_noisy_np = get_noisy_image(img_np,imgs_noise_sigma_)
            imgs_noisy_np_list.append(img_noisy_np)

        if PLOT:
            plot_image_grid(imgs_np_list + imgs_noisy_np_list, 4, 6, PLOT=True)

        # writer = SummaryWriter(comment= '_' + str(noisy_imgs_num)+"_noisy_images") # TB-log
        writer = SummaryWriter() # TB-log

        if LOG_IMAGES:
            fig = plot_image_grid(imgs_np_list + imgs_noisy_np_list, 4, 6)
            writer.add_image("clean and noisy images",fig,0)
            plt.close('all')

        


    ########## Net Setup ##########
    INPUT = 'noise' # 'meshgrid'
    pad = 'reflection'
    OPT_OVER = 'net' # 'net,input'
    # optimizer params
    LR = 0.01 # 0.005 # 0.01
    OPTIMIZER='adam' # 'LBFGS'
    exp_weight=0.99

    input_depth = 3 #32 
    # choose one of the following net configurations:
    # net_config 1
    net = get_net(input_depth, 'skip', pad,
                    skip_n33d=64, 
                    skip_n33u=64, 
                    skip_n11=8, 
                    num_scales= 5,
                    upsample_mode='bilinear').type(dtype)

    # net_config 2 - shorter depth, all have skip connections
    #                 results little less than config1
    # net = get_net(input_depth, 'skip', pad,
    #                 skip_n33d=128, 
    #                 skip_n33u=128, 
    #                 skip_n11=4, 
    #                 num_scales= 3,
    #                 upsample_mode='bilinear').type(dtype)

    # net_config 3 - same depth, drop 2 deeper skip connections. little bit like net 2
    #                 results little less than config1, 
    # net = skip(input_depth, 3, 
    #             num_channels_down = [128, 128, 128, 128, 128], 
    #             num_channels_up   = [128, 128, 128, 128, 128],
    #             num_channels_skip = [4, 4, 4, 0, 0], 
    #             upsample_mode='bilinear',
    #             need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)

    # net_config 4 - same depth, drop 2 shallow skip connections
    #                 results much less than config1. skip connections in the shallow layers is critical
    # net = skip(input_depth, 3, 
    #             num_channels_down = [128, 128, 128, 128, 128], 
    #             num_channels_up   = [128, 128, 128, 128, 128],
    #             num_channels_skip = [0, 0, 4, 4, 4], 
    #             upsample_mode='bilinear',
    #             need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)

    # # net_config 5 - same depth, more skip connections between layers
    #                  results little better than config1. TODO: try even more skip connections
    # net = get_net(input_depth, 'skip', pad,
    #                 skip_n33d=128, 
    #                 skip_n33u=128, 
    #                 skip_n11=16, 
    #                 num_scales= 5,
    #                 upsample_mode='bilinear').type(dtype)

    # # net_config 6 - like 1, more filters at shallow layers
    #                  results comparable to 7. 
    # net = skip(input_depth, 3, 
    #             num_channels_down = [192, 128, 128, 64, 64], 
    #             num_channels_up   = [192, 128, 128, 64, 64],
    #             num_channels_skip = [12, 8, 8, 4, 4], 
    #             upsample_mode='bilinear',
    #             need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)


    # net_config 7 - more skip connections on earlier phases, less filters in the depth
    #                BEST SO FAR. TODO: try even more skip connections?
    # net = skip(input_depth, 3, 
    #             num_channels_down = [128, 128, 64, 64, 64], 
    #             num_channels_up   = [128, 128, 64, 64, 64],
    #             num_channels_skip = [32, 32, 16, 8, 8], 
    #             upsample_mode='bilinear',
    #             need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)

    # net_config 8 - like config 5 with 2 more scales
    #                results barely different from config 5. the extra depth didn't help
    # net = skip(input_depth, 3, 
    #             num_channels_down = [128, 128, 128, 128, 128, 128, 128], 
    #             num_channels_up   = [128, 128, 128, 128, 128, 128, 128],
    #             num_channels_skip = [16, 16, 16, 16, 16, 16, 16], 
    #             upsample_mode='bilinear',
    #             need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)
    
    # net_config 9 - shit
    # net = skip(input_depth, 3, 
    #             num_channels_down = [32, 32, 64, 64, 128, 128, 128], 
    #             num_channels_up   = [32, 32, 64, 64, 128, 128, 128],
    #             num_channels_skip = [4, 4, 8, 8, 16, 16, 16], 
    #             upsample_mode='bilinear',
    #             need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)
        
    # net_config 10 - many filters with many skip connections at early stages
    #                   results not much better than net1.
    # net = skip(input_depth, 3, 
    #             num_channels_down = [192, 128, 128, 64], 
    #             num_channels_up   = [192, 128, 128, 64],
    #             num_channels_skip = [32, 32, 32, 16], 
    #             upsample_mode='bilinear',
    #             need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)

    # net_config 11 - many filters with many skip connections at early stages
    #                   results almost like 7
    # net = skip(input_depth, 3, 
    #             num_channels_down = [192, 160, 128, 64,32], 
    #             num_channels_up   = [192, 160, 128, 64,32],
    #             num_channels_skip = [16, 16, 16, 16, 16], 
    #             upsample_mode='bilinear',
    #             need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)

    ########### Changing convolution kernel sizes, stride  ##########

    # net_config 12 - less than config1
    # net = skip(input_depth, 3, 
    #             num_channels_down = [128, 128, 128, 128, 128], 
    #             num_channels_up   = [128, 128, 128, 128, 128],
    #             num_channels_skip = [4, 4, 4, 4, 4], 
    #             upsample_mode='bilinear',
    #             stride=2, filter_size_down=5, filter_size_up=5,
    #             need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)

    # net_config 13 - results not very different from 12 
    # net = skip(input_depth, 3, 
    #             num_channels_down = [128, 128, 128, 128, 128], 
    #             num_channels_up   = [128, 128, 128, 128, 128],
    #             num_channels_skip = [4, 4, 4, 4, 4], 
    #             upsample_mode='bilinear',
    #             stride=2,
    #             filter_size_down = [5, 5, 3, 3, 3], 
    #             filter_size_up   = [5, 5, 3, 3, 3],
    #             need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)  
    # net_config 14 - best so far
    # net = skip(input_depth, 3, 
    #             num_channels_down = [128, 128, 64, 64, 64], 
    #             num_channels_up   = [128, 128, 64, 64, 64],
    #             num_channels_skip = [32, 32, 16, 8, 8],
    #             upsample_mode='bilinear',
    #             stride=2,
    #             filter_size_down = [5, 5, 3, 3, 3], 
    #             filter_size_up   = [5, 5, 3, 3, 3],
    #             need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)
    
    # net_config 15 -
    # net = skip(input_depth, 3, 
    #             num_channels_down = [128, 128, 64, 64, 64], 
    #             num_channels_up   = [128, 128, 64, 64, 64],
    #             num_channels_skip = [32, 32, 16, 8, 8],
    #             upsample_mode='bilinear',
    #             stride=2,
    #             filter_size_down = [3, 3, 5, 5, 5], 
    #             filter_size_up   = [3, 3, 5, 5, 5],
    #             need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype) 

    # net_config 16 -
    # net = skip(input_depth, 3, 
    #             num_channels_down = [128, 128, 64, 64, 64], 
    #             num_channels_up   = [128, 128, 64, 64, 64],
    #             num_channels_skip = [32, 32, 16, 8, 8],
    #             upsample_mode='bilinear',
    #             stride=2,
    #             filter_size_down = [5, 5, 5, 5, 5], 
    #             filter_size_up   = [5, 5, 3, 5, 5],
    #             need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype) 
    

    ################# different stride ###############
    # net_config 17 - didn't work that well, didn't finish runs
    # net = skip(input_depth, 3, 
    #             num_channels_down = [128, 128, 64, 64, 64], 
    #             num_channels_up   = [128, 128, 64, 64, 64],
    #             num_channels_skip = [32, 32, 16, 8, 8],
    #             upsample_mode='bilinear',
    #             stride=3,
    #             filter_size_down = [5, 5, 5, 5, 5], 
    #             filter_size_up   = [5, 5, 5, 5, 5],
    #             need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)       
      
    


    # net_config XX
    # net = UNet(num_input_channels=input_depth, num_output_channels=3, 
    #             feature_scale=4, more_layers=0, concat_x=False,
    #             upsample_mode='deconv', pad=pad, norm_layer=nn.BatchNorm2d)

    # Compute number of parameters
    s  = sum([np.prod(list(p.size())) for p in net.parameters()]);
    writer.add_scalar("Number of params",s,0)
    print ('Number of params: %d' % s)

    # Loss
    mse = torch.nn.MSELoss().type(dtype)
    imgs_noisy_np = np.array(imgs_noisy_np_list)
    imgs_noisy_torch = np_to_torch(imgs_noisy_np).type(dtype)

    ########## Net Inputs Setup ##########
    net_inputs = np_to_torch(imgs_noisy_np).type(dtype).squeeze(dim=0)
    
    # dummy_input = torch.randn(net_inputs.shape).cuda()
    # writer.add_graph(net,dummy_input)

    ########## Optimize ##########
    net_inputs_saved = net_inputs.detach().clone() # saves globaly defind net inputs
    noise = net_inputs.detach().clone() # used only for shape
    out_avg = None
    last_net = None
    psnr_noisy_last = 0

    i = 0
    def closure():
        
        global i, out_avg, psnr_noisy_last, last_net, net_inputs

        if Z3_var > 0:
            net_inputs = net_inputs_saved + noise.normal_(mean=0, std=Z3_var)
        
        # foward_tic = timeit.default_timer()
        out = net(net_inputs).cuda()
        # foward_toc = timeit.default_timer()
        # writer.add_scalar('Foward time',foward_toc - foward_tic, i)

        # Smoothing
        if out_avg is None:
            out_avg = out.detach()
        else:
            out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)

        # backward_tic = timeit.default_timer()         
        
        # nadav_wp_loss 
        total_loss = mse(out[0,:,:,:], imgs_noisy_torch[0,1,:,:,:])
        for j in range (2,noisy_imgs_num):
            total_loss += mse(out[0,:,:,:], imgs_noisy_torch[0,j,:,:,:])
        for k in range (1,noisy_imgs_num):
            for j in range (0,noisy_imgs_num):
                if k != j:
                    total_loss += mse(out[k,:,:,:], imgs_noisy_torch[0,j,:,:,:])
        total_loss /= ((noisy_imgs_num-1) * noisy_imgs_num)

        #total_loss = mse(out, imgs_noisy_torch)
            
        writer.add_scalar('Loss', total_loss, i) # TB-log
        total_loss.backward()
        # backward_toc = timeit.default_timer() 
        # writer.add_scalar('Backward time',backward_toc - backward_tic, i)

        
        # nadav_wp-
        if single_img_seq and not add_patch_movement: 
            psnr_noisy = [compare_psnr(img_noisy_np, out.detach().cpu().numpy()[0]) for img_noisy_np in imgs_noisy_np_list] 
            min_psnr_noisy, max_psnr_noisy = min(psnr_noisy), max(psnr_noisy)
            psnr_gt = [compare_psnr(img_np, out.mean(dim=0).detach().cpu().numpy())]
            min_psnr_gt, max_psnr_gt = min(psnr_gt), max(psnr_gt)
            psnr_gt_sm = [compare_psnr(img_np, out_avg.mean(dim=0).detach().cpu().numpy())] 
            min_psnr_gt_sm, max_psnr_gt_sm = min(psnr_gt_sm), max(psnr_gt_sm)
            # psnr_noisy = [compare_psnr(img_noisy_np, out.detach().cpu().numpy()[0]) for img_noisy_np in imgs_noisy_np_list] 
            # min_psnr_noisy, max_psnr_noisy = min(psnr_noisy), max(psnr_noisy)
            # psnr_gt = [compare_psnr(img_np, out[j,:,:,:].detach().cpu().numpy()) for j in range(output_imgs_num)]
            # min_psnr_gt, max_psnr_gt = min(psnr_gt), max(psnr_gt)
            # psnr_gt_sm = [compare_psnr(img_np, out_avg[j,:,:,:].detach().cpu().numpy()) for j in range(output_imgs_num)] 
            # min_psnr_gt_sm, max_psnr_gt_sm = min(psnr_gt_sm), max(psnr_gt_sm)
        else:
            psnr_noisy = [compare_psnr(img_noisy_np, out.detach().cpu().numpy()[0]) for img_noisy_np in imgs_noisy_np_list] 
            min_psnr_noisy, max_psnr_noisy = min(psnr_noisy), max(psnr_noisy)
            psnr_gt = [compare_psnr(imgs_np_list[j], out[j,:,:,:].detach().cpu().numpy()) for j in range(noisy_imgs_num)]
            min_psnr_gt, max_psnr_gt = min(psnr_gt), max(psnr_gt)
            psnr_gt_sm = [compare_psnr(imgs_np_list[j], out_avg[j,:,:,:].detach().cpu().numpy()) for j in range(noisy_imgs_num)] 
            min_psnr_gt_sm, max_psnr_gt_sm = min(psnr_gt_sm), max(psnr_gt_sm)


        # nadav_wp-



        # TB-log
        writer.add_scalar('net_out vs noisy images - minimal [PSNR]', min_psnr_noisy, i)
        writer.add_scalar('net_out vs noisy images - maximal [PSNR]', max_psnr_noisy, i)
        writer.add_scalar('net_out vs gt image - minimal [PSNR]', min_psnr_gt, i)
        writer.add_scalar('net_out vs gt images - maximal [PSNR]', max_psnr_gt, i)
        writer.add_scalar('net_out smooth vs gt images - minimal [PSNR]', min_psnr_gt_sm, i)
        writer.add_scalar('net_out smooth vs gt images - maximal [PSNR]', max_psnr_gt_sm, i)
        # writer.add_scalar('net_out vs ground truth image [PSNR]', psnr_gt, i)
        # writer.add_scalar('smoothened net_out vs ground truth image [PSNR]', psnr_gt_sm, i)  

        
        # Note that we do not have GT for the "snail" example
        # So 'PSRN_gt', 'PSNR_gt_sm' make no sense
        #print ('Iteration %05d    Loss %f   PSNR_noisy: %f   PSRN_gt: %f PSNR_gt_sm: %f' % (i, total_loss.item(), psnr_noisy, psnr_gt, psnr_gt_sm), '\r', end='')
        # print('Iteration %05d    Loss %f   MAX_PSNR_noisy: %f   MAX_PSRN_gt: %f   MAX_PSNR_gt_sm: %f' % (i, total_loss.item(), max_psnr_noisy, max_psnr_gt, max_psnr_gt_sm), '\r', end='')
        print('Iteration %05d    Loss %f   MAX_PSRN_gt: %f   MAX_PSNR_gt_sm: %f' % (i, total_loss.item(), max_psnr_gt, max_psnr_gt_sm), '\r', end='')

        if  PLOT and i % show_every == 0:
            out_np = torch_to_np(out)
            plot_image_grid([np.clip(out_np, 0, 1), 
                            np.clip(torch_to_np(out_avg), 0, 1)], factor=figsize, nrow=1, PLOT=True)

        if LOG_IMAGES and i % show_every == 0:
            out_np = torch_to_np(out.mean(dim=0).unsqueeze(0))
            out_avg_np = torch_to_np(out_avg.mean(dim=0).unsqueeze(0))
            
            fig = plot_image_grid([np.clip(out_np, 0, 1), 
                                np.clip(out_avg_np, 0, 1)], 
                                factor=figsize, nrow=1)
            writer.add_image("net out and net out averaged",fig,i)
            plt.close('all')
    
        # # Backtracking
        # if i % show_every:
        #     if min_psnr_noisy - psnr_noisy_last < -5: 
        #         print('Falling back to previous checkpoint.')

        #         for new_param, net_param in zip(last_net, net.parameters()):
        #             net_param.data.copy_(new_param.cuda())

        #         return total_loss*0
        #     else:
        #         last_net = [x.detach().cpu() for x in net.parameters()]
        #         psnr_noisy_last = min_psnr_noisy
                
        i += 1

        return total_loss

    p = get_params(OPT_OVER, net, net_inputs)
    optimize(OPTIMIZER, p, closure, LR, num_iter)

    # result
    out_np = torch_to_np(net(net_inputs_saved))
    q = plot_image_grid([np.clip(out_np, 0, 1), img_np], factor=13, PLOT=PLOT)
    writer.add_image("final image - out",q,i)
    plt.close('all')

    out_avg_np = torch_to_np(out_avg) 
    q = plot_image_grid([np.clip(out_avg_np, 0, 1), img_np], factor=13, PLOT=PLOT)
    writer.add_image("final image - out averged",q,i)
    plt.close('all')
    writer.close() # TB-log
