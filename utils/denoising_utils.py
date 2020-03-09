import os
import random
from .common_utils import *


        
def get_noisy_image(img_np, sigma):
    """Adds Gaussian noise to an image.
    
    Args: 
        img_np: image, np.array with values from 0 to 1
        sigma: std of the noise
    """
    # img_noisy_np = img_np + np.random.normal(scale=sigma, size=img_np.shape)
    # img_noisy_np = img_noisy_np.astype(np.float32)
    img_noisy_np = np.clip(img_np + np.random.normal(scale=sigma, size=img_np.shape), 0, 1).astype(np.float32)
    img_noisy_pil = np_to_pil(img_noisy_np)

    return img_noisy_pil, img_noisy_np        

def get_noisy_images_list(org_img_np, sigma, img_num,add_gaussian_noise=True, 
                            add_img_enhancements=False,img_enhancements_params=[0,0,0],
                            add_homographies=False,homographies_params=[1,0,0,0,1,0,0,0],
                            add_patch_movement=False):
    """Adds Gaussian noise to an image.
    
    Args: 
        img_np: image, np.array with values from 0 to 1
        sigma: std of the noise
        img_num: number of output noised images
    """
    img_noisy_np_list = []
    img_noisy_pil_list = []
    img_np_list = []
    
    org_img_pil = np_to_pil(org_img_np)

    for i in range(0,img_num):
        if add_homographies:
            im_size = org_img_pil.size

            homographies_params_i = homographies_params
            for ind,param in enumerate(homographies_params_i):
                if param != 0 and param != 1:
                    homographies_params_i[ind] = random.uniform(0,param)
            homographies_params_i = tuple(homographies_params_i)
            
            trans_img_pil = org_img_pil.transform(size=im_size,method=Image.PERSPECTIVE,
                                                    data=homographies_params_i,resample=Image.BILINEAR)
        else:
            trans_img_pil = org_img_pil

        if add_img_enhancements: 
            # guassian with mean 1.0 and small std -> small changes of color, brightness and sharpness 
            color_factor = np.random.normal(1.0, img_enhancements_params[0])
            brightness_factor = np.random.normal(1.0, img_enhancements_params[1])
            sharpness_factor = np.random.normal(1.0, img_enhancements_params[2])
        else: # enhancements with factor=1.0 gives original image 
            color_factor = 1.0
            brightness_factor = 1.0
            sharpness_factor = 1.0

        enhancer = PIL.ImageEnhance.Color(trans_img_pil)
        enhanced_img_pil = enhancer.enhance(color_factor)
        enhancer = PIL.ImageEnhance.Brightness(enhanced_img_pil)
        enhanced_img_pil = enhancer.enhance(brightness_factor)
        enhancer = PIL.ImageEnhance.Brightness(enhanced_img_pil)
        enhanced_img_pil = enhancer.enhance(sharpness_factor)

        enhanced_img_np = pil_to_np(enhanced_img_pil)
        if add_patch_movement:
            x_pixels_shift = random.randint(-2,2)
            y_pixels_shift = random.randint(-2,2)
            im_size = org_img_pil.size
            shifted_img_np = enhanced_img_np[:,(16-x_pixels_shift):-(16+x_pixels_shift),(16-y_pixels_shift):-(16+y_pixels_shift)]
            img_np_list.append(shifted_img_np)

        else:
            shifted_img_np = enhanced_img_np
            shifted_img_pil = np_to_pil(shifted_img_np)
            img_np_list = [org_img_np]


        if add_gaussian_noise:
            img_noisy_np_list.append(np.clip(shifted_img_np + np.random.normal(scale=sigma, size=shifted_img_np.shape), 0, 1).astype(np.float32))
            img_noisy_pil_list.append(np_to_pil(img_noisy_np_list[i]))
        else:
            img_noisy_np_list.append(shifted_img_np)
            img_noisy_pil_list.append(shifted_img_pil)

    
    return img_noisy_pil_list, img_noisy_np_list, img_np_list
