import numpy as np
import imageio
import cv2
import os, random
from glob import glob


def denormalize(image):

    """
    Normalize image between 0 and 1
    """
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0

    #Denormalize -1000,400
    image = image * (MAX_BOUND - MIN_BOUND) + MIN_BOUND
    image = image.astype(np.int16) #integer
    
    return image


def clip(image, wl=40, ww=350):

    """
    Normalize image between 0 and 1
    """
    MIN_BOUND = wl - (ww/2) #40 - 175 = -135
    MAX_BOUND = wl + (ww/2) #40 + 175 = 215

    #Clip
    image[image>MAX_BOUND] = MAX_BOUND
    image[image<MIN_BOUND] = MIN_BOUND

    #Normalize 0,1
    # image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    # image = image.astype(np.float32)

    return image



if __name__ == '__main__':

    pid_list = ['L067', 'L506']
    noise_list = ['full', 'full_0.5', 'quarter', 'quarter_0.5', 'quarter_1.0', 'quarter_1.5', 'quarter_2.0']

    for pid in pid_list:
        for noise in noise_list:

            target_dir40 = '../../data/{}/{}/{}'.format('nimg_wl40', pid, noise)
            target_dir50 = '../../data/{}/{}/{}'.format('nimg_wl50', pid, noise)
            
            if not os.path.exists(target_dir40):
                os.makedirs(target_dir40)
            if not os.path.exists(target_dir50):
                os.makedirs(target_dir50)
                
            data_root = '../../data/nimg/{}/{}'.format(pid, noise)
            data_list = sorted(glob(data_root + '/*.tiff'))
            
            for data_path in data_list:
                
                filename = data_path.split('/')[-1]
                img = imageio.imread(data_path) #[0-1] float
                denorm_img = denormalize(img) #[-1000, 400] int
                
                a = denorm_img.copy()
                b = denorm_img.copy()
                clipped40_img = clip(a, wl=40, ww=350) #[0-1] float
                clipped50_img = clip(b, wl=50, ww=350) #[0-1] float
                
                imageio.imwrite(os.path.join(target_dir40, filename), clipped40_img)
                imageio.imwrite(os.path.join(target_dir50, filename), clipped50_img)

            print('>>> Saved as', pid, noise)
