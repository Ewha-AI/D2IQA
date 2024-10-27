import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import os
import skimage
import imageio
import cv2
import math

def noise2stat():

    ans_path = '../../data/doc_test/ans.csv'
    ans_df = pd.read_csv(ans_path, index_col='title')
    ans_idx = ans_df.index
    noise_dict = {'full': 0,
                'full_0.5': 1,
                'quarter' : 2,
                'quarter_0.5' : 3,
                'quarter_1.0' : 4,
                'quarter_1.5' : 5,
                'quarter_2.0' : 6} 

    noise_list = {'full': [],
                'full_0.5': [],
                'quarter' : [],
                'quarter_0.5' : [],
                'quarter_1.0' : [],
                'quarter_1.5' : [],
                'quarter_2.0' : []}  

    rows = []
    for i in range(len(ans_df)):
        row = list(ans_df.iloc[i, :])
        idx = ans_idx[i]
        
        csv_path = '../../Desktop/result_new/mayo_5_quarter/result_quarter_mAP_mayo_5_{}_mean.csv'.format(idx)
        df = pd.read_csv(csv_path, index_col='Unnamed: 0')[-7:]

        temp_values = list(np.round(df['4'].values,4))
        values = list(map(lambda x: x*1, temp_values))


        new_row = [values[noise_dict[i]] for i in row]

        for n, noise in enumerate(row):
            noise_list[noise].append(new_row[n])
    

    for noise in noise_list.keys():
        print(noise, '%5.4f %5.4f' %(np.mean(noise_list[noise]), np.std(noise_list[noise])))


def doctor2stat():

    ans_path = '../../data/doc_test/ans.csv'
    ans_df = pd.read_csv(ans_path, index_col='title')
    ans_idx = ans_df.index

    noise_list = {'full': [],
                'full_0.5': [],
                'quarter' : [],
                'quarter_0.5' : [],
                'quarter_1.0' : [],
                'quarter_1.5' : [],
                'quarter_2.0' : []}  

    rows = []
    for i in range(len(ans_df)):
        row = list(ans_df.iloc[i, :])
        idx = ans_idx[i]
        
        csv_path = 'test_result/result_beck.xlsx'
        df = pd.read_excel(csv_path, index_col='score')#.iloc[:,:-3]
        df_row = df.loc[idx].values


        for n, noise in enumerate(row):
            noise_list[noise].append(df_row[n])
    

    for noise in noise_list.keys():
        print(noise, '%5.4f %5.4f' %(np.mean(noise_list[noise]), np.std(noise_list[noise])))


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def get_psnr():

    file_root = '../..//Desktop/paper_prep/data/nimg'
    # fid_list = os.listdir('../../Desktop/paper_prep/data/doc_test/test')
    fid_list = os.listdir('../../data/doc_test_wwx')


    ans_path = '../../data/doc_test/ans.csv'
    ans_df = pd.read_csv(ans_path, index_col='title')
    ans_idx = ans_df.index

    noise_list = {'full': [],
                'full_0.5': [],
                'quarter' : [],
                'quarter_0.5' : [],
                'quarter_1.0' : [],
                'quarter_1.5' : [],
                'quarter_2.0' : []}  

    for i in range(len(ans_df)):
        row = list(ans_df.iloc[i, :])
        idx = ans_idx[i]

        pid, fid = idx.split('_')
        compare_path = os.path.join(file_root, pid, 'full', (fid + '.tiff'))
        compare_img = imageio.imread(compare_path)

        for noise in row:
            target_path = os.path.join(file_root, pid, noise, (fid + '.tiff'))
            target_img = imageio.imread(target_path)
            psnr = calculate_psnr(compare_img, target_img)
            # ssim = calculate_ssim(compare_img, target_img)
            
            noise_list[noise].append(psnr)
        
    
    for noise in noise_list.keys():
        print(noise, np.mean(noise_list[noise]))


    
    # print(noise_list)

if __name__ == '__main__':
    # noise2stat()
    # doctor2stat()
    get_psnr()