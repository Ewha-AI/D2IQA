"""
Reference : https://github.com/arifqodari/saliencyfilters
"""


from .saliencyfilters import SaliencyFilters
from sys import argv
import imageio
import time
import numpy as np
import os
from glob import glob
import cv2 as cv
import random


def remove_background(img):

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)

    # sure background area
    sure_bg = cv.dilate(opening,kernel,iterations=3)

    # Marker labelling
    _, markers = cv.connectedComponents(sure_bg)

    first_num = markers[0][0]
    last_num = markers[-1][0]
    img[markers == first_num] = [0, 0, 0]
    img[markers == last_num] = [0, 0, 0]

    no_bg_img = img

    return no_bg_img



def mayo2saliency(input_image, threshold):
    start_time = time.time()

    sf = SaliencyFilters()
    image = imageio.imread(input_image) #0-1
        
    #Add dimenstion
    image = np.expand_dims(image, axis=-1)
    image = np.concatenate([image, image, image], axis=-1)

    saliency = sf.compute_saliency(image)

    #Apply threshold 0.3
    saliency[saliency<threshold] = 0
    saliency[saliency>=threshold] = 1

    # io.imsave(output_image, (saliency * 255).astype('uint8'))
    # imageio.imsave(output_image, saliency.astype('float32'))

    return saliency.astype('float32')


def mayo_wo_background(input_image):
    
    image = imageio.imread(input_image) #0-1
        
    #Add dimenstion
    image = np.expand_dims(image, axis=-1)
    image = np.concatenate([image, image, image], axis=-1)

    data = image * 255
    gray = data.astype('uint8')
    saliency = remove_background(gray)
    saliency_mean = np.mean(saliency, axis=-1)

    # io.imsave(output_image, (saliency * 255).astype('uint8'))
    # imageio.imsave('b.png', saliency.astype('float32'))
    # print(saliency.astype('float32'))
    return saliency_mean.astype('float32')


