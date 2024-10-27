import os.path as osp
from glob import glob
import imageio
import mmcv
import torch
import numpy as np
import argparse

from ast import literal_eval
import os
import json
import re
import shutil

"""
make json file (coco format)
"""


def convert_mayo_to_coco(data_option, folder_name, data_root, test_option=0, multi_class=True):

    annotations = []
    images = []

    
    train_root = osp.join(data_root, folder_name, data_option, '*.tiff')
    train_paths = sorted(glob(train_root))
 
    count=0
    filenum = -1

    for idx, img_path in enumerate(train_paths):

        filenum +=1

        #Image
        # print(img_path)
        height, width = imageio.imread(img_path).shape
        filename = img_path.split('/')[-1]
        
        images.append(dict(
            id=filenum,
            file_name=filename,
            height=height,
            width=width))


        #Label
        label_path = img_path.replace(data_option, 'labels').replace('tiff', 'txt')
        # /home/eunbyeol/data/phantom_ge_0.07_test/labels/0_ge_chest_level1_100_000.txt
        bbox_list = np.loadtxt(label_path).reshape(-1, 5)
        box_num = bbox_list.shape[0]


        #Polygons

        poly_path = img_path.replace(data_option, 'polygons').replace('tiff', 'txt')
        # poly_list = np.loadtxt(poly_path)
        
        f = open(poly_path, 'r')
        raw_data = f.read()
        polygons = raw_data.split('\n')[:-1]
        poly_num = len(polygons)
        
                
        f.close()

        #Count obj
        obj_count = 0

        bboxes = []
        labels = []
        masks = []

        for num, bbox_coord in enumerate(bbox_list):
            if len(bbox_list) == 0:
                continue

            if box_num!= poly_num:
                raise AttributeError('Cannot happen')

            obj, x_min, x_max, y_min, y_max = bbox_coord
            
            r = re.findall('\([^)]*\)',polygons[num])
            pp = list(map(literal_eval, r))
            
            py = [p[0] for p in pp]
            px = [p[1] for p in pp]

            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

                
            data_anno = dict(
                image_id=filenum,
                id=count,
                category_id=int(obj),
                bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                area=(x_max - x_min) * (y_max - y_min),
                segmentation=[poly],
                iscrowd=0)
                
            annotations.append(data_anno)
            obj_count += 1
            count +=1

        # print(idx, '/', len(train_paths))

    if multi_class : 
        coco_format_json = dict(
            images=images,
            annotations=annotations,
            categories=[{'id':1, 'name': 'triangle'},
                        {'id':2, 'name': 'circle'},
                        {'id':3, 'name': 'square'},
                        {'id':4, 'name': 'star'}])
    else : #binary class
        coco_format_json = dict(
            images=images,
            annotations=annotations,
            categories=[{'id':1, 'name':'circle'}])
    
    if test_option == 1:
        g_level = data_option.split('/')[-1]
        output_name = osp.join(data_root, folder_name, data_option, 'mayo2coco_{}.json'.format(g_level)) 
    elif test_option == 2:
        output_name = osp.join(data_root, folder_name, 'mayo2coco_test.json')
    else:
        output_name = osp.join(data_root, folder_name, data_option, 'mayo2coco_{}.json'.format(data_option))    
    mmcv.dump(coco_format_json, output_name, indent=4)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MIQA dataset')

    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--root', type=str, required=True, help='Parent directory')
    args = parser.parse_args()

    folder_name = args.dataset
    data_root = args.root

    # folder_name= 'phantom_siemens_0.07_test'
    # convert_mayo_to_coco('chest/level4_022', folder_name, multi_class=True, test_option=1)

    convert_mayo_to_coco('test', folder_name, data_root, multi_class=True)
    print('test is done ...')
    convert_mayo_to_coco('val', folder_name, data_root, multi_class=True)
    print('val is done ...')
    convert_mayo_to_coco('train', folder_name, data_root, multi_class=True)
    print('train is done ...')
    # convert_mayo_to_coco('val', folder_name, multi_class=False)
    # print('val is done ...')
    # convert_mayo_to_coco('train', folder_name, multi_class=False)
    # print('train is done ...')
    
    


