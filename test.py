"""
python test.py --dataset mayo_5 --pid L067 --gid 0 --date quarter
python test_phantom.py --0810_1
"""

from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from mmdet.datasets import build_dataset
import os, glob
import csv, json
import numpy as np
import warnings
import argparse
import torch
import pandas as pd

warnings.filterwarnings(action='ignore')


def refine_prediction(outputs, image_num):

    pred_list = []
    for c, output in enumerate(outputs[0]):
        
        for pred_bbox in output:
            xmin, ymin, xmax, ymax, conf = pred_bbox
            pred = [image_num, c, conf, (xmin, ymin, xmax, ymax)]       
            # pred = [image_num, c, conf, (xmin, ymin, xmax, ymax)]       
            pred_list.append(pred)
        
    return pred_list


def json2bbox(json_path):

    with open(json_path, "r") as j:
        data = json.load(j)
        data = data['annotations']

    gt_list = []
    for info in data:
        image_id = info['image_id']
        x,y,h,w = info['bbox']
        c = int(info['category_id'])-1
        # c = int(info['category_id'])
        gt_list.append([image_id, c, 1, (x, y, x+w, y+h)])
        
    return gt_list



def json2coco(data_root, json_file, test_folder, noise_folder):
    
    """
    json -> coco format to use cocoEval
    """

    img_norm_cfg = dict(
        mean=[0, 0, 0], std=[1, 1, 1], to_rgb=False)

    test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(512, 512),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='Pad', size_divisor=32),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ])
    ]

    test=dict(
        type='CocoDataset',
        ann_file=(data_root + test_folder +'/' +noise_folder + '/' + json_file),
        img_prefix=(data_root + test_folder +'/' +noise_folder +'/'),
        pipeline=test_pipeline)

    datasets = build_dataset(test)

    return datasets


def test_mayo_datum(args, fid, c=None, r=None, sep=False, exp_num=None):
    
    device = torch.device('cuda:{}'.format(args.gid))
    
    model_name = args.modelname
    date = args.date
    

    if args.noise == 'p': #Poisson noise
        # folder_list = ['full', 'full_0.5', 'quarter', 'quarter_0.5', 'quarter_1.0','quarter_1.5', 'quarter_2.0', 'quarter_2.5', 'quarter_3.0', 'quarter_3.5']
        folder_list = ['full', 'full_0.5', 'quarter', 'quarter_0.5', 'quarter_1.0','quarter_1.5', 'quarter_2.0']
        data_root =  os.path.join(args.root, '{}/'.format(args.dataset))

    #Open csv to record results
    if sep:
        csv_path = os.path.join(args.root, 'results/result_mAP_{}_{}_{}_{}_{}.csv'.format(args.dataset, args.pid, fid, r, c))
        test_folder = '{}/{}_{}_{}_{}'.format(args.test_folder, args.pid, fid, r, c)
        # print(csv_path)
        # exit()
    elif exp_num != None: 
        csv_path = os.path.join(args.root, 'results/result_mAP_{}_{}_{}_{}.csv'.format(args.dataset, args.pid, fid, exp_num))
        test_folder = '{}/{}_{}/{}'.format(args.test_folder, args.pid, fid, exp_num)
    else :
        csv_path = os.path.join(args.root, 'results/result_mAP_{}_{}_{}.csv'.format(args.dataset, args.pid, fid))
        test_folder = '{}/{}_{}'.format(args.test_folder, args.pid, fid)

    # if not os.path.isfile(csv_path):   
    #     with open(csv_path, "w", newline='') as csvfile:
    #         wr = csv.writer(csvfile, dialect="excel")
    #         title = ['noise_type','epoch','bbox_mAP', 'bbox_mAP_50', 'bbox_mAP_75']
    #         wr.writerow(title)

    """
    Test
    """
    result_list =[]
    config_file = './work_dirs/default_runtime_{}_{}/default_runtime_{}_{}.py'.format(model_name, date, model_name, date)
    checkpoint_file = './work_dirs/default_runtime_{}_{}/epoch_{}.pth'.format(model_name, date, args.epoch)

    for noise in folder_list:
        print('>>>> ', args.epoch, noise, ' start!')
        
        #Read image & json(label)
        img_path = os.path.join(data_root, test_folder, noise) +'/*.tiff'
        img_list = sorted(glob.glob(img_path))

        json_file = 'mayo2coco_{}.json'.format(noise)
        json_path = data_root + test_folder +'/' + noise + '/' + json_file

        if args.cocoeval :
            datasets = json2coco(data_root, json_file, test_folder, noise)

        #Load model
        model = init_detector(config_file, checkpoint_file, device=device)
        
        #Test model
        outputs = []   
        
        if args.cocoeval :
            for i, img in enumerate(img_list):
                output = inference_detector(model, img)
                outputs.append(output)

            #Calculate mAP
            result = datasets.evaluate(outputs)
            result_list.append(result['bbox_mAP'])
    
        # data_csv = [noise, args.epoch, result['bbox_mAP'],result['bbox_mAP_50'],result['bbox_mAP_75'] ]
        # with open(csv_path, "a", newline='') as csvfile :
        #     wr = csv.writer(csvfile, dialect='excel')
        #     wr.writerow(data_csv)

    if args.cocoeval :
        for i, result in enumerate(result_list):
            print(result, end=',')

    return result_list



def test_mayo_data(args, c=None, r=None):
    
    #Define variables

    device = torch.device('cuda:{}'.format(args.gid))

    model_name = args.modelname
    date = args.date

    if c==None and r==None:
        test_folder = args.test_folder
        csv_path = os.path.join(args.root, 'results/result_mAP_{}_{}_{}.csv'.format(args.dataset, args.modelname, args.pid))
        
    else :
        test_folder = '{}/{}_{}_{}'.format(args.test_folder, args.pid, c, r)
        csv_path = os.path.join(args.root, 'results/result_mAP_{}_{}_{}_{}.csv'.format(args.dataset, args.pid, c, r))

    if args.noise == 'p': #Poisson noise
        # folder_list = ['full', 'full_0.5', 'quarter', 'quarter_0.5', 'quarter_1.0','quarter_1.5', 'quarter_2.0', 'quarter_2.5', 'quarter_3.0', 'quarter_3.5']
        folder_list = ['full', 'full_0.5', 'quarter', 'quarter_0.5', 'quarter_1.0','quarter_1.5', 'quarter_2.0']
        data_root =  os.path.join(args.root, '{}/'.format(args.dataset))
    

    if not os.path.isfile(csv_path):   
        with open(csv_path, "w", newline='') as csvfile:
            wr = csv.writer(csvfile, dialect="excel")
            title = ['noise_type','epoch','bbox_mAP', 'bbox_mAP_50', 'bbox_mAP_75']
            wr.writerow(title)

    """
    Test
    """
    result_list =[]
    config_file = './work_dirs/default_runtime_{}_{}/default_runtime_{}_{}.py'.format(model_name, date, model_name, date)
    checkpoint_file = './work_dirs/default_runtime_{}_{}/epoch_{}.pth'.format(model_name, date, args.epoch)

    for noise in folder_list:
        print('>>>> ', args.epoch, noise, ' start!')
        
        #Read image           
        img_path = os.path.join(data_root, test_folder, noise) +'/*.tiff'
        img_list = sorted(glob.glob(img_path))

        #Read json(label)
        json_file = 'mayo2coco_{}.json'.format(noise)
        json_path = data_root + test_folder +'/' + noise + '/' + json_file
        if args.cocoeval :
            datasets = json2coco(data_root, json_file, test_folder, noise)

        #Load model
        model = init_detector(config_file, checkpoint_file, device=device)
        
        #Test model
        outputs = []   
        
        if args.cocoeval :
            for i, img in enumerate(img_list):
                output = inference_detector(model, img)
                outputs.append(output)

            #Calculate mAP
            result = datasets.evaluate(outputs)
            result_list.append([result['bbox_mAP'],result['bbox_mAP_50'],result['bbox_mAP_75']])
    
        data_csv = [noise, args.epoch, result['bbox_mAP'],result['bbox_mAP_50'],result['bbox_mAP_75'] ]
        with open(csv_path, "a", newline='') as csvfile :
            wr = csv.writer(csvfile, dialect='excel')
            wr.writerow(data_csv)

    if args.cocoeval :
        for i, result in enumerate(result_list):
            print(result[0], end=',')


def test_main():
    """The main function."""
    
    parser = argparse.ArgumentParser(description='Test BIQA')
    
    parser.add_argument('--modelname', type=str, default='cascadercnn', help='cascadercnn|fasterrcnn')
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--root', type=str, required=True, help='Data root')
    parser.add_argument('--test_folder', type=str, default='ptest', help='ptest')
    parser.add_argument('--date', type=str, default='0805_5', required=True, help='Date when you trained models ex)0610')     
    parser.add_argument('--noise', type=str, default='p', help='p=poisson, g=gaussian')  
    parser.add_argument('--pid', type=str, default='L506', help='L067|L291|L506')  
    parser.add_argument('--epoch', type=int, default='10', help='10,20')
    parser.add_argument('--cocoeval', type=bool, default=True, help='Test with cocoEval')
    
    
    parser.add_argument('--r', type=int, nargs='+', default=[7,8,9,10], help='Radius list')  
    parser.add_argument('--c', type=float, nargs='+', default=[0.1, 0.09, 0.08], help='Content(mean diff) list')
    parser.add_argument('--gid', type=int, default=0, help='Gpu ID')
    args = parser.parse_args()


    # 1. If you made the dataset using make_ptestset(args, data_option)
    # test_mayo_data(args)
    """
    python test.py --date 0805_5 --dataset mayo_5 --test_folder ptest_L067 --pid L067
    python test.py --date circle --dataset mayo_5_circle --test_folder ptest_L506 --pid L506 --gid 1
    """

    #2. If you made the dataset using make_ptestset_sep(args, data_option)
    # for cont in args.c:
    #     for rad in args.r:
    #         print(cont, rad, ' start ...')
    #         test_mayo_data(args, cont, rad)


    #3. If tou made the dataset using make_ptest_all(args, data_option)?
    # if args.pid == 'L506':
    #     fid_list = [140.0, 170.0, 199.0, 208.0, 214.0, 245.0, 281.0, 286.0, 288.0, 300.0, 306.0, 310.0, 345.0, 347.0, 372.0] 
    # elif args.pid == 'L067':
    #     fid_list = [153.0, 176.0, 183.0, 199.0, 232.0, 248.0, 262.0, 269.0, 299.0, 302.0, 332.0, 344.0, 361.0, 377.0, 418.0] 
    # else:
    #     raise AssertionError("Wrong input pid")
    # for fid in fid_list:
    #     print(fid, ' start ...')
    #     test_mayo_datum(args, fid)


    #4. If you made the dataset using make_ptest_sep(args, data_option)
    # if args.pid == 'L506':
    #     fid_temp = [140.0, 170.0, 199.0, 208.0, 214.0, 245.0, 281.0, 286.0, 288.0, 300.0, 306.0, 310.0, 345.0, 347.0, 372.0] 
    #     fid_temp = [347.0, 372.0]
    # elif args.pid == 'L067':
    #     fid_temp = [153.0, 176.0, 183.0, 199.0, 232.0, 248.0, 262.0, 269.0, 299.0, 302.0, 332.0, 344.0, 361.0, 377.0, 418.0] 
    #     fid_temp = [377.0, 418.0]
    # else:
    #     raise AssertionError("Wrong input pid")

    # fid_list = [176.0]
    # for fid in fid_list:
    #     for cont in args.c:
    #         for rad in args.r:
    #             print(cont, rad, ' start ...')
    #             test_mayo_datum(args, fid, cont, rad, sep=True)
    

    # 5. If you made the dataset using make_ptest_all(args, data_option, exp=True)
    if args.pid == 'L506':
        fid_list = [140.0, 170.0, 199.0, 208.0, 214.0, 245.0, 281.0, 286.0, 288.0, 300.0, 306.0, 310.0, 345.0, 347.0, 372.0] 
        
    elif args.pid == 'L067':
        fid_list = [153.0, 176.0, 183.0, 199.0, 232.0, 248.0, 262.0, 269.0, 299.0, 302.0, 332.0, 344.0, 361.0, 377.0, 418.0] 
        fid_list = [153.0]
    else:
        raise AssertionError("Wrong input pid")
    
    for fid in fid_list:
        print(fid, ' start ...')
        result = []
        for t in range(5):
            result.append(test_mayo_datum(args, fid, exp_num=t))        
        
        # lv = ['full', 'full_0.5', 'quarter', 'quarter_0.5', 'quarter_1.0','quarter_1.5', 'quarter_2.0', 'quarter_2.5', 'quarter_3.0', 'quarter_3.5']
        lv = ['full', 'full_0.5', 'quarter', 'quarter_0.5', 'quarter_1.0','quarter_1.5', 'quarter_2.0']
        result_np = np.array(result)
        df = pd.DataFrame(result_np.T,  index=lv)
        cum_df = df.cumsum(axis=1)
        mean_df = cum_df.copy()
        for i in range(5):
            mean_df[cum_df.columns[i]] /= (i+1)
        total_df = pd.concat([df, cum_df, mean_df], axis=0)

        csv_path = os.path.join(args.root, 'results/result_quarter_mAP_{}_{}_{}_mean.csv'.format(args.dataset, args.pid, fid))
        total_df.to_csv(csv_path)

if __name__ == '__main__':
    test_main()
