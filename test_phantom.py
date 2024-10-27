from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from mmdet.datasets import build_dataset
import os, glob
import csv, json
import numpy as np
import warnings
import argparse
import torch

warnings.filterwarnings(action='ignore')



def refine_prediction(outputs, image_num):

    pred_list = []
    for c, output in enumerate(outputs[0]):
        
        for pred_bbox in output:
            xmin, ymin, xmax, ymax, conf = pred_bbox
            pred = [image_num, c, conf, (xmin, ymin, xmax, ymax)]       
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


def test_mayo_datum(args, fid):
    
    
    
    model_name = args.modelname
    date = args.date
    test_folder = '{}/{}_{}'.format(args.test_folder, args.company, args.anatomy)

    folder_list = sorted(os.listdir(test_folder))
    
    data_root =  '../../data/{}/'.format(args.dataset)

    #Open csv to record results
    csv_path = '../../data/results/result_mAP_{}_{}_{}.csv'.format(args.dataset, args.pid, fid)

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
        model = init_detector(config_file, checkpoint_file, device='cuda:2')
        
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



def test_mayo_data(args, c=None, r=None, fid=None, datum=False):

    #Define variables
    model_name = args.modelname
    date = args.date

    

    if datum :
        test_folder = 'testdata/{}_{}_{}'.format(args.company, args.anatomy, fid)
        csv_path = '../../data/results/result_mAP_{}_{}_{}_{}.csv'.format(args.company, args.anatomy, args.date, fid)
    else :
        test_folder = 'test'
        csv_path = '../../data/results/result_mAP_{}_{}_{}.csv'.format(args.company, args.anatomy, args.date)
    
    data_root =  '../../data/phantom_{}_{}/'.format(args.company, args.anatomy)
    folder_list = sorted(os.listdir(os.path.join(data_root,test_folder)))    

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
        model = init_detector(config_file, checkpoint_file, device='cuda:2')
        
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

    parser.add_argument('--test_folder', type=str, default='test', help='test')
    parser.add_argument('--date', type=str, default='0805_5', required=True, help='Date when you trained models ex)0610')     
    parser.add_argument('--pid', type=str, default='L506', help='L067|L291|L506')  
    parser.add_argument('--epoch', type=int, default='10', help='10')
    parser.add_argument('--cocoeval', type=bool, default=True, help='Test with cocoEval')
    
    parser.add_argument('--company', type=str, default='ge', help='ge|siemens|toshiba|cbct')
    parser.add_argument('--anatomy', type=str, default='chest', help='chest|pelvis|hn|catphan')
    parser.add_argument('--r', type=int, nargs='+', default=[7,8,9,10], help='Radius list')  
    parser.add_argument('--c', type=float, nargs='+', default=[0.1, 0.09, 0.08], help='Content(mean diff) list')
    args = parser.parse_args()

    # for cont in args.c:
    #     for rad in args.r:
    #         print(cont, rad, ' start ...')
    #         test_mayo_data(args, cont, rad)

    # fid_list = [154.0, 173.0, 229.0, 249.0, 275.0, 311.0, 332.0, 376.0]
    # for fid in fid_list:
    #     print(fid, ' start ...')
    #     test_mayo_datum(args, fid)


    # test_mayo_data(args)
    
    #GE
    # test_mayo_data(args, fid='084', datum=True)
    # test_mayo_data(args, fid='175', datum=True)
    # test_mayo_data(args, fid='259', datum=True)
    # idx_list = ['147', '154', '161', '168', '175', '182', '189', '196', '203', '210'] 
    # idx_list = ['217', '224', '231', '238', '245', '252', '259', '266', '273', '280']
    # idx_list = ['287', '294', '301', '308', '315', '322', '329', '336', '343', '350', '357','364', '371']

    # for idx in idx_list:
    #     test_mayo_data(args, fid=idx, datum=True)

    #Siemens
    # test_mayo_data(args, fid='077', datum=True)
    # test_mayo_data(args, fid='154', datum=True)
    # test_mayo_data(args, fid='231', datum=True)

    #Toshiba
    # test_mayo_data(args, fid='371', datum=True)
    # test_mayo_data(args, fid='245', datum=True)
    # test_mayo_data(args, fid='119', datum=True)


    
if __name__ == '__main__':
    test_main()
