import numpy as np
import imageio
import cv2
import os, random, shutil, time
from glob import glob
import argparse
import warnings

from utils import *
from abstract_shape import *
from save import save_trainset, save_testset
from data_split_pid import phantom_main
from mayo2coco import convert_mayo_to_coco
from make_data import generate_lesions, extract_bbox_poly

warnings.filterwarnings(action='ignore')



def make_phantom_trainset(args, data_option, trial=None):

    
    train_idx = {
        'ge':{
            'chest' : [0,300],
            'hn' : [33,79],
            'pelvis' : [44,233]
        },
        'siemens':{
            'chest' : [35,276],
            'hn' : [57,141],
            'pelvis' : [10,190]
        },
        'toshiba':{
            'chest' : [34,374],
            'hn' : [84,210],
            'pelvis' : [30,292]
        }
    }
    

    data_root = '../../Data/phantom_train_test_new/train/phantom/{}/{}'.format(args.company, args.anatomy)
    temp_folder_list = sorted(os.listdir(data_root))
    start_idx, end_idx = train_idx[args.company][args.anatomy]

    if args.company == 'cbct':
        folder_list = sorted(list(map(int, temp_folder_list)), reverse=True)
    else :
        folder_list = []
        for folder in temp_folder_list:
            if folder.startswith('level'):
                folder_list.append(folder)

    #Make new train directory
    dataset = 'phantom_{}_{}'.format(args.company, args.anatomy)

    if not os.path.isdir(os.path.join('../../data', dataset)):
        info_list = ['images', 'lesions', 'polygons', 'labels','setting', 'train', 'val']
        for info in info_list:
            os.makedirs(os.path.join(os.path.join('../../data', dataset), info))

    train_root = sorted(glob(os.path.join(data_root, folder_list[0], '*.tiff')))[start_idx:end_idx+1]
    for num in range(len(train_root)):

        blob_num = np.random.randint(2,6)
        radius_list = [random.choice(data_option['poss_radius']) for i in range(blob_num)]
        content_ratio_list =[random.choice(data_option['poss_content']) for i in range(blob_num)]
        shape_list = [random.choice(data_option['poss_shape']) for i in range(blob_num)]

        data_info = {'blob_num':blob_num, 'radius_list':radius_list,
                    'content_ratio_list':content_ratio_list, 'shape_list':shape_list}
        
        #Generate
        lesions, polygons = generate_lesions(train_root[num], blob_num, data_info)
        bbox_coord, poly_coord = extract_bbox_poly(polygons)

        img = imageio.imread(train_root[num]).copy() + lesions

        #Save image, label, polygon, setting
        bpsrc = (bbox_coord, poly_coord, shape_list, radius_list, content_ratio_list)
        
        if trial == None:
            filename = train_root[num].split('/')[-1]
        else :
            filename =  str(trial) + '_' + train_root[num].split('/')[-1]
        save_trainset(dataset, filename[:-5], img, lesions, bpsrc, multi_class=args.mc)

        print(num, '/', len(train_root))
    

    #Split train and valid data

    # dataset = 'phantom_{}_{}'.format(args.company, args.anatomy)

    # phantom_main(dataset)
    # convert_mayo_to_coco('val', dataset, multi_class=True)
    # print('val is done ...')
    # convert_mayo_to_coco('train', dataset, multi_class=True)
    # print('train is done ...')



def make_phantom_testset(args, data_option):

    test_idx = {
        'ge':{
            'chest' : [0,50],
            'hn' : [7,14],
            'pelvis' : [4,39]
        },
        'siemens':{
            'chest' : [7,38],
            'hn' : [11,24],
            'pelvis' : [3,32]
        },
        'toshiba':{
            'chest' : [7,64],
            'hn' : [15,36],
            'pelvis' : [6,49]
        }
    }

    
    data_root = '../../Data/phantom_train_test_new/test/phantom/{}/{}'.format(args.company, args.anatomy)
    temp_folder_list = sorted(os.listdir(data_root))
    start_idx, end_idx = test_idx[args.company][args.anatomy]

    start_idx +=1
    end_idx +=1

    if args.company == 'cbct':
        folder_list = sorted(list(map(int, temp_folder_list)), reverse=True)
    else :
        folder_list = []
        for folder in temp_folder_list:
            if folder.startswith('level'):
                folder_list.append(folder)
    

    #Make new directory
    dataset = 'phantom_{}_{}'.format(args.company, args.anatomy)
    fileroot= '../../data/{}/test'.format(dataset)

    if not os.path.isdir(fileroot):
        for folder in folder_list:
            os.makedirs(os.path.join(fileroot, folder))


    dose2path = dict()
    for dose in folder_list:
        dose2path[dose] = sorted(glob(os.path.join(data_root, dose, '*.tiff')))[start_idx:end_idx+1]

    test_root = sorted(glob(os.path.join(data_root, folder_list[0], '*.tiff')))[start_idx:end_idx+1]
    
    for num in range(len(test_root)):

        blob_num = np.random.randint(2,6)
        radius_list = [random.choice(data_option['poss_radius']) for i in range(blob_num)]
        content_ratio_list =[random.choice(data_option['poss_content']) for i in range(blob_num)]
        shape_list = [random.choice(data_option['poss_shape']) for i in range(blob_num)]

        data_info = {'blob_num':blob_num, 'radius_list':radius_list,
                    'content_ratio_list':content_ratio_list, 'shape_list':shape_list}
        
        #Generate
        lesions, polygons = generate_lesions(test_root[num], blob_num, data_info)
        bbox_coord, poly_coord = extract_bbox_poly(polygons)


        filename = test_root[num].split('/')[-1]

        for dose in folder_list:
            dose_test_path = dose2path[dose][num]
            test_img = imageio.imread(dose_test_path)
            
            new_test_img = test_img.copy()
            new_test_img += lesions
            
            target_img_path = os.path.join(fileroot, dose, filename)
            imageio.imwrite(target_img_path, new_test_img)

        #Save
        bpsrc = (bbox_coord, poly_coord, shape_list, radius_list, content_ratio_list)
        save_testset(filename[:-5], fileroot, new_test_img, lesions, bpsrc, multi_class=args.mc)
    
    
    #Convert mayo to coco format
    folder_name = fileroot[11:] +'/temp' #mayo_mmlab_random_21/test/000
    convert_mayo_to_coco('images', folder_name, test_option=2, multi_class=args.mc)

    #Copy json file
    json_path = '../../data/{}/mayo2coco_test.json'.format(folder_name)
    for folder in folder_list:
        json_name = folder+'.json'
        target_json_path = json_path.replace('temp', folder).replace('test.json', json_name)
        shutil.copy(json_path, target_json_path)




def make_phantom_test(args, data_option, fid):
    
    data_root = '../../Data/phantom_train_test_new/test/phantom/{}/{}'.format(args.company, args.anatomy)
    temp_folder_list = sorted(os.listdir(data_root))
    start_idx, end_idx = test_idx[args.company][args.anatomy]

    start_idx +=1
    end_idx +=1

    if args.company == 'cbct':
        folder_list = sorted(list(map(int, temp_folder_list)), reverse=True)
    else :
        folder_list = []
        for folder in temp_folder_list:
            if folder.startswith('level'):
                folder_list.append(folder)
    

    pimg_list = []
    for p in folder_list :
        test_path = sorted(glob(os.path.join(data_root, p, '*.tiff')))[fid]
        img = imageio.imread(test_path)
        pimg_list.append(img)
    
    file_item = test_path.split('/')[-1].split('_')[:2]
    file_item.append(test_path.split('/')[-1].split('_')[-1])
    filename = '_'.join(file_item)
    
    #Make new directory
    dataset = 'phantom_{}_{}'.format(args.company, args.anatomy)
    fileroot= '../../data/{}/testdata/{}'.format(dataset, filename[:-5])

    if not os.path.isdir(fileroot):
        for folder in folder_list:
            os.makedirs(os.path.join(fileroot, folder))
    
    for rp in range(args.rep):
    
        blob_num = np.random.randint(2,6)
        radius_list = [random.choice(data_option['poss_radius']) for i in range(blob_num)]
        content_ratio_list =[random.choice(data_option['poss_content']) for i in range(blob_num)]
        shape_list = [random.choice(data_option['poss_shape']) for i in range(blob_num)]

        data_info = {'blob_num':blob_num, 'radius_list':radius_list,
                    'content_ratio_list':content_ratio_list, 'shape_list':shape_list}
        
        #Generate
        lesions, polygons = generate_lesions(test_path, blob_num, data_info)
        bbox_coord, poly_coord = extract_bbox_poly(polygons)
        filename = '{0:04d}.tiff'.format(rp)


        for p, pimg in enumerate(pimg_list):
            new_pimg = pimg.copy()
            new_pimg += lesions
            
            target_img_path = os.path.join(fileroot, folder_list[p], filename)
            imageio.imwrite(target_img_path, new_pimg)

        #Save
        bpsrc = (bbox_coord, poly_coord, shape_list, radius_list, content_ratio_list)
        save_testset(filename[:-5], fileroot, new_pimg, lesions, bpsrc, multi_class=args.mc)
    
    
    #Convert mayo to coco format
    folder_name = fileroot[11:] +'/temp' #mayo_mmlab_random_21/test/000
    convert_mayo_to_coco('images', folder_name, test_option=2, multi_class=args.mc)

    #Copy json file
    json_path = '../../data/{}/mayo2coco_test.json'.format(folder_name)
    for folder in folder_list:
        json_name = folder+'.json'
        target_json_path = json_path.replace('temp', folder).replace('test.json', json_name)
        shutil.copy(json_path, target_json_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MIQA dataset')

    # parser.add_argument('--dataset', type=str, default='phantom_ge_chest', help='Dataset name')
    parser.add_argument('--mc', type=bool, default=True, help='Multi-classification?')
    parser.add_argument('--r', type=int, nargs='+', default=[7,8,9,10], help='Radius list')  
    parser.add_argument('--c', type=float, nargs='+', default=[0.06, 0.05, 0.04], help='Content(mean diff) list')
    parser.add_argument('--company', type=str, default='siemens', help='ge|siemens|toshiba|cbct')
    parser.add_argument('--anatomy', type=str, default='chest', help='chest|pelvis|hn|catphan')
    parser.add_argument('--rep', type=int, default='500', help='Repetition')
    #Test arguments
    # parser.add_argument('--pid', type=str, default='L067', help='L067|L291|L506')
    # parser.add_argument('--fid', type=int, default='100', help='File idx')
    
    args = parser.parse_args()


    # #If folder does not exist, make directories
    # if not os.path.isdir(root_dir):
    #     print('Make NEW directory')
    #     folder_list = ['lesions', 'polygons', 'labels','setting', 'catphan', 'chest', 'hn', 'pelvis']
    #     for folder in folder_list:
    #         os.makedirs(os.path.join(root_dir,folder))
    
    # # create level forders
    # for region in ['catphan', 'chest', 'hn', 'pelvis']:
    #     level_list = sorted(os.listdir(data_root+region))
    #     for level in level_list:
    #         os.makedirs(os.path.join(root_dir,region,level))

    #Set random number : shape type, size
    data_option={
            'poss_radius' : args.r,
            'poss_content' : args.c,
            'poss_shape':  [Triangle, Circle, Square, Star] if args.mc else [Circle]
    }
    
    print('*******************************')
    print('Radius : ', data_option['poss_radius'])
    print('Content : ', data_option['poss_content'])
    print('Multi-classification : ', args.mc)
    print('*******************************')


    #1.
    # for i in range(5):
    #     make_phantom_trainset(args, data_option,i+1)
    
    # dataset = 'phantom_{}_{}'.format(args.company, args.anatomy)
    
    # phantom_main(dataset)
    # convert_mayo_to_coco('val', dataset, multi_class=True)
    # print('val is done ...')
    # convert_mayo_to_coco('train', dataset, multi_class=True)
    # print('train is done ...')

    #2.
    # make_phantom_testset(args, data_option)
    

    3.
    test_idx = {
        'ge':{
            'chest' : [0,50],
            'hn' : [7,14],
            'pelvis' : [4,39]
        },
        'siemens':{
            'chest' : [7,38],
            'hn' : [11,24],
            'pelvis' : [3,32]
        },
        'toshiba':{
            'chest' : [7,64],
            'hn' : [15,36],
            'pelvis' : [6,49]
        }
    }

    start_idx, end_idx = test_idx[args.company][args.anatomy]

    for idx in range(start_idx+1, end_idx+1):
        make_phantom_test(args, data_option, idx)

    # fid1 = int((start_idx + end_idx)/4 * 1)
    # fid2 = int((start_idx + end_idx)/4 * 2)
    # fid3 = int((start_idx + end_idx)/4 * 3)

    # make_phantom_test(args, data_option, fid1)
    # make_phantom_test(args, data_option, fid2)
    # make_phantom_test(args, data_option, fid3)