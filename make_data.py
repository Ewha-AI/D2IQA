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
from mayo2coco import convert_mayo_to_coco

warnings.filterwarnings(action='ignore')


def generate_lesions(img_path, blob_num, data_info):

    img = imageio.imread(img_path)

    #Find the appropriate location to insert the lesion.
    #And extract center coordinates based on saliency map.
    border = find_app_location(img_path)
    centers = extract_centers(border, data_info['blob_num'], data_info['radius_list'])

    #Initialize
    img_copy = img.copy()
    lesions = np.zeros(img_copy.shape, dtype=img_copy.dtype)
    # im = np.zeros((512,512,3), dtype=np.uint8)
    polygons = []
    
    for i in range(data_info['blob_num']):

        shape = data_info['shape_list'][i]
        radius = data_info['radius_list'][i]
        content_ratio = data_info['content_ratio_list'][i]
        yoff, xoff = centers[i]

        #Generate shapes
        polygon = shape(xoff, yoff, radius).get_shape_rotated_coordinates()
        pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
        
        im = np.zeros((512,512,3), dtype=np.uint8)

        if shape == Circle :
            im = cv2.circle(im, (xoff, yoff), radius, (255,0,0), -1)
        elif shape == Star :
            im = cv2.fillPoly(im, [pts], (255,0,0))
            im = cv2.fillConvexPoly(im, pts[3:-1],(255,0,0))    
        else :
            im = cv2.fillConvexPoly(im, pts,(255,0,0))     

        im_np = np.sum(im, axis=-1)
        annot = np.where(im_np==255)
        x, y = annot[1],annot[0]

        #Generate lesion maps
        sign = 1 if random.random() <0.5 else -1
        lesion_map = fill_diff_cnr(x, y, img, sign, content_ratio)
        
        lesions += lesion_map
        # print('--------------')
        # print(lesions.max())
        # print(lesions.min())
        # print('--------------')
        polygons.append(polygon)

    
    return lesions, polygons


def extract_bbox_poly(polygons):

    bbox_coord, poly_coord, = [], []

    for b in range(len(polygons)):
        
        r_coordinates = []
        polygon = polygons[b]
                            
        for k in range(len(polygon)):

            x,y = polygon[k]
            coord = (y,x)
            r_coordinates.append(coord)
            
        ymin = int(np.min(r_coordinates,axis=0)[0] -2)
        xmin = int(np.min(r_coordinates,axis=0)[1] -2)
        ymax = int(np.max(r_coordinates, axis=0)[0] +2)
        xmax = int(np.max(r_coordinates, axis=0)[1] +2)

        bbox_coord.append((xmin, xmax, ymin, ymax))
        poly_coord.append(r_coordinates)

        # low[ymin:ymax,xmin] = 1
        # low[ymin:ymax,xmax] = 1
        # low[ymin,xmin:xmax] = 1
        # low[ymax,xmin:xmax] = 1

    return bbox_coord, poly_coord


def make_full_trainset(args, data_option):

    # hps = sorted(glob('/home/eunbyeol/Data/Mayo-Clinic/tiff/float32/full_1mm/*/*.tiff'))
    hps = sorted(glob(os.path.join(args.root, 'Mayo-Clinic/tiff/float32/full_1mm/*/*.tiff')))

    for img_num in range(len(hps)):

        blob_num = np.random.randint(2,6)
        radius_list = [random.choice(data_option['poss_radius']) for i in range(blob_num)]
        content_ratio_list =[random.choice(data_option['poss_content']) for i in range(blob_num)]
        shape_list = [random.choice(data_option['poss_shape']) for i in range(blob_num)]

        data_info = {'blob_num':blob_num, 'radius_list':radius_list,
                    'content_ratio_list':content_ratio_list, 'shape_list':shape_list}

        # print(" >>> generate lesions : ", blob_num, radius_list, content_ratio_list)
        
        #Generate
        lesions, polygons = generate_lesions(hps[img_num], blob_num, data_info)
        bbox_coord, poly_coord = extract_bbox_poly(polygons)

        high = imageio.imread(hps[img_num]).copy() + lesions
        
        #Save image, label, polygon, setting
        bpsrc = (bbox_coord, poly_coord, shape_list, radius_list, content_ratio_list)
        h_filename = hps[img_num].split('float32')[-1].replace('/', '_')[1:][:-5]
        save_trainset(args.dataset, args.root, h_filename, high, lesions, bpsrc, multi_class=args.mc)
        
        print(img_num, '/', len(hps))

def make_trainset(args, data_option):

    hps = sorted(glob(os.path.join(args.root, 'Mayo-Clinic/tiff/float32/full_1mm/*/*.tiff')))
    lps = sorted(glob(os.path.join(args.root, 'Mayo-Clinic/tiff/float32/quarter_1mm/*/*.tiff')))
    
    for img_num in range(len(lps)):

        blob_num = np.random.randint(2,6)
        radius_list = [random.choice(data_option['poss_radius']) for i in range(blob_num)]
        content_ratio_list =[random.choice(data_option['poss_content']) for i in range(blob_num)]
        shape_list = [random.choice(data_option['poss_shape']) for i in range(blob_num)]

        data_info = {'blob_num':blob_num, 'radius_list':radius_list,
                    'content_ratio_list':content_ratio_list, 'shape_list':shape_list}

        # print(" >>> generate lesions : ", blob_num, radius_list, content_ratio_list)
        
        #Generate
        lesions, polygons = generate_lesions(lps[img_num], blob_num, data_info)
        bbox_coord, poly_coord = extract_bbox_poly(polygons)

        high = imageio.imread(hps[img_num]).copy() + lesions
        low = imageio.imread(lps[img_num]).copy() + lesions

        #Save image, label, polygon, setting
        bpsrc = (bbox_coord, poly_coord, shape_list, radius_list, content_ratio_list)

        h_filename = hps[img_num].split('float32')[-1].replace('/', '_')[1:][:-5]
        l_filename = lps[img_num].split('float32')[-1].replace('/', '_')[1:][:-5]
        
        save_trainset(args.dataset, args.root, h_filename, high, lesions, bpsrc, multi_class=args.mc)
        save_trainset(args.dataset, args.root, l_filename, low, lesions, bpsrc, multi_class=args.mc)
        
        print(img_num, '/', len(hps))


def make_ptestset(args, data_option, fileroot_temp=None):

    pnoise_root = os.path.join(args.root, 'nimg/{}'.format(args.pid))
    folder_list = sorted(os.listdir(pnoise_root))

    #Make new directory
    if fileroot_temp == None:
        fileroot = os.path.join(args.root, '{}/ptest_{}'.format(args.dataset, args.pid))
    else : 
        fileroot = fileroot_temp

    if not os.path.isdir(fileroot):
        for folder in folder_list:
            os.makedirs(os.path.join(fileroot, folder))

    img_num = len(glob(os.path.join(pnoise_root, 'full', '*.tiff')))
    test_root = sorted(glob((os.path.join(args.root, 'nimg/{}/{}/*.tiff'.format(args.pid, 'full')))))

    for num in range(img_num):

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

        for p in folder_list:

            ptest_path = sorted(glob((os.path.join(args.root, 'nimg/{}/{}/*.tiff'.format(args.pid, p)))))[num]
            pimg = imageio.imread(ptest_path)
            
            new_pimg = pimg.copy()
            new_pimg += lesions
            
            target_img_path = os.path.join(fileroot, p, filename)
            imageio.imwrite(target_img_path, new_pimg)

        #Save          
        bpsrc = (bbox_coord, poly_coord, shape_list, radius_list, content_ratio_list)
        save_testset(filename[:-5], fileroot, new_pimg, lesions, bpsrc, multi_class=args.mc)

    #Convert mayo to coco format
    # folder_name = os.path.join(fileroot.split('/')[-2], fileroot.split('/')[-1], 'temp') #mayo_mmlab_random_21/test/000
    folder_name = os.path.join(fileroot.split(args.root)[1], 'temp')
    convert_mayo_to_coco('images', folder_name, args.root, test_option=2, multi_class=args.mc)

    #Copy json file
    json_path = os.path.join(args.root, '{}/mayo2coco_test.json'.format(folder_name))
    for folder in folder_list:
        json_name = folder+'.json'
        target_json_path = json_path.replace('temp', folder).replace('test.json', json_name)
        shutil.copy(json_path, target_json_path)


def make_ptestset_sep(args, data_option):

    """test dataset seperately"""

    r_list, c_list = data_option['poss_radius'], data_option['poss_content']
    pnoise_root = os.path.join(args.root, 'nimg/{}'.format(args.pid))
    folder_list = sorted(os.listdir(pnoise_root))

    for c in c_list:
        for r in r_list:

            fileroot = os.path.join(args.root, '{}/ptest/{}_{}_{}'.format(args.dataset, args.pid, c, r))

            data_option_temp = {
                'poss_radius' : [r],
                'poss_content' : [c],
                'poss_shape' : data_option['poss_shape']
            }

            print('*******************************')
            print('Radius : ', data_option_temp['poss_radius'])
            print('Content : ', data_option_temp['poss_content'])
            print('Multi-classification : ', args.mc)
            print('*******************************')

            start = time.time()
            make_ptestset(args, data_option_temp, fileroot)
            print('time :', time.time()-start)

    

def make_ptest(args, data_option, exp_num=None, fid=100, sep=False):

    pnoise_root = os.path.join(args.root, 'nimg/{}'.format(args.pid))
    folder_list = sorted(os.listdir(pnoise_root)) # nosie level
    # folder_list = ['full', 'full_0.5', 'quarter', 'quarter_0.5', 'quarter_1.0', 'quarter_1.5', 'quarter_2.0']

    pimg_list = []
    for p in folder_list :
        test_path = sorted(glob(os.path.join(args.root, 'nimg/{}/{}/*.tiff'.format(args.pid, p))))[fid]
        img = imageio.imread(test_path)
        pimg_list.append(img)

    filename = test_path.split('/')[-1]

    #Make new directory
    if sep:
        r = data_option['poss_radius']
        c = data_option['poss_content']

        if len(r) != 1 or len(c) !=1 :
            raise AssertionError('Wring input size')
        else :
            fileroot = os.path.join(args.root, '{}/ptest/{}_{}_{}_{}'.format(args.dataset, args.pid, filename[:-5], r[0], c[0]))

    elif exp_num != None: #rep and mean
        fileroot = os.path.join(args.root, '{}/ptest/{}_{}/{}'.format(args.dataset, args.pid, filename[:-5], exp_num))
    else :
        fileroot = os.path.join(args.root, '{}/ptest/{}_{}'.format(args.dataset, args.pid, filename[:-5]))
    

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

        bpsrc = (bbox_coord, poly_coord, shape_list, radius_list, content_ratio_list)
        save_testset(filename[:-5], fileroot, new_pimg, lesions, bpsrc, multi_class=args.mc)

    #Convert mayo to coco format
    # folder_name = fileroot[11:] +'/temp' #mayo_mmlab_random_21/test/000
    folder_name = os.path.join(fileroot.split(args.root)[1], 'temp')
    convert_mayo_to_coco('images', folder_name, args.root, test_option=2, multi_class=args.mc)

    #Copy json file
    json_path = os.path.join(args.root, '{}/mayo2coco_test.json'.format(folder_name))
    for folder in folder_list:
        json_name = folder+'.json'
        target_json_path = json_path.replace('temp', folder).replace('test.json', json_name)
        shutil.copy(json_path, target_json_path)


def make_ptest_all(args, data_option, exp=False):

    
    if args.pid == 'L506':
        fid_temp = [140.0, 170.0, 199.0, 208.0, 214.0, 245.0, 281.0, 286.0, 288.0, 300.0, 306.0, 310.0, 345.0, 347.0, 372.0] 
        # fid_temp = [208.0, 347.0, 300.0, 245.0, 288.0]

    elif args.pid == 'L067':
        fid_temp = [153.0, 176.0, 183.0, 199.0, 232.0, 248.0, 262.0, 269.0, 299.0, 302.0, 332.0, 344.0, 361.0, 377.0, 418.0] 
        # fid_temp = [248.0, 332.0, 176.0, 361.0, 199.0]
        
    else:
        raise AssertionError("Wrong input pid")

    fid_list = [int(i-120) for i in fid_temp]
    
    for fid in fid_list:
        print(fid, 'start...')
        if exp:
            for t in range(5):
                make_ptest(args, data_option, exp_num=t, fid=fid)
        else :
            make_ptest(args, data_option, fid)

    print(fid_list, 'is done...')



def make_ptest_sep(args, data_option):

    r_list, c_list = data_option['poss_radius'], data_option['poss_content']
    pnoise_root = os.path.join(args.root, 'nimg/{}'.format(args.pid))
    folder_list = sorted(os.listdir(pnoise_root))

    if args.pid == 'L506':
        fid_temp = [140.0, 170.0, 199.0, 208.0, 214.0, 245.0, 281.0, 286.0, 288.0, 300.0, 306.0, 310.0, 345.0, 347.0, 372.0] 

    elif args.pid == 'L067':
        fid_temp = [153.0, 176.0, 183.0, 199.0, 232.0, 248.0, 262.0, 269.0, 299.0, 302.0, 332.0, 344.0, 361.0, 377.0, 418.0] 
        
    else:
        raise AssertionError("Wrong input pid")

    fid_list = [int(i-120) for i in fid_temp] # FIXME
        
    for fid in fid_list:
        for c in c_list:
            for r in r_list:

                # fileroot = '../../data/{}/ptest/{}_{}_{}'.format(args.dataset, args.pid, c, r)

                data_option_temp = {
                    'poss_radius' : [r],
                    'poss_content' : [c],
                    'poss_shape' : data_option['poss_shape']
                }

                print('*******************************')
                print('Radius : ', data_option_temp['poss_radius'])
                print('Content : ', data_option_temp['poss_content'])
                print('Multi-classification : ', args.mc)
                print('*******************************')

                start = time.time()
                make_ptest(args, data_option_temp, fid=fid, sep=True)
                print('time :', time.time()-start)


        



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MIQA dataset')

    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--root', type=str, required=True, help='Data root')
    parser.add_argument('--mc', type=bool, default=False, help='Multi-classification?')
    parser.add_argument('--r', type=int, nargs='+', default=[7,8,9,10], help='Radius list')  
    parser.add_argument('--c', type=float, nargs='+', default=[0.1, 0.09, 0.08], help='Content(mean diff) list')
    #Test arguments
    parser.add_argument('--pid', type=str, default='L067', help='L067|L291|L506')
    parser.add_argument('--fid', type=int, default='100', help='File idx')
    parser.add_argument('--rep', type=int, default='100', help='Repetition')
    args = parser.parse_args()


    if not os.path.isdir(os.path.join('../../data', args.dataset)):
        print('Make NEW directory')

        folder_list = ['images', 'lesions', 'polygons', 'labels','setting', 'train', 'val', 'test']
        for folder in folder_list:
            os.makedirs(os.path.join(os.path.join('../../data', args.dataset), folder))

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

    # make trainset
    # make_trainset(args, data_option)
    # make_full_trainset(args, data_option)

    # make testset per patient
    # make_ptestset(args, data_option)
    # make_ptestset_sep(args, data_option)

    # make testset per slice image
    make_ptest_all(args, data_option, exp=True)
    # make_ptest_all(args, data_option) 
    # make_ptest_sep(args, data_option)   

    # make_ptest(args, data_option)