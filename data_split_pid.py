import os
from glob import glob
import random
import shutil
import argparse


"""
pid1 = ['L096', 'L109', 'L143', 'L192', 'L286','L310', 'L333']
pid2 =['L067', 'L291', 'L506']
Randomly split the images in pid1 into training and validation sets.
"""



def pair_split(train_filenames):

    half_idx = int(len(train_filenames) / 2)
    full_1mm = train_filenames[:half_idx]
    quarter_1mm = train_filenames[half_idx:]

    rand_idx = [idx for idx in range(len(full_1mm))]
    random.shuffle(rand_idx)


    train_len = int(0.8 * len(full_1mm))
    train_idx = rand_idx[:train_len]
    valid_idx = rand_idx[train_len:]


    train_full_1mm = [full_1mm[i] for i in train_idx]
    valid_full_1mm = [full_1mm[i] for i in valid_idx]
    train_quarter_1mm = [quarter_1mm[i] for i in train_idx]
    valid_quarter_1mm = [quarter_1mm[i] for i in valid_idx]

    train_data = train_full_1mm + train_quarter_1mm
    valid_data = valid_full_1mm + valid_quarter_1mm

    return train_data, valid_data


def non_pair_split(train_filenames):


    random.shuffle(train_filenames)
    train_len = int(0.8 * len(train_filenames))

    train_data = train_filenames[:train_len]
    valid_data = train_filenames[train_len:]

    return train_data, valid_data


def copy_folder(folder_name, root, data_root, data_filenames):

    dataset_type = data_root.split('/')[-1]
    temp_root = os.path.join(root, folder_name, 'images')
    txt_name = os.path.join(root, folder_name, '{}.txt'.format(dataset_type))
    # print(txt_name)
    # exit()

    out_file = open(txt_name, 'w')
    for data_name in data_filenames:
        data_path = os.path.join(temp_root, data_name)
        out_file.write(data_path + '\n')
        shutil.copy(data_path, data_root)


def mayo_main(folder_name, root):
        
    data_root = os.path.join(root, folder_name, 'images')
    data_samples = sorted(os.listdir(data_root))


    pid_train_list = ['L096', 'L109', 'L143', 'L192', 'L286','L310', 'L333']
    pid_test_list =['L067', 'L291', 'L506']


    #Split train and test
    train_filenames = []
    test_filenames = []

    for i in range(len(data_samples)):

        pid = data_samples[i].split('_')[-2]

        if pid in pid_train_list:
            train_filenames.append(data_samples[i])
        elif pid in pid_test_list:
            test_filenames.append(data_samples[i])
        else :
            raise AssertionError("Wrong")


    #Chhose pair or not
    train_data, valid_data = non_pair_split(train_filenames)

    #Copy folder
    train_root = os.path.join(root, folder_name, 'train')
    val_root = os.path.join(root, folder_name, 'val')
    test_root = os.path.join(root, folder_name, 'test')

    copy_folder(folder_name, root, train_root, train_data)
    copy_folder(folder_name, root, val_root, valid_data)
    copy_folder(folder_name, root, test_root, test_filenames)


def phantom_main(folder_name):

    # folder_name = 'phantom_ge_chest'
    data_root = '/home/eunbyeol/data/{}/images'.format(folder_name)
    train_filenames = sorted(os.listdir(data_root))
    
    #Chhose pair or not
    train_data, valid_data = non_pair_split(train_filenames)
    
    #Copy folder
    train_root = '/home/eunbyeol/data/{}/train'.format(folder_name)
    val_root = '/home/eunbyeol/data/{}/val'.format(folder_name)
    
    copy_folder(folder_name, train_root, train_data)
    copy_folder(folder_name, val_root, valid_data)


if __name__ == '__main__':
     
    parser = argparse.ArgumentParser(description='MIQA dataset')

    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--root', type=str, required=True, help='Parent directory')
    args = parser.parse_args()

    mayo_main(args.dataset, args.root)
    # phantom_main('phantom_ge_chest')
     