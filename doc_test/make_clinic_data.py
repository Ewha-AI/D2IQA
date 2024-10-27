import csv
import os
from glob import glob
import random
import imageio



remove_idx = {'L067' : [151,170], 'L506':[130,149] }
target_dir = '../../data/nimg_wl40/doc_test'

if not os.path.exists(os.path.join(target_dir, 'test')):
    os.makedirs(os.path.join(target_dir, 'test'))
if not os.path.exists(os.path.join(target_dir, 'full')):
    os.makedirs(os.path.join(target_dir, 'full'))
if not os.path.exists(os.path.join(target_dir, 'quarter_2.0')):
    os.makedirs(os.path.join(target_dir, 'quarter_2.0'))



#Record csv files
ans_csv_path = '../../data/nimg_wl40/doc_test/ans.csv'
prob_csv_path = '../../data/nimg_wl40/doc_test/prob.csv'

with open(ans_csv_path, "w", newline='') as csvfile:
    wr = csv.writer(csvfile, dialect="excel")
    title = ['title','1','2','3','4','5']
    wr.writerow(title)

with open(prob_csv_path, "w", newline='') as csvfile:
    wr = csv.writer(csvfile, dialect="excel")
    title = ['score','1','2','3','4','5']
    wr.writerow(title)


total_dose_list = []

for pid in remove_idx.keys():

    pid_root = '../../data/nimg_wl40/{}'.format(pid)    
    num_imgs = len(os.listdir(os.path.join(pid_root, 'full')))
    idx = [i for i in range(num_imgs)]

    #Remove idx of files
    srt_dix, end_idx = remove_idx[pid]
    temp_idx = idx[20:srt_dix] + idx[end_idx:-20]

    #Shuffle idx of files
    random.shuffle(temp_idx)
    test_idx = temp_idx[:15]

    for idx in test_idx:

        #Shuffle doses and choose 5 doses
        dose = os.listdir(pid_root)
        random.shuffle(dose)
        test_dose = dose[:5]
        total_dose_list += test_dose

        for pos_idx, pos in enumerate(test_dose):

            file_root = os.path.join(pid_root, pos)
            file_name = sorted(glob((file_root + '/*.tiff')))[idx]            
            
            target_folder = '{}_{}'.format(pid, file_name.split('/')[-1][:-5])
            if not os.path.isdir(os.path.join(target_dir,'test', target_folder)):
                os.makedirs(os.path.join(target_dir,'test', target_folder))

            prob_title = '{}_{}_{}'.format(pid, file_name.split('/')[-1][:-5], pos_idx+1)
            target_path = os.path.join(target_dir, 'test', target_folder, (prob_title + '.tiff'))

            imageio.imwrite(target_path, imageio.imread(file_name))
        
        #Write answer & porblem csv
        with open(prob_csv_path, "a", newline='') as csvfile :
            wr = csv.writer(csvfile, dialect='excel')
            wr.writerow([target_folder])
        
        data_csv = [target_folder] + test_dose
        with open(ans_csv_path, "a", newline='') as csvfile :
            wr = csv.writer(csvfile, dialect='excel')
            wr.writerow(data_csv)    

        #Save reference
        ref_list = ['full', 'quarter_2.0']

        for ref in ref_list:
            if not os.path.isdir(os.path.join(target_dir, ref)):
                    os.makedirs(os.path.join(target_dir, ref))

            file_root = os.path.join(pid_root, ref)
            file_name = sorted(glob((file_root + '/*.tiff')))[idx]

            target_name = '{}_{}'.format(pid, file_name.split('/')[-1])
            target_path = os.path.join(target_dir, ref, target_name)

            imageio.imwrite(target_path, imageio.imread(file_name))

        


pid_root = '../../data/nimg_wl40/{}'.format(pid)
folders = sorted(os.listdir(pid_root))
count_csv_path = '../../data/nimg_wl40/doc_test/countPerDose.csv'

with open(count_csv_path, "w", newline='') as csvfile:
    wr = csv.writer(csvfile, dialect="excel")
    title = ['dose', 'count']
    wr.writerow(title)

for folder in folders:
    
    data_csv = [folder, total_dose_list.count(folder)]
    print(data_csv)
    with open(count_csv_path, "a", newline='') as csvfile :
        wr = csv.writer(csvfile, dialect='excel')
        wr.writerow(data_csv)   

print(len(total_dose_list))


    