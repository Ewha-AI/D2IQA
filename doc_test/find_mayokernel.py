import os
from glob import glob
import numpy as np
import pydicom


def load_dcm_scans(dcm_dir):
    """
    dicom_dir: directory which contains all dicom files for one case
    return:
        dicom_slices with all dicom information
    """    
    # print(dcm_dir)
    # dcm_files = os.listdir(dcm_dir)
    # dcm_slices = [(pydicom.dcmread(os.path.join(dcm_dir, f)), os.path.basename(f)) for f in dcm_files]
    # dcm_slices.sort(key = lambda x: int(x[0].InstanceNumber))
    dcm_files = glob(os.path.join(dcm_dir, '*'))
    dcm_slices = [pydicom.dcmread(f) for f in dcm_files]
    dcm_slices.sort(key = lambda x: int(x.InstanceNumber))

    return dcm_slices

if __name__ == '__main__':


    # pid_list = {'L506' : [140.0, 170.0, 199.0, 208.0, 214.0, 245.0, 281.0, 286.0, 288.0, 300.0, 306.0, 310.0, 345.0, 347.0, 372.0],
    #             'L067': [153.0, 176.0, 183.0, 199.0, 232.0, 248.0, 262.0, 269.0, 299.0, 302.0, 332.0, 344.0, 361.0, 377.0, 418.0]}

    # category_list = ['full_1mm']
    # src_dcm_dir = '../../Data/Mayo-Clinic'
    
    # for pid in pid_list.keys():
    #     for category in category_list:
    #         dcm_dir = os.path.join(src_dcm_dir, pid, category)
    #         dcm_slices = load_dcm_scans(dcm_dir)

    #         fid_list = pid_list[pid]
    #         for fid in fid_list :
    #             ds = dcm_slices[int(fid)]
    #             print('Convolutional kernel : ', ds.ConvolutionKernel)
    

    pid_list = ['L067', 'L096', 'L109', 'L143', 'L192', 'L286',
                'L291', 'L310', 'L333', 'L506']

    category_list = ['full_1mm', 'quarter_1mm']
    src_dcm_dir = '../../Data/Mayo-Clinic'
    
    for pid in pid_list:
        for category in category_list:
            dcm_dir = os.path.join(src_dcm_dir, pid, category)
            dcm_slices = load_dcm_scans(dcm_dir)

            for i, ds in enumerate(dcm_slices):
                if ds.ConvolutionKernel != 'B30f':
                    # print(pid, category)
                    print('Convolutional kernel : ', ds.ConvolutionKernel)

                

            
            

