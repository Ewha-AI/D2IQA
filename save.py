import os
import imageio
from abstract_shape import *


def save_testset(filename, target_filepath, images, lesions, bpsrc, multi_class=False): 

    bbox_coord, poly_coord, shape_list, radius_list, content_list = bpsrc

    if multi_class :
        class_dict = {
            Triangle : 1,
            Circle : 2,
            Square : 3,
            Star : 4
        }
    else :
        class_dict = {Circle : 1}

    
    if not os.path.isdir(os.path.join(target_filepath, 'temp', 'images')):
        os.makedirs(os.path.join(target_filepath, 'temp', 'images'))
        os.makedirs(os.path.join(target_filepath, 'temp', 'labels'))
        os.makedirs(os.path.join(target_filepath, 'temp', 'polygons'))
        os.makedirs(os.path.join(target_filepath, 'temp', 'lesions'))
        os.makedirs(os.path.join(target_filepath, 'temp', 'setting'))
    
    image_filename = ('%s/temp/images/%s.tiff' %(target_filepath, filename))
    lesion_filename = ('%s/temp/lesions/%s.tiff' %(target_filepath, filename))
    bbox_filename =  open(('%s/temp/labels/%s.txt'  %(target_filepath, filename)), 'w')  
    poly_filename =  open(('%s/temp/polygons/%s.txt'  %(target_filepath, filename)), 'w')  
    setting_filename =  open(('%s/temp/setting/%s.txt'  %(target_filepath, filename)), 'w')  
        
    #Save
    imageio.imwrite(image_filename, images.astype('float32'))
    imageio.imwrite(lesion_filename, lesions.astype('float32'))

    for idx in range(len(shape_list)):

        class_id = str(class_dict[shape_list[idx]])
        radius = radius_list[idx]
        content = content_list[idx]

        string_labels = (class_id + " " + " ".join([str(a) for a in bbox_coord[idx]]) + '\n')        

        string_setting = (class_id + " " + " ".join([str(a) for a in bbox_coord[idx]]) 
                    + " " +str(radius) + " " + str(content) + '\n')
        string_polygons = (" ".join([str(a) for a in poly_coord[idx]]) + '\n')

        bbox_filename.write(string_labels)
        setting_filename.write(string_setting)
        poly_filename.write(string_polygons)


def save_trainset(dataset, root, filename, img, lesion, bpsrc, multi_class=False):
    
    bbox_coord, poly_coord, shape_list, radius_list, content_list = bpsrc

    if multi_class :
        class_dict = {
            Triangle : 1,
            Circle : 2,
            Square : 3,
            Star : 4
        }
    else :
        class_dict = {Circle : 1}


    out_images = ('%s/%s/images/%s.tiff' %(root, dataset, filename))
    out_lesions = ('%s/%s/lesions/%s.tiff' %(root, dataset, filename))      
    out_bbox_labels = open(('%s/%s/labels/%s.txt'  %(root, dataset, filename)), 'w')
    out_bbox_setting = open(('%s/%s/setting/%s.txt'  %(root, dataset, filename)), 'w')  
    out_bbox_polygons = open(('%s/%s/polygons/%s.txt'  %(root, dataset, filename)), 'w')  

    imageio.imwrite(out_images, img.astype('float32'))
    imageio.imwrite(out_lesions, lesion.astype('float32'))

    for idx in range(len(shape_list)):

        class_id = str(class_dict[shape_list[idx]])
        radius = radius_list[idx]
        content = content_list[idx]

        string_labels = (class_id + " " + " ".join([str(a) for a in bbox_coord[idx]]) + '\n')        

        string_setting = (class_id + " " + " ".join([str(a) for a in bbox_coord[idx]]) 
                    + " " +str(radius) + " " + str(content) + '\n')
        string_polygons = (" ".join([str(a) for a in poly_coord[idx]]) + '\n')

        out_bbox_labels.write(string_labels)
        out_bbox_setting.write(string_setting)
        out_bbox_polygons.write(string_polygons)

