import os
import numpy as np
import imageio
from skimage.filters import gaussian
from saliency.make_saliency import mayo2saliency, mayo_wo_background


def fill_diff_cnr(x, y, low, sign, content_ratio):
    
    """Apply same |mA-mB| but different cnr"""
    lesion_value = sign * content_ratio
    
    lesion_map = np.zeros(low.shape, dtype=low.dtype)
    lesion_map[y, x] = lesion_value #원래 분포가 그모양이여서 상수를 더함
    lesion_map = gaussian(lesion_map, sigma=1)

    return lesion_map

def extract_centers(intersect, num, radius_list, dist_limit=50):
            
    yx = [] # 좌표 저장 리스트
    i = 0
    while i < num:

        r = radius_list[i] + 25 # 여유두기
        rand_num = np.random.randint(len(intersect[0]))
                

        intersect_transpose = np.array(intersect).transpose()
        y,x = intersect_transpose[rand_num]

        if x > 502  or y >502:
            continue
        
        rota = [-1, 1]
        rotb = [-1, 1]
        circle = [(y+a*r, x+b*r) for a in rota for b in rotb]
        circle_in_intersect = [c for c in circle if c in intersect_transpose]


        if len(circle_in_intersect) == 4:

            if i == 0:
                yx.append((y,x))
                i += 1

            else :
                total_dist = []
                for k in range(len(yx)):
                    a = np.array(yx[k])
                    b = np.array((y,x))
                    dist = np.linalg.norm(a-b)
                    total_dist.append(dist)

                if min(total_dist) >= dist_limit:
                    yx.append((y,x))
                    i += 1            
        else :
            continue
            
    return yx

def find_app_location(lp):
    

    low_copy = imageio.imread(lp)
    
    #Remove background & Extract saliency map
    low_nobg = mayo_wo_background(lp)
    low_nobg /= 255
    low_nobg[low_nobg !=0] = 1

    threshold = 0.4   
    low_saliency_map = mayo2saliency(lp, threshold) * low_nobg
    
    border = np.where(low_saliency_map == 1)
    border = tuple(np.array(border))


    #To check
    # low_copy[low_copy==0] = 0.01
    # low_copy[border] = 0.5

    # imgname_hp = lp.split('float32')[-1].replace('/', '_')[1:]
    # out_img_hp = ('../../data/sample/%s' %imgname_hp)
    # imageio.imwrite(out_img_hp, low_copy.astype('float32'))

    return border