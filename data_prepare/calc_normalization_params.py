import cv2
import numpy as np
import os

import sys
sys.path.append('../')

import config


def calc_image_mean_and_std(img_dir):
    # img_dir = './source_data/images'
    # img_dir = './source_data/images_hsv'

    mean_B_list = []
    mean_G_list = []
    mean_R_list = []
    std_list = []

    img_fn_list = os.listdir(img_dir)
    for img_fn in img_fn_list:
        if not img_fn.lower().endswith('.jpg'):
            continue
        print("---------------------------------------------------------------------")
        print('img_fn: ', img_fn)
        img_path = os.path.join(img_dir, img_fn)
        img_cv2 = cv2.imread(img_path)
        img_data = np.asarray(img_cv2, dtype=np.float)

        b_layer = img_data[:, :, 0]
        g_layer = img_data[:, :, 1]
        r_layer = img_data[:, :, 2]

        mean_b = np.mean(b_layer)
        mean_g = np.mean(g_layer)
        mean_r = np.mean(r_layer)
        std = np.std(img_data)

        print('B/G/R/std: ', mean_b, mean_g, mean_r, std)
        mean_B_list.append(mean_b)
        mean_G_list.append(mean_g)
        mean_R_list.append(mean_r)
        std_list.append(std)

    print('---------------- calc mean ----------------')
    mean_B = np.mean(mean_B_list)
    mean_G = np.mean(mean_G_list)
    mean_R = np.mean(mean_R_list)
    g_std = np.mean(std_list)
    print('mean_BGR: ', mean_B, mean_G, mean_R)
    print('std: ', g_std)



def main():

    img_dir = config.img_dir
    calc_image_mean_and_std(img_dir)



if __name__=='__main__':


    main()


