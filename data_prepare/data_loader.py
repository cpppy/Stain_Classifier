import cv2
import os
import random
import numpy as np
import sys
sys.path.append('../')

import config
from data_prepare import parse_label


def image_preprocess_by_normality(img_cv2):
    '''
    mean_BGR:  58.420655528964474 62.79212985074819 108.6818206309374
    std:  62.55949780042519
    '''
    mean = config.MEAN_BGR
    std = config.STD
    img_data = np.asarray(img_cv2, dtype=np.float32)
    img_data = img_data - mean
    img_data = img_data / std
    img_data = img_data.astype(np.float32)
    return img_data


def preprocess(img_path):
    img_cv2 = cv2.imread(img_path)
    # img_cv2 = cv2.resize(img_cv2, dsize=(config.img_w, config.img_h), interpolation=cv2.INTER_NEAREST)
    img_cv2 = cv2.resize(img_cv2, dsize=(config.img_w, config.img_h), interpolation=cv2.INTER_LINEAR_EXACT)
    # normalization
    img_data = image_preprocess_by_normality(img_cv2)
    return img_data


class DataGenerator:
    def __init__(self,
                 img_dir,
                 img_w,
                 img_h,
                 img_ch,
                 batch_size):

        self.img_h = img_h
        self.img_w = img_w
        self.img_ch = img_ch
        self.img_dir = img_dir
        self.batch_size = batch_size

        self.img_fn_list = []
        self.n = 0  # number of images
        self.indexes = []
        self.cur_index = 0
        self.imgs = []
        self.label_dict = parse_label.get_isic_2019_labels(n_choose=config.n_train_samples)

    # samples
    def build_data(self):
        print("DataGenerator, build data ...")
        self.img_fn_list = [img_fn for img_fn in os.listdir(self.img_dir) if img_fn.endswith('.jpg')]
        self.img_fn_list = self.img_fn_list[0:config.n_train_samples]
        self.n = len(self.img_fn_list)
        print("sample size of current generator: ", self.n)
        self.indexes = list(range(self.n))
        random.shuffle(self.indexes)

    def next_sample(self):  ## index max -> 0
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.cur_index = 0
            random.shuffle(self.indexes)
        # load one image & label
        img_idx = self.indexes[self.cur_index]
        img_fn = self.img_fn_list[img_idx]
        img_path = os.path.join(self.img_dir, img_fn)
        img_data = preprocess(img_path)
        cls_name = self.label_dict[img_fn.split('.')[0]]
        label_data = config.cls_dict[cls_name]
        return img_data, label_data

    def next_batch(self):  ## batch size
        while True:
            X_data = np.zeros([self.batch_size, self.img_h, self.img_w, self.img_ch], dtype=np.float32)
            Y_data = np.zeros([self.batch_size], dtype=np.int)

            for i in range(self.batch_size):
                img_data, label_data = self.next_sample()
                X_data[i] = img_data.astype(np.float)
                Y_data[i] = label_data

            # dict
            inputs = {
                'images': X_data,  # (bs, h, w, 1)
            }
            outputs = {
                'labels': Y_data,  # (bs)
            }
            yield (inputs, outputs)


def test_preprocess():
    img_path = '/data/data/train_20/images/2_frontage_9.jpg'

    input_mask, output_mask = preprocess(img_path)
    hot_map = cv2.applyColorMap(np.uint8(output_mask * 125), cv2.COLORMAP_JET)
    cv2.imshow('mask', hot_map)
    cv2.waitKey(0)


def load_test():
    img_dir = config.img_dir
    train_data = DataGenerator(img_dir=img_dir,
                               label_fpath=config.label_fpath,
                               img_h=config.img_h,
                               img_w=config.img_w,
                               img_ch=config.img_ch,
                               batch_size=2)
    train_data.build_data()

    inputs, outputs = train_data.next_batch().__next__()
    images = inputs['images']
    labels = outputs['labels']
    print(images.shape)
    print(labels.shape)
    # print(labels)
    # hot_map = cv2.applyColorMap(np.uint8(images[0] * 100), cv2.COLORMAP_JET)
    # cv2.imshow('mask', hot_map)
    # cv2.waitKey(0)


if __name__ == "__main__":
    load_test()
