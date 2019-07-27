import tensorflow as tf
import os
import cv2
import numpy as np
import sys

from model_design import model_design_nn
import config
from data_prepare import data_loader

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__=='__main__':

    images = tf.placeholder(dtype=tf.float32, shape=[None, config.img_h, config.img_w, config.img_ch])

    net_out = model_design_nn.cls_net(images, is_training=False)
    scores = tf.keras.layers.Softmax()(net_out)

    sess = tf.Session()
    saver = tf.train.Saver(max_to_keep=3)
    model_save_dir = './run_output'
    ckpt_path = tf.train.latest_checkpoint(model_save_dir)
    print('latest_checkpoint_path: ', ckpt_path)
    if ckpt_path is not None:
        saver.restore(sess, ckpt_path)
    else:
        print('ckpt not exists, task over!')
        exit(0)

    with sess.as_default():
        img_dir = './data/logo_train_data'
        img_fn_list = os.listdir(img_dir)
        correct_cnt = 0
        for idx, img_fn in enumerate(img_fn_list):
            print('----------------- img_fn: %s'%img_fn)
            img_fpath = os.path.join(img_dir, img_fn)
            img_data = data_loader.preprocess(img_fpath)
            _images = np.expand_dims(img_data, axis=0)
            print('_images shape: ', images.shape)

            _scores = sess.run(scores, feed_dict={images: _images})

            print(_scores.shape)
            print(_scores)
            bg_prob, seat_belt_prob = _scores[0]
            if bg_prob > seat_belt_prob:
                cls_res = 0
                print('### BG: ', bg_prob)
            else:
                cls_res = 1
                print('### Seat_belt: ', seat_belt_prob)
            if cls_res == int(img_fn.startswith('pos_')):
                correct_cnt += 1
            else:
                # title = 'pos' if cls_res==1 else 'neg'
                # cv2.imshow('result_%s'%title, cv2.imread(img_fpath))
                # cv2.waitKey(0)
                print('######################################################### WRONG !!!')
            print('predict_result {}/{}'.format(str(correct_cnt), str(idx+1)))























