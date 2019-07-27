import numpy as np
import tensorflow as tf

from data_prepare import data_loader
import config
import cv2


def cnn_net(inputs):  # inputs (bs, 32, 128, 1)

    with tf.variable_scope('CNN_Module'):
        # ---------------------- part1 -------------------------
        conv1 = tf.keras.layers.Conv2D(filters=64,
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       padding='same',
                                       name='conv1')(inputs)

        bn1 = tf.keras.layers.BatchNormalization(name='bn1')(conv1)

        relu1 = tf.keras.activations.relu(x=bn1)

        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                             strides=(2, 2),
                                             name='pool1')(relu1)  # (bs, 16, 64, 64)

        # ---------------------- part2 -------------------------
        conv2 = tf.keras.layers.Conv2D(filters=128,
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       padding='same',
                                       name='conv2')(pool1)

        bn2 = tf.keras.layers.BatchNormalization(name='bn2')(conv2)

        relu2 = tf.keras.activations.relu(x=bn2)

        pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                             strides=(2, 2),
                                             name='pool2')(relu2)  # (bs, 16, 64, 64)

        # ---------------------- part3 -------------------------
        conv3 = tf.keras.layers.Conv2D(filters=256,
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       padding='same',
                                       name='conv3')(pool2)

        bn3 = tf.keras.layers.BatchNormalization(name='bn3')(conv3)

        relu3 = tf.keras.activations.relu(x=bn3)

        # ---------------------- part4 -------------------------
        conv4 = tf.keras.layers.Conv2D(filters=256,
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       padding='same',
                                       name='conv4')(relu3)

        bn4 = tf.keras.layers.BatchNormalization(name='bn4')(conv4)

        relu4 = tf.keras.activations.relu(x=bn4)

        pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                             strides=(2, 2),
                                             name='pool4')(relu4)  # (bs, 16, 64, 64)

        # ---------------------- part5 -------------------------
        conv5 = tf.keras.layers.Conv2D(filters=512,
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       padding='same',
                                       # kernel_initializer='he_normal',
                                       name='covn5')(pool4)  # (bs, 4, 16, 512)

        bn5 = tf.keras.layers.BatchNormalization(name='bn5')(conv5)

        relu5 = tf.keras.activations.relu(x=bn5)

        # ---------------------- part6 -------------------------
        conv6 = tf.keras.layers.Conv2D(filters=512,
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       padding='same',
                                       # kernel_initializer='he_normal',
                                       name='covn6')(relu5)  # (bs, 4, 16, 512)

        bn6 = tf.keras.layers.BatchNormalization(name='bn6')(conv6)

        relu6 = tf.keras.activations.relu(x=bn6)

        pool6 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                             strides=(2, 2),
                                             name='pool6')(relu6)  # (bs, 16, 64, 64)

        # ---------------------- part7 -------------------------
        conv7 = tf.keras.layers.Conv2D(filters=512,
                                       kernel_size=(1, 1),
                                       strides=(1, 1),
                                       padding='same',
                                       # kernel_initializer='he_normal',
                                       name='covn7')(pool6)  # (bs, 2, 8, 512)

        bn7 = tf.keras.layers.BatchNormalization(name='bn7')(conv7)

        relu7 = tf.keras.activations.relu(x=bn7)

    return relu7


def fc_net(cnn_out, is_training):
    with tf.variable_scope('FC_Module'):
        shape = cnn_out.get_shape().as_list()
        seq_out = tf.reshape(tensor=cnn_out, shape=[-1, shape[1] * shape[2] * shape[3]], name='reshape1')  # (bs, 8, 1024)
        seq_out.set_shape([None, shape[1]*shape[2]*shape[3]])

        # add dropout layer
        if is_training:
            seq_dr_out = tf.keras.layers.Dropout(rate=0.2)(seq_out)
        else:
            seq_dr_out = seq_out

        # fc depth-->256
        seq_fc_1 = tf.keras.layers.Dense(units=256, activation='relu', name='seq_fc_1')(seq_dr_out)
        seq_fc_2 = tf.keras.layers.Dense(units=len(config.cls_dict), name='seq_fc_2')(seq_fc_1)
        return seq_fc_2


def cls_net(inputs, is_training):
    f_inputs = tf.cast(inputs, tf.float32)
    rs_inputs = tf.reshape(tensor=f_inputs, shape=[-1, config.img_h, config.img_w, config.img_ch])

    cnn_out = cnn_net(rs_inputs)
    print('cnn_output: ', cnn_out.get_shape().as_list())
    fc_out = fc_net(cnn_out, is_training)
    print('fc_output: ', fc_out.get_shape().as_list())
    return fc_out


def main():
    inputs = tf.placeholder(dtype=tf.float32, shape=(None, config.img_h, config.img_w, config.img_ch), name='inputs')
    net_out = cls_net(inputs, is_training=True)


if __name__ == '__main__':
    main()
