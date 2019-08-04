import numpy as np
import tensorflow as tf

import sys
sys.path.append('../')

import config
from model_design.spatial_transformer import transformer, batch_transformer


def gen_summary_for_cnn(name, tensor):
    tensor = tf.reduce_sum(input_tensor=tensor, axis=-1)
    tensor = tf.expand_dims(input=tensor, axis=-1)
    tf.summary.image(name=name, tensor=tensor, max_outputs=1)


def stn_net_1(inputs):
    gen_summary_for_cnn('input', inputs)

    with tf.variable_scope('STN_Net'):
        # --------------------- part1 -------------------------
        x_flatten = tf.contrib.layers.flatten(inputs=inputs)
        stn_fc1 = tf.keras.layers.Dense(units=20,
                                        name='stn_fc1',
                                        kernel_initializer='zeros',
                                        bias_initializer='zeros',
                                        )(x_flatten)
        stn_act1 = tf.keras.activations.tanh(x=stn_fc1)
        stn_drop1 = tf.keras.layers.Dropout(rate=0.0)(stn_act1)
        # --------------------- part2 --------------------------
        stn_fc2_a = tf.keras.layers.Dense(units=6,
                                          name='stc_fc2_a',
                                          kernel_initializer='zeros',
                                          use_bias=False,
                                          )(stn_drop1)
        init_bias2 = tf.Variable(initial_value=[5, 0, 0, 0, 5, 0],
                                 dtype=tf.float32,
                                 trainable=True,
                                 name='stc_fc2_bias')
        stn_fc2_b = stn_fc2_a + init_bias2
        stc_act2 = tf.keras.activations.tanh(x=stn_fc2_b)
        print('stc_act2: ', stc_act2.get_shape().as_list())
        # --------------------- part3 ---------------------------
        stn_trans = transformer(U=inputs, theta=stc_act2, out_size=(config.img_h//2, config.img_w//2))

        stn_out = tf.reshape(tensor=stn_trans,
                             shape=[-1, config.img_h//2, config.img_w//2, config.img_ch],
                             name='stn_reshape')

        stn_out.set_shape([None, config.img_h//2, config.img_w//2, config.img_ch])

        gen_summary_for_cnn('stn_out', stn_out)
        return stn_out


def stn_net_2(inputs):
    gen_summary_for_cnn('input', inputs)

    with tf.variable_scope('STN_Net'):
        # --------------------- part1 -------------------------
        pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                          strides=(2, 2),
                                          padding='valid'
                                          )(inputs)

        conv1 = tf.keras.layers.Conv2D(filters=8,
                                       kernel_size=(3, 3),
                                       strides=(2, 2),
                                       padding='same',
                                       name='conv1'
                                       )(pool1)

        bn1 = tf.keras.layers.BatchNormalization(name='bn1')(conv1)

        act1 = tf.keras.activations.relu(x=bn1)

        pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                          strides=(2, 2),
                                          padding='valid'
                                          )(act1)

        conv2 = tf.keras.layers.Conv2D(filters=16,
                                       kernel_size=(3, 3),
                                       strides=(2, 2),
                                       padding='same',
                                       name='conv2'
                                       )(pool2)

        bn2 = tf.keras.layers.BatchNormalization(name='bn2')(conv2)

        act2 = tf.keras.activations.relu(x=bn2)

        conv3 = tf.keras.layers.Conv2D(filters=32,
                                       kernel_size=(2, 2),
                                       strides=(2, 2),
                                       padding='valid',
                                       name='conv3'
                                       )(act2)

        bn3 = tf.keras.layers.BatchNormalization(name='bn3')(conv3)

        act3 = tf.keras.activations.relu(x=bn3)

        ## -------------------- part2 ------------------------

        act3_sqz = tf.squeeze(input=act3, axis=1)
        # c3_shape = conv3.get_shape().as_list()
        # seq3 = tf.reshape(conv3, shape=[-1, c3_shape[2] * c3_shape[3]])
        # seq3.set_shape([None, c3_shape[2] * c3_shape[3]])

        n_hidden = 16
        fw_cell = tf.nn.rnn_cell.GRUCell(num_units=n_hidden, name='bi_rnn_fw')
        bw_cell = tf.nn.rnn_cell.GRUCell(num_units=n_hidden, name='bi_rnn_bw')
        rnn_out, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                         cell_bw=bw_cell,
                                                         inputs=act3_sqz,
                                                         dtype=tf.float32,
                                                         scope='bi_rnn')
        # fw_out, bw_out = rnn_out
        concat = tf.concat(values=rnn_out, axis=2, name='concat')  # (bs, time_steps, 255+255)
        concat_seq = tf.contrib.layers.flatten(inputs=concat)

        # --------------------- part3 --------------------------
        fc_a = tf.keras.layers.Dense(units=6,
                                     name='fc_a',
                                     kernel_initializer='zeros',
                                     use_bias=False,
                                     )(concat_seq)
        init_bias = tf.Variable(initial_value=[5, 0, 0, 0, 5, 0],
                                dtype=tf.float32,
                                trainable=True,
                                name='fc_bias')
        fc_b = fc_a + init_bias
        act3 = tf.keras.activations.tanh(x=fc_b)
        # --------------------- part4 ---------------------------
        stn_trans = transformer(U=inputs, theta=act3, out_size=(config.img_h, config.img_w))
        stn_out = tf.reshape(tensor=stn_trans,
                             shape=[-1, config.img_h, config.img_w, 1],
                             name='stn_reshape')
        stn_out.set_shape([None, config.img_h, config.img_w, 1])

        gen_summary_for_cnn('stn_out', stn_out)

        return stn_out


def cnn_net(inputs):  # inputs (bs, 32, 128, 1)

    with tf.variable_scope('CNN_Net'):
        # ---------------------- part1 -------------------------
        conv1 = tf.keras.layers.Conv2D(filters=64,
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       padding='same',
                                       name='conv1')(inputs)

        # gen_summary_for_cnn('cnn_conv1', conv1)

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

        # pool4 = tf.keras.layers.MaxPooling2D(pool_size=(1, 1),
        #                                      strides=(1, 1),
        #                                      name='pool4')(relu4)  # (bs, 16, 64, 64)

        # ---------------------- part5 -------------------------
        conv5 = tf.keras.layers.Conv2D(filters=512,
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       padding='same',
                                       # kernel_initializer='he_normal',
                                       name='covn5')(relu4)  # (bs, 4, 16, 512)

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
                                       kernel_size=(2, 2),
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
        seq_fc_1 = tf.keras.layers.Dense(units=1024, activation='relu', name='seq_fc_1')(seq_dr_out)
        seq_fc_2 = tf.keras.layers.Dense(units=len(config.cls_dict), name='seq_fc_2')(seq_fc_1)
        return seq_fc_2


def cls_net(inputs, is_training):
    f_inputs = tf.cast(inputs, tf.float32)
    rs_inputs = tf.reshape(tensor=f_inputs, shape=[-1, config.img_h, config.img_w, config.img_ch])

    stn_out = stn_net_1(rs_inputs)
    print('stn_out: ', stn_out.get_shape().as_list())
    cnn_out = cnn_net(stn_out)
    print('cnn_output: ', cnn_out.get_shape().as_list())
    fc_out = fc_net(cnn_out, is_training)
    print('fc_output: ', fc_out.get_shape().as_list())
    return fc_out


def main():
    inputs = tf.placeholder(dtype=tf.float32, shape=(None, config.img_h, config.img_w, config.img_ch), name='inputs')
    net_out = cls_net(inputs, is_training=True)


if __name__ == '__main__':
    main()
