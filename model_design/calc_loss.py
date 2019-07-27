import tensorflow as tf
import numpy as np
import os
import cv2

import sys

sys.path.append('../')

import config




def build_loss(logits, labels):
    logits = tf.expand_dims(input=logits, axis=-2)
    labels = tf.expand_dims(input=labels, axis=-1)
    print('logits: ', logits.get_shape().as_list())
    print('labels: ', labels.get_shape().as_list())
    cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits,
        labels=labels
    )
    print('ce_loss: ', cross_entropy_loss.get_shape().as_list())
    loss = tf.reduce_mean(cross_entropy_loss)
    return loss



if __name__ == '__main__':
    logits = tf.placeholder(dtype=tf.float32, shape=[3, 2])
    labels = tf.placeholder(dtype=tf.int32, shape=[3, 1])

    loss = build_loss(logits, labels)

    #
    # print('loss: ', loss.get_shape().as_list())
    # loss = tf.reduce_sum(loss) * 2e-6

    print('loss: ', loss.get_shape().as_list())





