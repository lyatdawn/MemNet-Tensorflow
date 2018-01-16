# -*- coding:utf-8 -*-
"""
Util module.
"""
import tensorflow as tf
import numpy as np
import cv2

def save_images(noisy, gene_output, label, image_path, max_samples=1):
    # batch_size is 1.
    image = np.concatenate([noisy, gene_output, label], axis=2) # concat 4D array, along width.
    image = image[0:max_samples, :, :, :]
    image = np.concatenate([image[i, :, :, :] for i in range(max_samples)], axis=0)
    # concat 3D array, along axis=0, w.t. along height. shape: (1024, 256, 3/1).

    # save image
    # scipy.misc.toimage(), array is 2D(gray, reshape to (H, W)) or 3D(RGB).
    # scipy.misc.toimage(image, cmin=0., cmax=1.).save(image_path) # image_path contain image path and name.
    # cv.imwrite() save image.
    cv2.imwrite(image_path, np.uint8(image.clip(0., 1.) * 255.))

def res_mod_layers(in_data, num_filters, kernel_size, strides, padding, is_training):
    # Batch Norm
    bn_out = tf.layers.batch_normalization(
        inputs=in_data,
        scale=False,
        training=is_training)
    # ReLU
    act_out = tf.nn.relu(bn_out)
    # conv
    conv_out = tf.layers.conv2d(
        inputs=act_out,
        filters=num_filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding)

    return conv_out