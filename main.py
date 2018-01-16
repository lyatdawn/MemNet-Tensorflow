# -*- coding:utf-8 -*-
"""
An implementation of acGAN using TensorFlow (work in progress).
"""

import tensorflow as tf
import numpy as np
from model import MemNet
import os
import glob
import cv2


def main(_):
    tf_flags = tf.app.flags.FLAGS
    # gpu config.
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9

    if tf_flags.phase == "train":
        with tf.Session(config=config) as sess: 
            # sess = tf.Session(config=config) # when use queue to load data, not use with to define sess
            train_model = MemNet.MemNet(sess, tf_flags)
            train_model.train(tf_flags.training_steps, tf_flags.summary_steps, 
                tf_flags.checkpoint_steps, tf_flags.save_steps)
    else:
        with tf.Session(config=config) as sess:
            test_model = MemNet.MemNet(sess, tf_flags)
            test_model.load(tf_flags.checkpoint)
            # test Set12
            # get psnr and ssim outside
            save_path = "./datasets/Set12_Recovery"
            for image_file in glob.glob(tf_flags.testing_set):
                print("testing {}...".format(image_file))
                # testing_set is path/*.jpg.
                c_image = np.reshape(cv2.resize(cv2.imread(image_file, 0), (tf_flags.img_size, tf_flags.img_size)), 
                    (1, tf_flags.img_size, tf_flags.img_size, 1)) / 255.
                # In Caffe, Tensorflow, might must divide 255.?
                recovery_image = test_model.test(c_image)
                # save image
                cv2.imwrite(os.path.join(save_path, image_file.split("/")[3]), 
                    np.uint8(recovery_image[0, :].clip(0., 1.) * 255.))
                # recovery_image[0, :], 3D array.
            print("Testing done.")


if __name__ == '__main__':
    tf.app.flags.DEFINE_string("output_dir", "model_output", 
                               "checkpoint and summary directory.")
    tf.app.flags.DEFINE_string("phase", "train", 
                               "model phase: train/test.")
    tf.app.flags.DEFINE_string("training_set", "", 
                               "dataset path for training.")
    tf.app.flags.DEFINE_string("testing_set", "", 
                               "dataset path for testing.")
    tf.app.flags.DEFINE_integer("img_size", 256, 
                                "testing image size.")
    tf.app.flags.DEFINE_integer("batch_size", 64, 
                                "batch size for training.")
    tf.app.flags.DEFINE_integer("training_steps", 100000, 
                                "total training steps.")
    tf.app.flags.DEFINE_integer("summary_steps", 100, 
                                "summary period.")
    tf.app.flags.DEFINE_integer("checkpoint_steps", 1000, 
                                "checkpoint period.")
    tf.app.flags.DEFINE_integer("save_steps", 100, 
                                "checkpoint period.")
    tf.app.flags.DEFINE_string("checkpoint", None, 
                                "checkpoint name for restoring.")
    tf.app.run(main=main)