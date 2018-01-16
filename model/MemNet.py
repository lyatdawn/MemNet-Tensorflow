# -*- coding:utf-8 -*-
"""
Load data.
Train and test MemNet with 6 Memory blocks, each contains 6 Recursive uints.
"""
import os
import logging
from datetime import datetime
import time
import tensorflow as tf
from MemNet_M6R6 import memnet_m6r6
from test_MemNet_M6R6 import test_memnet_m6r6
from utils import save_images

import sys
sys.path.append("../data")
from data.read_tfrecords import Read_TFRecords

class MemNet(object):
    def __init__(self, sess, tf_flags):
        self.sess = sess
        self.dtype = tf.float32

        # checkpoint and summary.
        self.output_dir = tf_flags.output_dir
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoint")
        self.checkpoint_prefix = "model"
        self.saver_name = "checkpoint"
        self.summary_dir = os.path.join(self.output_dir, "summary")

        self.is_training = (tf_flags.phase == "train") # train or test.
        self.sample_dir = "sample"

        # placeholder, clean_images and noisy_images.
        self.batch_size = tf_flags.batch_size
        self.image_h = 256
        self.image_w = 256
        assert self.image_h == self.image_w
        self.image_c = 1

        self.clean_images = tf.placeholder(self.dtype, [None, self.image_h, self.image_w, self.image_c])
        self.noisy_images = tf.placeholder(self.dtype, [None, self.image_h, self.image_w, self.image_c])

        # train
        if self.is_training:
            self.training_set = tf_flags.training_set

            # makedir aux dir
            self._make_aux_dirs()
            # compute and define loss
            self._build_training()
            # logging, only use in training
            log_file = os.path.join(self.output_dir, "MemNet.log")
            logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                                filename=log_file,
                                level=logging.DEBUG,
                                filemode='a+')
            logging.getLogger().addHandler(logging.StreamHandler())
        else:
            # test
            self.recovery_image = self._build_test()

    def _build_training(self):
        # MemNet_M6R6 output
        self.pre_images, self.loss = memnet_m6r6(name="MemNet_M6R6", clean_data=self.clean_images, 
            noisy_data=self.noisy_images, num_filters=64, image_c=self.image_c, is_training=self.is_training,
            reuse=False)
        # MemNet_M6R6 Variables
        self.memnet_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="MemNet_M6R6")

        # optim, SGD.
        '''
        Optim methods have:
        tf.train.AdadeltaOptimizer, tf.train.AdagradDAOptimizer, tf.train.AdagradOptimizer, tf.train.AdamOptimizer
        tf.train.MomentumOptimizer, tf.train.RMSPropOptimizer and so on.
        '''
        self.memnet_opt = tf.train.AdamOptimizer().minimize(
            self.loss, var_list=self.memnet_variables, name="memnet_opt")
        # summary. only add loss. 
        tf.summary.scalar('loss', self.loss)
        # It can add tf.summary.image() to save images in training.

         # merge
        self.summary = tf.summary.merge_all()

        # summary and checkpoint
        self.writer = tf.summary.FileWriter(
            self.summary_dir, graph=self.sess.graph)
        self.saver = tf.train.Saver(max_to_keep=10, name=self.saver_name)
        self.summary_proto = tf.Summary()

    def train(self, training_steps, summary_steps, checkpoint_steps, save_steps):
        step_num = 0 # save checkpoint format: model-num.*.
        # restore last checkpoint
        latest_checkpoint = tf.train.latest_checkpoint("model_output_20180115112852/checkpoint") # self.checkpoint_dir, or "", or appointed path.

        if latest_checkpoint:
            step_num = int(os.path.basename(latest_checkpoint).split("-")[1])
            assert step_num > 0, "Please ensure checkpoint format is model-*.*."
            self.saver.restore(self.sess, latest_checkpoint)
            logging.info("{}: Resume training from step {}. Loaded checkpoint {}".format(datetime.now(), 
                step_num, latest_checkpoint))
        else:
            self.sess.run(tf.global_variables_initializer()) # init all variables
            logging.info("{}: Init new training".format(datetime.now()))

        # data
        reader = Read_TFRecords(filename=self.training_set, batch_size=self.batch_size,
            image_h=self.image_h, image_w=self.image_w, image_c=self.image_c)
        tfrecord_clean_images, tfrecord_noisy_images = reader.read()

        self.coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)
        # train
        try:
            c_time = time.time()
            for c_step in xrange(step_num + 1, training_steps + 1):
                # Generate clean images and noisy images.
                batch_clean_images, batch_noisy_images = self.sess.run([tfrecord_clean_images, tfrecord_noisy_images])
                # print("clean image shape: {}".format(batch_clean_images.shape))
                # print("noisy image shape: {}".format(batch_noisy_images.shape))

                c_feed_dict = {
                    self.clean_images: batch_clean_images,
                    self.noisy_images: batch_noisy_images
                }
                self.ops = [self.memnet_opt]
                self.sess.run(self.ops, feed_dict=c_feed_dict)

                # save summary
                if c_step % summary_steps == 0:
                    c_summary = self.sess.run(self.summary, feed_dict=c_feed_dict)
                    self.writer.add_summary(c_summary, c_step)

                    e_time = time.time() - c_time
                    time_periter = e_time / summary_steps
                    logging.info("{}: Iteration_{} ({:.4f}s/iter) {}".format(
                        datetime.now(), c_step, time_periter,
                        self._print_summary(c_summary)))
                    c_time = time.time() # update time

                # save checkpoint
                if c_step % checkpoint_steps == 0:
                    self.saver.save(self.sess,
                        os.path.join(self.checkpoint_dir, self.checkpoint_prefix),
                        global_step=c_step)
                    logging.info("{}: Iteration_{} Saved checkpoint".format(
                        datetime.now(), c_step))

                # save images
                if c_step % save_steps == 0:
                    compress_images, recovery_images, real_images = self.sess.run([self.noisy_images, 
                        self.pre_images, self.clean_images], feed_dict=c_feed_dict)
                    # numpy ndarray.
                    save_images(compress_images, recovery_images, real_images, 
                        './{}/train_{:06d}.png'.format(self.sample_dir, c_step))
        
        except KeyboardInterrupt:
            print('Interrupted')
            self.coord.request_stop()
        except Exception as e:
            self.coord.request_stop(e)
        finally:
            # When done, ask the threads to stop.
            self.coord.request_stop()
            self.coord.join(threads)

        logging.info("{}: Done training".format(datetime.now()))

    def load(self, checkpoint_name=None):
        # restore checkpoint
        print("{}: Loading checkpoint...".format(datetime.now())),
        if checkpoint_name:
            checkpoint = os.path.join(self.checkpoint_dir, checkpoint_name)
            self.saver.restore(self.sess, checkpoint)
            print(" loaded {}".format(checkpoint_name))
        else:
            # restore latest model
            latest_checkpoint = tf.train.latest_checkpoint(
                self.checkpoint_dir)
            if latest_checkpoint:
                self.saver.restore(self.sess, latest_checkpoint)
                print(" loaded {}".format(os.path.basename(latest_checkpoint)))
            else:
                raise IOError(
                    "No checkpoints found in {}".format(self.checkpoint_dir))

    def test(self, noisy_image):
        # Handle single image.
        c_feed_dict = {
            self.noisy_images: noisy_image
        }
        
        recovery_image = self.sess.run(self.recovery_image, feed_dict=c_feed_dict)

        return recovery_image

    def _build_test(self):
        pre_image = test_memnet_m6r6(name="MemNet_M6R6", noisy_data=self.noisy_images, num_filters=64, 
            image_c=self.image_c, is_training=True, reuse=False)
        # With the DnCNN-Tensorflow have the same problem, the BN argument, is_training must true. What??

        # self.saver define after above!
        self.saver = tf.train.Saver(max_to_keep=10, name=self.saver_name)

        return pre_image

    def _make_aux_dirs(self):
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)

    def _print_summary(self, summary_string):
        self.summary_proto.ParseFromString(summary_string)
        result = []
        for val in self.summary_proto.value:
            result.append("({}={})".format(val.tag, val.simple_value))
        return " ".join(result)