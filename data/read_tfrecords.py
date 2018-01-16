# -*- coding:utf-8 -*-
"""
Read TFRecords file.
"""
import os
import tensorflow as tf

class Read_TFRecords(object):
    def __init__(self, filename, batch_size=64,
        image_h=256, image_w=256, image_c=3, num_threads=8, capacity_factor=3, min_after_dequeue=1000):
        '''
        filename: TFRecords file path.
        num_threads: TFRecords file load thread.
        capacity_factor: capacity_.
        '''
        self.filename = filename
        self.batch_size = batch_size
        self.image_h = image_h
        self.image_w = image_w
        self.image_c = image_c
        self.num_threads = num_threads
        self.capacity_factor = capacity_factor
        self.min_after_dequeue = min_after_dequeue

    def read(self):
        # Read a TFRecords file, return tf.train.batch/tf.train.shuffle_batch object.
        reader = tf.TFRecordReader()
        # Queue
        filename_queue = tf.train.string_input_producer([self.filename])
        key, serialized_example = reader.read(filename_queue)
        
        features = tf.parse_single_example(serialized_example,
            features={
                "image_clean": tf.FixedLenFeature([], tf.string),
                "image_noisy": tf.FixedLenFeature([], tf.string),
            })

        image_clean = tf.image.decode_jpeg(features["image_clean"], channels=self.image_c, name="decode_image")
        image_noisy = tf.image.decode_jpeg(features["image_noisy"], channels=self.image_c, name="decode_image")
        # print(image.shape) # (?, ?, 3).

        # not need Crop and other random augmentations
        # resize, transform type.
        if self.image_h is not None and self.image_w is not None:
            image_clean = tf.image.resize_images(image_clean, [self.image_h, self.image_w], 
                method=tf.image.ResizeMethod.BICUBIC)

            image_noisy = tf.image.resize_images(image_noisy, [self.image_h, self.image_w], 
                method=tf.image.ResizeMethod.BICUBIC)
        image_clean = tf.cast(image_clean, tf.float32)/255.0 # convert to float32
        image_noisy = tf.cast(image_noisy, tf.float32)/255.0 # convert to float32

        # tf.train.batch/tf.train.shuffle_batch.
        # Using asynchronous queues
        clean_data, noisy_data = tf.train.shuffle_batch([image_clean, image_noisy],
            batch_size=self.batch_size,
            capacity = self.min_after_dequeue + self.capacity_factor * self.batch_size,
            min_after_dequeue=self.min_after_dequeue,
            num_threads=self.num_threads,
            name='images')
       
        return clean_data, noisy_data # list or dictionary of tensors. 


if __name__ == '__main__':
    data_dir = "../datasets/tfrecords"
    reader = Read_TFRecords(filename=os.path.join(data_dir, "VOC0712.tfrecords"), batch_size=64,
        image_h=256, image_w=256, image_c=1)
    clean_images, noisy_images = reader.read()

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init) # Before QueueRunner, ues tf.global_variables_initializer()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        step = 0
        for i in range(2):
        # while not coord.should_stop():
            batch_clean_images, batch_noisy_images = sess.run([clean_images, noisy_images])
            print("clean image shape: {}".format(batch_clean_images.shape))
            print("noisy image shape: {}".format(batch_noisy_images.shape))
            step += 1
            print(step)
    except KeyboardInterrupt:
        print('Interrupted')
        coord.request_stop()
    except Exception as e:
        coord.request_stop(e)
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()
        coord.join(threads)
