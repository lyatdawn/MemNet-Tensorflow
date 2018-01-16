# -*- coding:utf-8 -*-
"""
MemNet_M6R6 network, include 6 Memory blocks, ecah block contains 6 Recursive units(ResBlock).
"""
import tensorflow as tf
from utils import res_mod_layers

def memnet_m6r6(name, clean_data=None, noisy_data=None, num_filters=64, image_c=1, is_training=False, reuse=False):
    # memnet_m6r6(): clean_image(label), noisy_image(data).
    # num_filters: 64. image_c: image channel, is 1. is_training: BN used.
    with tf.variable_scope(name, reuse=reuse):
        # FNet: bn + relu + conv
        conv1 = res_mod_layers(noisy_data, num_filters=num_filters, kernel_size=3, strides=[1, 1], 
                padding="SAME", is_training=is_training)

        # 1st Memory block.
        # 1st Recursive block: bn + relu + conv; bn + relu + conv; skip connection
        c_in = conv1
        for _ in range(2):
            conv01_01 = res_mod_layers(c_in, num_filters=num_filters, kernel_size=3, strides=[1, 1], 
                padding="SAME", is_training=is_training)
            c_in = conv01_01
        # skip connection, conv1 + c_in
        # c_in = conv01_01
        eltwise01_01 = conv1 + c_in

        # 2nd Recursive block:bn + relu + conv; bn + relu + conv; skip connection
        c_in = eltwise01_01
        for _ in range(2):
            conv01_02 = res_mod_layers(c_in, num_filters=num_filters, kernel_size=3, strides=[1, 1], 
                padding="SAME", is_training=is_training)
            c_in = conv01_02
        # skip connection, eltwise01_01 + c_in
        # c_in = conv01_02
        eltwise01_02 = eltwise01_01 + c_in

        # 3rd Recursive block:bn + relu + conv; bn + relu + conv; skip connection
        c_in = eltwise01_02
        for _ in range(2):
            conv01_03 = res_mod_layers(c_in, num_filters=num_filters, kernel_size=3, strides=[1, 1], 
                padding="SAME", is_training=is_training)
            c_in = conv01_03
        # skip connection, eltwise01_02 + c_in
        # c_in = conv01_03
        eltwise01_03 = eltwise01_02 + c_in

        # 4th Recursive block:bn + relu + conv; bn + relu + conv; skip connection
        c_in = eltwise01_03
        for _ in range(2):
            conv01_04 = res_mod_layers(c_in, num_filters=num_filters, kernel_size=3, strides=[1, 1], 
                padding="SAME", is_training=is_training)
            c_in = conv01_04
        # skip connection, eltwise01_03 + c_in
        # c_in = conv01_04
        eltwise01_04 = eltwise01_03 + c_in

        # 5th Recursive block:bn + relu + conv; bn + relu + conv; skip connection
        c_in = eltwise01_04
        for _ in range(2):
            conv01_05 = res_mod_layers(c_in, num_filters=num_filters, kernel_size=3, strides=[1, 1], 
                padding="SAME", is_training=is_training)
            c_in = conv01_05
        # skip connection, eltwise01_04 + c_in
        # c_in = conv01_05
        eltwise01_05 = eltwise01_04 + c_in

        # 6th Recursive block:bn + relu + conv; bn + relu + conv; skip connection
        c_in = eltwise01_05
        for _ in range(2):
            conv01_06 = res_mod_layers(c_in, num_filters=num_filters, kernel_size=3, strides=[1, 1], 
                padding="SAME", is_training=is_training)
            c_in = conv01_06
        # skip connection, eltwise01_05 + c_in
        # c_in = conv01_06
        eltwise01_06 = eltwise01_05 + c_in

        # concat
        concat01 = tf.concat([conv1, eltwise01_01, eltwise01_02, eltwise01_03, eltwise01_04, eltwise01_05, 
            eltwise01_06], axis=3)
        # tf.concat(). 参数为: values: A list of Tensor. axis.
        # Caffe, data format is NCHW(axis=1). TensorFlow, data format is NHWC(axis=3).

        conv_transition_01 = res_mod_layers(concat01, num_filters=num_filters, kernel_size=1, strides=[1, 1], 
            padding="SAME", is_training=is_training)
        # Tensorflow不能指定padding个数. kernel_size=1, stride=1, SAME=VALID

        # 2nd Memory block
        # 1st Recursive block: bn + relu + conv; bn + relu + conv; skip connection
        c_in = conv_transition_01
        for _ in range(2):
            conv02_01 = res_mod_layers(c_in, num_filters=num_filters, kernel_size=3, strides=[1, 1], 
                padding="SAME", is_training=is_training)
            c_in = conv02_01
        # skip connection, conv_transition_01 + c_in
        # c_in = conv02_01
        eltwise02_01 = conv_transition_01 + c_in

        # 2nd Recursive block: bn + relu + conv; bn + relu + conv; skip connection
        c_in = eltwise02_01
        for _ in range(2):
            conv02_02 = res_mod_layers(c_in, num_filters=num_filters, kernel_size=3, strides=[1, 1], 
                padding="SAME", is_training=is_training)
            c_in = conv02_02
        # skip connection, eltwise02_01 + c_in
        # c_in = conv02_02
        eltwise02_02 = eltwise02_01 + c_in

        # 3rd Recursive block: bn + relu + conv; bn + relu + conv; skip connection
        c_in = eltwise02_02
        for _ in range(2):
            conv02_03 = res_mod_layers(c_in, num_filters=num_filters, kernel_size=3, strides=[1, 1], 
                padding="SAME", is_training=is_training)
            c_in = conv02_03
        # skip connection, eltwise02_02 + c_in
        # c_in = conv02_03
        eltwise02_03 = eltwise02_02 + c_in

        # 4th Recursive block: bn + relu + conv; bn + relu + conv; skip connection
        c_in = eltwise02_03
        for _ in range(2):
            conv02_04 = res_mod_layers(c_in, num_filters=num_filters, kernel_size=3, strides=[1, 1], 
                padding="SAME", is_training=is_training)
            c_in = conv02_04
        # skip connection, eltwise02_03 + c_in
        # c_in = conv02_04
        eltwise02_04 = eltwise02_03 + c_in

        # 5th Recursive block: bn + relu + conv; bn + relu + conv; skip connection
        c_in = eltwise02_04
        for _ in range(2):
            conv02_05 = res_mod_layers(c_in, num_filters=num_filters, kernel_size=3, strides=[1, 1], 
                padding="SAME", is_training=is_training)
            c_in = conv02_05
        # skip connection, eltwise02_04 + c_in
        # c_in = conv02_05
        eltwise02_05 = eltwise02_04 + c_in

        # 6th Recursive block: bn + relu + conv; bn + relu + conv; skip connection
        c_in = eltwise02_05
        for _ in range(2):
            conv02_06 = res_mod_layers(c_in, num_filters=num_filters, kernel_size=3, strides=[1, 1], 
                padding="SAME", is_training=is_training)
            c_in = conv02_06
        # skip connection, eltwise02_05 + c_in
        # c_in = conv02_06
        eltwise02_06 = eltwise02_05 + c_in

        # concat
        concat02 = tf.concat([conv1, conv_transition_01, eltwise02_01, eltwise02_02, eltwise02_03, eltwise02_04, 
            eltwise02_05, eltwise02_06], axis=3)

        conv_transition_02 = res_mod_layers(concat02, num_filters=num_filters, kernel_size=1, strides=[1, 1], 
            padding="SAME", is_training=is_training)
        # kernel_size=1, stride=1, SAME=VALID

        # 3rd Memory block
        # 1st Recursive block: bn + relu + conv; bn + relu + conv; skip connection
        c_in = conv_transition_02
        for _ in range(2):
            conv03_01 = res_mod_layers(c_in, num_filters=num_filters, kernel_size=3, strides=[1, 1], 
                padding="SAME", is_training=is_training)
            c_in = conv03_01
        # skip connection, conv_transition_02 + c_in
        # c_in = conv03_01
        eltwise03_01 = conv_transition_02 + c_in

        # 2nd Recursive block: bn + relu + conv; bn + relu + conv; skip connection
        c_in = eltwise03_01
        for _ in range(2):
            conv03_02 = res_mod_layers(c_in, num_filters=num_filters, kernel_size=3, strides=[1, 1], 
                padding="SAME", is_training=is_training)
            c_in = conv03_02
        # skip connection, eltwise03_01 + c_in
        # c_in = conv03_02
        eltwise03_02 = eltwise03_01 + c_in

        # 3rd Recursive block: bn + relu + conv; bn + relu + conv; skip connection
        c_in = eltwise03_02
        for _ in range(2):
            conv03_03 = res_mod_layers(c_in, num_filters=num_filters, kernel_size=3, strides=[1, 1], 
                padding="SAME", is_training=is_training)
            c_in = conv03_03
        # skip connection, eltwise03_02 + c_in
        # c_in = conv03_03
        eltwise03_03 = eltwise03_02 + c_in

        # 4th Recursive block: bn + relu + conv; bn + relu + conv; skip connection
        c_in = eltwise03_03
        for _ in range(2):
            conv03_04 = res_mod_layers(c_in, num_filters=num_filters, kernel_size=3, strides=[1, 1], 
                padding="SAME", is_training=is_training)
            c_in = conv03_04
        # skip connection, eltwise03_03 + c_in
        # c_in = conv03_04
        eltwise03_04 = eltwise03_03 + c_in

        # 5th Recursive block: bn + relu + conv; bn + relu + conv; skip connection
        c_in = eltwise03_04
        for _ in range(2):
            conv03_05 = res_mod_layers(c_in, num_filters=num_filters, kernel_size=3, strides=[1, 1], 
                padding="SAME", is_training=is_training)
            c_in = conv03_05
        # skip connection, eltwise03_04 + c_in
        # c_in = conv03_05
        eltwise03_05 = eltwise03_04 + c_in

        # 6th Recursive block: bn + relu + conv; bn + relu + conv; skip connection
        c_in = eltwise03_05
        for _ in range(2):
            conv03_06 = res_mod_layers(c_in, num_filters=num_filters, kernel_size=3, strides=[1, 1], 
                padding="SAME", is_training=is_training)
            c_in = conv03_06
        # skip connection, eltwise03_05 + c_in
        # c_in = conv03_06
        eltwise03_06 = eltwise03_05 + c_in

        # concat
        concat03 = tf.concat([conv1, conv_transition_01, conv_transition_02, eltwise03_01, eltwise03_02, 
            eltwise03_03, eltwise03_04, eltwise03_05, eltwise03_06], axis=3)

        conv_transition_03 = res_mod_layers(concat03, num_filters=num_filters, kernel_size=1, strides=[1, 1], 
            padding="SAME", is_training=is_training)
        # kernel_size=1, stride=1, SAME=VALID

        # 4th Memory block
        # 1st Recursive block: bn + relu + conv; bn + relu + conv; skip connection
        c_in = conv_transition_03
        for _ in range(2):
            conv04_01 = res_mod_layers(c_in, num_filters=num_filters, kernel_size=3, strides=[1, 1], 
                padding="SAME", is_training=is_training)
            c_in = conv04_01
        # skip connection, conv_transition_03 + c_in
        # c_in = conv04_01
        eltwise04_01 = conv_transition_03 + c_in

        # 2nd Recursive block: bn + relu + conv; bn + relu + conv; skip connection
        c_in = eltwise04_01
        for _ in range(2):
            conv04_02 = res_mod_layers(c_in, num_filters=num_filters, kernel_size=3, strides=[1, 1], 
                padding="SAME", is_training=is_training)
            c_in = conv04_02
        # skip connection, eltwise04_01 + c_in
        # c_in = conv04_02
        eltwise04_02 = eltwise04_01 + c_in

        # 3rd Recursive block: bn + relu + conv; bn + relu + conv; skip connection
        c_in = eltwise04_02
        for _ in range(2):
            conv04_03 = res_mod_layers(c_in, num_filters=num_filters, kernel_size=3, strides=[1, 1], 
                padding="SAME", is_training=is_training)
            c_in = conv04_03
        # skip connection, eltwise04_02 + c_in
        # c_in = conv04_03
        eltwise04_03 = eltwise04_02 + c_in

        # 4th Recursive block: bn + relu + conv; bn + relu + conv; skip connection
        c_in = eltwise04_03
        for _ in range(2):
            conv04_04 = res_mod_layers(c_in, num_filters=num_filters, kernel_size=3, strides=[1, 1], 
                padding="SAME", is_training=is_training)
            c_in = conv04_04
        # skip connection, eltwise04_03 + c_in
        # c_in = conv04_04
        eltwise04_04 = eltwise04_03 + c_in

        # 5th Recursive block: bn + relu + conv; bn + relu + conv; skip connection
        c_in = eltwise04_04
        for _ in range(2):
            conv04_05 = res_mod_layers(c_in, num_filters=num_filters, kernel_size=3, strides=[1, 1], 
                padding="SAME", is_training=is_training)
            c_in = conv04_05
        # skip connection, eltwise04_04 + c_in
        # c_in = conv04_05
        eltwise04_05 = eltwise04_04 + c_in

        # 6th Recursive block: bn + relu + conv; bn + relu + conv; skip connection
        c_in = eltwise04_05
        for _ in range(2):
            conv04_06 = res_mod_layers(c_in, num_filters=num_filters, kernel_size=3, strides=[1, 1], 
                padding="SAME", is_training=is_training)
            c_in = conv04_06
        # skip connection, eltwise04_05 + c_in
        # c_in = conv04_06
        eltwise04_06 = eltwise04_05 + c_in

        # concat
        concat04 = tf.concat([conv1, conv_transition_01, conv_transition_02, conv_transition_03, eltwise04_01, 
            eltwise04_02, eltwise04_03, eltwise04_04, eltwise04_05, eltwise04_06], axis=3)

        conv_transition_04 = res_mod_layers(concat04, num_filters=num_filters, kernel_size=1, strides=[1, 1], 
            padding="SAME", is_training=is_training)
        # kernel_size=1, stride=1, SAME=VALID

        # 5th Memory block
        # 1st Recursive block: bn + relu + conv; bn + relu + conv; skip connection
        c_in = conv_transition_04
        for _ in range(2):
            conv05_01 = res_mod_layers(c_in, num_filters=num_filters, kernel_size=3, strides=[1, 1], 
                padding="SAME", is_training=is_training)
            c_in = conv05_01
        # skip connection, conv_transition_04 + c_in
        # c_in = conv05_01
        eltwise05_01 = conv_transition_04 + c_in

        # 2nd Recursive block: bn + relu + conv; bn + relu + conv; skip connection
        c_in = eltwise05_01
        for _ in range(2):
            conv05_02 = res_mod_layers(c_in, num_filters=num_filters, kernel_size=3, strides=[1, 1], 
                padding="SAME", is_training=is_training)
            c_in = conv05_02
        # skip connection, eltwise05_01 + c_in
        # c_in = conv05_02
        eltwise05_02 = eltwise05_01 + c_in

        # 3rd Recursive block: bn + relu + conv; bn + relu + conv; skip connection
        c_in = eltwise05_02
        for _ in range(2):
            conv05_03 = res_mod_layers(c_in, num_filters=num_filters, kernel_size=3, strides=[1, 1], 
                padding="SAME", is_training=is_training)
            c_in = conv05_03
        # skip connection, eltwise05_02 + c_in
        # c_in = conv05_03
        eltwise05_03 = eltwise05_02 + c_in

        # 4th Recursive block: bn + relu + conv; bn + relu + conv; skip connection
        c_in = eltwise05_03
        for _ in range(2):
            conv05_04 = res_mod_layers(c_in, num_filters=num_filters, kernel_size=3, strides=[1, 1], 
                padding="SAME", is_training=is_training)
            c_in = conv05_04
        # skip connection, eltwise05_03 + c_in
        # c_in = conv05_04
        eltwise05_04 = eltwise05_03 + c_in

        # 5th Recursive block: bn + relu + conv; bn + relu + conv; skip connection
        c_in = eltwise05_04
        for _ in range(2):
            conv05_05 = res_mod_layers(c_in, num_filters=num_filters, kernel_size=3, strides=[1, 1], 
                padding="SAME", is_training=is_training)
            c_in = conv05_05
        # skip connection, eltwise05_04 + c_in
        # c_in = conv05_05
        eltwise05_05 = eltwise05_04 + c_in

        # 6th Recursive block: bn + relu + conv; bn + relu + conv; skip connection
        c_in = eltwise05_05
        for _ in range(2):
            conv05_06 = res_mod_layers(c_in, num_filters=num_filters, kernel_size=3, strides=[1, 1], 
                padding="SAME", is_training=is_training)
            c_in = conv05_06
        # skip connection, eltwise05_05 + c_in
        # c_in = conv05_06
        eltwise05_06 = eltwise05_05 + c_in

        # concat
        concat05 = tf.concat([conv1, conv_transition_01, conv_transition_02, conv_transition_03, 
            conv_transition_04, eltwise05_01, eltwise05_02, eltwise05_03, eltwise05_04, eltwise05_05, 
            eltwise05_06], axis=3)

        conv_transition_05 = res_mod_layers(concat05, num_filters=num_filters, kernel_size=1, strides=[1, 1], 
            padding="SAME", is_training=is_training)
        # kernel_size=1, stride=1, SAME=VALID

        # 6th Memory block
        # 1st Recursive block: bn + relu + conv; bn + relu + conv; skip connection
        c_in = conv_transition_05
        for _ in range(2):
            conv06_01 = res_mod_layers(c_in, num_filters=num_filters, kernel_size=3, strides=[1, 1], 
                padding="SAME", is_training=is_training)
            c_in = conv06_01
        # skip connection, conv_transition_05 + c_in
        # c_in = conv06_01
        eltwise06_01 = conv_transition_05 + c_in

        # 2nd Recursive block: bn + relu + conv; bn + relu + conv; skip connection
        c_in = eltwise06_01
        for _ in range(2):
            conv06_02 = res_mod_layers(c_in, num_filters=num_filters, kernel_size=3, strides=[1, 1], 
                padding="SAME", is_training=is_training)
            c_in = conv06_02
        # skip connection, eltwise06_01 + c_in
        # c_in = conv06_02
        eltwise06_02 = eltwise06_01 + c_in

        # 3rd Recursive block: bn + relu + conv; bn + relu + conv; skip connection
        c_in = eltwise06_02
        for _ in range(2):
            conv06_03 = res_mod_layers(c_in, num_filters=num_filters, kernel_size=3, strides=[1, 1], 
                padding="SAME", is_training=is_training)
            c_in = conv06_03
        # skip connection, eltwise06_02 + c_in
        # c_in = conv06_03
        eltwise06_03 = eltwise06_02 + c_in

        # 4th Recursive block: bn + relu + conv; bn + relu + conv; skip connection
        c_in = eltwise06_03
        for _ in range(2):
            conv06_04 = res_mod_layers(c_in, num_filters=num_filters, kernel_size=3, strides=[1, 1], 
                padding="SAME", is_training=is_training)
            c_in = conv06_04
        # skip connection, eltwise06_03 + c_in
        # c_in = conv06_04
        eltwise06_04 = eltwise06_03 + c_in

        # 5th Recursive block: bn + relu + conv; bn + relu + conv; skip connection
        c_in = eltwise06_04
        for _ in range(2):
            conv06_05 = res_mod_layers(c_in, num_filters=num_filters, kernel_size=3, strides=[1, 1], 
                padding="SAME", is_training=is_training)
            c_in = conv06_05
        # skip connection, eltwise06_04 + c_in
        # c_in = conv06_05
        eltwise06_05 = eltwise06_04 + c_in

        # 6th Recursive block: bn + relu + conv; bn + relu + conv; skip connection
        c_in = eltwise06_05
        for _ in range(2):
            conv06_06 = res_mod_layers(c_in, num_filters=num_filters, kernel_size=3, strides=[1, 1], 
                padding="SAME", is_training=is_training)
            c_in = conv06_06
        # skip connection, eltwise06_05 + c_in
        # c_in = conv06_06
        eltwise06_06 = eltwise06_05 + c_in

        # concat
        concat06 = tf.concat([conv1, conv_transition_01, conv_transition_02, conv_transition_03, 
            conv_transition_04, conv_transition_05, eltwise06_01, eltwise06_02, eltwise06_03, eltwise06_04, 
            eltwise06_05, eltwise06_06], axis=3)

        conv_transition_06 = res_mod_layers(concat06, num_filters=num_filters, kernel_size=1, strides=[1, 1], 
            padding="SAME", is_training=is_training)
        # kernel_size=1, stride=1, SAME=VALID

        conv_end_01 = res_mod_layers(conv_transition_01, num_filters=image_c, kernel_size=3, strides=[1, 1], 
            padding="SAME", is_training=is_training)
        HR_recovery_01 = conv_end_01 + noisy_data

        if is_training:
            # loss_01, L2 loss
            loss_01 = tf.reduce_mean(tf.squared_difference(HR_recovery_01, clean_data))
        # scale, In Tensorflow, Scale use tf.matmul(x, W) + b don't satisfy. tf.matmul shape is same.
        # alpha and beta is Variable, shape=(channel_num,).
        alpha1 = tf.get_variable('alpha1', [image_c], initializer=tf.contrib.layers.xavier_initializer(), 
            trainable=True)
        weight_output_end_01 = alpha1 * HR_recovery_01

        conv_end_02 = res_mod_layers(conv_transition_02, num_filters=image_c, kernel_size=3, strides=[1, 1], 
            padding="SAME", is_training=is_training)
        HR_recovery_02 = conv_end_02 + noisy_data
        if is_training:
            # loss_02, L2 loss
            loss_02 = tf.reduce_mean(tf.squared_difference(HR_recovery_02, clean_data))
        # scale
        alpha2 = tf.get_variable('alpha2', [image_c], initializer=tf.contrib.layers.xavier_initializer(), 
            trainable=True)
        weight_output_end_02 = alpha2 * HR_recovery_02

        conv_end_03 = res_mod_layers(conv_transition_03, num_filters=image_c, kernel_size=3, strides=[1, 1], 
            padding="SAME", is_training=is_training)
        HR_recovery_03 = conv_end_03 + noisy_data
        if is_training:
            # loss_03, L2 loss
            loss_03 = tf.reduce_mean(tf.squared_difference(HR_recovery_03, clean_data))
        # scale
        alpha3 = tf.get_variable('alpha3', [image_c], initializer=tf.contrib.layers.xavier_initializer(), 
            trainable=True)
        weight_output_end_03 = alpha3 * HR_recovery_03

        conv_end_04 = res_mod_layers(conv_transition_04, num_filters=image_c, kernel_size=3, strides=[1, 1], 
            padding="SAME", is_training=is_training)
        HR_recovery_04 = conv_end_04 + noisy_data
        if is_training:
            # loss_04, L2 loss
            loss_04 = tf.reduce_mean(tf.squared_difference(HR_recovery_04, clean_data))
        # scale
        alpha4 = tf.get_variable('alpha4', [image_c], initializer=tf.contrib.layers.xavier_initializer(), 
            trainable=True)
        weight_output_end_04 = alpha4 * HR_recovery_04

        conv_end_05 = res_mod_layers(conv_transition_05, num_filters=image_c, kernel_size=3, strides=[1, 1], 
            padding="SAME", is_training=is_training)
        HR_recovery_05 = conv_end_05 + noisy_data
        if is_training:
            # loss_05, L2 loss
            loss_05 = tf.reduce_mean(tf.squared_difference(HR_recovery_05, clean_data))
        # scale
        alpha5 = tf.get_variable('alpha5', [image_c], initializer=tf.contrib.layers.xavier_initializer(), 
            trainable=True)
        weight_output_end_05 = alpha5 * HR_recovery_05

        conv_end_06 = res_mod_layers(conv_transition_06, num_filters=image_c, kernel_size=3, strides=[1, 1], 
            padding="SAME", is_training=is_training)
        HR_recovery_06 = conv_end_06 + noisy_data
        if is_training:
            # loss_06, L2 loss
            loss_06 = tf.reduce_mean(tf.squared_difference(HR_recovery_06, clean_data))
        # scale
        alpha6 = tf.get_variable('alpha6', [image_c], initializer=tf.contrib.layers.xavier_initializer(), 
            trainable=True)
        weight_output_end_06 = alpha6 * HR_recovery_06

        # HR_recovery, MemNet output.
        HR_recovery = weight_output_end_01 + weight_output_end_02 + weight_output_end_03 + weight_output_end_04 + \
            weight_output_end_05 + weight_output_end_06

        if is_training:
            # loss_07
            loss_07 = tf.reduce_mean(tf.squared_difference(HR_recovery, clean_data))

            # loss
            loss = loss_01 + loss_02 + loss_03 + loss_04 + loss_05 + loss_06 + loss_07
            # In every loss layer of caffe, loss_weight is 1, so total loss can be that. 
            # every subloss  coefficient is 1.
            return HR_recovery, loss
        else:
            return HR_recovery
