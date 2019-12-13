from backend.arch.basenet import BaseNet
from backend.layers import *


class Generator(BaseNet):
    def __init__(self, input_layer, is_train, reuse=False):
        super().__init__(input_layer, is_train)  # input = n
        num_res_block = 16
        ch = 64
        output_channel = 3
        with tf.variable_scope('generator', reuse=reuse):
            with tf.variable_scope('input_stage'):
                net = tf.layers.conv2d(input_layer, ch, 9, strides=1, padding='same', kernel_initializer=WEIGHT_INIT,
                                       name='conv')
                net = prelu_tf(net)

                stage1_output = net

            for i in range(num_res_block):
                net = p_resblock(net, ch, is_train, use_bias=False, scope='res_%d' % (i + 1))

            with tf.variable_scope('res_output'):
                net = tf.layers.conv2d(net, ch, 3, strides=1, use_bias=False, padding='same',
                                       kernel_initializer=WEIGHT_INIT, name='conv')

            net = net + stage1_output

            with tf.variable_scope('subpixel_stage_1'):
                net = tf.layers.conv2d(net, ch * 4, 3, strides=1, padding='same', kernel_initializer=WEIGHT_INIT,
                                       name='conv')
                net = pixel_shuffler(net)
                net = prelu_tf(net)

            with tf.variable_scope('subpixel_stage_2'):
                net = tf.layers.conv2d(net, ch * 4, 3, strides=1, padding='same', kernel_initializer=WEIGHT_INIT,
                                       name='conv')
                net = pixel_shuffler(net)
                net = prelu_tf(net)

            with tf.variable_scope('output_stage'):
                net = tf.layers.conv2d(net, output_channel, 9, strides=1, padding='same',
                                       kernel_initializer=WEIGHT_INIT, name='conv')

            self.arch_output = net


class Discriminator(BaseNet):
    def __init__(self, input_layer, is_train):
        super().__init__(input_layer, is_train)  # input = n
        ch = 64

        with tf.variable_scope('discriminator'):
            with tf.variable_scope('input_stage'):
                net = tf.layers.conv2d(input_layer, ch, 3, strides=1, name='conv')
                net = tf.nn.leaky_relu(net)

            net = discriminator_block(net, 3, ch, 2, is_train, name='dis_block_1')
            net = discriminator_block(net, 3, ch * 2, 1, is_train, name='dis_block_2')
            net = discriminator_block(net, 3, ch * 2, 2, is_train, name='dis_block_3')
            net = discriminator_block(net, 3, ch * 4, 1, is_train, name='dis_block_4')
            net = discriminator_block(net, 3, ch * 4, 2, is_train, name='dis_block_5')
            net = discriminator_block(net, 3, ch * 8, 1, is_train, name='dis_block_6')
            net = discriminator_block(net, 3, ch * 8, 2, is_train, name='dis_block_7')

            with tf.variable_scope('dense_stage_1'):
                net = flatten(net)
                net = dense(net, 1024, bn=False)
                net = tf.nn.leaky_relu(net)

            with tf.variable_scope('dense_stage_2'):
                net = dense(net, 1, bn=False)
                net = tf.nn.sigmoid(net)

            self.arch_output = net
