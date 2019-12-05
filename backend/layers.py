import platform

import numpy as np
import tensorflow as tf

if platform.system() == 'Linux':
    DEVICE = '/gpu:0'
    CUDNN_ON_GPU = True
else:
    DEVICE = '/cpu:0'
    CUDNN_ON_GPU = False
D_TYPE = tf.float32
BIAS_INIT = tf.constant_initializer(0.0)
ONE_INIT = tf.constant_initializer(1.0)
WEIGHT_INIT = tf.contrib.layers.xavier_initializer()
# WEIGHT_INIT = tf.truncated_normal_initializer(stddev=0.1)
REGULARIZER = tf.contrib.layers.l2_regularizer(0.01)
ACT_FUNC = tf.nn.relu6


def resblock(x_init, ch, is_train=False, use_bias=True, down_sample=False, scope='resblock'):
    with tf.variable_scope(scope):

        if down_sample:
            shortcut = tf.layers.conv2d(x_init, ch, kernel_size=1, strides=2, padding='same', use_bias=use_bias,
                                        name='shortcut')
            shortcut = tf.layers.batch_normalization(shortcut, training=is_train, name='bn_shortcut')
            x = tf.layers.conv2d(x_init, ch, kernel_size=3, strides=2, padding='same', use_bias=use_bias, name='conv_0')

        else:
            shortcut = x_init
            x = tf.layers.conv2d(x_init, ch, kernel_size=3, strides=1, padding='same', use_bias=use_bias, name='conv_0')

        x = tf.layers.batch_normalization(x, training=is_train, name='bn_0')
        x = tf.nn.relu(x)

        x = tf.layers.conv2d(x, ch, kernel_size=3, strides=1, padding='same', use_bias=use_bias, name='conv_1')
        x = tf.layers.batch_normalization(x, training=is_train, name='bn_1')

        x = x + shortcut

        return tf.nn.relu(x)


def bottle_resblock(x, ch, is_train=False, use_bias=True, down_sample=False, scope='bottle_resblock'):
    with tf.variable_scope(scope):

        if down_sample:
            strides = 2
        else:
            strides = 1

        shortcut = tf.layers.conv2d(x, ch * 4, kernel_size=1, strides=strides, padding='same', use_bias=use_bias,
                                    name='shortcut')
        shortcut = tf.layers.batch_normalization(shortcut, training=is_train, name='bn_shortcut')

        x = tf.layers.conv2d(x, ch, kernel_size=1, strides=1, padding='same', use_bias=use_bias, name='conv_0')
        x = tf.layers.batch_normalization(x, training=is_train, name='bn_0')
        x = tf.nn.relu(x)

        x = tf.layers.conv2d(x, ch, kernel_size=3, strides=strides, padding='same', use_bias=use_bias, name='conv_1')
        x = tf.layers.batch_normalization(x, training=is_train, name='bn_1')
        x = tf.nn.relu(x)

        x = tf.layers.conv2d(x, ch * 4, kernel_size=1, strides=1, padding='same', use_bias=use_bias, name='conv_2')
        x = tf.layers.batch_normalization(x, training=is_train, name='bn_2')

        x = x + shortcut

        return tf.nn.relu(x)


def conv2d(x,
           kernel,
           num_filter,
           stride,
           bn=True,
           act=tf.identity,
           name='conv',
           padding='SAME',
           is_train=False):
    stride = [1, stride[0], stride[1], 1]
    pre_channel = int(x.get_shape()[-1])
    shape = [kernel[0], kernel[1], pre_channel, num_filter]
    with tf.device(DEVICE):
        with tf.variable_scope(name):
            w = tf.get_variable(
                name='w_conv',
                shape=shape,
                initializer=WEIGHT_INIT,
                dtype=D_TYPE,
                regularizer=REGULARIZER)
            out = tf.nn.conv2d(
                x, w, stride, padding, use_cudnn_on_gpu=CUDNN_ON_GPU)
            if bn:
                out = tf.layers.batch_normalization(
                    out, name='bn', training=is_train)
            return act(out)


def conv2d_bias(x,
                kernel,
                num_filter,
                stride,
                bn=True,
                act=tf.identity,
                name='conv',
                padding='SAME',
                is_train=False):
    stride = [1, stride[0], stride[1], 1]
    pre_channel = int(x.get_shape()[-1])
    shape = [kernel[0], kernel[1], pre_channel, num_filter]
    with tf.device(DEVICE):
        with tf.variable_scope(name):
            w = tf.get_variable(
                name='w_conv',
                shape=shape,
                initializer=WEIGHT_INIT,
                dtype=D_TYPE,
                regularizer=REGULARIZER)
            b = tf.get_variable(
                name='b_conv',
                shape=[num_filter],
                initializer=WEIGHT_INIT,
                dtype=D_TYPE)
            out = tf.nn.conv2d(
                x, w, stride, padding, use_cudnn_on_gpu=CUDNN_ON_GPU)
            out = tf.nn.bias_add(out, b)
            if bn:
                out = tf.layers.batch_normalization(
                    out, name='bn', training=is_train)
            return act(out)


def depthwise_conv2d(x,
                     kernel,
                     stride,
                     bn=True,
                     act=tf.identity,
                     name='conv_dw',
                     is_train=False):
    stride = [1, stride[0], stride[1], 1]
    num_filter = x.shape[-1]
    kernel_shape = [kernel[0], kernel[1], num_filter, 1]

    with tf.device(DEVICE):
        with tf.variable_scope(name):
            w = tf.get_variable(
                name='w_dw',
                shape=kernel_shape,
                initializer=WEIGHT_INIT,
                dtype=D_TYPE,
                regularizer=REGULARIZER)
            out = tf.nn.depthwise_conv2d(
                x, w, stride, name='d_conv', padding='SAME')
            if bn:
                out = tf.layers.batch_normalization(
                    out, name='bn', training=is_train)
            return act(out)


def group_conv2d(x,
                 kernel,
                 num_filter,
                 stride,
                 num_groups=1,
                 bn=True,
                 act=tf.identity,
                 name='group_conv',
                 padding='SAME',
                 is_train=False):
    pre_channel = int(x.get_shape()[-1])
    shape = [kernel[0], kernel[1], pre_channel // num_groups, num_filter]

    output_list = []

    with tf.device(DEVICE):
        with tf.variable_scope(name):
            w = tf.get_variable(
                name='w_conv',
                shape=shape,
                initializer=WEIGHT_INIT,
                dtype=D_TYPE,
                regularizer=REGULARIZER)
            input_list = tf.split(x, num_groups, axis=-1)
            filter_list = tf.split(w, num_groups, axis=-1)

            for conv_idx, (input_tensor, filter_tensor) in enumerate(
                    zip(input_list, filter_list)):
                conv = tf.nn.convolution(
                    input_tensor, filter_tensor, padding, stride, name=name)
                output_list.append(conv)
            out = tf.concat(output_list, axis=-1)
            if bn:
                out = tf.layers.batch_normalization(
                    out, name='bn', training=is_train)
            return act(out)


def flatten(x, name='flatten'):
    all_dims_exc_first = np.prod([v.value for v in x.get_shape()[1:]])
    out = tf.reshape(x, [-1, all_dims_exc_first], name=name)
    return out


def dense(x,
          num_classes,
          bn=True,
          act=tf.identity,
          name='dense',
          is_train=False):
    n_in = int(x.get_shape()[-1])
    with tf.device(DEVICE):
        with tf.variable_scope(name):
            w = tf.get_variable(
                name='w_dense',
                shape=[n_in, num_classes],
                initializer=WEIGHT_INIT,
                dtype=D_TYPE,
                regularizer=REGULARIZER)
            b = tf.get_variable(
                name='b_dense',
                shape=num_classes,
                initializer=BIAS_INIT,
                dtype=D_TYPE)
            out = tf.nn.bias_add(tf.matmul(x, w), b)
            if bn:
                out = tf.layers.batch_normalization(
                    out, training=is_train, gamma_initializer=ONE_INIT)
            return act(out)


def inv_res_block(x,
                  expansion_ratio,
                  output_dim,
                  stride,
                  name='inv_res_block',
                  is_train=False,
                  shortcut=True):
    in_dim = int(x.get_shape()[-1])
    with tf.variable_scope(name):
        # pw
        bottleneck_dim = round(expansion_ratio * in_dim)
        net = conv2d(
            x, (1, 1),
            bottleneck_dim,
            stride,
            act=ACT_FUNC,
            name='pw',
            is_train=is_train)
        # dw
        net = depthwise_conv2d(
            net, (3, 3), stride, act=ACT_FUNC, name='dw', is_train=is_train)
        # pw & linear
        net = conv2d(
            net, (1, 1),
            output_dim,
            stride,
            act=tf.identity,
            name='pw_linear',
            is_train=is_train)

        # element wise add, only for stride==1
        if shortcut and stride == (1, 1):
            if in_dim != output_dim:
                ins = conv2d(
                    x, (1, 1),
                    output_dim, (1, 1),
                    bn=False,
                    act=tf.identity,
                    name='shortcut',
                    is_train=is_train)
                net = net + ins
            else:
                net = net + x

        return net


def global_avg(x):
    with tf.name_scope('global_avg'):
        net = tf.layers.average_pooling2d(x, x.get_shape()[1:-1], 1)
        return net


def fire_module(x, s1, e1, e3, name="fire_module"):
    pre_channel = int(x.get_shape()[-1])
    strides = [1, 1, 1, 1]

    with tf.variable_scope(name):
        s1_w = tf.get_variable(
            name='s1_w',
            shape=[1, 1, pre_channel, s1],
            initializer=WEIGHT_INIT,
            dtype=D_TYPE,
            regularizer=REGULARIZER)
        s1_b = tf.get_variable(
            name='s1_b',
            shape=[s1],
            initializer=BIAS_INIT,
            dtype=D_TYPE)

        e1_w = tf.get_variable(
            name='e1_w',
            shape=[1, 1, s1, e1],
            initializer=WEIGHT_INIT,
            dtype=D_TYPE,
            regularizer=REGULARIZER)
        e1_b = tf.get_variable(
            name='e1_b',
            shape=[e1],
            initializer=BIAS_INIT,
            dtype=D_TYPE)

        e3_w = tf.get_variable(
            name='e3_w',
            shape=[3, 3, s1, e3],
            initializer=WEIGHT_INIT,
            dtype=D_TYPE,
            regularizer=REGULARIZER)
        e3_b = tf.get_variable(
            name='e3_b',
            shape=[e1],
            initializer=BIAS_INIT,
            dtype=D_TYPE)

        # squeeze layer
        squeeze_out = tf.nn.conv2d(
            x, s1_w, strides, padding="VALID", use_cudnn_on_gpu=CUDNN_ON_GPU)

        squeeze_out = tf.nn.relu(tf.nn.bias_add(squeeze_out, s1_b))

        # expand layer
        expand1 = tf.nn.conv2d(
            squeeze_out,
            e1_w,
            strides,
            padding="VALID",
            use_cudnn_on_gpu=CUDNN_ON_GPU)
        expand1 = tf.nn.relu(tf.nn.bias_add(expand1, e1_b))

        expand3 = tf.nn.conv2d(
            squeeze_out,
            e3_w,
            strides,
            padding="SAME",
            use_cudnn_on_gpu=CUDNN_ON_GPU)
        expand3 = tf.nn.relu(tf.nn.bias_add(expand3, e3_b))

        return tf.concat([expand1, expand3], axis=3)


def max_pool(x, filter_size, strides, padding='SAME', name='max_pool'):
    ksize = [1, filter_size[0], filter_size[1], 1]
    strides = [1, strides[0], strides[1], 1]
    return tf.nn.max_pool(
        x, ksize=ksize, strides=strides, padding=padding, name=name)


def avg_pool(x, filter_size, strides, padding='SAME', name='avg_pool'):
    ksize = [1, filter_size[0], filter_size[1], 1]
    strides = [1, strides[0], strides[1], 1]
    return tf.nn.avg_pool(
        x, ksize=ksize, strides=strides, padding=padding, name=name)


def prelu_tf(x, name='Prelu'):
    with tf.variable_scope(name):
        alphas = tf.get_variable('alpha', x.get_shape()[-1], initializer=tf.zeros_initializer(), dtype=tf.float32)
    pos = tf.nn.relu(x)
    neg = alphas * (x - abs(x)) * 0.5

    return pos + neg


def pixel_shuffler(x, scale=2):
    size = tf.shape(x)
    batch_size = size[0]
    h = size[1]
    w = size[2]
    c = x.get_shape().as_list()[-1]

    # Get the target channel size
    channel_target = c // (scale * scale)
    channel_factor = c // channel_target

    shape_1 = [batch_size, h, w, channel_factor // scale, channel_factor // scale]
    shape_2 = [batch_size, h * scale, w * scale, 1]

    # Reshape and transpose for periodic shuffling for each channel
    input_split = tf.split(x, channel_target, axis=3)
    output = tf.concat([phase_shift(x, shape_1, shape_2) for x in input_split], axis=3)

    return output


def p_resblock(x_init, ch, is_train=False, use_bias=True, down_sample=False, scope='resblock'):
    with tf.variable_scope(scope):

        if down_sample:
            shortcut = tf.layers.conv2d(x_init, ch, kernel_size=1, strides=2, padding='same',
                                        kernel_initializer=WEIGHT_INIT, use_bias=use_bias,
                                        name='shortcut')
            shortcut = tf.layers.batch_normalization(shortcut, training=is_train, name='bn_shortcut')
            x = tf.layers.conv2d(x_init, ch, kernel_size=3, strides=2, padding='same',
                                 kernel_initializer=WEIGHT_INIT, use_bias=use_bias, name='conv_0')

        else:
            shortcut = x_init
            x = tf.layers.conv2d(x_init, ch, kernel_size=3, strides=1, padding='same',
                                 kernel_initializer=WEIGHT_INIT, use_bias=use_bias, name='conv_0')

        x = tf.layers.batch_normalization(x, training=is_train, name='bn_0')
        x = prelu_tf(x)

        x = tf.layers.conv2d(x, ch, kernel_size=3, strides=1, padding='same',
                             kernel_initializer=WEIGHT_INIT, use_bias=use_bias, name='conv_1')
        x = tf.layers.batch_normalization(x, training=is_train, name='bn_1')

        return x + shortcut


def discriminator_block(x, kernel, num_filter, stride, is_train, name='dis_block'):
    with tf.variable_scope(name):
        x = conv2d(x, kernel, num_filter, stride, bn=True, is_train=is_train)
        x = tf.nn.leaky_relu(x)

        return x


def phase_shift(x, shape_1, shape_2):
    x = tf.reshape(x, shape_1)
    x = tf.transpose(x, [0, 1, 3, 2, 4])

    return tf.reshape(x, shape_2)
