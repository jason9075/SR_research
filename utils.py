import os

import tensorflow as tf


def random_flip(input, decision):
    f1 = tf.identity(input)
    f2 = tf.image.flip_left_right(input)
    output = tf.cond(tf.less(decision, 0.5), lambda: f2, lambda: f1)

    return output


def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2


def preprocessLR(image):
    with tf.name_scope("preprocessLR"):
        return tf.identity(image)


def deprocessLR(image):
    with tf.name_scope("deprocessLR"):
        return tf.identity(image)


def save_images(contents, filename='test.jpg'):
    image_dir = os.path.join("images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    out_path = os.path.join(image_dir, filename)
    with open(out_path, "wb") as f:
        f.write(contents)
