import glob
import os
import random

import cv2
import numpy as np
import tensorflow as tf

from backend.arch.resnet import Generator
from hr2lr import gen_dataset
from utils import preprocessLR, preprocess, deprocess, save_images

BATCH_SIZE = 20
INPUT_SIZE = (56, 56)
CAPACITY = 4000
THREAD = 4
EPOCH = 500
SAVER_MAX_KEEP = 10

SHOW_INFO_INTERVAL = 100
SAVE_MODEL_INTERVAL = 1000
VALIDATE_INTERVAL = 500


def purge():
    for f in glob.glob('valid_output/*.jpg'):
        os.remove(f)


def _parse_lr(image_path):
    file_contents = tf.read_file(image_path)
    image = tf.image.decode_image(file_contents, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image.set_shape((INPUT_SIZE[0], INPUT_SIZE[0], 3))

    return image


def _parse_hr(image_path):
    file_contents = tf.read_file(image_path)
    image = tf.image.decode_image(file_contents, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image.set_shape((INPUT_SIZE[0] * 4, INPUT_SIZE[0] * 4, 3))

    return image


def main():
    purge()

    global_step = tf.train.get_or_create_global_step()
    input_node = tf.placeholder(
        name='input_images',
        shape=[1, None, None, 3],
        dtype=tf.float32)
    target_node = tf.placeholder(
        name='target_images',
        shape=[1, None, None, 3],
        dtype=tf.float32)
    is_train = tf.placeholder_with_default(False, (), name='is_training')
    lr_image_paths = tf.placeholder(tf.string, shape=(None,), name='lr_image_paths')
    hr_image_paths = tf.placeholder(tf.string, shape=(None,), name='hr_image_paths')

    lr_valid_image = cv2.imread('dataset/lr_valid/jason.jpg')
    lr_valid_image = cv2.cvtColor(lr_valid_image, cv2.COLOR_BGR2RGB)
    lr_valid_image = np.expand_dims(lr_valid_image, axis=0)
    lr_valid_image = lr_valid_image / 255
    hr_valid_image = cv2.imread('dataset/hr_valid/jason.jpg')
    hr_valid_image = cv2.cvtColor(hr_valid_image, cv2.COLOR_BGR2RGB)
    hr_valid_image = (hr_valid_image / 255) * 2 - 1
    hr_valid_image = np.expand_dims(hr_valid_image, axis=0)

    with tf.variable_scope('load_image'):

        input_image_lr = tf.map_fn(_parse_lr, lr_image_paths, dtype=tf.float32)
        input_image_hr = tf.map_fn(_parse_hr, hr_image_paths, dtype=tf.float32)

        # Normalize the low resolution image to [0, 1], high resolution to [-1, 1]
        inputs_batch = preprocessLR(input_image_lr)
        targets_batch = preprocess(input_image_hr)

    with tf.name_scope('train'):
        net_train = Generator(inputs_batch, is_train)
        gen_output = net_train.arch_output
        gen_output.set_shape([BATCH_SIZE, INPUT_SIZE[0] * 4, INPUT_SIZE[1] * 4, 3])

    with tf.name_scope('valid'):
        net_valid = Generator(input_node, tf.constant(False), reuse=True)
        gen_valid = net_valid.arch_output
        gen_valid.set_shape([1, INPUT_SIZE[0] * 4, INPUT_SIZE[1] * 4, 3])
        gen_valid = deprocess(gen_valid)

        valid_diff = gen_valid - target_node
        valid_loss = tf.reduce_mean(tf.reduce_sum(tf.square(valid_diff), axis=[3]))
        converted_outputs = tf.image.convert_image_dtype(gen_valid, dtype=tf.uint8, saturate=True)
        outputs_node = tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name='output_pngs')

    extracted_feature_gen = gen_output
    extracted_feature_target = targets_batch

    with tf.variable_scope('generator_loss'):
        # Content loss
        with tf.variable_scope('content_loss'):
            diff = extracted_feature_gen - extracted_feature_target
            content_loss = tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis=[3]))

        gen_loss = content_loss

    with tf.variable_scope('generator_train'):
        # Need to wait discriminator to perform train step
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            gen_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
            gen_optimizer = tf.train.AdamOptimizer(0.01)
            gen_grads_and_vars = gen_optimizer.compute_gradients(gen_loss, gen_tvars)
            gen_train = gen_optimizer.apply_gradients(gen_grads_and_vars, global_step=global_step)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    saver = tf.train.Saver(max_to_keep=SAVER_MAX_KEEP)

    sv = tf.train.Supervisor(logdir='./save', save_summaries_secs=0, saver=None)
    step = 1
    with sv.managed_session(config=config) as sess:
        # Performing the training
        for epoch_idx in range(0, EPOCH):
            gen_dataset()
            lr_paths = glob.glob('dataset/lr/*.jpg')
            hr_paths = glob.glob('dataset/hr/*.jpg')
            steps_per_epoch = len(lr_paths) // BATCH_SIZE
            all_paths = list(zip(lr_paths, hr_paths))
            lr_paths = [lr for lr, _ in all_paths]
            hr_paths = [hr for _, hr in all_paths]
            random.shuffle(all_paths)
            epoch_step = 0
            for idx in range(0, len(lr_paths), BATCH_SIZE):
                lr_batch = lr_paths[idx: idx + BATCH_SIZE]
                hr_batch = hr_paths[idx: idx + BATCH_SIZE]
                fetches = {
                    "global_step": global_step,
                    "train": gen_train,
                    "gen_output": gen_output,
                    "gen_loss": gen_loss
                }
                feed_dict = {
                    is_train: True,
                    lr_image_paths: lr_batch,
                    hr_image_paths: hr_batch,
                }

                results = sess.run(fetches, feed_dict=feed_dict)

                if step % SHOW_INFO_INTERVAL == 0:
                    loss = results['gen_loss']
                    print('[%d][%d/%d] step:%d, loss:%.2f' % (epoch_idx, epoch_step, steps_per_epoch, step, loss))

                if step % VALIDATE_INTERVAL == 0:
                    fetches = {
                        "valid_loss": valid_loss,
                        "outputs_node": outputs_node,
                        "gen_valid": gen_valid
                    }
                    feed_dict = {
                        is_train: False,
                        input_node: lr_valid_image,
                        target_node: hr_valid_image,
                    }
                    val_results = sess.run(fetches, feed_dict=feed_dict)
                    val_loss = val_results['valid_loss']
                    print('valid loss: %.2f' % val_loss)
                    save_images(val_results['outputs_node'][0],
                                filename='step_%d_loss_%.2f.jpg' % (step, val_loss))

                if step % SAVE_MODEL_INTERVAL == 0:
                    print('saving ckpt.')
                    filename = f'step_{step}.ckpt'
                    saver.save(sess, f'model_out/{filename}')

                epoch_step += 1
                step += 1


if __name__ == '__main__':
    main()
