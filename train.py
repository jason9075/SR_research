import glob
import math

from backend.arch.resnet import Generator
import tensorflow as tf

from utils import preprocessLR, preprocess, random_flip, deprocess, save_images

BATCH_SIZE = 32
INPUT_SIZE = (56, 56)
CAPACITY = 4000
THREAD = 4
EPOCH = 100


def main():
    global_step = tf.train.get_or_create_global_step()
    # input_node = tf.placeholder(
    #     name='input_images',
    #     shape=[1, None, None, 3],
    #     dtype=tf.float32)
    # target_node = tf.placeholder(
    #     name='target_images',
    #     shape=[1, None, None, 3],
    #     dtype=tf.float32)
    is_train = tf.placeholder_with_default(False, (), name='is_training')

    lr_image_list = glob.glob('dataset/lr/*.jpg')
    hr_image_list = glob.glob('dataset/hr/*.jpg')
    lr_image_list_tensor = tf.convert_to_tensor(lr_image_list, dtype=tf.string)
    hr_image_list_tensor = tf.convert_to_tensor(hr_image_list, dtype=tf.string)

    with tf.variable_scope('load_image'):
        output = tf.train.slice_input_producer([lr_image_list_tensor, hr_image_list_tensor],
                                               shuffle=False, capacity=CAPACITY)

        image_lr = tf.read_file(output[0])
        image_hr = tf.read_file(output[1])
        input_image_lr = tf.image.decode_png(image_lr, channels=3)
        input_image_hr = tf.image.decode_png(image_hr, channels=3)
        input_image_lr = tf.image.convert_image_dtype(input_image_lr, dtype=tf.float32)
        input_image_hr = tf.image.convert_image_dtype(input_image_hr, dtype=tf.float32)

        assertion = tf.assert_equal(tf.shape(input_image_lr)[2], 3, message="image does not have 3 channels")
        with tf.control_dependencies([assertion]):
            input_image_lr = tf.identity(input_image_lr)
            input_image_hr = tf.identity(input_image_hr)

        # Normalize the low resolution image to [0, 1], high resolution to [-1, 1]
        a_image = preprocessLR(input_image_lr)
        b_image = preprocess(input_image_hr)

        inputs, targets = [a_image, b_image]

    with tf.name_scope('data_preprocessing'):
        with tf.variable_scope('random_flip'):
            # Produce the decision of random flip
            decision = tf.random_uniform([], 0, 1, dtype=tf.float32)

            input_images = random_flip(inputs, decision)
            target_images = random_flip(targets, decision)

    input_images.set_shape([INPUT_SIZE[0], INPUT_SIZE[1], 3])
    target_images.set_shape([INPUT_SIZE[0] * 4, INPUT_SIZE[1] * 4, 3])

    paths_lr_batch, paths_hr_batch, inputs_batch, targets_batch = tf.train.shuffle_batch(
        [output[0], output[1], input_images, target_images],
        batch_size=BATCH_SIZE, capacity=CAPACITY + 4 * BATCH_SIZE,
        min_after_dequeue=CAPACITY, num_threads=THREAD)

    steps_per_epoch = int(math.ceil(len(lr_image_list) / BATCH_SIZE))
    inputs_batch.set_shape([BATCH_SIZE, INPUT_SIZE[0], INPUT_SIZE[1], 3])
    targets_batch.set_shape([BATCH_SIZE, INPUT_SIZE[0] * 4, INPUT_SIZE[1] * 4, 3])

    net = Generator(inputs_batch, is_train)
    gen_output = net.arch_output
    gen_output.set_shape([BATCH_SIZE, INPUT_SIZE[0] * 4, INPUT_SIZE[1] * 4, 3])

    outputs = deprocess(gen_output)
    converted_outputs = tf.image.convert_image_dtype(outputs, dtype=tf.uint8, saturate=True)
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

    sv = tf.train.Supervisor(logdir='./save', save_summaries_secs=0, saver=None)
    with sv.managed_session(config=config) as sess:
        # Performing the training
        total_step = EPOCH * steps_per_epoch
        print('Optimization starts!!!')
        for _ in range(total_step):
            fetches = {
                "global_step": global_step,
                "train": gen_train,
                "gen_loss": gen_loss,
                "test": gen_output,
                "outputs_node": outputs_node
            }

            results = sess.run(fetches, feed_dict={is_train: True})
            print(results['gen_loss'])
            print(results['test'][0, 100:105, 100:105, 0])
            if results['global_step'] % 1000 == 0:
                save_images(results['outputs_node'][0], filename='step_%d.jpg' % results['global_step'])

        print('Optimization done!!!!!!!!!!!!')


if __name__ == '__main__':
    main()
