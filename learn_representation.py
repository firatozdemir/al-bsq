########################
# Author:
# Firat Ozdemir (fozdemir@gmail.com), Copyright (R) 2019, ETH Zurich
########################
# This script trains an MMD VAE on a public dataset to learn a low dimensional representation of a potentially high dimensional input space.

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import argparse
import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

import utils

def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--logdir", type=str, help="directory to save model and export tensorboard", default='./tmp_logs')
    parser.add_argument("--ndims_latent", type=int, help="#latent_dims. Default is 5.", default=5)
    parser.add_argument("--lambda_mmd", type=int, help="#latent_dims. Default is 1000.", default=1e3)
    parser.add_argument("--batch_size", type=int, help="Mini batch size. Default is 100.", default=100)
    args = parser.parse_args()

    assert args.logdir is not None
    train_infoVAE(args)

def train_infoVAE(args):

    log_dir = args.logdir
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    continue_training = True
    g = tf.Graph()
    input_size = (28, 28, 1)
    latent_dims = int(args.ndims_latent)
    lambda_mmd = float(args.lambda_mmd)
    max_steps = int(1e7)
    freq_model_save = 5000
    saving_unlimited_ckpts = False
    freq_regular_save = int(5 * 1e4)
    freq_train_log = 300
    batch_size = int(args.batch_size)
    dataset_str = 'MNIST'


    with g.as_default():
        tf.set_random_seed(1990)
        load_dir_tfds = '~/.'
        # Construct a tf.data.Dataset
        dataset_mnist = tfds.load(name="mnist", split=tfds.Split.TRAIN, data_dir=load_dir_tfds).map(lambda x: {'image': tf.cast(x['image'], tf.float32), 'label': x['label']})  # MNIST [28,28,1]
        if dataset_str == 'MNIST':
            dataset = dataset_mnist
        elif dataset_str == 'MNIST + EMNIST':
            dataset_emnist = tfds.load(name="emnist", split=tfds.Split.TRAIN, data_dir=load_dir_tfds).map(lambda x: {'image': tf.cast(tf.transpose(x['image'], perm=[1,0,2]), tf.float32), 'label': x['label']})  # extended MNIST [28,28,1]
            dataset = dataset_emnist.concatenate(dataset_mnist)
        dataset = dataset.shuffle(1024).repeat(count=None).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        iterator = dataset.make_initializable_iterator()
        batch_op = iterator.get_next()
        normalize_x = lambda x: (tf.cast(x, tf.float32) / 127.5) - 1.0 # normalize to range [-1,1] from [0,255]
        x_batch_op = normalize_x(batch_op['image'])
        y_batch_op = batch_op['label']  # shape (?,), int64 # kind of irrelevant for training except for class separation analysis in Z.

        is_training_pl = tf.placeholder_with_default(True, shape=[], name='is_training_pl')
        learning_rate_pl = tf.placeholder_with_default(1e-3, shape=[], name='learning_rate')

        # Create a session for running Ops on the Graph.
        config = tf.ConfigProto()
        # Do not assign whole gpu memory, just use it on the go
        config.gpu_options.allow_growth = True
        # If a operation is not define it the default device, let it execute in another.
        config.allow_soft_placement = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.999
        sess = tf.Session(config=config)

        model_obj = utils.InfoVAE(input_size=input_size, is_training_pl=is_training_pl, latent_dims=latent_dims, graph=g, session=sess)
        model_obj.build_model(x_batch_op)
        model = model_obj.model
        model.summary(print_fn=logging.info)
        pred_z = model.outputs[0]  # shape [?, latent_dims]
        pred_x = model.outputs[1]  # shape [?, 28, 28, 1]

        # Create a saver for writing training checkpoints.
        max_to_keep = 2
        saver = tf.train.Saver(max_to_keep=max_to_keep)
        saver_regular_save = tf.train.Saver(max_to_keep=None)  # keep all

        step_init = 0
        if continue_training:
            try:
                ckpt_path, step_init = utils.check_tf_ckpt(log_dir, 'model')
                if ckpt_path is None:
                    logging.info('Could not find model ckpt in %s. Restarting training.' % (log_dir))
                    continue_training = False
                    step_init = 0
                else:
                    saver.restore(sess, ckpt_path)
            except:
                logging.info('Could not find model ckpt in %s. Restarting training.' % (log_dir))
                continue_training = False
                step_init = 0

        # define losses
        prior_z = tf.random.normal(tf.stack([300, latent_dims]))  # default is mean=0, std=1. 300 samples is picked randomly.
        loss_mmd = utils.compute_mmd(x=pred_z, y=prior_z)
        loss_reconstruction_l2 = tf.reduce_mean(tf.keras.losses.MSE(y_true=x_batch_op, y_pred=pred_x))
        loss_reconstruction_l1 = tf.reduce_mean(tf.keras.losses.MAE(y_true=x_batch_op, y_pred=pred_x))
        loss_reconstruction = 0.5 * loss_reconstruction_l1 + 0.5 * loss_reconstruction_l2
        loss_total = lambda_mmd * loss_mmd + loss_reconstruction + tf.losses.get_regularization_loss()

        # Define gradient compute and apply:
        list_vars = model_obj.variables
        list_ops = model_obj.update_ops
        with tf.control_dependencies(list_ops):
            train_op = tf.train.AdamOptimizer(learning_rate=learning_rate_pl, name='Adam_Optimizer').minimize(
                loss_total, var_list=list_vars)

        # reate tensorboard summaries 

        # Add the variable initializer Op.
        init = tf.global_variables_initializer()

        # Run the Op to initialize the variables.
        sess.run(init)
        sess.run(iterator.initializer)

        logging.info('Logging experiments under: %s' % log_dir)
        for step in range(step_init, max_steps):

            if step % freq_train_log == freq_train_log - 1:
                _, mmd_tmp, reconstruction_tmp = sess.run([train_op, loss_mmd, loss_reconstruction])
                print('Step %d. MMD: %.5f, MSE+MAE: %.5f' % (step + 1, mmd_tmp, reconstruction_tmp))
            else:
                _ = sess.run(train_op)

            if step % freq_model_save == freq_model_save - 1:
                ckpt_name = os.path.join(log_dir, 'model.ckpt')
                saver.save(sess, ckpt_name, global_step=step + 1)

            if step % freq_regular_save == freq_regular_save -1:
                if saving_unlimited_ckpts:
                    reg_save_file = os.path.join(log_dir, 'model_regular_save.ckpt')
                    saver_regular_save.save(sess, reg_save_file, global_step=step+1)

if __name__ == "__main__":
    main(0)
