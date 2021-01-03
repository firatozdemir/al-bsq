########################
# Author:
# Firat Ozdemir (fozdemir@gmail.com),
# Zixuan Peng,
# Copyright (R) 2017, ETH Zurich
########################
# This module contains all necessary utils which will be used in the demo.

import math
import tensorflow as tf
import tensorflow_datasets as tfds
import glob
import os
import numpy as np

def check_tf_ckpt(path, pattern):
    '''Function checks path for TF model checkpoints that match the pattern and return the latest, if any.'''
    l_candidates = glob.glob(os.path.join(path, '*%s*.index' % (pattern)))
    if len(l_candidates) == 0:
        return None, None
    else:
        curr_latest = 0
        full_path_name = ''
        for full_path in l_candidates:
            fname = os.path.basename(full_path)
            _, ckpt, _ = fname.split('.')
            ckpt_num = int(ckpt.split('-')[-1])
            if ckpt_num > curr_latest:
                full_path_name = full_path[:-1*len('.index')]
                curr_latest = ckpt_num
        return full_path_name, curr_latest

def cdf(x, mu, std):
    '''Cumulative distribution function for normal distribution'''
    v = (x - mu) / std
    return (1.0 + math.erf(v / math.sqrt(2.0))) / 2.0
def sample_cdf(x,mu,std):
    if x > mu:
        return 2 * cdf(x=2*mu-x, mu=mu, std=std)
    else:
        return 2 * cdf(x=x, mu=mu, std=std)

def compute_kernel(x, y):
    '''Compute k(z,z')'''
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
    tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
    return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))

def compute_mmd(x, y):
    ''' Compute MMD '''
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)

class InfoVAE:
    def __init__(self, input_size, is_training_pl, latent_dims=10, output_size=None, **kwargs):
        self.input_size = input_size
        self.latent_dims = latent_dims
        self.output_size = output_size if output_size is not None else input_size
        self.train_pl = is_training_pl
        self.graph = kwargs.get('graph', None)
        self.sess = kwargs.get('session', None)
        self.input = None
        self.latent_space = None
        self.decoder_input = None
        self.output = None
        self.l1r = 0.0
        self.l2r = 0.0
        self.use_BN = True
        self.kernel_init = tf.keras.initializers.he_normal()
        self.enc_act = tf.keras.layers.ReLU()
        self.dec_act = tf.keras.layers.LeakyReLU()
        self.network_scope = 'net'
        self.model_enc = None
        self.model_dec = None
        self.model = None
        self.variables = None
        self.update_ops = None

    def build_model(self, input):
        '''Similarly to the experiment setup in infoVAE, using DC-GAN architecture.'''
        with self.graph.as_default():
            with tf.variable_scope(name_or_scope=self.network_scope):
                padding = 'VALID'
                with tf.variable_scope(name_or_scope='enc'):
                    self.input = tf.keras.layers.Input(tensor=input, name='Input')
                    conv1_1 = self.conv2d(input=self.input, nf=64, act=self.enc_act, name='conv1_1', strides=(1, 1), kernel_size=(5, 5), padding=padding)  # [28,28] OR [24,24 x64
                    conv2_1 = self.conv2d(input=conv1_1, nf=128, act=self.enc_act, name='conv2_1', strides=(2, 2), kernel_size=(5, 5), padding=padding)  # size: [14 OR [10
                    conv3_1 = self.conv2d(input=conv2_1, nf=256, act=self.enc_act, name='conv3_1', strides=(2, 2), kernel_size=(5, 5), padding=padding)  # size: [7 OR [3 x 256
                    conv_flat = tf.keras.layers.Reshape(target_shape=[3 * 3 * 256], name='flatten_final_conv')(conv3_1)  # size [batch size, N], N=3*3*256

                    self.latent_space = tf.keras.layers.Dense(units=self.latent_dims, activation=None, use_bias=False,
                                                              name='Z')(conv_flat)

                with tf.variable_scope(name_or_scope='dec'):
                    match_conv_size = tf.keras.layers.Dense(units=7 * 7 * 256, use_bias=False, name='Match_decoder_conv_shape')(self.latent_space)
                    match_conv_size_bn = tf.keras.layers.BatchNormalization(name='Match_decoder_conv_shape' + '_BN')(match_conv_size, training=self.train_pl)
                    match_conv_size_act = tf.keras.layers.Activation(self.dec_act, name='Match_decoder_conv_shape' + '_act')(match_conv_size_bn)
                    flat_to_image = tf.keras.layers.Reshape(target_shape=(7, 7, 256), name='flat_to_image')(match_conv_size_act)

                    padding = 'SAME'
                    deconv1 = self.deconv2d(input=flat_to_image, nf=256, act=self.dec_act, name='deconv1_1', strides=(1, 1), kernel_size=(5, 5), padding=padding)  # size(7,7,128)
                    deconv2 = self.deconv2d(input=deconv1, nf=128, act=self.dec_act, name='deconv2_1', strides=(2, 2), kernel_size=(5, 5), padding=padding)  # size(14,14,128)
                    self.output = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(5, 5), strides=(2, 2), padding=padding, use_bias=False, activation='tanh', name='output')(deconv2)  # size(28,28,128)

            self.model = tf.keras.Model(inputs=self.input, outputs=[self.latent_space, self.output])
            self.model_enc = tf.keras.Model(inputs=self.input, outputs=self.latent_space)

            self.variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.network_scope)
            self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.network_scope)
            self.update_ops += self.model.updates

    def conv2d(self, input, nf, act, name, strides=(1, 1), kernel_size=(3, 3), padding='SAME'):
        c = tf.keras.layers.Conv2D(activation=None, filters=nf, name=name,
                                   kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1r, l2=self.l2r),
                                   kernel_size=kernel_size, padding=padding, use_bias=False,
                                   strides=strides,
                                   kernel_initializer=self.kernel_init,
                                   activity_regularizer=tf.keras.layers.ActivityRegularization(l1=1e-5))(input)

        c1 = tf.keras.layers.BatchNormalization(name=name + '_BN')(c, training=self.train_pl)
        c2 = tf.keras.layers.Activation(act, name=name + '_act')(c1)
        return c2

    def deconv2d(self, input, nf, act, name, strides=(1, 1), kernel_size=(3, 3), padding='SAME'):
        c = tf.keras.layers.Conv2DTranspose(activation=None, filters=nf, name=name,
                                            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1r, l2=self.l2r),
                                            kernel_size=kernel_size, padding=padding, use_bias=False,
                                            strides=strides,
                                            kernel_initializer=self.kernel_init,
                                            activity_regularizer=tf.keras.layers.ActivityRegularization(l1=1e-5))(input)

        c1 = tf.keras.layers.BatchNormalization(name=name + '_BN')(c, training=self.train_pl)
        c2 = tf.keras.layers.Activation(act, name=name + '_act')(c1)
        return c2


class AL_BSQ_Sampler:
    def __init__(self, log_dir, **kwargs):
        self.inds_an = kwargs.get('inds_an', None) # Indices of the initially annotated dataset
        self.latent_dims = kwargs.get('latent_dims', 5)
        self.dataset_str = kwargs.get('dataset_str', 'MNIST')
        self.prng = kwargs.get('prng', np.random.RandomState(1999))
        self.load_dir_tfds = kwargs.get('load_dir_tfds', '~/') #directory to download and unpack tfds datasets
        self.log_dir = log_dir
        self.sess = None
        self.g = None
        self.pred_x = None
        self.pred_z = None
        self.iterator = None
        self.x_pool = None
        self.y_pool = None
        self.z_pool = None
        self.num_pool_size = None
        self.restore_latent_mapping() # Restore learned weights for mapping samples to the latent space
        self.map_pool_to_latent_space() # Map all samples in D_pool to latent space
        self.synthetic_initial_annotated_indices(num_an_init_size=10) # create a synthetic initial annotated dataset. !!! This is strictly for the demo !!!

    def restore_latent_mapping(self):
        self.g = tf.Graph()
        input_size = (28, 28, 1)
        with self.g.as_default():
            tf.set_random_seed(1990)
            # Construct a tf.data.Dataset
            dataset_mnist = tfds.load(name="mnist", split=tfds.Split.TRAIN, data_dir=self.load_dir_tfds).map(
                lambda x: {'image': tf.cast(x['image'], tf.float32), 'label': x['label']})  # MNIST [28,28,1]
            if self.dataset_str == 'MNIST':
                dataset = dataset_mnist
            elif self.dataset_str == 'MNIST + EMNIST':
                dataset_emnist = tfds.load(name="emnist", split=tfds.Split.TRAIN, data_dir=self.load_dir_tfds).map(
                    lambda x: {'image': tf.cast(tf.transpose(x['image'], perm=[1, 0, 2]), tf.float32),
                               'label': x['label']})  # extended MNIST [28,28,1]
                dataset = dataset_emnist.concatenate(dataset_mnist)
            dataset = dataset.repeat(count=1).batch(50).prefetch(tf.data.experimental.AUTOTUNE)
            self.iterator = dataset.make_initializable_iterator()
            batch_op = self.iterator.get_next()
            self.x_batch_op = tf.cast(batch_op['image'], tf.float32, name='x_batch')  # shape (?, 28, 28, 1), float32
            self.y_batch_op = batch_op['label']  # shape (?,), int64 # kind of irrelevant for training except for class separation analysis in Z.

            is_training_pl = tf.placeholder_with_default(False, shape=[], name='is_training_pl')

            self.sess = tf.Session()

            model_obj = InfoVAE(input_size=input_size, is_training_pl=is_training_pl, latent_dims=self.latent_dims, graph=self.g, session=self.sess)
            model_obj.build_model(self.x_batch_op)
            model = model_obj.model
            self.pred_z = model.outputs[0]  # shape [?, latent_dims]
            self.pred_x = model.outputs[1]  # shape [?, 28, 28, 1]

            # Create a saver for writing training checkpoints.
            max_to_keep = 2
            saver = tf.train.Saver(max_to_keep=max_to_keep)

            ckpt_path, step_init = check_tf_ckpt(self.log_dir, 'model')
            if ckpt_path is None:
                raise AssertionError('Could not find model ckpt in %s.' % (self.log_dir))
            saver.restore(self.sess, ckpt_path)

    def map_pool_to_latent_space(self):
        self.z_pool = []
        # self.x_pool = []
        # self.y_pool = []
        with self.g.as_default():
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(self.iterator.initializer)
            while True:
                try:
                    # x_tmp, y_tmp, z_tmp = self.sess.run([self.x_batch_op, self.y_batch_op, self.pred_z])
                    # y_tmp, z_tmp = self.sess.run([self.y_batch_op, self.pred_z])
                    z_tmp = self.sess.run(self.pred_z)
                except tf.errors.OutOfRangeError:
                    print('batch finished')
                    break
                # self.x_pool.append(x_tmp)
                # self.y_pool.append(y_tmp)
                self.z_pool.append(z_tmp)
        # self.x_pool = np.concatenate(self.x_pool, axis=0)  # mnist: [60k, 28,28]
        # self.y_pool = np.concatenate(self.y_pool, axis=0)  # mnist: [60k]
        self.z_pool = np.concatenate(self.z_pool, axis=0)  # mnist: [60k, #latent_dims]

    def synthetic_initial_annotated_indices(self, y_pool=None, num_an_init_size=10):
        if y_pool is None:
            is_non_uniform_sample_weights = False
        else:
            is_non_uniform_sample_weights = True
        num_pool_size = np.shape(self.z_pool)[0]
        self.num_pool_size = num_pool_size
        #compute z_annotated
        num_labels = len(np.unique(y_pool))
        inds_an_ = np.arange(num_pool_size)
        if is_non_uniform_sample_weights: #Requires knowing y_pool
            probability_ratio_list = np.ones((num_labels,))
            i_low_prob = self.prng.choice(np.arange(num_labels), 3, replace=False)
            probability_ratio_list[i_low_prob] = 0.1
            probability_pool = np.asarray([probability_ratio_list[y] for y in y_pool])
            probability_pool /= np.sum(probability_pool)
            inds_an = self.prng.choice(inds_an_, size=num_an_init_size, replace=False, p=probability_pool)
        else:
            inds_an = self.prng.choice(inds_an_, size=num_an_init_size, replace=False)
        self.inds_an = inds_an

    def query_samples(self, inds_an, num_query=None):
        '''Function queries #num_query indices with highest representation of the pool.
        if num_query is None, all candidates will be returned with decreasing representativeness order.'''

        z_an = self.z_pool[inds_an]
        # x_an = self.x_pool[inds_an]
        # y_an = self.y_pool[inds_an]
        inds_available_mask = np.ones((self.num_pool_size,), dtype=np.bool)
        inds_available_mask[inds_an] = False
        inds_available = np.where(inds_available_mask)[0]
        # y_candidates = self.y_pool[inds_available_mask]
        # x_candidates = self.x_pool[inds_available_mask]
        z_candidates = self.z_pool[inds_available_mask]

        eps = 1e-10
        latent_dims = self.z_pool.shape[-1]
        assert latent_dims == self.latent_dims
        num_candidates = np.shape(z_candidates)[0]
        mu_an = np.mean(z_an, axis=0)
        sigma_an = np.std(z_an, axis=0)
        mu_pool = np.mean(self.z_pool, axis=0)
        sigma_pool = np.std(z_an, axis=0)
        p_z_an = np.zeros((num_candidates,))
        p_z_pool = np.zeros((num_candidates,))
        for i_sample in range(num_candidates):
            p_z_an[i_sample] = np.prod([sample_cdf(x=z_candidates[i_sample, i], mu=mu_an[i], std=sigma_an[i]) + eps for i in range(latent_dims)])
            p_z_pool[i_sample] = np.prod([sample_cdf(x=z_candidates[i_sample, i], mu=mu_pool[i], std=sigma_pool[i]) + eps for i in range(latent_dims)])
        p_z = np.log(p_z_an) - np.log(p_z_pool)
        inds_candidates_ = np.argsort(p_z)  # argmin of log p(z|x,Xan) - log p(z|x,Xpool)
        if num_query is None:
            return inds_available[inds_candidates_]
        else:
            return inds_available[inds_candidates_[:num_query]]

    def compute_posterior_on_Xnon(self, inds_an):
        '''Function returns associated probability for each sample in X_non; i.e., compute Eqn (7) for all X_non, hence ignore the argmax part. 
        This would be useful when number of samples to be annotated are sufficiently large, and one can sample the most important, yet not redundant
        using the list of probabilities from this function as a prior.
        Example:
        sampler_obj = utils.AL_BSQ_Sampler(log_dir=logdir)
        priors_for_Xnon = sampler_obj.compute_posterior_on_Xnon(inds_an=indices_of_samples_that_are_already_annotated)
        inds_query = np.random.choice(inds_Xnon, size=num_query, replace=False, p=priors_for_Xnon)
        '''

        z_an = self.z_pool[inds_an]
        inds_available_mask = np.ones((self.num_pool_size,), dtype=np.bool)
        inds_available_mask[inds_an] = False
        inds_available = np.where(inds_available_mask)[0]
        z_candidates = self.z_pool[inds_available_mask]

        eps = 1e-10
        latent_dims = self.z_pool.shape[-1]
        assert latent_dims == self.latent_dims
        num_candidates = np.shape(z_candidates)[0]
        mu_an = np.mean(z_an, axis=0)
        sigma_an = np.std(z_an, axis=0)
        mu_pool = np.mean(self.z_pool, axis=0)
        sigma_pool = np.std(z_an, axis=0)
        p_z_an = np.zeros((num_candidates,))
        p_z_pool = np.zeros((num_candidates,))
        for i_sample in range(num_candidates):
            p_z_an[i_sample] = np.prod([sample_cdf(x=z_candidates[i_sample, i], mu=mu_an[i], std=sigma_an[i]) + eps for i in range(latent_dims)])
            p_z_pool[i_sample] = np.prod([sample_cdf(x=z_candidates[i_sample, i], mu=mu_pool[i], std=sigma_pool[i]) + eps for i in range(latent_dims)])
        p_z = np.log(p_z_an) - np.log(p_z_pool)
        return p_z
