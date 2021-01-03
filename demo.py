########################
# Author:
# Firat Ozdemir (fozdemir@gmail.com), Copyright (R) 2019, ETH Zurich
########################
# This demo script trains an MMD VAE on a public dataset, then selects representative samples for N iterations.

import argparse
import numpy as np
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

from learn_representation import train_infoVAE
import utils

def main(args):

    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--logdir", type=str, help="directory to save model and export tensorboard", default='./tmp_logs')
    parser.add_argument("--ndims_latent", type=int, help="#latent_dims. Default is 5.", default=5)
    parser.add_argument("--lambda_mmd", type=int, help="#latent_dims. Default is 1000.", default=1e3)
    parser.add_argument("--batch_size", type=int, help="Mini batch size. Default is 100.", default=100)
    args = parser.parse_args()

    assert args.logdir is not None
    log_dir = args.logdir

    # Learn representation space
    train_infoVAE(args)

    # Sample representative samples
    sampler_obj = utils.AL_BSQ_Sampler(log_dir=log_dir)
    num_al_iterations = 1  # sample most representative samples 1 time in a row for this demo.
    num_query = 10  # query 10 most representative samples at a time.
    inds_an = sampler_obj.inds_an.copy()
    l_inds_an = [inds_an]
    for i_al in range(num_al_iterations):
        inds_query = sampler_obj.query_samples(inds_an=inds_an, num_query=num_query)
        inds_an = np.concatenate((inds_an, inds_query), axis=0)
        l_inds_an.append(inds_an.copy())
        logging.info('Queried indices for step %d:\n%s' % (i_al + 1, str(inds_query)))


if __name__ == "__main__":
    main(0)