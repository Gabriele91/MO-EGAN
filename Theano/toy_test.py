from dataset.datasets import toy_dataset
import h5py
from models import models_uncond
from lib.rng import py_rng, np_rng, t_rng
from lib.theano_utils import floatX, sharedX
from lib.data_utils import processing_img, convert_img_back, convert_img, Batch, shuffle, iter_data, ImgRescale, OneHot
from PIL import Image
from time import time
import shutil
import lasagne
import json
import theano.tensor as T
import theano
import numpy as np
import os
from matplotlib.pyplot import imshow, imsave, imread
import matplotlib.pyplot as plt
import matplotlib
import sys
import math
from sklearn.metrics import pairwise_kernels, pairwise_distances
import argparse
sys.path.append('..')
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def MMD2u(K, m, n, biased = False):
    """The MMD^2 statistic.
    """
    Kx = K[:m, :m]
    Ky = K[m:, m:]
    Kxy = K[:m, m:]
    if biased:
        return 1.0 / (m * n) * (Kx.sum() - Kx.diagonal().sum()) + \
               1.0 / (n * n) * (Ky.sum() - Ky.diagonal().sum()) - \
               2.0 / (m * n) * Kxy.sum()
    else:
        #unbaised
        return 1.0 / (m * (m - 1.0)) * (Kx.sum() - Kx.diagonal().sum()) + \
               1.0 / (n * (n - 1.0)) * (Ky.sum() - Ky.diagonal().sum()) - \
               2.0 / (m * n) * Kxy.sum()


def compute_metric_mmd2(X,Y):
    m = len(X)
    n = len(Y)
    sigma2 = np.median(pairwise_distances(X, Y, metric='euclidean'))**2
    XY = np.vstack([X, Y])
    K = pairwise_kernels(XY, metric='rbf',gamma=1.0/sigma2)
    mmd2u = MMD2u(K, m, n, False)
    return mmd2u


def create_G(DIM=64):
    noise = T.matrix('noise')
    generator = models_uncond.build_generator_toy(noise, nd=DIM)
    Tgimgs = lasagne.layers.get_output(generator)
    gen_fn = theano.function([noise],lasagne.layers.get_output(generator, deterministic=True))
    return gen_fn, generator

def generate_image(true_dist, generate_dist, num=0, desc=None, postfix=""):
    """
    Generates and saves a plot of the true distribution, the generator, and the
    critic.
    """
    N_POINTS = 128
    RANGE = 3

    points = np.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
    points[:, :, 0] = np.linspace(-RANGE, RANGE, N_POINTS)[:, None]
    points[:, :, 1] = np.linspace(-RANGE, RANGE, N_POINTS)[None, :]
    points = points.reshape((-1, 2))

    plt.clf()

    x = y = np.linspace(-RANGE, RANGE, N_POINTS)
    #plt.contour(x, y, disc_map.reshape((len(x), len(y))).transpose())

    plt.scatter(true_dist[:, 0], true_dist[:, 1], c='orange', marker='+')
    # if not FIXED_GENERATOR:
    plt.scatter(generate_dist[:, 0],
                generate_dist[:, 1], c='green', marker='+')

    if not os.path.isdir('tmp'):
        os.mkdir(os.path.join('tmp/'))
    if not os.path.isdir('tmp/'+desc):
        os.mkdir(os.path.join('tmp/', desc))

    #plt.savefig('tmp/' + DATASET + '/' + prefix + 'frame' + str(frame_index[0]) + '.jpg')
    plt.savefig('tmp/' + desc + '/frame_' + str(num) + postfix + '.jpg')

    #frame_index[0] += 1


def main(path,datasetname):
    #params
    DIM = 512
    SAMPLES = 25000
    nz = 2
    #load
    gen_fn, generator = create_G(DIM = DIM)
    #load samples from db
    xmb = toy_dataset(DATASET=datasetname, size=SAMPLES)
    #for all in the path:
    for root, dirs, files in os.walk(path):
        mmd_list = []
        files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
        for filename in files: 
            try:
                genpath = os.path.join(root,filename)
                params_map = dict(np.load(genpath))
                params=list()
                for key,vals in sorted(params_map.items(),key=lambda x: int(x[0].split("_")[1])):
                    params.append(np.float32(vals))
                #set params
                lasagne.layers.set_all_param_values(generator, params)
                # generate sample
                s_zmb = floatX(np_rng.uniform(-1., 1., size=(SAMPLES, nz)))
                g_imgs = gen_fn(s_zmb)
                mmd = abs(compute_metric_mmd2(g_imgs,xmb))
                print("MMD: ",mmd, genpath)
                mmd_list.append((mmd,genpath))
            except:
                pass
        mmd_list.sort(key=lambda v:v[0])
        i = 0
        for val,name in mmd_list[:10]:
            i += 1
            print("Best MMD["+str(i)+"]", val, math.sqrt(val), name)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])