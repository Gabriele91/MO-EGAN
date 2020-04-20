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

def MMD2u(K, m, n):
    """The MMD^2_u unbiased statistic.
    """
    Kx = K[:m, :m]
    Ky = K[m:, m:]
    Kxy = K[:m, m:]
    return 1.0 / (m * (m - 1.0)) * (Kx.sum() - Kx.diagonal().sum()) + \
        1.0 / (n * (n - 1.0)) * (Ky.sum() - Ky.diagonal().sum()) - \
        2.0 / (m * n) * Kxy.sum()

def compute_metric_mmd2(X,Y):
    m = len(X)
    n = len(Y)
    sigma2 = np.median(pairwise_distances(X, Y, metric='euclidean'))**2
    XY = np.vstack([X, Y])
    K = pairwise_kernels(XY, metric='rbf',gamma=1.0/sigma2)
    mmd2u = MMD2u(K, m, n)
    return mmd2u

class GeneratorTrainer:
    def __init__(self, noise, generator, discriminator, lr, b1):
        self.noise=noise
        self.generator=generator
        self.discriminator=discriminator
        self.Tgimgs = lasagne.layers.get_output(generator)
        self.Tfake_out = lasagne.layers.get_output(discriminator, self.Tgimgs)
        self.generator_params = lasagne.layers.get_all_params(generator, trainable=True)
        self.g_loss_logD = lasagne.objectives.binary_crossentropy(self.Tfake_out, 1).mean()
        self.g_loss_minimax = -lasagne.objectives.binary_crossentropy(self.Tfake_out, 0).mean()
        self.g_loss_ls = T.mean(T.sqr((self.Tfake_out - 1)))
        self.up_g_logD = lasagne.updates.adam(self.g_loss_logD, self.generator_params, learning_rate=lr, beta1=b1)
        self.up_g_minimax = lasagne.updates.adam(self.g_loss_minimax, self.generator_params, learning_rate=lr, beta1=b1)
        self.up_g_ls = lasagne.updates.adam(self.g_loss_ls, self.generator_params, learning_rate=lr, beta1=b1)
        self.train_g = theano.function([noise], self.g_loss_logD, updates=self.up_g_logD)
        self.train_g_minimax = theano.function([noise], self.g_loss_minimax, updates=self.up_g_minimax)
        self.train_g_ls = theano.function([noise], self.g_loss_ls, updates=self.up_g_ls)
        self.gen_fn = theano.function([noise],  lasagne.layers.get_output(generator, deterministic=True))
    
    def train(self,loss_type,zmb):
        if loss_type == 'trickLogD':
            return self.train_g(zmb)
        elif loss_type == 'minimax':
            return self.train_g_minimax(zmb)
        elif loss_type == 'ls':
            cost = self.train_g_ls(zmb)
        else:
            raise "{} is invalid loss".format(loss_type)

    def gen(self,zmb):
        return self.gen_fn(zmb)

    def set(self,params):
        lasagne.layers.set_all_param_values(self.generator, params)

    def get(self):
        return lasagne.layers.get_all_param_values(self.generator)


def create_G(noise=None, discriminator=None, lr=0.0002, b1=0.5, DIM=64):
    alias_noise=T.matrix('noise') if noise==None else noise
    generator = models_uncond.build_generator_toy(alias_noise, nd=DIM)
    return GeneratorTrainer(alias_noise, 
                            generator, 
                            discriminator, 
                            lr, 
                            b1)


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

# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

'''-------non-dominated sorting function-------'''      
def non_dominated_sorting(population_size,chroms_obj_record):
    s,n={},{}
    front,rank={},{}
    front[0]=[]     
    for p in range(population_size):
        s[p]=[]
        n[p]=0
        for q in range(population_size):
            
            if ((chroms_obj_record[p][0]<chroms_obj_record[q][0] and chroms_obj_record[p][1]<chroms_obj_record[q][1]) \
                or (chroms_obj_record[p][0]<=chroms_obj_record[q][0] and chroms_obj_record[p][1]<chroms_obj_record[q][1])\
                or (chroms_obj_record[p][0]<chroms_obj_record[q][0] and chroms_obj_record[p][1]<=chroms_obj_record[q][1])):
                if q not in s[p]:
                    s[p].append(q)
            elif ((chroms_obj_record[p][0]>chroms_obj_record[q][0] and chroms_obj_record[p][1]>chroms_obj_record[q][1]) \
                or (chroms_obj_record[p][0]>=chroms_obj_record[q][0] and chroms_obj_record[p][1]>chroms_obj_record[q][1])\
                or (chroms_obj_record[p][0]>chroms_obj_record[q][0] and chroms_obj_record[p][1]>=chroms_obj_record[q][1])):
                n[p]=n[p]+1
        if n[p]==0:
            rank[p]=0
            if p not in front[0]:
                front[0].append(p)
    
    i=0
    while (front[i]!=[]):
        Q=[]
        for p in front[i]:
            for q in s[p]:
                n[q]=n[q]-1
                if n[q]==0:
                    rank[q]=i+1
                    if q not in Q:
                        Q.append(q)
        i=i+1
        front[i]=Q
                
    del front[len(front)-1]
    return front
'''--------calculate crowding distance function---------'''
def calculate_crowding_distance(front,chroms_obj_record):
    
    distance={m:0 for m in front}
    for o in range(2):
        obj={m:chroms_obj_record[m][o] for m in front}
        sorted_keys=sorted(obj, key=obj.get)
        distance[sorted_keys[0]]=distance[sorted_keys[len(front)-1]]=999999999999
        for i in range(1,len(front)-1):
            if len(set(obj.values()))==1:
                distance[sorted_keys[i]]=distance[sorted_keys[i]]
            else:
                distance[sorted_keys[i]]=distance[sorted_keys[i]]+(obj[sorted_keys[i+1]]-obj[sorted_keys[i-1]])/(obj[sorted_keys[len(front)-1]]-obj[sorted_keys[0]])
            
    return distance            
'''----------selection----------'''
def selection(population_size,front,chroms_obj_record,total_chromosome):   
    N=0
    new_pop=[]
    while N < population_size:
        for i in range(len(front)):
            N=N+len(front[i])
            if N > population_size:
                distance=calculate_crowding_distance(front[i],chroms_obj_record)
                sorted_cdf=sorted(distance, key=distance.get)
                sorted_cdf.reverse()
                for j in sorted_cdf:
                    if len(new_pop)==population_size:
                        break                
                    new_pop.append(j)              
                break
            else:
                new_pop.extend(front[i])
    
    population_list=[]
    for n in new_pop:
        population_list.append(total_chromosome[n])
        
    return population_list,new_pop
'''---------NSGA-2 pass --------'''
def nsga_2_pass(N, chroms_obj_record, chroms_total):
    front = non_dominated_sorting(len(chroms_obj_record),chroms_obj_record)
    distance = calculate_crowding_distance(front,chroms_obj_record)
    population_list,new_pop=selection(N, front, chroms_obj_record, chroms_total)
    return new_pop
        

def main(problem, 
         popsize,
         moegan, 
         freq, 
         loss_type = ['trickLogD','minimax', 'ls'],
         postfix = None,
         nPassD = 1, #backpropagation pass for discriminator
         inBatchSize = 64
         ):

    # Parameters
    task = 'toy'
    name = '{}_{}_{}MMDu2'.format(problem,"MOEGAN" if moegan else "EGAN", postfix + "_" if postfix is not None else "") #'8G_MOEGAN_PFq_NFd_t2'

    DIM = 512
    begin_save = 0
    nloss = len(loss_type)
    batchSize = inBatchSize

    if problem == "8G":
        DATASET = '8gaussians'
    elif problem == "25G":
        DATASET = '25gaussians'
    else:
        exit(-1)

    ncandi = popsize
    kD = nPassD       # # of discrim updates for each gen update
    kG = 1            # # of discrim updates for each gen update
    ntf = 256
    b1 = 0.5          # momentum term of adam
    nz = 2          # # of dim for Z
    niter = 4       # # of iter at starting learning rate
    lr = 0.0001       # initial learning rate for adam G
    lrd = 0.0001       # initial learning rate for adam D
    N_up = 100000
    save_freq = freq
    show_freq = freq 
    test_deterministic = True
    beta = 1.
    GP_norm = False     # if use gradients penalty on discriminator
    LAMBDA = 2.       # hyperparameter sudof GP
    NSGA2 = moegan
    # Load the dataset

    # MODEL D
    print("Building model and compiling functions...")
    # Prepare Theano variables for inputs and targets
    real_imgs = T.matrix('real_imgs')
    fake_imgs = T.matrix('fake_imgs')
    # Create neural network model
    discriminator = models_uncond.build_discriminator_toy(nd=DIM, GP_norm=GP_norm)
    # Create expression for passing real data through the discriminator
    real_out = lasagne.layers.get_output(discriminator, real_imgs)
    # Create expression for passing fake data through the discriminator
    fake_out = lasagne.layers.get_output(discriminator, fake_imgs)
    # Create loss expressions
    discriminator_loss = (lasagne.objectives.binary_crossentropy(real_out, 1) + lasagne.objectives.binary_crossentropy(fake_out, 0)).mean()

    # Gradients penalty norm
    if GP_norm is True:
        alpha = t_rng.uniform((batchSize, 1), low=0., high=1.)
        differences = fake_imgs - real_imgs
        interpolates = real_imgs + (alpha*differences)
        gradients = theano.grad(lasagne.layers.get_output(
            discriminator, interpolates).sum(), wrt=interpolates)
        slopes = T.sqrt(T.sum(T.sqr(gradients), axis=(1)))
        gradient_penalty = T.mean((slopes-1.)**2)

        D_loss = discriminator_loss + LAMBDA*gradient_penalty
        b1_d = 0.
    else:
        D_loss = discriminator_loss
        b1_d = 0.

    # Create update expressions for training
    discriminator_params = lasagne.layers.get_all_params(discriminator, trainable=True)
    lrtd = theano.shared(lasagne.utils.floatX(lrd))
    updates_d = lasagne.updates.adam(D_loss, discriminator_params, learning_rate=lrtd, beta1=b1_d)
    lrt = theano.shared(lasagne.utils.floatX(lr))

    # Fd Socre
    Fd = theano.gradient.grad(discriminator_loss, discriminator_params)
    Fd_score = beta*T.log(sum(T.sum(T.sqr(x)) for x in Fd))

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_d = theano.function([real_imgs, fake_imgs],  discriminator_loss, updates=updates_d)

    # Compile another function generating some data
    dis_fn = theano.function([real_imgs, fake_imgs], [
                             (fake_out).mean(), Fd_score])
    disft_fn = theano.function([real_imgs, fake_imgs],
                               [real_out.mean(), fake_out.mean(), (real_out > 0.5).mean(),  (fake_out > 0.5).mean(),  Fd_score])

    #main MODEL G
    noise = T.matrix('noise')
    generator_trainer = create_G(noise=noise, discriminator=discriminator, lr=lr, b1=b1, DIM=DIM)

    # Finally, launch the training loop.
    print("Starting training...")
    desc = task + '_' + name
    print(desc)

    if not os.path.isdir('front'):
        os.mkdir(os.path.join('front'))
    if not os.path.isdir('front/'+desc):
        os.mkdir(os.path.join('front/', desc))
    if not os.path.isdir('logs'):
        os.mkdir(os.path.join('logs'))
    f_log = open('logs/%s.ndjson' % desc, 'wb')
    if not os.path.isdir('models'):
        os.mkdir(os.path.join('models/'))
    if not os.path.isdir('models/'+desc):
        os.mkdir(os.path.join('models/', desc))

    instances = []
    class Instance:
        def __init__(self,fq,fd,params, img_values):
            self.fq = fq
            self.fd = fd
            self.params = params
            self.img = img_values

        def f(self):
            return self.fq - self.fd

    # We iterate over epochs:
    for n_updates in range(N_up):
        xmb = toy_dataset(DATASET=DATASET, size=batchSize*kD)
        xmb = xmb[0:batchSize*kD]
        # initial G cluster
        if n_updates == 0:
            for can_i in range(0, ncandi):
                init_generator_trainer = create_G(noise=noise, discriminator=discriminator, lr=lr, b1=b1, DIM=DIM)
                zmb = floatX(np_rng.uniform(-1., 1., size=(batchSize, nz)))
                cost = init_generator_trainer.train(loss_type[can_i%nloss],zmb)
                sample_zmb = floatX(np_rng.uniform(-1., 1., size=(ntf, nz)))
                gen_imgs = init_generator_trainer.gen(sample_zmb)
                frr_score, fd_score = dis_fn(xmb[0:ntf], gen_imgs)
                instances.append(Instance(frr_score, 
                                          fd_score,
                                          lasagne.layers.get_all_param_values(init_generator_trainer.generator), 
                                          gen_imgs))
        else:
            instances_old = instances
            instances = []
            for can_i in range(0, ncandi):
                for type_i in range(0,nloss):
                    generator_trainer.set(instances_old[can_i].params)
                    #train
                    zmb = floatX(np_rng.uniform(-1., 1., size=(batchSize, nz)))
                    generator_trainer.train(loss_type[type_i],zmb)
                    #score
                    sample_zmb = floatX(np_rng.uniform(-1., 1., size=(ntf, nz)))
                    gen_imgs = generator_trainer.gen(sample_zmb)
                    frr_score, fd_score = dis_fn(xmb[0:ntf], gen_imgs)
                    #save
                    instances.append(Instance(frr_score, 
                                              fd_score,
                                              generator_trainer.get(), 
                                              gen_imgs))
            if ncandi <= (len(instances)+len(instances_old)):
                if NSGA2==True:
                    #add parents in the pool
                    for inst in instances_old:
                        generator_trainer.set(inst.params)
                        sample_zmb = floatX(np_rng.uniform(-1., 1., size=(ntf, nz)))
                        gen_imgs = generator_trainer.gen(sample_zmb)
                        frr_score, fd_score = dis_fn(xmb[0:ntf], gen_imgs)
                        instances.append(Instance(
                            frr_score, 
                            fd_score,
                            generator_trainer.get(),
                            gen_imgs
                        ))
                    #cromos = { idx:[float(inst.fq),-0.5*float(inst.fd)] for idx,inst in enumerate(instances) } # S1
                    cromos = { idx:[-float(inst.fq),0.5*float(inst.fd)] for idx,inst in enumerate(instances) } # S2
                    cromos_idxs = [ idx for idx,_ in enumerate(instances) ]
                    finalpop = nsga_2_pass(ncandi, cromos, cromos_idxs)
                    instances = [instances[p] for p in finalpop]
                    with open('front/%s.tsv' % desc, 'wb') as ffront:
                        for inst in instances:
                            ffront.write((str(inst.fq) + "\t" + str(inst.fd)).encode())
                            ffront.write("\n".encode())
                elif nloss>1:
                    #sort new
                    instances.sort(key=lambda inst: -inst.f()) #wrong def in the paper
                    #print([inst.f() for inst in instances])
                    #cut best ones
                    instances = instances[len(instances)-ncandi:]
                    #print([inst.f() for inst in instances])



        sample_xmb = toy_dataset(DATASET=DATASET, size=ncandi*ntf)
        sample_xmb = sample_xmb[0:ncandi*ntf]
        for i in range(0, ncandi):
            xfake = instances[i].img[0:ntf, :]
            xreal = sample_xmb[i*ntf:(i+1)*ntf, :]
            tr, fr, trp, frp, fdscore = disft_fn(xreal, xfake)
            fake_rate = np.array([fr]) if i == 0 else np.append(fake_rate, fr)
            real_rate = np.array([tr]) if i == 0 else np.append(real_rate, tr)
            fake_rate_p = np.array([frp]) if i == 0 else np.append(fake_rate_p, frp)
            real_rate_p = np.array([trp]) if i == 0 else np.append(real_rate_p, trp)
            FDL = np.array([fdscore]) if i == 0 else np.append(FDL, fdscore)

        print(fake_rate, fake_rate_p, FDL)
        print(n_updates, real_rate.mean(), real_rate_p.mean())
        f_log.write((str(fake_rate)+' '+str(fake_rate_p)+'\n' + str(n_updates) + ' ' + str(real_rate.mean()) + ' ' + str(real_rate_p.mean())+'\n').encode())
        f_log.flush()

        # train D
        #for xreal, xfake in iter_data(xmb, shuffle(fmb), size=batchSize):
        #    cost = train_d(xreal, xfake)
        imgs_fakes = instances[0].img[0:int(batchSize/ncandi*kD), :];
        for i in range(1,len(instances)):
            img = instances[i].img[0:int(batchSize/ncandi*kD), :]
            imgs_fakes = np.append(imgs_fakes, img, axis=0)
        for xreal, xfake in iter_data(xmb, shuffle(imgs_fakes), size=batchSize):
            cost = train_d(xreal, xfake)

        if (n_updates % show_freq == 0 and n_updates!=0) or n_updates==1:
            id_update = int(n_updates/save_freq)
            #metric
            s_zmb = floatX(np_rng.uniform(-1., 1., size=(512, nz)))
            xmb = toy_dataset(DATASET=DATASET, size=512)
            #compue mmd for all points
            mmd2_all = []
            for i in range(0, ncandi):
                generator_trainer.set(instances[i].params)
                g_imgs = generator_trainer.gen(s_zmb)
                mmd2_all.append(abs(compute_metric_mmd2(g_imgs,xmb)))
            mmd2_all = np.array(mmd2_all)
            #print pareto front
            if NSGA2==True:
                front_path=os.path.join('front/', desc)
                with open('%s/%d_%s_mmd2u.tsv' % (front_path,id_update, desc), 'wb') as ffront:
                    for idx in range(0, ncandi):
                        ffront.write((str(instances[idx].fq) + "\t" + str(instances[idx].fd) + "\t" + str(mmd2_all[idx])).encode())
                        ffront.write("\n".encode())
            #mmd2 output
            print(n_updates, "mmd2u:", np.min(mmd2_all), "id:", np.argmin(mmd2_all))
            #save best
            params = instances[np.argmin(mmd2_all)].params
            generator_trainer.set(params)
            g_imgs_min = generator_trainer.gen(s_zmb)
            generate_image(xmb, g_imgs_min, id_update, desc, postfix="_mmu2d_best")
            np.savez('models/%s/gen_%d.npz'%(desc,id_update), *lasagne.layers.get_all_param_values(discriminator))
            np.savez('models/%s/dis_%d.npz'%(desc,id_update), *generator_trainer.get())
            #worst_debug
            params = instances[np.argmax(mmd2_all)].params
            generator_trainer.set(params)
            g_imgs_max = generator_trainer.gen(s_zmb)
            generate_image(xmb, g_imgs_max, id_update, desc, postfix="_mmu2d_worst")


        #if n_updates % save_freq == 0 and n_updates > begin_save - 1:
            # Optionally, you could now dump the network weights to a file like this:
            #    np.savez('models/%s/gen_%d.npz'%(desc,n_updates/save_freq), *lasagne.layers.get_all_param_values(generator))
            #    np.savez('models/%s/dis_%d.npz'%(desc,n_updates/save_freq), *lasagne.layers.get_all_param_values(discriminator))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm","-a", choices=["egan","moegan"], default="moegan")
    parser.add_argument("--loss_type","-l", nargs="+", choices=['trickLogD','minimax', 'ls'], default=['trickLogD','minimax', 'ls'])
    parser.add_argument("--problem","-p", choices=["8G","25G"],default="8G")
    parser.add_argument("--population_size","-mu", type=int, default=8)
    parser.add_argument("--save_frequency","-freq", type=int, default=1000)
    parser.add_argument("--post_fix","-pfix", type=str, default=None)
    parser.add_argument("--update_discrminator","-ud", type=int, default=1)
    parser.add_argument("--batch_size","-bs", type=int, default=64)
    arguments = parser.parse_args()
    print("_"*42)
    print(" "*14+"> ARGUMENTS <")
    print("problem:", arguments.problem)
    print("population_size:", arguments.population_size)
    print("algorithm:", arguments.algorithm)
    print("loss_type:", arguments.loss_type)
    print("save_frequency:", arguments.save_frequency)
    print("post_fix:", arguments.post_fix)    
    print("update_discrminator:", arguments.update_discrminator)    
    print("batch_size:", arguments.batch_size)    
    print("_"*42)
    main(problem=arguments.problem,
         popsize=arguments.population_size,
         moegan=arguments.algorithm=="moegan",
         freq=arguments.save_frequency,
         loss_type=arguments.loss_type,
         postfix=arguments.post_fix,
         nPassD=arguments.update_discrminator,
         inBatchSize=arguments.batch_size)
