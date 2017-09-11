import os
import time
import argparse
import importlib
import tensorflow as tf
import tensorflow.contrib as tc
import matplotlib
matplotlib.use('Agg')
import cPickle as pickle
from numpy import linalg, argmin, array, amax, arange, clip, load, random, ceil
import matplotlib.gridspec as gridspec
from utilize import data_readf, dwp
import logging # these 2 lines ar used in GPU3
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from visualize import *


class MIMIC_WGAN(object):
    def __init__(self,
                 g_net,
                 d_net,
                 ae_net,
                 x_sampler,
                 z_sampler,
                 data,
                 model,
                 decompressDims,
                 aeActivation,
                 dataType,
                 _VALIDATION_RATIO,
                 nEpochs,
                 batch_size,
                 n_discriminator_update,
                 n_generator_update
                 ): # changed
        self.g_net = g_net
        self.d_net = d_net
        self.ae_net = ae_net
        self.x_sampler = x_sampler
        self.z_sampler = z_sampler
        self.x_dim = self.d_net.inputDim
        self.z_dim = self.g_net.randomDim
        self.decompressDims = decompressDims
        self.aeActivation = aeActivation
        self.dataType = dataType
        self.x = tf.placeholder(tf.float32, [None, self.x_dim], name='x')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.keep_prob = tf.placeholder('float')
        self.bn_train = tf.placeholder('bool')
        self._VALIDATION_RATIO = _VALIDATION_RATIO
        self.nEpochs = nEpochs
        self.batch_size = batch_size
        self.n_discriminator_update = n_discriminator_update
        self.n_generator_update = n_generator_update

        self.loss_ae, self.decodeVariables = self.ae_net(self.x) # AE
        self.x_ = self.g_net(self.z) # G, get generated data
        self.d = self.d_net(self.x, self.keep_prob, reuse=False) # D, in the beginning, no reuse

        all_regs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        self.trainX, self.testX, self.num_data, self.dim_data = self.loadData()
        self.nBatches = int(np.ceil(float(self.trainX.shape[0]) / float(self.batch_size)))

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.optimize_ae = tf.train.AdamOptimizer().minimize(self.loss_ae + sum(all_regs), var_list=self.ae_net.vars)
            # self.d_rmsprop = tf.train.AdamOptimizer()  # DP case
            # grads_and_vars = self.d_rmsprop.compute_gradients(self.d_loss_reg, var_list=self.d_net.vars)
            # dp_grads_and_vars = []  # noisy version
            # for gv in grads_and_vars:  # for each pair
            #     g = gv[0]  # get the gradient, type in loop one: Tensor("gradients/AddN_37:0", shape=(4, 4, 1, 64), dtype=float32)
            #     #print g # shape of all vars
            #     if g is not None:  # skip None case
            #         g = self.dpnoise(g, self.batch_size)  # add noise on the tensor, type in loop one: Tensor("Add:0", shape=(4, 4, 1, 64), dtype=float32)
            #     dp_grads_and_vars.append((g, gv[1]))
            # self.d_rmsprop_new = self.d_rmsprop.apply_gradients(dp_grads_and_vars) # should assign to a new optimizer
            self.d_rmsprop = tf.train.AdamOptimizer() \
                .minimize(self.d_loss + sum(all_regs), var_list=self.d_net.vars) # non-DP case
            self.g_rmsprop = tf.train.AdamOptimizer() \
                .minimize(self.g_loss + sum(all_regs), var_list=self.g_net.vars+self.decodeVariables.values())

        self.d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in self.d_net.vars]
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.wdis_store = []  # store Wasserstein distance, new added

    def train_autoencoder(self, pretrainEpochs, pretrainBatchSize):
        '''Pre-training autoencoder'''
        for epoch in range(pretrainEpochs):
            idx = random.permutation(self.trainX.shape[0])
            trainLossVec = []
            nTrainBatches = int(ceil(float(self.trainX.shape[0])) / float(pretrainBatchSize))
            for i in range(nTrainBatches):
                batchX = self.trainX[idx[i * pretrainBatchSize:(i + 1) * pretrainBatchSize]]
                _, loss = self.sess.run([self.optimize_ae, self.loss_ae], feed_dict={self.x: batchX})
                trainLossVec.append(loss)
            print 'Pretrain_Epoch:%d, trainLoss:%f' % (epoch, np.mean(trainLossVec))

    def train(self, nEpochs=500, batch_size=1000):
        plt.ion()
        self.sess.run(tf.initialize_all_variables())
        start_time = time.time()

        idx = np.arange(self.trainX.shape[0])
        for epoch in range(nEpochs):
            for i in range(self.nBatches):
                for _ in range(self.discriminatorTrainPeriod): # train discriminator
                    batchIdx = random.choice(idx, size=batch_size, replace=False)
                    batchX = self.trainX[batchIdx]
                    randomX = random.normal(size=(batch_size, self.randomDim))
                    self.sess.run(self.d_clip)
                    #_, rd_loss = self.sess.run([self.d_rmsprop_new, self.d_loss], feed_dict={self.x: batchX, self.z: randomX}) # DP case
                    _, rd_loss = self.sess.run([self.d_rmsprop, self.d_loss], feed_dict={self.x: batchX, self.z: randomX}) # non-DP case
                randomX = random.normal(size=(batch_size, self.randomDim))
                _, rg_loss = self.sess.run([self.g_rmsprop, self.g_loss], feed_dict={self.x: batchX, self.z: randomX, self.keep_prob: 1.0, bn_train: True})

                print('Time [%5.4f] d_loss [%.4f] g_loss [%.4f]' %
                        (time.time() - start_time, rd_loss, rg_loss))

                # store rd_loss and rg_loss, new added
                self.g_loss_store.append(rg_loss)  # g_loss will decrease, here is not self.g_loss nor self.g_loss_reg
                self.d_loss_store.append(rd_loss)  # d_loss will increase
                self.wdis_store.append(rd_loss)  # Wasserstein distance will decrease

        z_sample = self.z_sampler(self.num_data, self.z_dim) # generate EHR from generator, after finish training
        x_gene = self.sess.run(self.x_, feed_dict={self.z: z_sample}) # type(x_gene): , x_gene[0].shape:

        # store generated EHR and figures
        with open('./result/genefinalfig/x_gene.pickle', 'wb') as fp:
            pickle.dump(x_gene, fp)

        rv, gv = dwp(self.trainX, x_gene, self.testX) #


    def loadData(self):
        MIMIC_data, num_data, dim_data = data_readf()
        if self.dataType == 'binary':
            MIMIC_data = clip(MIMIC_data, 0, 1)
        trainX, testX = train_test_split(MIMIC_data, test_size=self._VALIDATION_RATIO, random_state=0)
        return trainX, testX, num_data, dim_data

    def dpnoise(self, tensor, batch_size):
        '''add noise to tensor'''
        s = tensor.get_shape().as_list()  # get shape of the tensor
        sigma = 0.0  # assign it manually
        cg = 0.0
        rt = tf.random_normal(s, mean=0.0, stddev=sigma * cg)
        t = tf.add(tensor, tf.scalar_mul((1.0 / batch_size), rt))
        return t

    def loss_store(self):
        '''store everything new added'''
        # store figure
        t = arange(len(self.g_loss_store))
        plt.close() # clears the entire current figure with all its axes
        plt.plot(t, self.wdis_store, 'r--')
        plt.xlabel('Generator iterations (*10^{2})')
        plt.ylabel('Wasserstein distance')
        plt.savefig('./result/lossfig/wdis.jpg')
        # store to file
        wpick = file('./result/lossfile/wdis.pckl', "w")
        pickle.dump(self.wdis_store, wpick)
        wpick.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--data', type=str, default='MIMIC-III')
    parser.add_argument('--model', type=str, default='fc')
    parser.add_argument('--gpus', type=str, default='0')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    data = importlib.import_module(args.data) # from parser
    model = importlib.import_module(args.data + '.' + args.model)
    dataType = 'binary' # some parameters
    inputDim = 942
    embeddingDim = 128
    randomDim = 128
    generatorDims = list((128, 128)) + [embeddingDim]
    discriminatorDims = (256, 128, 1)
    compressDims = list(()) + [embeddingDim]
    decompressDims = list(()) + [inputDim]
    bnDecay = 0.99
    l2scale = 0.001
    bn_train = True
    _VALIDATION_RATIO = 0.1
    n_pretrain_epoch = 100
    pretrain_batch_size = 128
    nEpochs = 1000
    batch_size = 1024
    n_discriminator_update = 2
    n_generator_update = 1
    if dataType == 'binary':
        aeActivation = tf.nn.tanh
    else:
        aeActivation = tf.nn.relu
    generatorActivation = tf.nn.relu
    discriminatorActivation = tf.nn.relu
    xs = data.DataSampler()
    zs = data.NoiseSampler()
    ae_net = model.Autoencoder(inputDim, l2scale, compressDims, aeActivation, decompressDims, dataType) # ? autoencoder network
    g_net = model.Generator(randomDim, l2scale, generatorDims, bn_train, generatorActivation, bnDecay, dataType)# ?
    d_net = model.buildDiscriminator(inputDim, discriminatorDims, discriminatorActivation)# ?
    wgan = MIMIC_WGAN(g_net, d_net, ae_net, xs, zs, args.data, args.model, batch_size, decompressDims, aeActivation, dataType, _VALIDATION_RATIO, nEpochs, batch_size, n_discriminator_update, n_generator_update)
    wgan.train_autoencoder(n_pretrain_epoch, pretrain_batch_size) # Pre-training autoencoder
    wgan.train(batch_size)
    wgan.loss_store() # new added