import os
import time
import argparse
import importlib
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib as plt
import cPickle as pickle
from numpy import arange, random, ceil, mean
from utilize import load_MIMICIII, dwp
import logging # these 2 lines are used in GPU3
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from visualize import *


class MIMIC_WGAN(object):
    def __init__(self, g_net, d_net, ae_net, z_sampler, decompressDims, aeActivation, dataType, _VALIDATION_RATIO, top, batchSize, cilpc, n_discriminator_update): # changed
        self.g_net = g_net
        self.d_net = d_net
        self.ae_net = ae_net
        self.z_sampler = z_sampler
        self.x_dim = self.d_net.inputDim
        self.z_dim = self.g_net.randomDim
        self.decompressDims = decompressDims
        self.aeActivation = aeActivation
        self.dataType = dataType
        self.x = tf.placeholder(tf.float32, [None, self.x_dim], name='x')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.keep_prob = tf.placeholder('float')
        self._VALIDATION_RATIO = _VALIDATION_RATIO
        self.top = top
        self.batchSize = batchSize
        self.cilpc = cilpc
        self.n_discriminator_update = n_discriminator_update

        self.loss_ae, self.decodeVariables = self.ae_net(self.x) # AE, autoencoder
        self.x_ = self.g_net(self.z) # G, get generated data
        self.d_loss, self.g_loss, self.y_hat_real, self.y_hat_fake, _ = self.d_net(self.x, self.x_, self.keep_prob, self.decodeVariables, reuse=False) # D, in the beginning, no reuse
        self.trainX, self.testX, _ = load_MIMICIII(self.dataType, self._VALIDATION_RATIO, self.top) # load whole dataset and split into training and testing set
        self.nBatches = int(ceil(float(self.trainX.shape[0]) / float(self.batchSize))) # number of batch if using training set

        all_regs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.optimize_ae = tf.train.AdamOptimizer().minimize(self.loss_ae + sum(all_regs), var_list=self.ae_net.vars)
            # self.d_rmsprop = tf.train.AdamOptimizer()  # DP case
            # grads_and_vars = self.d_rmsprop.compute_gradients(self.d_loss_reg, var_list=self.d_net.vars)
            # dp_grads_and_vars = []  # noisy version
            # for gv in grads_and_vars:  # for each pair
            #     g = gv[0]  # get the gradient, type in loop one: Tensor("gradients/AddN_37:0", shape=(4, 4, 1, 64), dtype=float32)
            #     #print g # shape of all vars
            #     if g is not None:  # skip None case
            #         g = self.dpnoise(g, self.batchSize)  # add noise on the tensor, type in loop one: Tensor("Add:0", shape=(4, 4, 1, 64), dtype=float32)
            #     dp_grads_and_vars.append((g, gv[1]))
            # self.d_rmsprop_new = self.d_rmsprop.apply_gradients(dp_grads_and_vars) # should assign to a new optimizer
            self.d_rmsprop = tf.train.AdamOptimizer() \
                .minimize(self.d_loss + sum(all_regs), var_list=self.d_net.vars) # non-DP case
            self.g_rmsprop = tf.train.AdamOptimizer() \
                .minimize(self.g_loss + sum(all_regs), var_list=self.g_net.vars+self.decodeVariables.values())

        self.d_clip = [v.assign(tf.clip_by_value(v, -1*self.cilpc,  self.cilpc)) for v in self.d_net.vars]
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(tf.initialize_all_variables())
        self.g_loss_store = []  # store loss of generator
        self.d_loss_store = []  # store loss of discriminator
        self.wdis_store = []  # store Wasserstein distance, new added

    def train_autoencoder(self, pretrainEpochs, pretrainBatchSize):
        '''Pre-training autoencoder'''
        nTrainBatches = int(ceil(float(self.trainX.shape[0])) / float(pretrainBatchSize))
        nTestBatches = int(ceil(float(self.testX.shape[0])) / float(pretrainBatchSize))
        for epoch in range(pretrainEpochs):
            idx = random.permutation(self.trainX.shape[0]) # shuffle training data in each epoch
            trainLossVec = []
            for i in range(nTrainBatches):
                batchX = self.trainX[idx[i * pretrainBatchSize:(i + 1) * pretrainBatchSize]]
                _, loss = self.sess.run([self.optimize_ae, self.loss_ae], feed_dict={self.x: batchX})
                trainLossVec.append(loss)
            idx = random.permutation(self.testX.shape[0])
            testLossVec = []
            for i in range(nTestBatches):
                batchX = self.testX[idx[i * pretrainBatchSize:(i + 1) * pretrainBatchSize]]
                loss = self.sess.run(self.loss_ae, feed_dict={self.x: batchX})
                testLossVec.append(loss)
            print 'Pretrain_Epoch:%d, trainLoss:%f, validLoss:%f' % (epoch, mean(trainLossVec), mean(testLossVec))

    def train(self, nEpochs, batchSize):
        start_time = time.time()
        idx = arange(self.trainX.shape[0])
        for epoch in range(nEpochs):
            for i in range(self.nBatches):
                rd_loss = 0
                rg_loss = 0
                for _ in range(self.n_discriminator_update): # train discriminator
                    batchIdx = random.choice(idx, size=batchSize, replace=False)
                    batchX = self.trainX[batchIdx]
                    randomZ = self.z_sampler(batchSize, self.z_dim)
                    #_, rd_loss = self.sess.run([self.d_rmsprop_new, self.d_loss], feed_dict={self.x: batchX, self.z: randomZ}) # DP case
                    _, rd_loss = self.sess.run([self.d_rmsprop, self.d_loss], feed_dict={self.x: batchX, self.z: randomZ, self.keep_prob: 1.0}) # non-DP case
                    self.sess.run(self.d_clip)

                randomZ = self.z_sampler(batchSize, self.z_dim) # train generator
                _, rg_loss = self.sess.run([self.g_rmsprop, self.g_loss], feed_dict={self.x: batchX, self.z: randomZ, self.keep_prob: 1.0})

                if i % 50 == 0: # print out loss
                    print('Time [%5.4f] d_loss [%.4f] g_loss [%.4f]' %
                          (time.time() - start_time, rd_loss, rg_loss))

                # store rd_loss and rg_loss, new added
                self.g_loss_store.append(rg_loss)  # g_loss will decrease, here is not self.g_loss nor self.g_loss_reg
                self.d_loss_store.append(rd_loss)  # d_loss will increase
                self.wdis_store.append(-1*rd_loss)  # Wasserstein distance will decrease

        z_sample = self.z_sampler(self.trainX.shape[0], self.z_dim) # generate EHR from generator, after finish training
        x_gene = self.sess.run(self.x_, feed_dict={self.z: z_sample})
        self.dec = self.decoder(x_gene)
        x_gene_dec = self.sess.run(self.dec)
        # print "please check this part, make sure it is correct"
        # print self.trainX.shape, x_gene.shape, x_gene_dec.shape, self.testX.shape
        return x_gene_dec, dwp(self.trainX, x_gene_dec, self.testX) # Dimension-wise prediction, note that self.trainX and self.testX are numpy array but self.decoder(x_gene) is tensor

    def decoder(self, x_fake): # this function is specifically to make sure the output of generator goes through the decoder
        tempVec = x_fake
        i = 0
        for _ in self.decompressDims[:-1]:
            tempVec = self.aeActivation(tf.add(tf.matmul(tempVec, self.decodeVariables['aed_W_' + str(i)]), self.decodeVariables['aed_b_' + str(i)]))
            i += 1
        if self.dataType == 'binary':
            x_decoded = tf.nn.sigmoid(tf.add(tf.matmul(tempVec, self.decodeVariables['aed_W_' + str(i)]), self.decodeVariables['aed_b_' + str(i)]))
        else:
            x_decoded = tf.nn.relu(tf.add(tf.matmul(tempVec, self.decodeVariables['aed_W_' + str(i)]), self.decodeVariables['aed_b_' + str(i)]))
        return x_decoded

    def dpnoise(self, tensor, batchSize):
        '''add noise to tensor'''
        s = tensor.get_shape().as_list()  # get shape of the tensor
        sigma = 6000.0  # assign it manually
        cg = 0.0
        rt = tf.random_normal(s, mean=0.0, stddev=sigma * cg)
        t = tf.add(tensor, tf.scalar_mul((1.0 / batchSize), rt))
        return t

    def loss_store(self, x_gene, rv, gv):
        '''store everything new added'''
        with open('./result/genefinalfig/real.pickle', 'wb') as fp:
            pickle.dump(rv, fp)
        with open('./result/genefinalfig/generated.pickle', 'wb') as fp:
            pickle.dump(gv, fp)
        t = arange(len(self.wdis_store))
        plt.close() # clears the entire current figure with all its axes
        plt.plot(t, self.wdis_store, 'r--')
        plt.xlabel('Generator iterations (*10^{2})')
        plt.ylabel('Wasserstein distance')
        plt.savefig('./result/lossfig/wdis.jpg')
        with open('./result/lossfile/wdis.pckl', 'wb') as fp:
            pickle.dump(self.wdis_store, fp)
        with open('./result/genefinalfig/x_gene.pickle', 'wb') as fp: # store generated EHR and figures
            pickle.dump(x_gene, fp)
        plt.close()
        plt.scatter(rv, gv)
        plt.title('Scatter plot of dimension-wise MSE')
        plt.xlabel('Real')
        plt.ylabel('Generated')
        plt.savefig('./result/genefinalfig/dwp.jpg')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--data', type=str, default='MIMIC-III')
    parser.add_argument('--model', type=str, default='fc')
    parser.add_argument('--gpus', type=str, default='0')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    data = importlib.import_module(args.data) # from parser
    model = importlib.import_module(args.data + '.' + args.model)

    # some parameters
    dataType = 'binary'
    inputDim = 512
    embeddingDim = 128
    randomDim = 128
    generatorDims = list((128, 128)) + [embeddingDim]
    discriminatorDims = (256, 128, 1)
    compressDims = list(()) + [embeddingDim]
    decompressDims = list(()) + [inputDim]
    bnDecay = 0.99
    l2scale = 0.001
    pretrainEpochs = 100 #2, 100
    pretrainBatchSize = 128 #2, 128
    nEpochs = 1000 #2, 1000
    batchSize = 1024 #2, 1024
    cilpc = 0.01
    n_discriminator_update = 2
    bn_train = True
    _VALIDATION_RATIO = 0.25
    top = 512
    if dataType == 'binary':
        aeActivation = tf.nn.tanh
    else:
        aeActivation = tf.nn.relu
    generatorActivation = tf.nn.relu
    discriminatorActivation = tf.nn.relu

    zs = data.NoiseSampler()
    ae_net = model.Autoencoder(inputDim, l2scale, compressDims, aeActivation, decompressDims, dataType)
    g_net = model.Generator(randomDim, l2scale, generatorDims, bn_train, generatorActivation, bnDecay, dataType)
    d_net = model.buildDiscriminator(inputDim, discriminatorDims, discriminatorActivation, decompressDims, aeActivation, dataType, l2scale)
    wgan = MIMIC_WGAN(g_net, d_net, ae_net, zs, decompressDims, aeActivation, dataType, _VALIDATION_RATIO, top, batchSize, cilpc, n_discriminator_update)
    wgan.train_autoencoder(pretrainEpochs, pretrainBatchSize) # Pre-training autoencoder
    x_gene, tuplerg = wgan.train(nEpochs, batchSize)
    wgan.loss_store(x_gene, tuplerg[0], tuplerg[1])