import tensorflow as tf
import tensorflow.contrib.layers as tcl
from tensorflow.contrib.layers import batch_norm


class Autoencoder(object):
    def __init__(self, inputDim, l2scale, compressDims, aeActivation, decompressDims, dataType):
        self.x_dim = inputDim
        self.l2scale = l2scale
        self.compressDims = compressDims
        self.aeActivation = aeActivation
        self.decompressDims = decompressDims
        self.dataType = dataType
        self.name = 'mimiciii/fc/autoencoder'

    def __call__(self, x_input):
        decodeVariables = {}
        with tf.variable_scope(self.name, regularizer=tcl.l2_regularizer(self.l2scale)):
            tempVec = x_input
            tempDim = self.x_dim
            i = 0
            for compressDim in self.compressDims:
                # W = tf.get_variable('aee_W_' + str(i), initializer=tf.random_normal([tempDim, compressDim], stddev=0.05)) # in medgan
                # b = tf.get_variable('aee_b_' + str(i), initializer=tf.random_normal([compressDim], stddev=0.05))
                W = tf.get_variable('aee_W_' + str(i), shape=[tempDim, compressDim])
                b = tf.get_variable('aee_b_' + str(i), shape=[compressDim])
                tempVec = self.aeActivation(tf.add(tf.matmul(tempVec, W), b))
                tempDim = compressDim
                i += 1

            i = 0
            for decompressDim in self.decompressDims[:-1]:
                # W = tf.get_variable('aed_W_' + str(i), initializer=tf.random_normal([tempDim, decompressDim], stddev=0.05)) # in medgan
                # b = tf.get_variable('aed_b_' + str(i), initializer=tf.random_normal([decompressDim], stddev=0.05))
                W = tf.get_variable('aed_W_' + str(i), shape=[tempDim, decompressDim])
                b = tf.get_variable('aed_b_' + str(i), shape=[decompressDim])
                tempVec = self.aeActivation(tf.add(tf.matmul(tempVec, W), b))
                tempDim = decompressDim
                decodeVariables['aed_W_' + str(i)] = W
                decodeVariables['aed_b_' + str(i)] = b
                i += 1
            # W = tf.get_variable('aed_W_' + str(i), initializer=tf.random_normal([tempDim, self.decompressDims[-1]], stddev=0.05)) # in medgan
            # b = tf.get_variable('aed_b_' + str(i), initializer=tf.random_normal([self.decompressDims[-1]], stddev=0.05))
            W = tf.get_variable('aed_W_' + str(i), shape=[tempDim, self.decompressDims[-1]])
            b = tf.get_variable('aed_b_' + str(i), shape=[self.decompressDims[-1]])
            decodeVariables['aed_W_' + str(i)] = W
            decodeVariables['aed_b_' + str(i)] = b

            if self.dataType == 'binary':
                x_reconst = tf.nn.sigmoid(tf.add(tf.matmul(tempVec, W), b))
                loss = tf.reduce_mean(-tf.reduce_sum(x_input * tf.log(x_reconst + 1e-12) + (1. - x_input) * tf.log(1. - x_reconst + 1e-12), 1), 0)
            else:
                x_reconst = tf.nn.relu(tf.add(tf.matmul(tempVec, W), b))
                loss = tf.reduce_mean((x_input - x_reconst) ** 2)

        return loss, decodeVariables

    @property
    def vars(self):
        return [var for var in tf.trainable_variables() if self.name in var.name]


class Generator(object):
    def __init__(self, randomDim, l2scale, generatorDims, bn_train, generatorActivation, bnDecay, dataType):
        self.randomDim = randomDim
        self.l2scale = l2scale
        self.generatorDims = generatorDims
        self.bn_train = bn_train
        self.generatorActivation = generatorActivation
        self.bnDecay = bnDecay
        self.dataType = dataType
        self.name = 'mimiciii/fc/g_net'

    def __call__(self, z):
        tempVec = z
        tempDim = self.randomDim
        with tf.variable_scope(self.name, regularizer=tcl.l2_regularizer(self.l2scale)):
            for i, genDim in enumerate(self.generatorDims[:-1]):
                W = tf.get_variable('W_' + str(i), shape=[tempDim, genDim])
                h = tf.matmul(tempVec, W)
                h2 = batch_norm(h) # GAN: batch_norm(h, decay=self.bnDecay, scale=True, is_training=self.bn_train, updates_collections=None)
                h3 = self.generatorActivation(h2)
                # tempVec = h3 + tempVec # need in GAN
                tempDim = genDim
            W = tf.get_variable('W' + str(i), shape=[tempDim, self.generatorDims[-1]])
            h = tf.matmul(tempVec, W)
            h2 = h # GAN: batch_norm(h, decay=self.bnDecay, scale=True, is_training=self.bn_train, updates_collections=None)

            if self.dataType == 'binary':
                h3 = tf.nn.sigmoid(h2) # GAN: tf.nn.tanh(h2)
            else:
                h3 = tf.nn.relu(h2)

            output = h3 # GAN: + tempVec
        return output

    @property
    def vars(self):
        return [var for var in tf.trainable_variables() if self.name in var.name]


class Discriminator(object):
    def __init__(self, inputDim, discriminatorDims, discriminatorActivation, l2scale):
        self.inputDim = inputDim
        self.discriminatorDims = discriminatorDims
        self.discriminatorActivation = discriminatorActivation
        self.l2scale = l2scale
        self.name = 'mimiciii/fc/d_net'

    def __call__(self, x_input, keepRate, reuse=False):
        # batchSize = tf.shape(x_input)[0]
        # inputMean = tf.reshape(tf.tile(tf.reduce_mean(x_input, 0), [batchSize]), (batchSize, self.inputDim))
        # tempVec = tf.concat(axis = 1, values = [x_input, inputMean]) # https://stackoverflow.com/questions/41813665/tensorflow-slim-typeerror-expected-int32-got-list-containing-tensors-of-type
        # tempDim = self.inputDim * 2 # need in GAN
        tempVec = x_input
        tempDim = self.inputDim # remove in GAN
        with tf.variable_scope(self.name, reuse=reuse): # GAN: regularizer=tcl.l2_regularizer(self.l2scale)
            for i, discDim in enumerate(self.discriminatorDims[:-1]):
                W = tf.get_variable('W_' + str(i), shape=[tempDim, discDim])
                b = tf.get_variable('b_' + str(i), shape=[discDim])
                h = self.discriminatorActivation(tf.add(tf.matmul(tempVec, W), b))
                # h = tf.nn.dropout(h, keepRate) # need in GAN
                tempVec = h
                tempDim = discDim
            W = tf.get_variable('W', shape=[tempDim, 1])
            b = tf.get_variable('b', shape=[1])
            y_hat = tf.squeeze(tf.add(tf.matmul(tempVec, W), b)) # need sigmoid in GAN

        return y_hat, self.name


class buildDiscriminator(object):
    '''Generated data need to go through a decoder before enter discriminator, real data enter discriminator directly'''
    def __init__(self, inputDim, discriminatorDims, discriminatorActivation, decompressDims, aeActivation, dataType, l2scale):
        self.d = Discriminator(inputDim, discriminatorDims, discriminatorActivation, l2scale) # it contains a discriminator
        self.inputDim = inputDim
        self.decompressDims = decompressDims
        self.aeActivation = aeActivation
        self.dataType = dataType
        self.name = 'mimiciii/fc/build_d_net'

    def __call__(self, x_real, x_fake, keepRate, decodeVariables, reuse=True):
        y_hat_real, self.name = self.d(x_real, keepRate, reuse=False)
        tempVec = x_fake
        i = 0
        for _ in self.decompressDims[:-1]:
            tempVec = self.aeActivation(tf.add(tf.matmul(tempVec, decodeVariables['aed_W_' + str(i)]), decodeVariables['aed_b_' + str(i)]))
            i += 1
        if self.dataType == 'binary':
            x_decoded = tf.nn.sigmoid(tf.add(tf.matmul(tempVec, decodeVariables['aed_W_' + str(i)]), decodeVariables['aed_b_' + str(i)]))
        else:
            x_decoded = tf.nn.relu(tf.add(tf.matmul(tempVec, decodeVariables['aed_W_' + str(i)]), decodeVariables['aed_b_' + str(i)]))
        y_hat_fake, self.name = self.d(x_decoded, keepRate, reuse=True)
        d_loss = -tf.reduce_mean(y_hat_real) + tf.reduce_mean(y_hat_fake)
        g_loss = -tf.reduce_mean(y_hat_fake)

        return d_loss, g_loss, y_hat_real, y_hat_fake, x_decoded

    @property
    def vars(self):
        return [var for var in tf.trainable_variables() if self.name in var.name]