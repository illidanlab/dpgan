import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl
from tensorflow.contrib.layers import batch_norm
import matplotlib

def leaky_relu(x, alpha=0.1):
    return tf.maximum(tf.minimum(0.0, alpha * x), x) # this is equivalent to: x \leq 0: \alpha x, x > 0: x, here \alpha \leq 1

class Autoencoder(object):
    def __init__(self, inputDim, l2scale, compressDims, aeActivation, decompressDims, dataType):
        self.name = 'mimiciii/fc/autoencoder'
        self.x_dim = inputDim
        self.l2scale = l2scale
        self.compressDims = compressDims
        self.aeActivation = aeActivation
        self.decompressDims = decompressDims
        self.dataType = dataType

    def __call__(self, x_input, reuse=True):
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


class Discriminator(object):
    def __init__(self):
        self.x_dim = 784
        self.name = 'mimiciii/fc/d_net'

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name) as vs: # all variable below are prefixed with self.name = 'mnist/mlp/d_net'
            if reuse:
                vs.reuse_variables()
            bs = tf.shape(x)[0] # batch size
            x = tf.reshape(x, [bs, 28, 28, 1]) # reshape to [bs, 28, 28, 1] = [batch size, width, length, channel]
            conv1 = tc.layers.convolution2d(
                x, 64, [4, 4], [2, 2], # 64: number of output filters, [4, 4]: spatial dimensions of of the filters, [2, 2]: stride
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            ) # output shape: [bs, 16, 16, 64]
            conv1 = leaky_relu(conv1)
            conv2 = tc.layers.convolution2d(
                conv1, 128, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            ) # output shape: [bs, 10, 10, 128]
            conv2 = leaky_relu(tc.layers.batch_norm(conv2))
            conv2 = tcl.flatten(conv2)
            fc1 = tc.layers.fully_connected(
                conv2, 1024,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            )
            fc1 = leaky_relu(tc.layers.batch_norm(fc1))
            fc2 = tc.layers.fully_connected(fc1, 1, activation_fn=tf.identity)
            return fc2

    @property # decorator, a issue see: https://stackoverflow.com/questions/42817388/typeerror-list-object-is-not-callable-when-using-a-property
    def vars(self):
        return [var for var in tf.all_variables() if self.name in var.name] # string var.name contains substring self.name

    def loss(self, prediction, target):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(prediction, target))


class Generator(object):
    def __init__(self, bn_train, randomDim, l2scale, generatorDims, generatorActivation, bnDecay, dataType):
        self.randomDim = randomDim
        self.l2scale = l2scale
        self.generatorDims = generatorDims
        self.bn_train = bn_train
        self.generatorActivation = generatorActivation
        self.bnDecay = bnDecay
        self.name = 'mimiciii/fc/g_net'

    def __call__(self, z):
        tempVec = z
        tempDim = self.randomDim
        with tf.variable_scope(self.name, regularizer=tcl.l2_regularizer(self.l2scale)):
            for i, genDim in enumerate(self.generatorDims[:-1]):
                W = tf.get_variable('W_' + str(i), shape=[tempDim, genDim])
                h = tf.matmul(tempVec, W)
                h2 = batch_norm(h, decay=self.bnDecay, scale=True, is_training=self.bn_train, updates_collections=None)
                h3 = self.generatorActivation(h2)
                tempVec = h3 + tempVec
                tempDim = genDim
            W = tf.get_variable('W' + str(i), shape=[tempDim, self.generatorDims[-1]])
            h = tf.matmul(tempVec, W)
            h2 = batch_norm(h, decay=self.bnDecay, scale=True, is_training=self.bn_train, updates_collections=None)

            if self.dataType == 'binary':
                h3 = tf.nn.tanh(h2)
            else:
                h3 = tf.nn.relu(h2)

            output = h3 + tempVec
        return output

    @property
    def vars(self):
        return [var for var in tf.all_variables() if self.name in var.name]