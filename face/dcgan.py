import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl


def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x) # this is equivalent to: x \leq 0: \alpha x, x > 0: x, here \alpha \leq 1


class Discriminator(object):
    def __init__(self):
        self.x_dim = 784 #[28, 28, 1]
        self.name = 'face/dcgan/d_net'

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            bs = tf.shape(x)[0] # batch size
            x = tf.reshape(x, [bs, 28, 28, 1])
            conv1 = tc.layers.convolution2d(
                x, 64, [4, 4], [2, 2], # 64: number of output filters, [4, 4]: spatial dimensions of of the filters, [2, 2]: stride
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            ) # output shape: [bs, 16, 16, 64], the padding in this layer is P = 1, 16*16*64*4*4*3 = 786432
            conv1 = leaky_relu(conv1)
            conv2 = tc.layers.convolution2d(
                conv1, 128, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            ) # output shape: [bs, 8, 8, 128], the padding in this layer is P = 1  we will use this shape in generator, 8*8*128*4*4*64 = 8388608
            conv2 = leaky_relu(conv2)
            conv2 = tcl.flatten(conv2)
            fc1 = tc.layers.fully_connected(
                conv2, 1024,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            ) # 8*8*128*1024 = 8388608
            fc1 = leaky_relu(fc1)
            fc2 = tc.layers.fully_connected(fc1, 1, activation_fn=tf.identity)
            return fc2

    @property
    def vars(self):
        return [var for var in tf.all_variables() if self.name in var.name] # string var.name contains substring self.name

    def loss(self, prediction, target):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(prediction, target))


class Generator(object):
    def __init__(self):
        self.z_dim = 100
        self.x_dim = 784 #[28, 28, 1]
        self.name = 'face/dcgan/g_net'

    def __call__(self, z):
        with tf.variable_scope(self.name) as vs:
            bs = tf.shape(z)[0]
            fc1 = tc.layers.fully_connected(
                z, 1024,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            fc1 = tc.layers.batch_norm(fc1)
            fc1 = tf.nn.relu(fc1)
            fc2 = tc.layers.fully_connected(
                fc1, 7 * 7 * 128,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            fc2 = tf.reshape(fc2, tf.stack([bs, 7, 7, 128]))
            fc2 = tc.layers.batch_norm(fc2)
            fc2 = tf.nn.relu(fc2)
            conv1 = tc.layers.convolution2d_transpose( # convolution2d_transpose see "https://datascience.stackexchange.com/questions/6107/what-are-deconvolutional-layers", just reverse of input and output, remember to replace the number of output filter to the number of input layer
                fc2, 64, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            ) # output shape: [bs, 16, 16, 64]
            conv1 = tc.layers.batch_norm(conv1)
            conv1 = tf.nn.relu(conv1)
            conv2 = tc.layers.convolution2d_transpose(
                conv1, 1, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.sigmoid
            ) # output shape: [bs, 32, 32, 3]
            conv2 = tf.reshape(conv2, tf.stack([bs,784]))
            return conv2

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]