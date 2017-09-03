import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl

def leaky_relu(x, alpha=0.1):
    return tf.maximum(tf.minimum(0.0, alpha * x), x) # this is equivalent to: x \leq 0: \alpha x, x > 0: x, here \alpha \leq 1


class Discriminator(object):
    def __init__(self):
        self.x_dim = 784
        self.name = 'mnist/mlp/d_net'

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
            )
            conv1 = leaky_relu(conv1)
            conv2 = tc.layers.convolution2d(
                conv1, 128, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            )
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
    def __init__(self):
        self.z_dim = 100
        self.x_dim = 784
        self.name = 'mnist/mlp/g_net'

    def __call__(self, z):
        with tf.variable_scope(self.name) as vs:
            fc = z
            fc = tcl.fully_connected(
                fc, 512,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tcl.batch_norm
            )
            fc = leaky_relu(fc)
            fc = tcl.fully_connected(
                fc, 512,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tcl.batch_norm
            )
            fc = leaky_relu(fc)
            fc = tcl.fully_connected(
                fc, 512,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tcl.batch_norm
            )
            fc = leaky_relu(fc)
            fc = tc.layers.fully_connected(
                fc, 784,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.sigmoid
            )
            return fc

    @property
    def vars(self):
        return [var for var in tf.all_variables() if self.name in var.name]