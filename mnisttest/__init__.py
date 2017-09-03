import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./mnisttest/MNIST')


class DataSampler(object):
    def __init__(self):
        self.shape = [28, 28, 1]

    def __call__(self, batch_size): # __call__ method is executed when the instance is called
        return mnist.train.next_batch(batch_size)
        # mnist.train.next_batch(batch_size) is a tuple with shape: (batch_size*784, batch_size), 2nd component is the row array for label
        # the data here is already normlized (/255)

    def data2img(self, data):
        return np.reshape(data, [data.shape[0]] + self.shape)


class NoiseSampler(object):
    def __call__(self, batch_size, z_dim):
        return np.random.uniform(-1.0, 1.0, [batch_size, z_dim]) # "changed"
        # the shape of return is: batch_size*z_dim