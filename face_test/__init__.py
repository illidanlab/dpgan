import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./face_test/MNIST')



class DataSampler(object):
    def __init__(self):
        self.shape = [28, 28, 1]

    def __call__(self, batch_size): # __call__ method is executed when the instance is called
        return mnist.train.next_batch(batch_size)


    def data2img(self, data):
        return np.reshape(data, [data.shape[0]] + self.shape)


class NoiseSampler(object):
    def __call__(self, batch_size, z_dim):
        return np.random.uniform(-1.0, 1.0, [batch_size, z_dim]) # "changed"
        # the shape of return is: batch_size*z_dim