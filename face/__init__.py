import numpy as np
from utilize import loaddata_face


path = "./face/CelebA/img_align_celeba_10000_1st_r_28/"

class DataSampler(object):
    def __init__(self):
        self.shape = [28, 28, 1]

    def __call__(self, batch_size):
        return loaddata_face(path, batch_size) # loaddata_face(path, batch_size).shape: (batch_size, 784) or (batch_size, 32, 32, 1)

    def data2img(self, data):
        return np.reshape(data, [data.shape[0]] + self.shape)


class NoiseSampler(object):
    def __call__(self, batch_size, z_dim):
        return np.random.uniform(-1.0, 1.0, [batch_size, z_dim]) # "changed"
        # the shape of return is: batch_size*z_dim