from numpy import random, clip
from utilize import *



class DataSampler(object):
    def __init__(self):
        self.MIMIC_data, self.num_data, self.dim_data = data_readf()
        self.shape = [self.dim_data]

    def __call__(self, batch_size, dataType):
        indices = random.choice(self.num_data, batch_size, replace=False) # sample a batch data points (without repeat)
        if dataType == 'binary':
            return clip(self.MIMIC_data[indices], 0, 1) # no label
        else:
            return self.MIMIC_data[indices]  # no label


class NoiseSampler(object):
    def __call__(self, batch_size, z_dim):
        return random.normal(size=(batch_size, z_dim)) # "changed"
        # the shape of return is: batch_size*z_dim
        # see Medgan line 209, use np.random.normal(), which has defauld std = 1.0