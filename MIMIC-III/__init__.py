from numpy import random, clip
from utilize import *


class NoiseSampler(object):
    def __call__(self, batch_size, z_dim):
        return random.normal(size=(batch_size, z_dim))
        # the shape of return is: batch_size*z_dim
        # see Medgan line 209, use np.random.normal(), which has defauld std = 1.0