import numpy as np
from utilize import loaddata_face


path = "/home/xieliyan/Dropbox/GPU/Data/face_test/LFW/lfw_aligned_cropped_64641/"



class DataSampler(object):
    def __init__(self):
        self.shape = [64, 64, 1]

    def __call__(self, batch_size):
        return loaddata_face(path, batch_size) # no longer use this due to speed reason


    def data2img(self, data):
        return np.reshape(data, [data.shape[0]] + self.shape)


class NoiseSampler(object):
    def __call__(self, batch_size, z_dim):
        return np.random.uniform(-1.0, 1.0, [batch_size, z_dim]) # "changed"
        # the shape of return is: batch_size*z_dim