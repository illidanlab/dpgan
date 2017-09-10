from numpy import random, clip

class DataSampler(object):
    def __init__(self):
        self.shape = [1, dim_data, 1]

    def __call__(self, batch_size, dataType): # __call__ method is called when the instance is called
        indices = random.choice(num_data, batch_size, replace=False) # sample a batch data points (without repeat)
        if dataType == 'binary':
            return clip(MIMIC_ICD9[indices], 0, 1) # no label
        else:
            return MIMIC_ICD9[indices]  # no label


class NoiseSampler(object):
    def __call__(self, batch_size, z_dim):
        return random.uniform(-1.0, 1.0, [batch_size, z_dim]) # "changed"
        # the shape of return is: batch_size*z_dim