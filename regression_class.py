# CSE 491 Introduction to Machine Learning
# Python Demo for Linear Regression.
#
# By Jiayu Zhou, Oct 11, 2016

import time
import numpy as np
import matplotlib.pyplot as plt

def rand_split_train_test(data, label, train_perc):
    """
    Randomly split training and testing data by specify a percentage of training.
    :param data: X
    :param label: y
    :param train_perc: training percentage
    :return: training X, testing X, training y, testing y.
    """
    if train_perc >= 1 or train_perc <= 0:
        raise Exception('train_perc should be between (0,1).')
    sample_size = data.shape[0]
    if sample_size < 2:
        raise Exception('Sample size should be larger than 1. ')

    train_sample = np.max([np.floor(sample_size * train_perc).astype(int), 1])
    idx = np.random.permutation(data.shape[0])
    idx_tr = idx[0: train_sample]
    idx_te = idx[train_sample:]

    data_tr = data[idx_tr, :]
    data_te = data[idx_te, :]
    label_tr = label[idx_tr, :]
    label_te = label[idx_te, :]

    return data_tr, data_te, label_tr, label_te


def subsample_data(data, label, subsample_size):
    """
    Subsample a portion of data
    :param data: X
    :param label: y
    :param subsample_size: size of the subsample.
    :return: sampled X, sampled y
    """
    # protected sample size
    subsample_size = np.max([1, np.min([data.shape[0], subsample_size])])

    idx = np.random.permutation(data.shape[0])
    idx = idx[0: subsample_size]
    data = data[idx, :]
    label = label[idx, :]
    return data, label


def generate_rnd_data(feature_size, sample_size, bias=False):
    """
    Generate random data
    :param feature_size: number of features
    :param sample_size:  number of sample size
    :param bias:  do we include an extra bias term and .
    :return: data (sample_size X feature_size), label (sample_size X 1), truth_model (feature_size X 1)
    """
    # Generate X matrix.
    data = np.concatenate((np.random.randn(sample_size, feature_size), np.ones((sample_size, 1))), axis=1) \
        if bias else np.random.randn(sample_size, feature_size)  # the first dimension is sample_size (n X d)
    # # data normalization
    # data2 = []
    # for i in range(len(data)):
    #     data2.append(data[i]/np.linalg.norm(data[i]))
    #     # print np.linalg.norm(data2[i])
    # data2 = np.asarray(data2)
    # # print data2.shape

    # Generate ground truth (oracle) model.
    truth_model = np.random.randn(feature_size + 1, 1) * 10 \
        if bias else np.random.randn(feature_size, 1) * 10
    # # ground truth (oracle) model normalization
    # truth_model = truth_model/np.linalg.norm(truth_model)
    # # print truth_model.shape
    # Generate label.
    label = np.dot(data, truth_model)

    # add element-wise gaussian noise to each label.
    label += np.random.randn(sample_size, 1)
    return data, label, truth_model

def con2bi(arr):
    '''continues target to binary label'''
    res = np.sign(arr)
    return res

def mean_squared_error(true_label, predicted_label):
    """
    Compute the mean squared error given a set of predictive values and their ground truth.
    :param true_label: true target
    :param predicted_label: predicted target
    :return: mean squared error.
    """
    return np.sqrt(np.sum((true_label - predicted_label)**2)/true_label.size)


def least_squares(feature, target):
    """
    Compute least squares using closed form
    :param feature: X
    :param target: y
    :return: computed weight vector
    """
    return np.dot(np.linalg.inv(np.dot(feature.T, feature)), np.dot(feature.T, target))



def least_squares_gd(feature, target, max_iter=1000, step_size=0.00001, tol=1e-3): # check tol
    feature_dim = feature.shape[1]
    weight = np.zeros((feature_dim, 1))
    obj_val = []
    for i in range(max_iter):
        weight_old = weight.copy()

        err = np.dot(feature, weight) - target
        obj_val += [np.linalg.norm(err, 'fro')**2/2]
        grad = np.dot(feature.T, err)
        grad = dpnoise(grad)
        # print grad
        # print weight
        weight -= step_size * grad
        # print weight
        # weight = np.clip(weight, -0.05, +0.05) # clip the value of weight into [-5,+5]
        # stop criteria.
        if np.linalg.norm(weight - weight_old, 'fro') < tol:
            break
    return weight, obj_val


def ridge_regression(feature, target, lam=1e-17):
    """
    Compute ridge regression using closed form
    :param feature: X
    :param target: y
    :param lam: lambda
    :return:
    """
    feature_dim = feature.shape[1]
    return np.dot(np.linalg.inv(np.dot(feature.T, feature) + np.eye(feature_dim)*lam), np.dot(feature.T, target))


def dpnoise(arr):
    '''add noise and return result'''
    nv = np.random.normal(loc=0, scale=50, size = arr.shape)
    print np.linalg.norm(nv), np.linalg.norm(arr)

    return nv + arr


def exp5():
    # gradient descent.
    (feature_all, target_all, model) = generate_rnd_data(feature_size=30, sample_size=100, bias=True)
    # print feature_all.shape, target_all.shape, model.shape # (10000, 301) (10000, 1) (301, 1)
    # print feature_all[0]
    feature_train, feature_test, target_train, target_test = \
        rand_split_train_test(feature_all, target_all, train_perc=0.9)
    reg_model = least_squares(feature_train, target_train)
    reg_model_gd, obj_val = least_squares_gd(feature_train, target_train, max_iter=300, step_size=0.01) # 0.0001, 0.00001
    print obj_val
    print 'Model difference {}'.format(np.linalg.norm(reg_model - reg_model_gd, 'fro')/reg_model.size)
    plt.figure()
    plt.plot(range(len(obj_val)), obj_val, linestyle='-', color='r', label='Objective Value')
    plt.xlabel("Iteration")
    plt.ylabel("Objective")
    plt.title("Convergence.")
    plt.show()

    lsqr_obj = lambda model: np.linalg.norm(np.dot(feature_train, model) - target_train, 'fro') ** 2 / 2
    print 'Closed Form Objective: ', lsqr_obj(reg_model)
    print 'Gradient Descent Objective: ', lsqr_obj(reg_model)

if __name__ == '__main__':
    plt.interactive(False)

    # set seeds to get repeatable results.
    np.random.seed(491)

    exp5()

