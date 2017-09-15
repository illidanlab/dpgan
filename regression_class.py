# CSE 491 Introduction to Machine Learning
# Python Demo for Linear Regression.
#
# By Jiayu Zhou, Oct 11, 2016
# This is a test for the DPGAN code. It shows how noise and clip affect convergence
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

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


def generate_rnd_data(feature_size, sample_size, bias=False):
    """
    Generate random data
    :param feature_size: number of features
    :param sample_size:  number of sample size
    :param bias:  do we include an extra bias term and .
    :return: data (sample_size X feature_size), label (sample_size X 1), truth_model (feature_size X 1)
    """
    # Generate X matrix.
    data = np.random.randn(sample_size, feature_size)
    # data zscore, along each feature (column)
    data = data.transpose()
    data = stats.zscore(data, axis=1)
    data = data.transpose()
    if bias:
        data = np.concatenate((data, np.ones((sample_size, 1))), axis=1)

    # Generate ground truth (oracle) model.
    truth_model = np.random.randn(feature_size + 1, 1) * 10 \
        if bias else np.random.randn(feature_size, 1) * 10
    # ground truth (oracle) model normalization
    truth_model = truth_model/np.linalg.norm(truth_model)
    # print truth_model.shape
    # Generate label.
    label = np.dot(data, truth_model)

    # add element-wise gaussian noise to each label.
    label += np.random.randn(sample_size, 1)
    return data, label, truth_model


def least_squares(feature, target):
    """
    Compute least squares using closed form
    :param feature: X
    :param target: y
    :return: computed weight vector
    """
    return np.dot(np.linalg.inv(np.dot(feature.T, feature)), np.dot(feature.T, target))*len(feature) # Divide objective function by the number of data point

def least_squares_gd1(feature, target, max_iter=1000, step_size=0.00001, tol=1e-3): # noise-free
    feature_dim = feature.shape[1]
    weight = np.zeros((feature_dim, 1))
    obj_val = []
    for i in range(max_iter):
        weight_old = weight.copy()
        err = np.dot(feature, weight) - target
        obj_val += [np.linalg.norm(err, 'fro')**2/2]
        grad = np.dot(feature.T, err)
        # print np.linalg.norm(grad)
        weight -= step_size * grad
        # if np.linalg.norm(weight - weight_old, 'fro') < tol:
        #     break
    return weight, obj_val

def least_squares_gd2(feature, target, max_iter=1000, step_size=0.00001, tol=1e-3): # noise-only
    feature_dim = feature.shape[1]
    weight = np.zeros((feature_dim, 1))
    obj_val = []
    for i in range(max_iter):
        weight_old = weight.copy()
        err = np.dot(feature, weight) - target
        obj_val += [np.linalg.norm(err, 'fro')**2/2]
        grad = np.dot(feature.T, err)
        noise = np.random.normal(loc=0, scale=10.0, size = grad.shape) # mind the relation between noise and step_size, scale=50 for unnormalized data
        # print np.linalg.norm(grad), np.linalg.norm(noise)
        grad = grad + noise
        weight -= step_size * grad
        # # stop criteria.
        # if np.linalg.norm(weight - weight_old, 'fro') < tol:
        #     break
    return weight, obj_val

def least_squares_gd3(feature, target, max_iter=1000, step_size=0.00001, tol=1e-3): # clip-only
    feature_dim = feature.shape[1]
    weight = np.zeros((feature_dim, 1))
    obj_val = []
    for i in range(max_iter):
        weight_old = weight.copy()
        err = np.dot(feature, weight) - target
        obj_val += [np.linalg.norm(err, 'fro')**2/2]
        grad = np.dot(feature.T, err)
        print np.linalg.norm(grad)
        weight -= step_size * grad
        weight = np.clip(weight, -0.05, +0.05) # clip the value of weight into [-5,+5]
        # # stop criteria.
        # if np.linalg.norm(weight - weight_old, 'fro') < tol:
        #     break
    return weight, obj_val

def least_squares_gd4(feature, target, max_iter=1000, step_size=0.00001, tol=1e-3): # noise_clip
    feature_dim = feature.shape[1]
    weight = np.zeros((feature_dim, 1))
    obj_val = []
    for i in range(max_iter):
        weight_old = weight.copy()
        err = np.dot(feature, weight) - target
        obj_val += [np.linalg.norm(err, 'fro')**2/2]
        grad = np.dot(feature.T, err)
        noise = np.random.normal(loc=0, scale=10.0, size = grad.shape) # mine the relation between noise and step_size
        # print np.linalg.norm(grad), np.linalg.norm(noise)
        grad = grad + noise
        # print weight
        weight -= step_size * grad
        weight = np.clip(weight, -0.05, +0.05) # clip the value of weight into [-5,+5]
        # # stop criteria.
        # if np.linalg.norm(weight - weight_old, 'fro') < tol:
        #     break
    return weight, obj_val


def exp5():
    # gradient descent.
    (feature_all, target_all, model) = generate_rnd_data(feature_size=30, sample_size=100, bias=True)
    # print feature_all.shape, target_all.shape, model.shape, type(feature_all)
    # print feature_all
    # print target_all
    # print feature_all[0]
    feature_train, feature_test, target_train, target_test = \
        rand_split_train_test(feature_all, target_all, train_perc=0.9)
    reg_model = least_squares(feature_train, target_train)
    max_iter_set = 1000
    step_size_set = 0.001 # 0.01 for unnormalized data
    reg_model_gd1, obj_val1 = least_squares_gd1(feature_train, target_train, max_iter=max_iter_set, step_size=step_size_set)
    reg_model_gd2, obj_val2 = least_squares_gd2(feature_train, target_train, max_iter=max_iter_set, step_size=step_size_set)
    reg_model_gd3, obj_val3 = least_squares_gd3(feature_train, target_train, max_iter=max_iter_set, step_size=step_size_set)
    reg_model_gd4, obj_val4 = least_squares_gd4(feature_train, target_train, max_iter=max_iter_set, step_size=step_size_set)

    print 'Model difference of noise-free {}'.format(np.linalg.norm(reg_model - reg_model_gd1, 'fro')/reg_model.size)
    print 'Model difference of noise-only {}'.format(np.linalg.norm(reg_model - reg_model_gd2, 'fro')/reg_model.size)
    print 'Model difference of clip-only {}'.format(np.linalg.norm(reg_model - reg_model_gd3, 'fro')/reg_model.size)
    print 'Model difference of noise_clip {}'.format(np.linalg.norm(reg_model - reg_model_gd4, 'fro')/reg_model.size)

    # plt.figure()
    # plt.plot(range(len(obj_val3)), obj_val3, linestyle='-', color='r', label='Objective Value')
    # plt.xlabel("Iteration")
    # plt.ylabel("Objective")
    # plt.title("Convergence, ")
    # plt.show()

    # plt.subplot(2, 2, 1)
    # plt.plot(range(len(obj_val1)), obj_val1, linestyle='-', color='r', label='Objective Value')
    # plt.xlabel("Iteration")
    # plt.ylabel("Objective")
    # plt.title("Convergence, noise-free")
    # plt.subplot(1, 3, 2)
    # plt.plot(range(len(obj_val2)), obj_val2, linestyle='-', color='r', label='Objective Value')
    # plt.xlabel("Iteration")
    # plt.ylabel("Objective")
    # plt.title("Convergence, noise-only")
    # plt.subplot(1, 3, 3)
    # plt.plot(range(len(obj_val3)), obj_val3, linestyle='-', color='r', label='Objective Value')
    # plt.xlabel("Iteration")
    # plt.ylabel("Objective")
    # plt.title("Convergence, clip-only")
    # plt.subplot(1, 3, 3)
    # plt.plot(range(len(obj_val4)), obj_val4, linestyle='-', color='r', label='Objective Value')
    # plt.xlabel("Iteration")
    # plt.ylabel("Objective")
    # plt.title("Convergence, noise_clip")
    # plt.show()

    # plot with various axes scales
    plt.figure(1)

    # linear
    plt.subplot(221)
    plt.plot(range(len(obj_val1)), obj_val1)
    plt.yscale('linear')
    plt.xlabel("Iteration")
    plt.ylabel("Objective")
    plt.title("Convergence, noise-free")
    plt.grid(True)

    # log
    plt.subplot(222)
    plt.plot(range(len(obj_val2)), obj_val2)
    plt.yscale('linear')
    plt.xlabel("Iteration")
    plt.ylabel("Objective")
    plt.title("Convergence, noise-only")
    plt.grid(True)

    # symmetric log
    plt.subplot(223)
    plt.plot(range(len(obj_val3)), obj_val3)
    plt.yscale('linear')
    plt.xlabel("Iteration")
    plt.ylabel("Objective")
    plt.title("Convergence, clip-only")
    plt.grid(True)

    # logit
    plt.subplot(224)
    plt.plot(range(len(obj_val4)), obj_val4)
    plt.yscale('linear')
    plt.xlabel("Iteration")
    plt.ylabel("Objective")
    plt.title("Convergence, noise_clip")
    plt.grid(True)


    plt.show()


    # lsqr_obj = lambda model: np.linalg.norm(np.dot(feature_train, model) - target_train, 'fro') ** 2 / 2
    # print 'Closed Form Objective: ', lsqr_obj(reg_model)
    # print 'Gradient Descent Objective: ', lsqr_obj(reg_model)

if __name__ == '__main__':
    plt.interactive(False)

    # set seeds to get repeatable results.
    np.random.seed(491)

    exp5()

