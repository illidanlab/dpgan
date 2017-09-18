# This file is to debug the curve in DPGAN paper, experiment 2, convergence of wdis


from __future__ import print_function

import tensorflow as tf
from numpy import random, sign, array
import matplotlib.pyplot as plt
from regression_class import generate_rnd_data
rng = random

def noise(tensor):
    '''add noise to tensor'''
    s = tensor.get_shape().as_list()  # get shape of the tensor
    rt = tf.random_normal(s, mean=0.0, stddev=0.5) # .04 for ls, 0.5 for lr
    t = tf.add(tensor, rt)
    return t

def s2hot(arr):
    '''scalar to one-hot'''
    h = [] # store one-hot vector
    for i in range(len(arr)):
        if arr[i][0] == 1.0:
            h.append([1, 0])
        else:
            h.append([0, 1])
    return array(h)

# least square, synthetic data, https://www.datahubbs.com/tensorflow-intro-linear-regression/
# logistic regression tensorflow, synthetic data, https://medium.com/all-of-us-are-belong-to-machines/gentlest-intro-to-tensorflow-4-logistic-regression-2afd0cabc54

# Parameters
n = 1000 # number of iterations
alpha = 0.1 # learning rate, 0.5 for ls, 0.1 for lr
c = 0.8 # clip value, 0.5 for ls, 0.8 for lr
class_num = 2 # number of class in lr

# Training Data
(feature_all, target_all, model) = generate_rnd_data(feature_size=30, sample_size=100, bias=True)
target_all = s2hot(sign(target_all)) # coutinues target to binary label, scalar to one-hot

# noise-free
# W = tf.Variable(tf.random_uniform([feature_all.shape[1], 1], -1.0, 1.0))
# b = tf.Variable(tf.zeros([feature_all.shape[0], 1]))
# x = tf.placeholder(tf.float32, [None, feature_all.shape[1]])
# y = tf.placeholder(tf.float32, [None, 1])
# y_hat = tf.matmul(x, W) + b
# loss = tf.reduce_mean(tf.square(y - y_hat))/(2*feature_all.shape[0])

W = tf.Variable(tf.random_uniform([feature_all.shape[1], class_num], -1.0, 1.0))
b = tf.Variable(tf.zeros([class_num]))
x = tf.placeholder(tf.float32, [None, feature_all.shape[1]])
y = tf.placeholder(tf.float32, [None, class_num])
y_hat = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(y_hat), reduction_indices=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
train = optimizer.minimize(loss)
feed = {x: feature_all, y: target_all}
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
cost_val_1 = [] # store cost value
for step in range(n):
    _, loss_val = sess.run([train, loss],
             feed_dict=feed)
    cost_val_1.append(loss_val)
    if n % 20 == 0:
        print(loss_val)
sess.close()

# noise-only
# W = tf.Variable(tf.random_uniform([feature_all.shape[1], 1], -1.0, 1.0))
# b = tf.Variable(tf.zeros([feature_all.shape[0], 1]))
# x = tf.placeholder(tf.float32, [None, feature_all.shape[1]])
# y = tf.placeholder(tf.float32, [None, 1])
# y_hat = tf.matmul(x, W) + b
# loss = tf.reduce_mean(tf.square(y - y_hat))/(2*feature_all.shape[0])

W = tf.Variable(tf.random_uniform([feature_all.shape[1], class_num], -1.0, 1.0))
b = tf.Variable(tf.zeros([class_num]))
x = tf.placeholder(tf.float32, [None, feature_all.shape[1]])
y = tf.placeholder(tf.float32, [None, class_num])
y_hat = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(y_hat), reduction_indices=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
grads_and_vars = optimizer.compute_gradients(loss, var_list=tf.global_variables())
dp_grads_and_vars = []
for gv in grads_and_vars:  # for each pair
    g = gv[0]  # get the gradient
    if g is not None:  # skip None case
        g = noise(g)  # add noise on the tensor
    dp_grads_and_vars.append((g, gv[1]))
optimizer_new = optimizer.apply_gradients(dp_grads_and_vars) # should assign to a new optimizer
feed = {x: feature_all, y: target_all}
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
cost_val_2 = []
for step in range(n):
    _, loss_val = sess.run([optimizer_new, loss],
             feed_dict=feed)
    cost_val_2.append(loss_val)
    if n % 20 == 0:
        print(loss_val)
sess.close()

# clip-only
# W = tf.Variable(tf.random_uniform([feature_all.shape[1], 1], -1.0, 1.0))
# b = tf.Variable(tf.zeros([feature_all.shape[0], 1]))
# x = tf.placeholder(tf.float32, [None, feature_all.shape[1]])
# y = tf.placeholder(tf.float32, [None, 1])
# y_hat = tf.matmul(x, W) + b
# loss = tf.reduce_mean(tf.square(y - y_hat))/(2*feature_all.shape[0])

W = tf.Variable(tf.random_uniform([feature_all.shape[1], class_num], -1.0, 1.0))
b = tf.Variable(tf.zeros([class_num]))
x = tf.placeholder(tf.float32, [None, feature_all.shape[1]])
y = tf.placeholder(tf.float32, [None, class_num])
y_hat = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(y_hat), reduction_indices=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
train = optimizer.minimize(loss)
graph_clip = [v.assign(tf.clip_by_value(v, -c, c)) for v in tf.global_variables()]
feed = {x: feature_all, y: target_all}
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
cost_val_3 = [] # store cost value
for step in range(n):
    _, loss_val = sess.run([train, loss],
             feed_dict=feed)
    sess.run(graph_clip)
    cost_val_3.append(loss_val)
    if n % 20 == 0:
        print(loss_val)
sess.close()

# noise-clip
W = tf.Variable(tf.random_uniform([feature_all.shape[1], 1], -1.0, 1.0))
b = tf.Variable(tf.zeros([feature_all.shape[0], 1]))
x = tf.placeholder(tf.float32, [None, feature_all.shape[1]])
y = tf.placeholder(tf.float32, [None, 1])
y_hat = tf.matmul(x, W) + b
loss = tf.reduce_mean(tf.square(y - y_hat))/(2*feature_all.shape[0])

W = tf.Variable(tf.random_uniform([feature_all.shape[1], class_num], -1.0, 1.0))
b = tf.Variable(tf.zeros([class_num]))
x = tf.placeholder(tf.float32, [None, feature_all.shape[1]])
y = tf.placeholder(tf.float32, [None, class_num])
y_hat = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(y_hat), reduction_indices=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
grads_and_vars = optimizer.compute_gradients(loss, var_list=tf.global_variables())
dp_grads_and_vars = []
for gv in grads_and_vars:  # for each pair
    g = gv[0]  # get the gradient
    if g is not None:  # skip None case
        g = noise(g)  # add noise on the tensor
    dp_grads_and_vars.append((g, gv[1]))
optimizer_new = optimizer.apply_gradients(dp_grads_and_vars) # should assign to a new optimizer
graph_clip = [v.assign(tf.clip_by_value(v, -c, c)) for v in tf.global_variables()]
feed = {x: feature_all, y: target_all}
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
cost_val_4 = []
for step in range(n):
    _, loss_val = sess.run([optimizer_new, loss],
             feed_dict=feed)
    sess.run(graph_clip)
    cost_val_4.append(loss_val)
    if n % 20 == 0:
        print(loss_val)
sess.close()



plt.figure(1)
plt.subplot(221)
plt.plot(range(len(cost_val_1)), cost_val_1, 'b')
plt.xlabel("Iteration")
plt.ylabel("Objective")
plt.title("Noise-free")
plt.grid(True)
plt.subplot(222)
plt.plot(range(len(cost_val_2)), cost_val_2)
plt.yscale('linear')
plt.xlabel("Iteration")
plt.ylabel("Objective")
plt.title("Noise-only")
plt.grid(True)
plt.subplot(223)
plt.plot(range(len(cost_val_3)), cost_val_3)
plt.yscale('linear')
plt.xlabel("Iteration")
plt.ylabel("Objective")
plt.title("Clip-only")
plt.grid(True)
plt.subplot(224)
plt.plot(range(len(cost_val_4)), cost_val_4)
plt.yscale('linear')
plt.xlabel("Iteration")
plt.ylabel("Objective")
plt.title("Noise_clip")
plt.grid(True)
plt.show()

