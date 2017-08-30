# test pre-train of autoencoder
# the norm_gradient_variables converge, see "./testf/norm_gradient_variables.jpg (norm_gradient_variables for autoencoder (no bias).jpg)"
# for autoencoder.py

from __future__ import division, print_function, absolute_import

import tensorflow as tf
from numpy import linalg, arange, reshape
import matplotlib.pyplot as plt
from testf.fc import Autoencoder

def norm_w(v):
    ''' input must be a list with numpy array type elements'''
    return sum([linalg.norm(i) for i in v])

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/home/decs/2017-DPGAN/code/wgan/mnist/MNIST", one_hot=True)

# Parameters, use the ones in medgan
# learning_rate = 0.001
# training_epochs = 100
# batch_size = 100

# Parameters, use the ones in "Auto Encoder Example"
learning_rate = 0.001
training_epochs = 20
batch_size = 256

display_step = 1
examples_to_show = 10
dataType = 'count'
inputDim = 784
embeddingDim = 128
compressDims = list(()) + [embeddingDim]
decompressDims = list(()) + [inputDim]
l2scale = 0.001
aeActivation = tf.nn.relu
ae_net = Autoencoder(inputDim, l2scale, compressDims, aeActivation, decompressDims,
                           dataType)  # autoencoder network

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, inputDim])

# Construct model
decoder_op, _ = ae_net(X)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
# optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost) # in medgan
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Explore trainable variables (weight_bias)
var = [v for v in tf.trainable_variables() if 'mimiciii/fc/autoencoder' in v.name] # (784, 128), (128,), (128, 784), (784,)
var_grad = tf.gradients(cost, var) # gradient of cost w.r.t. trainable variables, len(var_grad): 8, type(var_grad): list
norm_gradient_variables = []

# Launch the graph
with tf.Session() as sess:
    writer = tf.train.SummaryWriter("./graph/my_graph", sess.graph)
    sess.run(init)
    total_batch = int(mnist.train.num_examples/batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
            var_grad_val = sess.run(var_grad, feed_dict={X: batch_xs})
            # var_grad_val = [var_grad_val[0], var_grad_val[2]] # no bias, change for different network
            if type(var_grad_val) != type([0]):  # if a is not a list, which indicate it contains only one weight matrix
                var_grad_val = [var_grad_val]
            norm_gradient_variables.append(norm_w(var_grad_val))  # compute the norm of all trainable variables
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")

    plt.plot(arange(len(norm_gradient_variables)), norm_gradient_variables, 'b--')
    plt.xlabel('Iterations')
    plt.ylabel('Norm of gradients of all trainable variables')
    plt.savefig('./norm_gradient_variables.jpg')
    print("Gradient compuation Finished!")

    # Applying encode and decode over test set
    encode_decode = sess.run(
        y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})
    # print('encode_decode')
    # print(encode_decode)
    # Compare original images with their reconstructions
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(reshape(encode_decode[i], (28, 28)))
    f.show()
    plt.draw()
    plt.waitforbuttonpress()
