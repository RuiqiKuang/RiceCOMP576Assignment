import imageio
import matplotlib as mp
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow.compat.v1 as tf
from scipy import misc

tf.disable_v2_behavior()


# --------------------------------------------------
# setup

def weight_variable(shape):
    """
    Initialize weights
    :param shape: shape of weights, e.g. [w, h ,Cin, Cout] where
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters
    Cout: the number of filters
    :return: a tensor variable for weights with initial values
    """

    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """
    Initialize biases
    :param shape: shape of biases, e.g. [Cout] where
    Cout: the number of filters
    :return: a tensor variable for biases with initial values
    """

    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    """
    Perform 2-D convolution
    :param x: input tensor of size [N, W, H, Cin] where
    N: the number of images
    W: width of images
    H: height of images
    Cin: the number of channels of images
    :param W: weight tensor [w, h, Cin, Cout]
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters = the number of channels of images
    Cout: the number of filters
    :return: a tensor of features extracted by the filters, a.k.a. the results after convolution
    """

    h_conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    return h_conv


def max_pool_2x2(x):
    """
    Perform non-overlapping 2-D maxpooling on 2x2 regions in the input data
    :param x: input data
    :return: the results of maxpooling (max-marginalized + downsampling)
    """

    h_max = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
    return h_max


ntrain = 1000  # per class
ntest = 100  # per class
nclass = 10  # number of classes
imsize = 28
nchannels = 1
batchsize = 64

Train = np.zeros((ntrain * nclass, imsize, imsize, nchannels))
Test = np.zeros((ntest * nclass, imsize, imsize, nchannels))
LTrain = np.zeros((ntrain * nclass, nclass))
LTest = np.zeros((ntest * nclass, nclass))

itrain = -1
itest = -1
for iclass in range(0, nclass):
    for isample in range(0, ntrain):
        path = '~/CIFAR10/Train/%d/Image%05d.png' % (iclass, isample)
        im = imageio.imread(path);  # 28 by 28
        im = im.astype(float) / 255
        itrain += 1
        Train[itrain, :, :, 0] = im
        LTrain[itrain, iclass] = 1  # 1-hot lable
    for isample in range(0, ntest):
        path = '~/CIFAR10/Test/%d/Image%05d.png' % (iclass, isample)
        im = imageio.imread(path);  # 28 by 28
        im = im.astype(float) / 255
        itest += 1
        Test[itest, :, :, 0] = im
        LTest[itest, iclass] = 1  # 1-hot lable

sess = tf.InteractiveSession()

tf_data = tf.placeholder(dtype=tf.float32, shape=[None, imsize, imsize,
                                                  nchannels])  # tf variable for the data, remember shape is [None,
# width, height, numberOfChannels]
tf_labels = tf.placeholder(dtype=tf.float32, shape=[None, nclass])  # tf variable for labels

# --------------------------------------------------
# model
# create your model
x_0 = tf_data
# Convolutional layer with kernel 5 x 5 and 32 filter maps followed by ReLU
W_1 = weight_variable([5, 5, 1, 32])
b_1 = bias_variable([32])
h_1 = tf.nn.relu(conv2d(x_0, W_1) + b_1)
# Max Pooling layer subsampling by 2
x_1 = max_pool_2x2(h_1)

# Convolutional layer with kernel 5 x 5 and 64 filter maps followed by ReLU
W_2 = weight_variable([5, 5, 32, 64])
b_2 = bias_variable([64])
h_2 = tf.nn.relu(conv2d(x_1, W_2) + b_2)
# Max Pooling layer subsampling by 2
x_2 = max_pool_2x2(h_2)

# Fully Connected layer that has input 7*7*64 and output 1024
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(x_2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Fully Connected layer that has input 1024 and output 10 (for the classes)
# Softmax layer (Softmax Regression + Softmax Nonlinearity)
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# --------------------------------------------------
# loss
# set up the loss, optimization, evaluation, and accuracy
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=tf_labels, logits=h_fc2))
opt = tf.train.AdamOptimizer(1e-3)
optimizer_name = "AdamOptimizer"
optimizer = opt.minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(h_fc2, 1), tf.argmax(tf_labels, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# --------------------------------------------------
# optimization

sess.run(tf.initialize_all_variables())
batch_xs = np.zeros(
    (batchsize, imsize, imsize, nchannels))  # setup as [batchsize, width, height, numberOfChannels] and use np.zeros()
batch_ys = np.zeros((batchsize, nclass))  # setup as [batchsize, the how many classes]
loss_list = []
acc_list = []
weight_first_layer = None
for i in range(2000):
    perm = np.arange(ntrain * nclass)
    np.random.shuffle(perm)
    for j in range(batchsize):
        batch_xs[j, :, :, :] = Train[perm[j], :, :, :]
        batch_ys[j, :] = LTrain[perm[j], :]
    loss = cross_entropy.eval(feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 0.5})
    acc = accuracy.eval(feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 0.5})
    weight_first_layer = W_1.eval()
    loss_list.append(loss)
    acc_list.append(acc)
    if i % 10 == 0:
        # calculate train accuracy and print it
        print('step %d, loss %g, training accuracy %g' % (i, loss, acc))
    # dropout only during training
    optimizer.run(feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 0.5})

# --------------------------------------------------
# test


print("test accuracy %g" % accuracy.eval(feed_dict={tf_data: Test, tf_labels: LTest, keep_prob: 1.0}))

sess.close()

# Plot the accuracy and loss under different parameters
_, ax = plt.subplots()
ax.plot(range(len(acc_list)), acc_list, 'k', label='accuracy')
ax.legend(loc='upper right', shadow=True)
plt.savefig('./pictures/' + optimizer_name + '_acc.png')

_, bx = plt.subplots()
bx.plot(range(len(loss_list)), loss_list, 'k', label='loss')
bx.legend(loc='upper right', shadow=True)
plt.savefig('./pictures/' + optimizer_name + '_loss.png')

# Plot the filters of the first layer
fig = plt.figure()
for i in range(32):
    ax = fig.add_subplot(6, 6, 1 + i)
    ax.imshow(weight_first_layer[:, :, 0, i], cmap='gray')
plt.savefig('./pictures/weight_first_layer.png')

