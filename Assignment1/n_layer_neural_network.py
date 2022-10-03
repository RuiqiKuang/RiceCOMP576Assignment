__author__ = 'RK Kuang'

import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model


def generate_data():
    """
    generate data
    :return: X: input data, y: given labels
    """
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    return X, y


def plot_decision_boundary(pred_func, X, y):
    """
    plot the decision boundary
    :param pred_func: function used to predict the label
    :param X: input data
    :param y: given labels
    :return:
    """
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()

class Layer(object):
    def __init__(self, layerID, in_dim, out_dim, actFun_type, reg_lambda, seed):
        self.db = None
        self.dw = None
        self.delta = None
        self.layer_a = None
        self.layer_z = None
        self.input = None
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layerID = layerID
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda
        np.random.seed(seed)
        self.layer_W = np.random.randn(self.in_dim, self.out_dim) / np.sqrt(self.in_dim)
        self.layer_b = np.zeros((1, self.out_dim))

    def actFun(self, z, type):
        """
        actFun computes the activation functions
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: activations
        """
        res = 0.0
        if type == 'tanh':
            res = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
        elif type == 'sigmoid':
            res = 1.0 / (1.0 + np.exp(-z))
        elif type == 'relu':
            res = np.maximum(0, z)
        return res

    def diff_actFun(self, z, type):
        """
        diff_actFun compute the derivatives of the activation functions wrt the net input
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: the derivatives of the activation functions wrt the net input
        """
        res = 0
        if type == 'tanh':
            res = 4 * np.exp(2 * z) / ((np.exp(2 * z) + 1) ** 2)
        elif type == 'sigmoid':
            res = np.exp(-z) / ((1 + np.exp(-z)) ** 2)
        elif type == 'relu':
            res = np.where(z > 0, 1.0, 0.0)
        return res

    def feedforward(self, input):
        self.input = input
        self.layer_z = np.dot(input, self.layer_W) + self.layer_b
        self.layer_a = self.actFun(self.layer_z, self.actFun_type)

    def backprop(self, W, lastDelta):
        self.delta = np.dot(lastDelta, np.transpose(W)) * self.diff_actFun(self.layer_z, self.actFun_type)
        self.dw = np.dot(self.input.T, self.delta)
        self.db = np.sum(self.delta, axis=0, keepdims=True)


class NeuralNetwork(object):
    """
    This class builds and trains a neural network
    """

    def __init__(self, nn_input_dim, nn_hidden_dim, nn_num_layers, nn_output_dim, actFun_type='tanh', reg_lambda=0.01,
                 seed=0):
        """
        :param nn_input_dim: input dimension
        :param nn_hidden_dim: the number of hidden units
        :param nn_output_dim: output dimension
        :param actFun_type: type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        """
        self.probs = None
        self.z2 = None
        self.a1 = None
        self.z1 = None
        self.nn_input_dim = nn_input_dim
        self.nn_hidden_dim = nn_hidden_dim
        self.nn_num_layers = nn_num_layers
        self.nn_output_dim = nn_output_dim
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda

        # initialize the weights and biases in the network
        np.random.seed(seed)
        self.W1 = np.random.randn(self.nn_input_dim, self.nn_hidden_dim) / np.sqrt(self.nn_input_dim)
        self.b1 = np.zeros((1, self.nn_hidden_dim))
        self.W2 = np.random.randn(self.nn_hidden_dim, self.nn_output_dim) / np.sqrt(self.nn_hidden_dim)
        self.b2 = np.zeros((1, self.nn_output_dim))

        self.layers = []
        for i in range(self.nn_num_layers):
            if i == 0:
                self.layers.append(Layer(i, nn_input_dim, nn_hidden_dim, actFun_type, reg_lambda, seed))
            elif i == self.nn_num_layers - 1:
                self.layers.append(Layer(i, nn_hidden_dim, nn_output_dim, actFun_type, reg_lambda, seed))
            else:
                self.layers.append(Layer(i, nn_hidden_dim, nn_hidden_dim, actFun_type, reg_lambda, seed))

    def actFun(self, z, type):
        """
        actFun computes the activation functions
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: activations
        """
        res = 0.0
        if type == 'tanh':
            res = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
        elif type == 'sigmoid':
            res = 1.0 / (1.0 + np.exp(-z))
        elif type == 'relu':
            res = np.maximum(0, z)
        return res

    def diff_actFun(self, z, type):
        """
        diff_actFun compute the derivatives of the activation functions wrt the net input
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: the derivatives of the activation functions wrt the net input
        """
        res = 0
        if type == 'tanh':
            res = 4 * np.exp(2 * z) / ((np.exp(2 * z) + 1) ** 2)
        elif type == 'sigmoid':
            res = np.exp(-z) / ((1 + np.exp(-z)) ** 2)
        elif type == 'relu':
            res = np.where(z > 0, 1.0, 0.0)
        return res

    def feedforward(self, X, actFun):
        """
        feedforward builds a 3-layer neural network and computes the two probabilities,
        one for class 0 and one for class 1
        :param X: input data
        :param actFun: activation function
        :return:
        """

        # YOU IMPLEMENT YOUR feedforward HERE

        last_a = X
        for layer in self.layers:
            layer.feedforward(last_a)
            last_a = layer.layer_a

        self.z2 = self.layers[-1].layer_z
        exp_sum = np.exp(self.z2)
        self.probs = exp_sum / np.sum(exp_sum, axis=1, keepdims=True)
        return None

    def calculate_loss(self, X, y):
        """
        calculate_loss compute the loss for prediction
        :param X: input data
        :param y: given labels
        :return: the loss for prediction
        """
        num_examples = len(X)
        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))

        # Calculating the loss

        data_loss = np.sum(np.dot(y, np.log(self.probs)))

        # Add regularization term to loss (optional)
        sum_reg = sum([np.sum(np.square(layer.layer_W)) for layer in self.layers])
        sum_reg += np.sum(np.square(self.W1)) + np.sum(np.square(self.W2))
        data_loss += self.reg_lambda / 2 * sum_reg
        # data_loss += self.reg_lambda / 2 * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)))
        return (1. / num_examples) * data_loss

    def predict(self, X):
        """
        predict infers the label of a given data point X
        :param X: input data
        :return: label inferred
        """
        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
        return np.argmax(self.probs, axis=1)

    def backprop(self, X, y):
        """
        backprop run backpropagation to compute the gradients used to update the parameters in the backward step
        :param X: input data
        :param y: given labels
        :return: dL/dW1, dL/b1, dL/dW2, dL/db2
        """

        # IMPLEMENT YOUR BACKPROP HERE
        dldz2 = self.probs.copy()
        # print(self.probs.shape[0])
        dldz2[range(len(X)), y] -= 1
        dldz2 = dldz2 / len(X)

        self.layers[-1].delta = dldz2
        self.layers[-1].dw = np.dot(self.layers[-1].input.T, dldz2)  # dWO
        self.layers[-1].db = np.sum(dldz2, axis=0, keepdims=True)  # dbO
        for i in range(self.nn_num_layers - 2, -1, -1):
            self.layers[i].backprop(self.layers[i + 1].layer_W, self.layers[i + 1].delta)

    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        """
        fit_model uses backpropagation to train the network
        :param epsilon:
        :param X: input data
        :param y: given labels
        :param num_passes: the number of times that the algorithm runs through the whole dataset
        :param print_loss: print the loss or not
        :return:
        """
        # Gradient descent.
        for i in range(0, num_passes):
            # Forward propagation
            self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
            # Backpropagation
            self.backprop(X, y)

            for layer in self.layers:
                layer.dw += self.reg_lambda * layer.layer_W
                layer.layer_W -= epsilon * layer.dw
                layer.layer_b -= epsilon * layer.db

            self.W1 = self.layers[0].layer_W
            self.b1 = self.layers[0].layer_b
            self.W2 = self.layers[-1].layer_W
            self.b2 = self.layers[-1].layer_b
            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))

    def visualize_decision_boundary(self, X, y):
        """
        visualize_decision_boundary plot the decision boundary created by the trained network
        :param X: input data
        :param y: given labels
        :return:
        """
        plot_decision_boundary(lambda x: self.predict(x), X, y)


def main():
    # generate and visualize Make-Moons dataset
    X, y = generate_data()
    # plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    # plt.show()
    model = NeuralNetwork(nn_input_dim=2, nn_hidden_dim=3, nn_num_layers=5, nn_output_dim=2, actFun_type='tanh')
    model.fit_model(X, y)
    model.visualize_decision_boundary(X, y)


if __name__ == "__main__":
    main()
