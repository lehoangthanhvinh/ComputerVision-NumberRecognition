import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Layer:
    def __init__(self):
        self.weights = None
        self.bias = None

    def predict(self, features):
        return features

    def take_gradient(self, gradients):
        pass

    def update(self, learning_rate):
        pass

    def backward(self, loss):
        return None

class Perceptron(Layer):
    def __init__(self, num_weights, num_perceptrons=1, scale=1):
        super().__init__()
        self.catch_features = None
        self.gradients = None
        self.weights = np.random.randn(num_weights, num_perceptrons) * scale
        self.bias = np.random.randn(1, num_perceptrons)
        #self.weights = np.ones((num_weights, num_perceptrons))
        self.bias = np.zeros((1, num_perceptrons))

    def predict(self, features):
        if self.weights.shape[0] != features.shape[1]:
            raise ValueError('Unmatched weights and features')

        self.catch_features = features.T
        return np.dot(features, self.weights) + self.bias

    def take_gradient(self, gradients):
        if self.catch_features is None:
            raise ValueError('No catch features')
        self.gradients = np.dot(self.catch_features, gradients)
        self.bias_gradient = gradients.sum()
        self.catch_features = None

    def update(self, learning_rate):
        if (self.gradients is None) or (self.bias_gradient is None):
            raise ValueError('No gradients')
        self.weights += learning_rate * self.gradients
        #self.bias += learning_rate * self.bias_gradient

        #print('Bias')
        #print(self.bias)
        self.gradients = None
        self.bias_gradient = None

    def backward(self, loss):
        return np.dot(loss, self.weights.T)

class Sigmoid(Layer):
    def __init__(self, A=1, k=1, c=0):
        super().__init__()
        self.A = A
        self.k = k
        self.c = c

    def predict(self, features):
        return self.A / (1 + np.exp(-self.k * (features - self.c)))

    def backward(self, loss):
        return loss * self.predict(loss) * (1 - self.predict(loss))

class Softmax(Layer):
    def __init__(self, T=1):
        super().__init__()
        self.T = T

    def predict(self, features):
        return np.exp(self.T * features) / np.exp(self.T * features).sum()

class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def predict(self, features):
        return (features > 0) * features