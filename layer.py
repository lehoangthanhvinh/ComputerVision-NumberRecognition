import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Layer:
    def __init__(self):
        self.weights = None
        self.bias = None

    def forward(self, features):
        return features

    def update(self, learning_rate):
        pass

    def backward(self, gradients):
        return None

class Perceptron(Layer):
    def __init__(self, num_weights, num_perceptrons=1, scale=1):
        super().__init__()
        self.catch_features = None
        self.gradients = None
        self.weights = np.random.randn(num_weights, num_perceptrons) * scale
        self.bias = np.random.randn(1, num_perceptrons)
        #self.weights = np.ones((num_weights, num_perceptrons))
        #self.bias = np.zeros((1, num_perceptrons))

    def forward(self, features):
        if self.weights.shape[0] != features.shape[1]:
            raise ValueError('Unmatched weights and features')

        self.catch_features = features.T
        return np.dot(features, self.weights) + self.bias

    def backward(self, gradients):
        if self.catch_features is None:
            raise ValueError('No catch features')
        self.gradients = np.dot(self.catch_features, gradients) / gradients.shape[0]
        self.bias_gradient = gradients.mean(axis=0)
        self.catch_features = None

        return np.dot(gradients, self.weights.T)

    def update(self, learning_rate):
        if (self.gradients is None) or (self.bias_gradient is None):
            raise ValueError('No gradients')
        self.weights -= learning_rate * self.gradients
        self.bias -= learning_rate * self.bias_gradient

        #print('weight')
        #print(self.weights)
        #print('bias')
        #print(self.bias)
        self.gradients = None
        self.bias_gradient = None
        

class Sigmoid(Layer):
    def __init__(self, A=1, k=1, c=0):
        super().__init__()
        self.A = A
        self.k = k
        self.c = c

    def forward(self, features):
        self.predictions = self.A / (1 + np.exp(-self.k * (features - self.c)))
        return self.predictions

    def backward(self, loss):
        return loss * self.predictions * (1 - self.predictions)

class Softmax(Layer):
    def __init__(self, T=1):
        super().__init__()
        self.T = T

    def forward(self, features):
        f_max = features.max(axis=1, keepdims=True)
        f_exp = np.exp(features - f_max)
        f_sum = f_exp.sum(axis=1, keepdims=True)
        self.predictions = f_exp / f_sum
        return self.predictions

    def backward(self, gradients):
        pred_grad = self.predictions * gradients
        return pred_grad - self.predictions * pred_grad.sum(axis=1, keepdim=True)

class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, features):
        return (features > 0) * features