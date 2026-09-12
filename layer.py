import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Layer:
    def __init__(self):
        self.info = {
            'class': 'Layer'
        }

    def forward(self, features):
        return features

    def backward(self, gradients):
        return None

    def update(self, learning_rate):
        pass

class WeightedLayer(Layer):
    def __init__(self):
        super().__init__()
        self.info = {
            'class': 'WeightedLayer',
            'input': 0,
            'output': 0
        }
        self.weights = None
        self.bias = None

    def get_states(self):
        return {
            'weights': self.weights,
            'bias': self.bias
        }

    def load_states(self, state_dict):
        if state_dict is None:
            return {
                'weights': None,
                'bias': None
            }
        try:
            self.weights = state_dict['weights']
            self.bias = state_dict['bias']
        except KeyError:
            raise KeyError('Weighted Layer requires load files to have both \'weights\' and \'bias\' as it\'s entry')

class Dense(WeightedLayer):
    def __init__(self, num_weights=1, num_perceptrons=1, scale=1):
        super().__init__()
        self.catch_features = None
        self.gradients = None
        self.weights = np.random.randn(num_weights, num_perceptrons) * scale
        self.bias = np.random.randn(1, num_perceptrons)
        self.info = {
            'class': 'Dense',
            'input': num_weights,
            'output': num_perceptrons
        }
        
        #Set all weights and bias to 1 (for debug)
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

        #Track weights and bias after update (for debug)
        #print('weight')
        #print(self.weights)
        #print('bias')
        #print(self.bias)

        self.gradients = None
        self.bias_gradient = None

    def load_states(self, state_dict):
        if state_dict:
            super().load_states(state_dict)
            self.info['input'] = self.weights.shape[0]
            self.info['output'] = self.weights.shape[1]
        else: return super().load_states(state_dict)
        
class ActivateLayer(Layer):
    def __init__(self):
        super().__init__()
        self.info = {
            'class': 'ActivateLayer'
        }

    def load_para(self, para_dict):
        pass

class Sigmoid(ActivateLayer):
    def __init__(self, A=1, k=1, c=0):
        super().__init__()
        self.A = A
        self.k = k
        self.c = c
        self.info = {
            'class': 'Sigmoid',
            'A': A,
            'k': k,
            'c': c
        }

    def forward(self, features):
        self.predictions = self.A / (1 + np.exp(-self.k * (features - self.c)))
        return self.predictions

    def backward(self, gradients):
        return gradients * self.predictions * (self.A - self.predictions) * self.k / self.A

    def load_para(self, para_dict):
        super().load_para(para_dict)
        try:
            self.A = para_dict['A']
            self.k = para_dict['k']
            self.c = para_dict['c']
            self.info = {
                'class': 'Sigmoid',
                'A': self.A,
                'k': self.k,
                'c': self.c
            }
        except KeyError:
            raise KeyError('Unmatched parameter(s) for Activate Layer')

class Softmax(ActivateLayer):
    def __init__(self, T=1):
        super().__init__()
        self.T = T
        self.info = {
            'class': 'Softmax',
            'T': T
        }

    def forward(self, features):
        f_max = features.max(axis=1, keepdims=True)
        f_exp = np.exp(features - f_max)
        f_sum = f_exp.sum(axis=1, keepdims=True)
        self.predictions = f_exp / f_sum
        return self.predictions

    def backward(self, gradients):
        pred_grad = self.predictions * gradients
        return pred_grad - self.predictions * pred_grad.sum(axis=1, keepdims=True)

    def load_para(self, para_dict):
        super().load_para(para_dict)
        try:
            self.A = para_dict['T']
            self.info = {
                'class': 'Softmax',
                'T': self.T,
            }
        except KeyError:
            raise KeyError('Unmatched parameter(s) for Activate Layer')

class Tanh(ActivateLayer):
    def __init__(self, A=1, k=1, c=0):
        super().__init__()
        self.A = A
        self.k = k
        self.c = c
        self.info = {
            'class': 'Tanh',
            'A': A,
            'k': k,
            'c': c
        }

    def forward(self, features):
        self.predictions = self.A * np.tanh(self.k * (features - self.c))
        return self.predictions

    def backward(self, gradients):
        return gradients  * (self.k * (self.A - np.pow(self.predictions, 2) / self.A))

    def load_para(self, para_dict):
        super().load_para(para_dict)
        try:
            self.A = para_dict['A']
            self.k = para_dict['k']
            self.c = para_dict['c']
            self.info = {
                'class': 'Tanh',
                'A': self.A,
                'k': self.k,
                'c': self.c
            }
        except KeyError:
            raise KeyError('Unmatched parameter(s) for Activate Layer')
    
class ReLU(ActivateLayer):
    def __init__(self):
        super().__init__()
        self.info = {
            'class': 'ReLU'
        }

    def forward(self, features):
        self.predictions = (features > 0) * features
        return self.predictions

    def backward(self, gradients):
        return gradients * (self.predictions > 0)

layer_dict = {
    'Layer': Layer,
    'WeightedLayer': WeightedLayer,
    'Dense': Dense,
    'ActivateLayer': ActivateLayer,
    'Sigmoid': Sigmoid,
    'Softmax': Softmax,
    'Tanh': Tanh,
    'ReLU': ReLU,
}
