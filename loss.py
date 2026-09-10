import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Loss:
    def __init__(self):
        self.info = {
            'class': 'Loss'
        }

    def forward(self, predictions, labels):
        self.predictions = predictions
        self.labels = labels

    def backward(self):
        pass

class MAE(Loss):
    def __init__(self):
        super().__init__()
        self.info = {
            'class': 'MAE'
        }

    def forward(self, predictions, labels):
        super().forward(predictions, labels)
        return np.abs(self.labels - self.predictions).mean()

    def backward(self):
        return np.sign(self.predictions - self.labels)
    
class MSE(Loss):
    def __init__(self):
        super().__init__()
        self.info = {
            'class': 'MSE'
        }

    def forward(self, predictions, labels):
        super().forward(predictions, labels)
        return (np.pow(self.labels - self.predictions, 2) / 2).mean()

    def backward(self):
        return self.predictions - self.labels

class BCE(Loss):
    def __init__(self):
        super().__init__()
        self.info = {
            'class': 'BCE'
        }

    def forward(self, predictions, labels):
        super().forward(predictions, labels)
        self.clipped = np.clip(self.predictions, 1e-7, 1.0-1e-7)
        return -(self.labels * np.log(self.clipped) + (1 - self.labels) * np.log(1 - self.clipped)).mean()

    def backward(self):
        return - self.labels / self.clipped + (1 - self.labels) / (1 - self.clipped)

class CCE(Loss):
    def __init__(self):
        super().__init__()
        self.info = {
            'class': 'CCE'
        }

    def forward(self, predictions, labels):
        super().forward(predictions, labels)
        self.clipped = np.clip(self.predictions, 1e-7, 1.0-1e-7)
        return -(self.labels * np.log(self.clipped)).mean()

    def backward(self):
        return -(self.labels / self.clipped)

loss_dict = {
    'Loss': Loss,
    'MAE': MAE,
    'MSE': MSE,
    'BCE': BCE,
    'CCE': CCE,
}