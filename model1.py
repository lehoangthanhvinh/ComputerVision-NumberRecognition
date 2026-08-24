import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Model:
    def __init__(self, weights):
        self.weights = np.zeros((weights, 1))
        self.bias = 0
        self.snapshots = []
        self.best_result = 0
        self.best_weights = None

    def predict(self, features):
        if len(features) != self.weights.shape[0]:
            print('Unmatched weight')
            raise ValueError("predicttion failed")
        res = np.dot(features, self.weights) + self.bias
        res = self.activate(res)
        return res

    def activate(self, val):
        return np.round(val, decimals=0)

    def predict_batch(self, batch):
        labels = []
        for features in batch:
            try: label = self.predict(features)[0]
            except ValueError as e: 
                print(e)
                return
            labels.append(label)
        return np.array(labels)

    def train(self, features, labels, num_iterations=1000, learning_rate=0.01, report_frequency=50):
        self.snapshots = []
        self.best_result = 0
        self.best_weights = None

        for epoch in range(num_iterations):
            for i, feature in enumerate(features):
                try: res = self.predict(feature)[0]
                except ValueError as e: 
                    print(e)
                    return

                self.update(feature, res, labels[i], learning_rate)

            if epoch % report_frequency == 0:
                correct, mse = self.report(features, labels)
                print(f'Epoch {epoch}: Correct: {correct} ({(100 * correct/len(features)):.2f}%), MSE: {mse:.2f}')

                self.snapshots.append(correct)
                if correct > self.best_result:
                    self.best_result = correct
                    self.best_weights = self.weights

    def take_best_weight(self):
        self.weights = self.best_weights

    def update(self, feature, prediction, label, learning_rate):
        ammount = (label - prediction) * learning_rate * feature
        ammount = ammount.reshape(self.weights.shape)

        self.weights += ammount
        self.bias += (label-prediction) * learning_rate * 3

    def report(self, features, labels):
        total = 0; correct = 0
        for i, feature in enumerate(features):
            res = self.predict(feature)[0]
            if labels[i] - res == 0: correct+=1
            total += pow(labels[i]-res, 2)
        mse = total/len(features)

        return (correct, mse)

    def save_weights(self, filename='weights.npz'):
        np.savez(
            filename,
            weights = self.weights,
            bias = self.bias
        )

    def load_weights(self, filename, weights_key='weights', bias_key='bias'):
        if not filename:
            raise ValueError('Filename required')
        
        try:
            with np.load(filename) as data:
                if weights_key not in data.keys():
                    raise ValueError('Weight key not found')
                if bias_key not in data.keys():
                    raise ValueError('Bias key not found')
                self.weights = data[weights_key]
                self.bias = data[bias_key]
        except FileNotFoundError: raise ValueError('File not found')


'''
def make_new_model():
    model = Model(784)
    model.train(train_set[0], train_set[1])
    correct, mse = model.report(train_set[0], train_set[1])
    print(f'Final: Correct: {correct} ({(100 * correct/len(train_set[0])):.2f}%), MSE: {mse:.2f}')
    model.save_weights()

def load_model_for_train():
    model = Model(784)
    model.load_weights('weights.npz')
    model.train(train_set[0], train_set[1])
    correct, mse = model.report(train_set[0], train_set[1])
    print(f'Final: Correct: {correct} ({(100 * correct/len(train_set[0])):.2f}%), MSE: {mse:.2f}')
    model.save_weights()


def check_model_by_labels():
    model = Model(784)
    model.load_weights('weights.npz')
    for key in clasified_train_set.keys():
        labels = model.predict_batch(clasified_train_set[key])
        correct = (labels == int(key)).sum() / labels.shape[0]
        print(f'Label {int(key)}: Correct: {correct*100:.2f}%')
'''