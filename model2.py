import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from loss import MSE

class Model:
    def __init__(self, learning_rate=0.01, num_iterations=1000, report_frequency=100, pipeline=[], loss_func=MSE()):
        self.learning_rate = learning_rate
        self.report_frequency = report_frequency
        self.num_iterations = num_iterations
        self.pipeline = pipeline
        self.loss_func = loss_func

        self.snapshots = []
        self.label_dict = {}
        self.reversed_label_dict = {}

    def config(self, learning_rate=0, num_iterations=0, report_frequency=0, pipeline=[], loss_func=None):
        if learning_rate:
            self.learning_rate = learning_rate

        if num_iterations:
            self.num_iterations = num_iterations

        if report_frequency:
            self.report_frequency = report_frequency

        if pipeline:
            self.pipeline = pipeline

        if loss_func:
            self.loss_func = loss_func

    def fit(self, features, labels):
        for epoch in range(self.num_iterations):
            predictions = self.predict(features)
            loss = self.loss_func.forward(predictions, labels)
            gradients = self.loss_func.backward()
            
            for layer in reversed(self.pipeline):
                gradients = layer.backward(gradients)
            for layer in self.pipeline:
                layer.update(self.learning_rate)

            if epoch % self.report_frequency == 0:
                correct, loss = self.report(features, labels)
                print(f'Epoch: {epoch}, Correct: {correct}, ({correct / len(labels) * 100:.2f}%), loss={loss:.4f}')

    def predict(self, features):
        if not self.pipeline:
            raise ValueError('Pipeline configuration required')
        
        input = features
        for layer in self.pipeline:
            res = layer.forward(input)
            input = res
        return res

    def decide(self, prediction):
        decisions = np.array([i == i.max() for i in prediction])

        for decision in decisions:
            if decision.sum() > 1:
                first = next(i for i, a in enumerate(decision) if a)
                for i in range(len(decision)):
                    decision[i] = False if i > first else decision[i]

        return decisions

    def fit_catergorize_label(self, labels):
        self.label_dict = {}
        self.reversed_label_dict = {}

        labels_uniques = set(labels)
        for idx, label in enumerate(labels_uniques):
            tranformed = np.zeros(len(labels_uniques))
            tranformed[idx] = 1
            self.label_dict[str(label)] = tranformed

        self.reversed_label_dict = {tuple(v): k for k, v in self.label_dict.items()}
        return self.catergorize_label(labels)

    def catergorize_label(self, labels):
        if not self.label_dict:
            raise KeyError('Labels dict has not been made')
        
        catergorized = []
        for label in labels:
            if str(label) not in self.label_dict.keys():
                raise KeyError('Label values not found', label)
            catergorized.append(self.label_dict[str(label)])

        return np.array(catergorized)

    def reverse_labels(self, labels):
        if not self.label_dict:
            raise KeyError('Labels dict has not been made')
        
        return [self.reversed_label_dict[tuple(i)] for i in labels]

    def report(self, features, labels):
        predictions = self.predict(features)
        decisions = self.decide(predictions)
        loss = self.loss_func.forward(predictions, labels)
        correct = 0
        for i, label in enumerate(labels):
            if all(label == decisions[i]):
                correct += 1

        return (correct, loss)

    def save_model(self, filename='weight.npz'):
        pass

    def load_model():
        pass