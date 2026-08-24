import dataset as dts
from model2 import Model
from layer import Perceptron, Sigmoid
import numpy as np

MNIST = dts.read_data('data/mnist.pkl.gz')

train_set = MNIST[0]
validate_set = MNIST[1]
test_set = MNIST[2]

clasified_train_set = dts.classify_dataset(train_set)

model = Model()
model.config(pipeline=[Perceptron(784, num_perceptrons=10, scale=0.1), Sigmoid()])
model.config(learning_rate=0.01, num_iterations=100, report_frequency=10)
X = train_set[0]
y = model.fit_catergorize_label(train_set[1])
model.fit(X, y)
correct = model.report(X, y)
print(f'Final, Correct: {correct}, ({correct / len(y) * 100:.2f}%)')

