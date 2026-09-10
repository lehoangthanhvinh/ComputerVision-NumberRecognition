How to use:
1. Create a main.py file
2. Import dataset, read data by using function read_data('data/mnist.pkl.gz')
    (dataset.py is created SPECIFICLY for data/mnist.pkl.gz, currently not supporting other forrm of data)
Ex:
import dataset as dts
MNIST = dts.read_data('data/mnist.pkl.gz')

3. After read_data the structure of the data should look like this
    mnist-------dataset1--------features1(50000, 784)
        |              |--------labels1(50000, )
        |
        --------dataset2--------features2(10000, 784)
        |              |--------labels2(10000, )
        |
        --------dataset3--------features3(10000, 784)
                       |--------labels3(10000, )
Ex: (Split data to train_set, validate_set, test_set)
train_set = MNIST[0]
validate_set = MNIST[1]
test_set = MNIST[2]

4. Import model2, layer, create a model and config pipelines
Ex:
model = Model()
model.config(pipeline=[Dense(784, num_perceptrons=10, scale=0.1), Softmax()])
model.config(learning_rate=0.1, num_iterations=1000, report_frequency=100)

5. Train model, and see results
Ex:
X = train_set[0]
y = model.fit_catergorize_label(train_set[1])
model.fit(X, y)
correct = model.report(X, y)
print(f'Final, Correct: {correct}, ({correct / len(y) * 100:.2f}%)')

Full example code:
------------------------------------------------------------------------------------------
import dataset as dts
from model2 import Model
from layer import Perceptron, Softmax
import numpy as np

MNIST = dts.read_data('data/mnist.pkl.gz')

train_set = MNIST[0]
validate_set = MNIST[1]
test_set = MNIST[2]

model = Model()
model.config(pipeline=[Dense(784, num_perceptrons=10, scale=0.1), Softmax()])
model.config(learning_rate=0.1, num_iterations=1000, report_frequency=100)

X = train_set[0]
y = model.fit_catergorize_label(train_set[1])
model.fit(X, y)
correct = model.report(X, y)
print(f'Final, Correct: {correct}, ({correct / len(y) * 100:.2f}%)')
-----------------------------------------------------------------------------------------