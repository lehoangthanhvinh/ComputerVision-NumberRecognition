import numpy as np

class Sigmoid():
    def __init__(self, A=1, k=1, c=0):
        super().__init__()
        self.A = A
        self.k = k
        self.c = c

    def forward(self, features):
        self.predictions = self.A / (1 + np.exp(-self.k * (features - self.c)))
        return self.predictions

    def backward(self, loss):
        return loss * self.forward(loss) * (1 - self.forward(loss))

sig = Sigmoid()
forth = sig.forward(np.array([[1]]))
print(forth)
back = sig.backward(forth - 1)
print(back)
