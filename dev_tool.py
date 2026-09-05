import numpy as np
import matplotlib.pyplot as plt

def activate_func_check(func, sample=np.linspace(-10, 10, 101)):
    #Created to check functionality of activate function
    #Create a func as a member of activate class defined in layer.py
    #And it will draw a graph of that function and it's derivative
    #Can also add customized range of the graph
    X = sample
    y = func.forward(X)
    z = func.backward(np.ones_like(sample))

    plt.plot(X, y, label='Function', color='red', linestyle='-')
    plt.plot(X, z, label='Derivative', color='blue', linestyle='--')
    plt.legend()
    plt.grid(True)
    plt.show()

'''
Ex: draw a graph of Sigmoid function and it's derivative
import layer

func = layer.Sigmoid(A=3, k=5, c=-1)
activate_func_check(func)
------------------------------------------------------------
'''