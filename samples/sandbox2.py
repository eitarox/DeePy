import math
import numpy as np
from deepy import Function
from deepy import Variable
from deepy.utils import plot_dot_graph


class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * np.cos(x)
        return gx


def sin(x):
    return Sin()(x)


def my_sin(x, threshold=0.0001):
    y = 0
    for i in range(100000):
        c = (-1) ** i / math.factorial(2 * i + 1)
        t = c * x ** (2 * i + 1)
        y = y + t
        if abs(t.data) < threshold:
            break
    return y


x0 = Variable(np.array(np.pi/4))
y0 = my_sin(x0)
y0.backward()

print(y0.data)
print(x0.grad)
plot_dot_graph(y0, verbose=False, to_file='my_sin0.png')

x1 = Variable(np.array(np.pi/4))
y1 = my_sin(x1, threshold=1e-150)
y1.backward()

print(y1.data)
print(x1.grad)
plot_dot_graph(y1, verbose=False, to_file='my_sin1.png')

