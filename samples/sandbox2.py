import numpy as np
from deepy import Function
from deepy import Variable

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




x = Variable(np.array(np.pi/4))
y = sin(x)
y.backward()

print(x.data)
print(x.grad)