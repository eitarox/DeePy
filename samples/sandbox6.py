import numpy as np
from deepy import Variable
import deepy.functions as F

x = Variable(np.array(([1, 2, 3], [4, 5, 6])))
# y = F.reshape(x, (6,))
y = F.transpose(x)
y.backward()
print(x.grad)
