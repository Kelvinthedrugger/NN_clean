# deprecated, not useful
# it's a experiment template rather
import module as nn
import numpy as np
"""haven't invoke topo so far"""
"""
(784,128)  (28,28)
l11          l12

(128,10)   (28,10)
l21          l22
   
    l.concat: adder (128+28,10)

      loss
"""
# dataset
x = np.array([[0, 0, 1, 1, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 1, 1, 1, 0, 0], [
             0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0]])
x1 = np.concatenate([[[ele]*4 for ele in row]*4 for row in x])
x2 = x1.reshape(28, 28)
y = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])  # it's a five

# model
layer11 = nn.Tensor(784, 128)
layer12 = nn.Tensor(128, 10)
layer21 = nn.Tensor(28, 28)
layer22 = nn.Tensor(28, 10)
lossfn = nn.Loss().crossentropy
optim = nn.Optimizer().Adam


def train_model(x, y, layer1, layer2, lossfn, optim):
    layer1.forward = x
    x = x @ layer1.weight
    layer2.forward = x
    x = x @ layer2.weight

    loss, grad = lossfn(x, y)
    layer2.grad = layer2.forward.T @ grad
    grad = grad @ (layer2.weight.T)
    optim(layer2)

    layer1.grad = layer1.forward.T @ grad
    optim(layer1)
