"""
No Activation, No Accuracy
# try to implement training on batch
"""

import numpy as np


def layer_init(row, col):
    return np.random.uniform(-1., 1., size=(row, col))/np.sqrt(row*col)


class Tensor:
    def __init__(self, weight):
        self.weight = weight  # layer
        self.forward = None  # to save forward pass from previous layer
        self.grad = None  # d_layer

    def ReLU(self):
        self.weight = np.maximum(self.weight, 0)

    def Sigmoid(self, xx):
        # slow, hasn't checked
        S = np.array(map(lambda x: 1/(1+np.exp(-x) for x in xx)))
        return S, np.multiply(S, (1-S))


class Model:
    def __init__(self, layers):
        model = []
        for layer in layers:
            # add const time checker to implement Activation
            model.append(Tensor(layer))
        self.model = model

    def forward(self, x):
        for layer in self.model:
            layer.forward = x
            x = x @ layer.weight
        return x

    def backward(self, bpass):
        for layer in self.model[::-1]:
            layer.grad = layer.forward.T @ bpass
            bpass = bpass @ (layer.weight.T)
            self.optim(self, layer)

    def compile(self, lossfn, optim):
        self.lossfn = lossfn
        self.optim = optim

    def fit(self, x, y, epoch):
        history = {"loss": [], "accuracy": []}
        for _ in range(epoch):
            yhat = self.forward(x)
            loss, gradient = self.lossfn(self, y, yhat)
            self.backward(gradient)
            history["loss"].append(loss.mean())
            history["accuracy"].append(
                (yhat.argmax(axis=1) == y).astype(np.float32).mean())
        return history


class Loss:
    def mse(self, y, yhat, supervised=True, num_class=10):
        """read num_class when supervised"""
        if supervised:
            label = np.zeros((len(y), num_class), dtype=np.float32)
            label[range(label.shape[0]), y] = 1
            y = label
        loss = np.square(np.subtract(yhat, y))  # vector form
        diff = 2*np.subtract(yhat, y)/(y.shape[-1])
        return loss, diff

    def crossentropy(self, y, yhat, supervised=True, num_class=10):
        label = np.zeros((len(y), num_class), dtype=np.float32)
        label[range(label.shape[0]), y] = 1
        los = (-yhat + np.log(np.exp(yhat).sum(axis=1)).reshape((-1, 1)))
        loss = (label*los)  # .mean(axis=1)
        d_out = label/len(y)
        diff = -d_out + np.exp(-los)*d_out.sum(axis=1).reshape((-1, 1))
        return loss, diff


class Optimizer:
    # all void func's
    # can use __init__ to set learning rate manually from test file
    def SGD(self, layer, learning_rate=1e-6):
        layer.weight -= learning_rate * layer.grad

    def Adam(self, layer, learning_rate=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        m, v, t = 0, 0, 0
        tmp = 0  # to record weight change
        while np.abs((tmp-layer.weight).sum()/layer.weight.sum()) > 1e-1:
            t += 1
            g = layer.grad
            m = b1*m + (1-b1)*g
            v = b2*v + (1-b2)*g**2
            mhat = m/(1-b1**t)
            vhat = v/(1-b2**t)
            # prev weight
            tmp = layer.weight
            # current weight
            layer.weight -= learning_rate*mhat/(vhat**0.5+eps)
