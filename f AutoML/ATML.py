import numpy as np
from module import Tensor, layer_init

"""# AutoML class
    L1_s   L2_s
      v     v
    L1_g   L2_g
      v     v
 x -> L1 -> L2 -> y

forward pass:
  side chains -> main model

inputs:
  layers and the shape
  (optional) hyperparams: learning_rate, batchsize
"""


class AutoML:
    def __init__(self, layers, layer_depth=None):
        """# random weight -> Tensor -> Tensor"""
        # layer: shape of layers
        # seed -> generator layer -> the layer (output)
        # not scalable yet
        if layer_depth is None:
            layer_depth = [1] * len(layers)
        model = [[] for _ in range(len(layers))]
        for i in range(len(layers)):
            assert isinstance(layers[i], tuple)
            model[i].append(layer_init(layers[i][0], layer_depth[i]))
            model[i].append(Tensor(layer_depth[i], layers[i][1]))
            model[i].append(Tensor(layers[i][0], layers[i][1]))
        self.model = model

    def forward_layer(self):
        """forward pass to generate weights in main model"""
        for model in self.model:
            # model[-1].weight = model[0] @ (model[1].weight)
            x = model[0]
            for i in range(1, len(model)-1):
                x = x @ model[i].weight
            model[-1].weight = x

    def forward(self, x):
        """actual forward pass on the dataset"""
        # forget about activation function right now
        # so set the learning rate lower
        for i in range(len(self.model)):
            # last layer is the layer in the main model
            layer = self.model[i][-1]
            layer.forward = x
            x = x @ layer.weight
        return x

    def backward(self, bpass):
        """actual backprop on the dataset"""
        # no update in weights of the main model via backprop!
        for i in range(len(self.model)-1, -1, -1):
            layer = self.model[i][-1]
            layer.grad = layer.forward.T @ bpass
            bpass = bpass @ (layer.weight.T)

    def backward_layer(self):
        """backprop to generate weights"""
        for model in self.model:
            for i in range(len(model)-2, 1, -1):
                model[i].grad = model[i-1].weight.T @ model[i+1].grad
                self.optimizer(model[i])
            model[1].grad = model[0].T @ model[2].grad
            self.optimizer(model[1])

    def compile(self, lossfn, optimizer):
        self.lossfn = lossfn
        self.optimizer = optimizer

    def fit(self, X, Y, epoch=5, batch_size=32):
        history = {"loss": [], "accuracy": []}
        for _ in range(epoch):
            self.forward_layer()  # create the layer in main model
            samp = np.random.randint(0, len(X), size=batch_size)
            x = X[samp]
            y = Y[samp]
            self.forward(x)
            yhat = self.forward(x)
            loss, gradient = self.lossfn(self, y, yhat)
            self.backward(gradient)
            self.backward_layer()
            history["loss"].append(loss.mean())
            history["accuracy"].append(
                (yhat.argmax(axis=1) == y).astype(np.float32).mean())
        return history
