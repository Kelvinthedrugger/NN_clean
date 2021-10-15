from module import layer_init
import numpy as np


class Layer:
    """a qualified tensor based on tree structure"""

    def __init__(self, h=1, w=1, weight=None):
        if weight is None:
            # figure out how to take tuple argument and parse it automatically
            self.weight = layer_init(h, w)
        else:
            self.weight = weight
        # topo
        self.prev = None
        self.child = None
        # autograd
        self.forward = None  # to save forward pass from previous layer
        self.grad = None  # d_layer
        self.trainable = True

    def __call__(self, x):
        """
        x: the next layer
        let loss(last layer) be the root node (tree-like structure)
        """
        x.child = self
        self.prev = x
        return x

    def forward(self, x):
        # x: input data here
        return x @ self.weight


if __name__ == "__main__":
    np.random.seed(1337)
    layer1 = Layer(12, 3)
    layer2 = Layer(3, 2)
    x = layer2(layer1)
    assert layer1.child == layer2
