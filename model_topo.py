from module import layer_init
import numpy as np


class Layer:
    """a qualified tensor based on tree structure"""

    def __init__(self, weight=None):
        if isinstance(weight, tuple):
            # figure out how to take tuple argument and parse it automatically
            self.weight = layer_init(weight[0], weight[1])
        else:
            self.weight = weight
        # topo
        self.prev = None
        self.child = None
        # autograd
        self.forward = None  # to save forward pass from previous layer
        self.grad = None  # d_layer
        self.trainable = True


class Model:
    def __init__(self, layers):
        """
        construct topo available model here
        , consider started from layer class
        """
        # layers: List
        for i in range(len(layers)):
            if i == 0:
                layers[i].child = layers[i+1]
            elif i < len(layers)-2:
                layers[i].prev = layers[i-1]
                layers[i].child = layers[i+1]
            else:
                layers[i].prev = layers[i-1]

        self.model = layers


if __name__ == "__main__":
    layer1 = Layer((12, 3))
    layer2 = Layer((3, 2))
