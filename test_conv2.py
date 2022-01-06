from nn.module import layer_init, Loss, Optimizer
import numpy as np
from matplotlib import pyplot as plt
from nn.topo import ReLU

class Conv:
    def __init__(self, filters, kernel_size, stride=1, padding=False):
        self.filters = filters
        self.ks = kernel_size

        # slow in pure python
        # fast: np.random.uniform(filters,rows,cols)?
        weight = np.zeros((filters, kernel_size, kernel_size))
        for r in range(filters):
            weight[r, :, :] = layer_init(kernel_size, kernel_size)
        self.weight = weight

        self.st = stride
        self.padding = padding  # bool

        # similar to Tensor, can be replaced by inheriting from class Layer
        self.forward = None
        self.grad = np.zeros_like(weight)  # zeros with same shape as weight
        self.trainable = True

        self.child = None

    def __call__(self,layer):
        self.child = layer
        return layer

    def forwards(self, x): 
        ks = self.ks
        st = self.st
        for r in range(self.filters):
            for k in range(0, (x.shape[1]-ks)//st + 1, st):
                for m in range(0, (x.shape[2]-ks)//st + 1, st):
                    pass

        # forward pass
        return x

    def backwards(self):
        ks = self.ks
        st = self.st
        for r in range(self.filters):
            for k in range(0, (x.shape[1]-ks)//st + 1, st):
                for m in range(0, (x.shape[2]-ks)//st + 1, st):
                    pass



if __name__ == "__main__":
    np.random.seed(1337)
    # five
    x = np.array(
        [[0, 0, 1, 1, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0]])
    x = np.concatenate(
        [[[ele]*4 for ele in row]*4 for row in x]).reshape(28, 28)
 
