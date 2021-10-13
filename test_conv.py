from module import layer_init, Tensor
from module.Loss import mse
import numpy as np


"""
construct a image (say, a 5) and label it
-> establish a 1-conv_layer model
-> define loss as mse (since it's the simplest)
-> train on the image & figure conv backprop out

integrate initialization with Tensor
"""


class Conv:
    def __init__(self, filters, kernel_size, stride=1, padding=None):
        self.filters = filters
        self.kernel_size = kernel_size
        weight = np.zeros((filters, kernel_size, kernel_size))
        for r in range(filters):
            weight[r, :, :] = layer_init(kernel_size, kernel_size)
        self.weight = weight
        self.stride = stride
        self.padding = padding  # bool
        # below similar to Tensor
        self.forward = None
        self.grad = [[] for _ in range(filters)]
        self.trainable = True

    def __call__(self, x):
        rows, cols = x.shape
        out = np.zeros((self.filters, rows-2, cols-2), dtype=np.float32)
        # now, it's fake gradient
        for r in range(self.filters):
            grad = 0
            for k in range(0, (rows-self.kernel_size)//self.stride + 1, self.stride):
                for m in range(0, (cols-self.kernel_size)//self.stride+1, self.stride):
                    tmp = x[k:k+self.kernel_size, m:m+self.kernel_size]
                    ret = np.multiply(self.weight[r], tmp)
                    out[r, k, m] = ret.sum()
                    grad += tmp.T @ ret  # gradient wrt filter
            self.grad[r].append(grad)
        return out


if __name__ == "__main__":
    np.random.seed(1337)
    layer = Conv(filters=2, kernel_size=3)
    x = np.random.uniform(-1, 1, size=(5, 5))
    out = layer(x)
    print(out)
    print(layer.grad[0])
    print(layer.grad[1])
