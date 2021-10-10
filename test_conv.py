from module import layer_init
import numpy as np


class Conv:
    def __init__(self, filters, kernel_size, stride=1, padding=None):
        self.filters = filters
        self.kernel_size = kernel_size
        self.weight = layer_init(kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding  # bool
        self.forward = None
        self.grad = None
        self.trainable = True

    def __call__(self, x):
        rows, cols = x.shape
        out = np.zeros((self.filters, rows-2, cols-2), dtype=np.float32)
        for r in range(self.filters):
            for k in range(0, (rows-self.kernel_size)//self.stride + 1, self.stride):
                for m in range(0, (cols-self.kernel_size)//self.stride+1, self.stride):
                    tmp = x[k:k+self.kernel_size, m:m+self.kernel_size]
                    out[r, k, m] = np.multiply(self.weight, tmp).sum()
        del tmp
        return out


if __name__ == "__main__":
    np.random.seed(1337)
    layer = Conv(1, 3)
    x = np.random.uniform(-1, 1, size=(5, 5))
    out = layer(x)
