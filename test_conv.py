from module import Tensor, layer_init
import numpy as np


class Conv:
    def __init__(self, filters, kernel_size, stride=1, padding=None):
        self.filters = filters
        self.kernel_size = kernel_size
        self.weight = layer_init(kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding  # bool

    def __call__(self, x):
        rows, cols = x.shape
        out = np.zeros((rows-2, cols-2), dtype=np.float32)
        for i in range(0, (rows-self.kernel_size)//self.stride + 1, self.stride):
            for j in range(0, (cols-self.kernel_size)//self.stride+1, self.stride):
                out[i][j] = j+i*3

        return out


if __name__ == "__main__":
    np.random.seed(1337)
    layer = Conv(1, 3)
    x = layer_init(5, 5)
    print(layer(x))
