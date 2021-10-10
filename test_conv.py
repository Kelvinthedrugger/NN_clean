from module import layer_init
import numpy as np


class Conv:
    def __init__(self, filters, kernel_size, stride=1, padding=None):
        self.filters = filters
        self.kernel_size = kernel_size
        weight = np.zeros((filters, kernel_size, kernel_size))
        for r in range(filters):
            weight[r, :, :] = layer_init(kernel_size, kernel_size)
        self.weight = weight
        #self.weight = layer_init(kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding  # bool
        # below similar to Tensor
        self.forward = None
        self.grad = [[] for _ in range(filters)]
        self.trainable = True

    def __call__(self, x):
        rows, cols = x.shape
        out = np.zeros((self.filters, rows-2, cols-2), dtype=np.float32)
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
    layer = Conv(2, 3)
    x = np.random.uniform(-1, 1, size=(5, 5))
    out = layer(x)
    print(out)
    print(layer.grad[0])
    print(layer.grad[1])
