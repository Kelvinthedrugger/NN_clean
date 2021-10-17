from module import layer_init, Tensor, Loss, Optimizer
import numpy as np

"""backprop not done"""


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
        self.grad = np.zeros_like(weight)  # zeros with same shape as weight
        self.trainable = True

    def __call__(self, x):
        """add padding"""
        rows, cols = x.shape
        out = np.zeros((self.filters, rows-2, cols-2), dtype=np.float32)
        for r in range(self.filters):
            for k in range(0, (rows-self.kernel_size)//self.stride + 1, self.stride):
                for m in range(0, (cols-self.kernel_size)//self.stride+1, self.stride):
                    tmp = x[k:k+self.kernel_size, m:m+self.kernel_size]
                    ret = np.multiply(self.weight[r], tmp)
                    out[r, k, m] = ret.sum()
        return out


if __name__ == "__main__":
    np.random.seed(1337)
    from matplotlib.pyplot import imshow, show
    # five
    x = np.array(
        [[0, 0, 1, 1, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0]])
    print(x.shape)
    x = np.concatenate(
        [[[ele]*4 for ele in row]*4 for row in x]).reshape(28, 28)
    print(x.shape)
    # layer init
    layer = Conv(filters=1, kernel_size=3)  # filters=1 means nothing
    lossfn = Loss().mse
    optim = Optimizer(learning_rate=1e-3).SGD
    # training loop
    losses = []
    for _ in range(2):
        # forward pass
        out = layer(x)
        # backprop
        loss, gradient = lossfn(x, out)
        layer.grad = x.T @ gradient
        optim(layer)
        losses.append(loss)
    print(losses)
