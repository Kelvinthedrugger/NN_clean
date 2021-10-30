from nn.module import layer_init, Tensor, Loss, Optimizer
import numpy as np


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
        # similar to Tensor
        self.forward = None
        self.grad = np.zeros_like(weight)  # zeros with same shape as weight
        self.trainable = True

    def __call__(self, x):
        rows, cols = x.shape
        out = np.zeros((self.filters, rows, cols), dtype=np.float32)
        for r in range(self.filters):
            for k in range(0, (rows-self.kernel_size)//self.stride + 1, self.stride):
                for m in range(0, (cols-self.kernel_size)//self.stride+1, self.stride):
                    tmp = x[k:k+self.kernel_size, m:m+self.kernel_size]
                    ret = np.multiply(self.weight[r], tmp)
                    out[r, k, m] = ret.sum()
        return out


if __name__ == "__main__":
    np.random.seed(1337)
    from matplotlib import pyplot as plt
    # five
    x = np.array(
        [[0, 0, 1, 1, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0]])
    x = np.concatenate(
        [[[ele]*4 for ele in row]*4 for row in x]).reshape(28, 28)
    # layer init
    layer = Conv(filters=1, kernel_size=3)  # filters=1 means nothing
    lossfn = Loss().mse
    optim = Optimizer(learning_rate=1e-3).SGD
    # training loop
    losses = []
    out = None
    for _ in range(10):
        # forward pass
        out = layer(x)
        assert out.shape == (1, 28, 28)
        # backprop
        loss, gradient = lossfn(x, out, supervised=False)
        tmpgrad = (x.T @ gradient).reshape((28, 28))
        # do conv update
        tmpker = np.zeros((3, 3), dtype=np.float32)
        for r in range(1):
            for k in range(0, (x.shape[0]-3)//1+1, 3):
                for m in range(0, (x.shape[1]-3)//1+1, 3):
                    tmpker += tmpgrad[k:k+3, m:m+3]
        layer.grad = tmpker
        optim(layer)
        losses.append(loss.mean())

    print("epoch  loss")
    for i in range(len(losses)):
        print("%d      %.4f" % (i, losses[i]))

    #plt.figure(figsize=(5, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(x.reshape(28, 28))
    plt.title("original image")
    plt.subplot(1, 2, 2)
    plt.imshow(out.reshape(28, 28))
    plt.title("forward pass after training")
    plt.show()
