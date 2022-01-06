from nn.module import layer_init, Loss, Optimizer
import numpy as np
from matplotlib import pyplot as plt
from nn.topo import ReLU

class Conv:
    def __init__(self, filters, kernel_size, stride=1, padding=None):
        self.filters = filters
        self.kernel_size = kernel_size
        # slow in pure python
        # fast: np.random.uniform(filters,rows,cols)
        weight = np.zeros((filters, kernel_size, kernel_size))
        for r in range(filters):
            weight[r, :, :] = layer_init(kernel_size, kernel_size)
        self.weight = weight
        self.stride = stride
        self.padding = padding  # bool
        # similar to Tensor, can be replaced by inheriting from class Layer
        self.forward = None
        self.grad = np.zeros_like(weight)  # zeros with same shape as weight
        self.trainable = True

        self. child = None

    def __call__(self,layer):
        self.child = layer
        return layer

    def forwards(self, x):
        rows, cols = x.shape
        out = np.zeros((self.filters, rows, cols), dtype=np.float32)
        for r in range(self.filters):
            for k in range(0, (rows-self.kernel_size)//self.stride + 1, self.stride):
                for m in range(0, (cols-self.kernel_size)//self.stride+1, self.stride):
                    tmp = x[k:k+self.kernel_size, m:m+self.kernel_size]
                    ret = np.multiply(self.weight[r], tmp)
                    out[r, k, m] = ret.sum()
        # store forward pass in layer
        self.forward = out
        return out

    def backwards(self,bpass,optim):
        if self.trainable:
           tmpker = np.zeros(self.weight.shape, dtype=np.float32)
           ks = self.kernel_size
           st = self.stride
           for r in range(self.filters):
              # calculate grad wrt filters
              tmpgrad = self.forward[r].T @ bpass[r]
              """
              print(self.forward.shape,bpass.shape)
              plt.subplot(1,2,1)
              plt.imshow(self.forward[0])
              plt.subplot(1,2,2)
              plt.imshow(bpass[0])
              plt.show()"""
              for k in range(0, (self.forward[r].shape[0]-self.kernel_size)//st+1, self.kernel_size):
                  for m in range(0, (self.forward[r].shape[1]-self.kernel_size)//st+1, self.kernel_size):
                      tmpker[r] += tmpgrad[k:k+ks, m:m+ks]

              self.grad = tmpker
           # update backward pass before update the weight
           #print(bpass.shape, self.weight.shape)
           #bpass = bpass @ self.weight
           # update the weights of filters at once
           optim(self)

        if self.child is not None:
            self.child.backwards(bpass,optim)

    """
    # works but it's wrong 
    def backwards(self,bpass,optim):
        if self.trainable:
            # do conv update, slow, in pure python
            # reshape for grad calculation 
            tmpgrad = (x.T @ gradient).reshape((28, 28))
            tmpker = np.zeros((self.kernel_size, self.kernel_size), dtype=np.float32)
            for r in range(1):
                for k in range(0, (x.shape[0]-self.kernel_size)//1+1, self.kernel_size):
                    for m in range(0, (x.shape[1]-self.kernel_size)//1+1, self.kernel_size):
                        tmpker += tmpgrad[k:k+self.kernel_size, m:m+self.kernel_size]
            layer.grad = tmpker
            optim(layer)
        if self.child is not None:
            self.child.backwards(bpass,optim)"""

if __name__ == "__main__":
    np.random.seed(1337)
    # five
    x = np.array(
        [[0, 0, 1, 1, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0]])
    x = np.concatenate(
        [[[ele]*4 for ele in row]*4 for row in x]).reshape(28, 28)
    # layer init
    layer = Conv(filters=1, kernel_size=3)  # filters=1 means nothing
    lossfn = Loss().mse
    optim = Optimizer(learning_rate=5e-5).SGD
    # training loop
    losses = []
    out = None
    print(layer.weight)
    for _ in range(1):
        # forward pass
        out = layer.forwards(x)
        assert out.shape == (1, 28, 28)
        # backprop
        loss, gradient = lossfn(x, out, supervised=False)
        layer.backwards(gradient,optim)
        losses.append(loss.mean())
    print(layer.weight)
    print("epoch  loss")
    for i in range(len(losses)):
        print("%d      %.4f" % (i, losses[i]))

    """

    #plt.figure(figsize=(5, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(x.reshape(28, 28))
    plt.title("original image")
    plt.subplot(1, 2, 2)
    plt.imshow(out.reshape(28, 28))
    plt.title("forward pass after training")
    plt.show()"""
