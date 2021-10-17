from module import layer_init, Loss, Optimizer
import numpy as np


class Layer:
    """a qualified tensor based on tree structure"""

    def __init__(self, h=1, w=1, weight=None):
        if weight is None:
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
        self.child = x
        return x

    def forwards(self, ds):
        if self.child is not None:
            ds = self.child.forwards(ds)
        self.forward = ds
        return ds @ self.weight

    def backwards(self, bpass):
        if self.trainable:
            self.grad = self.forward.T @ bpass
            optim(self)
        bpass = bpass @ (self.weight.T)
        if self.child is not None:
            self.child.backwards(bpass)


def random_ds():
    # model
    layer1 = Layer(12, 3)
    layer2 = Layer(3, 2)
    layer2(layer1)
    assert layer2.child == layer1
    # forward pass
    ds1 = np.random.uniform(-1., 1., size=(1, 12)).astype(np.float32)
    ans1 = ds1 @ layer1.weight @ layer2.weight
    ans2 = layer2.forwards(ds1)
    assert ans1.all() == ans2.all()
    # backprop
    target = np.array([1], dtype=np.uint8)
    lossfn = Loss().mse
    optim = Optimizer().Adam
    losses = []
    # training loop
    for _ in range(100):
        ans2 = layer2.forwards(ds1)
        loss, gradient = lossfn(target, ans2, num_class=2)
        layer2.backwards(gradient)
        # optim(layer1)
        # optim(layer2)
        losses.append(loss.mean())

    from matplotlib.pyplot import plot, show, title, legend, xlabel, ylabel
    plot(losses)
    title("random dataset")
    legend("mse_loss")
    xlabel("epoch")
    ylabel("loss")
    show()


if __name__ == "__main__":
    np.random.seed(1337)

    # on mnist
    from fetch_it import mnist
    x_train, y_train, x_test, y_test = mnist()
    layer3 = Layer(784, 128)
    layer4 = Layer(128, 10)
    layer4(layer3)

    lossfn = Loss().mse
    optim = Optimizer(learning_rate=1e-6).Adam

    batch_size = 128
    mnist_loss = {"loss": [], "val_loss": []}
    from time import time
    start = time()
    for epoch in range(10):
        for _ in range(0, len(x_train)//batch_size, batch_size):
            samp = np.random.randint(0, len(x_train), size=batch_size)
            X = x_train[samp].reshape((-1, 28*28))
            Y = y_train[samp]

            out = layer4.forwards(X)
            lossess, grad = lossfn(Y, out)
            layer4.backwards(grad)

            # val loss
            ss = np.random.randint(0, len(x_test), size=batch_size)
            outf = layer4.forwards(x_test[ss].reshape((-1, 28*28)))
            val_loss, _ = lossfn(y_test[ss], outf)

            mnist_loss["loss"].append(lossess.mean())
            mnist_loss["val_loss"].append(val_loss.mean())
    end = time()
    from matplotlib.pyplot import plot, show, title, legend, xlabel, ylabel
    print("time spent: %.4f" % (end-start))
    plot(mnist_loss["loss"])
    plot(mnist_loss["val_loss"])
    legend(["loss", "val_loss"])
    xlabel("num of data")
    ylabel("loss")
    show()
