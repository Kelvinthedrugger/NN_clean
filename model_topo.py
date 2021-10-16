from module import layer_init, Loss, Optimizer
import numpy as np


class Layer:
    """a qualified tensor based on tree structure"""

    def __init__(self, h=1, w=1, weight=None):
        if weight is None:
            # figure out how to take tuple argument and parse it automatically
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
        x.prev = self
        return x

    def forwards(self, ds):
        if self.child is not None:
            ds = self.child.forwards(ds)
        self.forward = ds
        return ds @ self.weight

    def backwards(self, bpass):
        if self.trainable:
            self.grad = self.forward.T @ bpass
        bpass = bpass @ (self.weight.T)
        if self.child is not None:
            self.child.backwards(bpass)


if __name__ == "__main__":
    np.random.seed(1337)
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
        optim(layer1)
        optim(layer2)
        losses.append(loss.mean())

    from matplotlib.pyplot import plot, show, title, legend, xlabel, ylabel
    plot(losses)
    title("random dataset")
    legend("mse_loss")
    xlabel("epoch")
    ylabel("loss")
    show()
