
class Activations:
    def __init__(self):
        self.child = None
        self.grad = None
        self.trainable = False

    def backwards(self, bpass):
        bpass = np.multiply(self.grad, bpass)
        if self.child is not None:
            self.child.backwards(bpass)


class ReLU(Activations):
    def __call__(self, layer):
        self.child = layer
        return layer

    def forwards(self, x):
        if self.child is not None:
            x = self.child.forwards(x)
        out = np.maximum(x, 0)
        self.grad = (out > 0).astype(np.float32)
        return out


class Layer:
    """a qualified tensor based on tree structure, loss being the root node"""

    def __init__(self, h=1, w=1, weight=None):
        if weight is None:
            self.weight = layer_init(h, w)
        else:
            self.weight = weight
        # topo
        self.child = None
        # autograd
        self.forward = None  # save forward pass from previous layer
        self.grad = None  # d_layer
        self.trainable = True

    def __call__(self, layer):
        self.child = layer
        return layer

    def forwards(self, ds):
        if self.child is not None:
            ds = self.child.forwards(ds)
        if self.trainable:
            self.forward = ds
        return ds @ self.weight

    def backwards(self, bpass, optim):
        if self.trainable:
            self.grad = self.forward.T @ bpass
            optim(self)
        bpass = bpass @ (self.weight.T)
        if self.child is not None:
            self.child.backwards(bpass, optim)
