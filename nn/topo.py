from nn.module import layer_init
import numpy as np

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

class Optimizer:
    def __init__(self, learning_rate=1e-3):
        self.learning_rate = learning_rate

    def call(self,layer,opt):
        if layer.child is not None:
            opt(layer.child)

    def SGD(self, layer):
        layer.weight -= self.learning_rate * layer.grad
        # update the params in model recursively
        self.call(layer, self.SGD)

    def Adam(self, layer, b1=0.9, b2=0.999, eps=1e-8):
        m, v, t = 0, 0, 0
        tmp = 0  # to record weight change
        while np.abs(((tmp-layer.weight).sum())/layer.weight.sum()) > 1e-1:
            t += 1
            g = layer.grad
            m = b1*m + (1-b1)*g
            v = b2*v + (1-b2)*g**2
            mhat = m/(1-b1**t)
            vhat = v/(1-b2**t)
            # prev weight
            tmp = layer.weight
            # current weight
            layer.weight -= self.learning_rate*mhat/(vhat**0.5+eps)

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

class Conv:
    def __init__(self, filters, kernel_size, stride=1, padding=False):
      self.filters = filters
      self.ks = kernel_size

      # fast in built-in, consider merge in layer_init()
      weight = np.random.uniform(-1., 1.,size=(filters,kernel_size,kernel_size))/np.sqrt(kernel_size**2)
      self.weight = weight.astype(np.float32)

      self.st = stride
      self.padding = padding  # bool

      # similar to Tensor, can be replaced by inheriting from class Layer
      self.forward = None
      self.grad = np.zeros(weight.shape,dtype=np.float32)
      self.trainable = True

      self.child = None

    def __repr__(self):
      return f"filters: {self.filters}, ks: {self.ks}"

    def __call__(self,layer):
      self.child = layer
      return layer

    def forwards(self, x): 
      ks = self.ks
      st = self.st
      # output[0]: batchsize -> No. of filter
      # not the real conv, which doesn't require padding
      # try to remove padding when forward, and add padding
      # when backward
      out = np.zeros((self.filters,x.shape[1],x.shape[2]))
      for r in range(self.filters):
        for k in range(0, (x.shape[1]-ks) + 1, st):
          for m in range(0, (x.shape[2]-ks) + 1, st):
            tmp = x[:, k:k+ks, m:m+ks]
            ret = np.multiply(self.weight[r], tmp)
            out[r, k, m] = ret.sum()

      self.forward = out 
      return out 

    def backwards(self,bpass):
      # d_weight = forward.T @ bpass
      ks = self.ks
      st = self.st
      rk = self.forward.shape[1]
      rm = self.forward.shape[2]

      for r in range(self.filters):
        # calculate the grad of each filter
        tmpgrad = self.forward[r].T @ bpass[r] 
        tmpout = np.zeros(self.weight[0].shape)
        for k in range(0, rk, st):
          for m in range(0, rm, st):
            tmpout += tmpgrad[k:ks+k, m:ks+k].sum()
        self.grad[r] += tmpout

      # pass the grad to the front first
      # difficult
      # fpass: grid formed by where center of filter has passed
      #  in self.forwards
      # bpass = bpass * fpass

      # update the weights at once
      #optim(self)

      # call child for model backprop later
      # self.child.backwards()

