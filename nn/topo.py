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
      self.grad = np.zeros(weight.shape)  # zeros with same shape as weight
      self.trainable = True

      self.child = None

    def __call__(self,layer):
      self.child = layer
      return layer

    def forwards(self, x): 
      ks = self.ks
      st = self.st
      # output[0]: batchsize -> No. of filter
      out = np.zeros((self.filters,x.shape[1],x.shape[2]))
      for r in range(self.filters):
        for k in range(0, (x.shape[1]-ks) + 1, st):
          for m in range(0, (x.shape[2]-ks) + 1, st):
            tmp = x[:, k:k+ks, m:m+ks]
            ret = np.multiply(self.weight[r], tmp)
            out[r, k, m] = ret.sum()

      # forward pass: return the x' = layer(x)
      # x is put here to pass compile
      self.forward = out 
      return out 

    def backwards(self,bpass,optim):
      # d_weight = forward.T @ bpass
      ks = self.ks
      st = self.st
      rk = self.forward.shape[1]
      rm = self.forward.shape[2]
      # will all the filters learned exactly
      # the same features ? ans: no, see bottom for visualization
      for r in range(self.filters):
        # calculate the grad of each filter
        tmpgrad = self.forward[r].T @ bpass[r] 
        tmpout = np.zeros((3,3))
        for k in range(0, rk, st):
          for m in range(0, rm, st):
            tmpout += tmpgrad[k:ks+k, m:ks+k].sum()
        self.grad[r] += tmpout

      # pass the grad to the front first
      # bpass = bpass @ (weight.T)
      # difficult
      # fpass: grid formed by where center of filter has passed
      #  on self.forwards
      # bpass = bpass * fpass

      # update the weights at once
      optim(self)

      # call child for model backprop later
      # self.child.backwards()


