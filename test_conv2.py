from nn.module import Loss, Optimizer
import numpy as np
from matplotlib import pyplot as plt
from nn.topo import ReLU

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
      # out = np.zeros((self.filters,(x.shape[1]-ks)//st + 1, (x.shape[2]-ks)//st + 1))
      out = np.zeros(x.shape)
      for r in range(self.filters):
        for k in range(0, (x.shape[1]-ks)//st + 1, st):
          for m in range(0, (x.shape[1]-ks)//st + 1, st):
            tmp = x[r, k:k+ks, m:m+ks]
            ret = np.multiply(self.weight[r], tmp)
            out[r, k, m] = ret.sum()

      # forward pass: return the x' = layer(x)
      # x is put here to pass compile
      self.forward = out 
      return out 

    # calculate gradient only, don't update the weights here first
    # , then make use of self.child and update the weights 
    # right after gradient is calculated
    # pass back the grad -> calculate d_weight -> update weight -= lr*d_weight
    def backwards(self,bpass,optim=Optimizer().SGD):

      # calc. the d_weight
      # d_weight = forward.T @ bpass
      ks = self.ks
      st = self.st
      rk = self.forward.shape[1]
      rm = self.forward.shape[2]
      # will all the filters learned exactly
      # the same features ?
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

from matplotlib import pyplot as plt
if __name__ == "__main__":
  np.random.seed(1337)
  # five
  x = np.array(
      [[0, 0, 1, 1, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0]])
  x = np.concatenate(
      [[[ele]*4 for ele in row]*4 for row in x]).reshape(1, 28, 28)
  layer = Conv(filters=1, kernel_size=3,stride=2)
  lossfn = Loss().mse
  opt = Optimizer(1e-6).SGD

  losses = []
  # training loop
  for i in range(10):
   out = layer.forwards(x)

   loss, grad = lossfn(out,x,supervised=False)
   losses.append(loss.mean())
   layer.backwards(grad,opt)
 
  print(losses)
  plt.subplot(1,3,1)
  plt.imshow(x[0])
  plt.subplot(1,3,2)
  plt.imshow(out[0])
  plt.subplot(1,3,3)
  plt.imshow(layer.weight[0] * 100 + 100)
  plt.show()
