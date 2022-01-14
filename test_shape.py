# to fix shape/channel problem when convoluting 
from nn.module import Loss 
from nn.topo import ReLU, Conv, Optimizer, Linear
from matplotlib import pyplot as plt

import numpy as np

def plot_filters(weight):
  for r in range(weight.filters):
    plt.subplot(1,weight.filters,r+1)
    plt.imshow(weight.weight[r] * 100 + 100)
  plt.show()

class Convb:
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
      # remove padding when forward, 
      # and add padding when backward
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
        tmpgrad = self.forward[r].T @ bpass[r] 
        tmpout = np.zeros(self.weight[0].shape)
        for k in range(0, rk, st):
          for m in range(0, rm, st):
            tmpout += tmpgrad[k:ks+k, m:ks+k].sum()
        self.grad[r] += tmpout


def debug():
  # five
  x = np.array(
      [[0, 0, 1, 1, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0]])
  x = np.concatenate(
      [[[ele]*4 for ele in row]*4 for row in x]).reshape(1, 28, 28)

  # pure conv model
  layer1 = Conv(filters=3, kernel_size=3,stride=2)
  act = ReLU()
  layer = Conv(filters=1, kernel_size=5, stride=1)
  layer(act(layer1))
  lossfn = Loss().mse
  opt = Optimizer(2e-6).SGD

  losses = []

  # training loop
  from time import time
  start = time()
  for i in range(10):
   out = layer.forwards(x)

   loss, grad = lossfn(out,x,supervised=False)
   # callbacks: early stop
   if len(losses) > 0 and loss.mean() > losses[-1]: break

   losses.append(loss.mean())

   # callbacks: early stop also
   if losses[-1] > 0.5: break

   layer.backwards(grad)
   opt(layer)
  end = time()

  # print model
  print(layer, layer1)

  print(x.shape, out.shape)
 
  for i in range(len(losses)):
    print("epoch: %d, loss: %.4f" % (i, losses[i]))
  print("total time: %.4f, time per epoch: %.4f" % (end-start, (end-start)/(i+1)))

  # input/output
  plt.subplot(1,2,1)
  plt.imshow(x[0])
  plt.subplot(1,2,2)
  plt.imshow(out[0])
  plt.show()
  # filters
  #plot_filters(layer)
  #plot_filters(layer1)

if __name__ == "__main__":
  np.random.seed(1337)
  debug()
  #large()


