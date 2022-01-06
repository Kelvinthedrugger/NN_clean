from nn.module import Loss, Optimizer
from nn.topo import ReLU, Conv
from matplotlib import pyplot as plt

import numpy as np

if __name__ == "__main__":
  np.random.seed(1337)
  # five
  x = np.array(
      [[0, 0, 1, 1, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0]])
  x = np.concatenate(
      [[[ele]*4 for ele in row]*4 for row in x]).reshape(1, 28, 28)
  layer = Conv(filters=3, kernel_size=3,stride=2)
  lossfn = Loss().mse
  opt = Optimizer(1e-6).SGD

  losses = []
  # training loop
  for i in range(10):
   out = layer.forwards(x)

   loss, grad = lossfn(out,x,supervised=False)
   layer.backwards(grad,opt)

   losses.append(loss.mean())
 
  for i in range(len(losses)):
    print("epoch: %d, loss: %.4f" % (i, losses[i]))
  """
  # input/output
  plt.subplot(1,2,1)
  plt.imshow(x[0])
  plt.subplot(1,2,2)
  plt.imshow(out[0])
  plt.show()
  # filters
  plt.subplot(1,3,1)
  plt.imshow(layer.weight[0] * 100 + 100)
  plt.subplot(1,3,2)
  plt.imshow(layer.weight[1] * 100 + 100)
  plt.subplot(1,3,3)
  plt.imshow(layer.weight[2] * 100 + 100)
  plt.show()
  """
