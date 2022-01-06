from nn.module import Loss 
from nn.topo import ReLU, Conv, Optimizer 
from matplotlib import pyplot as plt

import numpy as np

def plot_filters(weight):
  for r in range(weight.filters):
    plt.subplot(1,weight.filters,r+1)
    plt.imshow(weight.weight[r] * 100 + 100)
  plt.show()

if __name__ == "__main__":
  #np.random.seed(1337)
  # five
  x = np.array(
      [[0, 0, 1, 1, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0]])
  x = np.concatenate(
      [[[ele]*4 for ele in row]*4 for row in x]).reshape(1, 28, 28)
  layer1 = Conv(filters=3, kernel_size=3,stride=2)
  layer = Conv(filters=5, kernel_size=5, stride=1)
  layer(layer1)
  lossfn = Loss().mse
  opt = Optimizer(2e-6).SGD

  losses = []
  # training loop
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

  # print model
  print(layer, layer1)
 
  for i in range(len(losses)):
    print("epoch: %d, loss: %.4f" % (i, losses[i]))
  """
  # input/output
  plt.subplot(1,2,1)
  plt.imshow(x[0])
  plt.subplot(1,2,2)
  plt.imshow(out[0])
  plt.show()"""
  # filters
  #plot_filters(layer)
  #plot_filters(layer1)

