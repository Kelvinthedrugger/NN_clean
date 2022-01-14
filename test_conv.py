from nn.module import Loss 
from nn.topo import ReLU, Conv, Optimizer, Linear
from matplotlib import pyplot as plt

import numpy as np

def plot_filters(weight):
  for r in range(weight.filters):
    plt.subplot(1,weight.filters,r+1)
    plt.imshow(weight.weight[r] * 100 + 100)
  plt.show()


# reshape the gradient when forward and backward
class Flatten:
  def __init__(self):
    pass

  def __call__(self):
    pass

  def forwards(self):
    pass
  
  def backwards(self):
    pass

def debug():
  # five
  x = np.array(
      [[0, 0, 1, 1, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0]])
  x = np.concatenate(
      [[[ele]*4 for ele in row]*4 for row in x]).reshape(1, 28, 28)

  # pure conv model
  layer1 = Conv(filters=3, kernel_size=3,stride=2)
  act = ReLU()
  layer = Conv(filters=5, kernel_size=5, stride=1)
  layer(act(layer1))
  """
  # conv-dense model (DNN ? )
  layer1 = Conv()
  act1 = ReLU()
  layer2 = Conv()
  layer3 = Flatten()
  layer4 = Linear()
  layer4(layer3(layer2(act1(layer1))))
  """
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
 
  for i in range(len(losses)):
    print("epoch: %d, loss: %.4f" % (i, losses[i]))
  print("total time: %.4f, time per epoch: %.4f" % (end-start, (end-start)/(i+1)))

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



if __name__ == "__main__":
  np.random.seed(1337)
  debug()
  #large()



"""
# train on mnist dataset
def large():
  from fetch_it import mnist
  x_train,_ ,x_test, _ = mnist() 

  # auto encoder sort of
  # five layer doesn't work, either
  # which i guess deconvolution should be added
  # to do an autoencoder
  layer1 = Conv(filters=24,kernel_size=5,stride=2)
  layer = Conv(filters=24,kernel_size=5,stride=2)
  layer(layer(1))

  lossfn = Loss().mse
  opt = Optimizer(1e-1).SGD
  batch_size = 1#24 

  losses = []
  # training loop
  epoch = 5
  from time import time
  start = time()
  for _ in range(epoch):
    for i in range(0, len(x_train), batch_size):
     x = x_train[i:i+batch_size]
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
 
  for i in range(len(losses)):
    print("epoch: %d, loss: %.4f" % (i, losses[i]))
  print("total time: %.4f, time per epoch: %.4f" % (end-start, (end-start)/(i+1)))

"""
