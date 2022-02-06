# folder shxt
import os, sys
curdir = os.path.dirname(os.path.realpath(__file__))
pardir = os.path.dirname(curdir)
sys.path.append(pardir)

import nn.module as nn
import numpy as np
from nn.topo import Linear, ReLU

# dataset: y = a*x + b, (a,b) = (1,2) in this case
x_train = np.arange(0.1,10.1,0.1)
x_train = x_train.reshape((len(x_train),1,1))
y_train = x_train + 2

# from scratch
layer1 = Linear(1,10)
layer2 = Linear(10,1)
act = ReLU()

lossfn = nn.Loss().mse
optim = nn.Optimizer(learning_rate=1e-4).SGD

def cap():
  # build model
  layer2(act(layer1))
  # train
  losses = []
  for _ in range(10):
    out = layer2.forwards(x_train[_])
    loss, grad = lossfn(y_train[_],out,supervised=False)
    layer2.backwards(grad,optim)
    losses.append(loss.mean())

  print(losses)


# forward pass and backprop
def fb(x,y,lr=1e-4):
  # don't build model mofo
  f1 = layer1.forwards(x) 
  f11 = np.maximum(f1,0)
  #print(x.shape, f1.shape, layer1.weight.shape, f11.shape, layer2.weight.shape)
  f2 = layer2.forwards(f11)
  #print(f2.shape)
  #return
  # backprop
  loss, grad = lossfn(y,f2,supervised=False)
  df2 =  f11.T @ grad 
  dfr = grad @ (layer2.weight.T)
  df11 = (dfr > 0.).astype(np.float32)
  df1 = x.T @ df11

  # update
  layer2.weight -= lr * df2

  layer1.weight -= lr * df1

  return loss

def raw():
  # check model is not build
  assert layer2.child is None
  loss = []
  for _ in range(2000):
    idx = np.random.randint(0,len(x_train[0]))
    loss.append(fb(x_train[idx],y_train[idx],lr=1e-1))

  for i in range(0, len(loss),100):
    print("epoch: %4d, loss: %.6f" % (i, loss[i]))
  print("\nlast loss: %.6f" % loss[-1])
if __name__ == "__main__":
  raw()
#import matplotlib.pyplot as plt
#plt.show()
# we'll pick arbitrary point on the line as test dataset. right now, we'll start proving the ability to fit an arbitrary ds now

# the neural net
# target: learn (a,b).T, which happens to be 
# , the params in this only layer of the model
# workflow:
#  feed x_train inside the model
#  -> calculate the loss wrt y_train
#  , so, the 1st input dim and last output dim should be 1 or batch_size
# abstraction: randomly initialize the model
# feed in the ds until the model learns the pattern (which is y=ax+b in this case)
# point of deep learning is to NOT solve the problem analytically,
# therefore, asking for (a,b) is usually not the problem we're dealing with
# since the patterns in real world is just to complicated to be solved analytically


