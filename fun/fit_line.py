# folder shxt
import os, sys
curdir = os.path.dirname(os.path.realpath(__file__))
pardir = os.path.dirname(curdir)
sys.path.append(pardir)

from nn.module import Loss, Optimizer

from nn.topo import ReLU, Linear
import numpy as np


# dataset: y = a*x + b, (a,b) = (1,2) in this case
x_train = np.arange(0.1,10,0.1)
x_train = x_train.reshape((len(x_train),1,1)
y_train = x_train + 2

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
layer = Linear(1,2)

lossfn = Loss().mse
optim = Optimizer(learning_rate=1e-2).SGD

bs = 1 
loss = []
print(layer1.weight)

for epoch in range(10):
  #buggy
  #idx = np.random.randint(0,len(x_train),size=bs)
  idx = epoch 
  #print(layer1.weight)
  #print(x_train[0].shape,y_train[0].shape)

  out = layer2.forwards(x_train[idx])
  #print(out)

  losss, grad = lossfn(y_train[idx], out,supervised=False)
  #print(losss, grad)
  layer2.backwards(grad,optim)
  #print(layer1.weight)
  loss.append(losss.sum())

print(layer1.weight)

for i in range(len(loss)):
  print("epoch: %d, loss: %.4f" % (i+1,loss[i]))

from matplotlib import pyplot as plt

