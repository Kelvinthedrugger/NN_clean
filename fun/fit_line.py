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
x_train = np.expand_dims(x_train,axis=-1) 
x_train = np.expand_dims(x_train,axis=-1) 
y_train = x_train + 2

# we'll pick arbitrary point on the line as test dataset. right now, we'll start proving the ability to fit an arbitrary ds now

# the neural net
# target: learn (a,b)
layer1 = Linear(1,2)
act = ReLU()
layer2 = Linear(2,2)
layer2(act(layer1))

lossfn = Loss().mse
optim = Optimizer(learning_rate=1e-2).SGD

bs = 4 
loss = []

for epoch in range(10):
  #idx = np.random.randint(0,len(x_train),size=bs)
  idx = 0
  #print(layer1.weight)
  #print(x_train[0].shape,y_train[0].shape)

  out = layer2.forwards(x_train[idx])
  #print(out)

  losss, grad = lossfn(y_train[idx], out,supervised=False)
  #print(losss, grad)
  layer2.backwards(grad,optim)
  #print(layer1.weight)
  loss.append(losss.sum())

for i in range(len(loss)):
  print("epoch: %d, loss: %.4f" % (i+1,loss[i]))

from matplotlib import pyplot as plt
# loss wrt epoch/batch
#plt.plot(list(range(len(loss))), loss)

# on test set
xt = np.arange(11.,20.,1.)
xt = np.expand_dims(xt,axis=-1)
xt = np.expand_dims(xt,axis=-1)
yt = xt + 2


