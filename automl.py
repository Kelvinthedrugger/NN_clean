import numpy as np
import module as nn
from fetch_it import mnist
from matplotlib import pyplot as plt
from ATML import AutoML

"""
we have to implement model concatenate first (or Adder)

automl model:

    L1_s   L2_s
      v     v
    L1_g   L2_g
      v     v
 x -> L1 -> L2 -> y

forward pass:
  side chains -> main model

"""

if __name__ == "__main__":
    xtrain, ytrain, _, _ = mnist()
    x = xtrain[:1000].reshape(-1, 28*28)
    y = ytrain[:1000]

    model = AutoML([(784, 128), (128, 10)])
    """# the arithmetic"""
    model.compile(nn.Loss.crossentropy, nn.Optimizer(1e-4).Adam)
    from time import time
    start = time()
    hist = model.fit(x, y, epoch=1000, batch_size=128)
    end = time()

    print("loss: %.4f accuracy: %.4f" %
          (hist["loss"][-1], sum(hist["accuracy"])/len(hist["accuracy"])))
    print("time spent: %.4f sec" % (end-start))
    plt.plot(hist["loss"])
    plt.plot(hist["accuracy"])
    plt.legend(["loss", "accuracy"])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("AutoML Mnist")
    plt.show()
