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
    np.random.seed(1337)

    xtrain, ytrain, _, _ = mnist()
    x = xtrain[:100].reshape(-1, 28*28)
    y = ytrain[:100]

    model = AutoML([(784, 128), (128, 10)])
    """# the arithmetic"""
    model.compile(nn.Loss.crossentropy, nn.Optimizer(5e-6).Adam)
    # model.fit(x, y)
    # print(model.model[0][-1].weight)
    # print("")
    # """after"""
    # # plt.imshow(np.clip(model.model[0][-1].weight, 0, 255))
    # # plt.show()
    # model.fit(x, y)
    # # plt.imshow(np.clip(model.model[0][-1].weight, 0, 255))
    # # plt.show()
    # print(model.model[0][-1].weight)
    from time import time
    start = time()  # found out to be linear time wrt epoch
    hist = model.fit(x, y, epoch=1000, batch_size=32)
    end = time()

    print("loss: %.4f accuracy: %.4f" %
          (hist["loss"][-1], sum(hist["accuracy"])/len(hist["accuracy"])))
    print("time spent: %.4f sec" % (end-start))
    plt.plot(hist["loss"])
    plt.plot(hist["accuracy"])
    plt.legend(["loss", "accuracy"])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Mnist")
    plt.show()
