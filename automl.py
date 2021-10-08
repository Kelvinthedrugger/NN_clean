import numpy as np
import module as nn
from fetch_it import mnist
from matplotlib import pyplot as plt
from ATML import AutoML


if __name__ == "__main__":
    xtrain, ytrain, _, _ = mnist()
    x = xtrain[:1000].reshape(-1, 28*28)
    y = ytrain[:1000]

    model = AutoML([(784, 128), (128, 10)])
    """# the arithmetic"""
    model.compile(nn.Loss.crossentropy, nn.Optimizer(1e-2).SGD)
    print(model.model[0][-1].weight[0])
    from time import time
    start = time()
    hist = model.fit(x, y, epoch=1000, batch_size=128)
    end = time()
    print("")
    print(model.model[0][-1].weight[:10])
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
