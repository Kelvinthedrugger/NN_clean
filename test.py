from fetch_it import mnist
import module as nn
import numpy as np
from matplotlib import pyplot as plt
from time import time

if __name__ == "__main__":
    xtrain, ytrain, _, _ = mnist()
    x = xtrain[:].reshape(-1, 28*28)
    y = ytrain[:]

    layer1 = nn.Tensor(784, 128)
    layer2 = nn.Tensor(128, 10)

    model = nn.Model([
        layer1,
        nn.Activation.ReLU,
        layer2,
    ])

    lossfn = nn.Loss.crossentropy
    optimizer = nn.Optimizer(5e-4).Adam
    model.compile(lossfn, optimizer)
    start = time()  # found out to be linear time wrt epoch
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
    plt.title("Mnist")
    plt.show()
