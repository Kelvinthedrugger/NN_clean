from fetch_it import mnist
import module as nn
import numpy as np
from matplotlib import pyplot as plt
from time import time

if __name__ == "__main__":
    xtrain, ytrain, _, _ = mnist()
    x = xtrain[:100].reshape(-1, 28*28)
    y = ytrain[:100]
    layer1 = nn.layer_init(784, 128)
    layer2 = nn.layer_init(128, 10)
    model = nn.Model([layer1, layer2])
    lossfn = nn.Loss.crossentropy
    optimizer = nn.Optimizer.Adam
    model.compile(lossfn, optimizer)
    # found out to be linear time wrt epoch
    start = time()
    hist = model.fit(x, y, epoch=100)
    end = time()
    print("accuracy: %.4f" % (sum(hist["accuracy"])/len(hist["accuracy"])))
    print("time spent: %.4f sec" % (end-start))
    plt.plot(hist["loss"])
    plt.plot(hist["accuracy"])
    plt.legend(["loss", "accuracy"])
    plt.xlabel("num of data")
    plt.ylabel("loss")
    plt.title("Mnist")
    plt.show()
