import numpy as np
import module as nn
from fetch_it import mnist
from matplotlib import pyplot as plt

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
    x = xtrain[:64].reshape(-1, 28*28)
    y = ytrain[:64]

    # seed: initialize inside the model ?
    L1_seed = nn.layer_init(784, 1)
    L2_seed = nn.layer_init(128, 1)

    # automl layer: in the Model argument
    L1_g = nn.Tensor(1, 128)
    L2_g = nn.Tensor(1, 10)

    """
    layer1: L1_seed @ L1_g
    layer2: L2_seed @ L2_g
    """

    # model layer
    layer1 = nn.Tensor(784, 128)
    layer2 = nn.Tensor(128, 10)

    layer1.trainable = False
    layer2.trainable = False

    model = nn.Model([
        layer1,
        nn.Activation.ReLU,
        layer2,
    ])

    lossfn = nn.Loss.crossentropy
    optimizer = nn.Optimizer(5e-4).Adam
    model.compile(lossfn, optimizer)
    hist = model.fit(x, y, epoch=5, batch_size=32)

    print("loss: %.4f accuracy: %.4f" %
          (hist["loss"][-1], sum(hist["accuracy"])/len(hist["accuracy"])))
    plt.plot(hist["loss"])
    plt.plot(hist["accuracy"])
    plt.legend(["loss", "accuracy"])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Mnist")
    plt.show()
