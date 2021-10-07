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
