import numpy as np
from module import Tensor, layer_init

"""# AutoML class"""
"""    
    # seed: initialize inside the model ?
    L1_seed = nn.layer_init(784, 1)
    L2_seed = nn.layer_init(128, 1)

    # automl layer: in the Model argument
    L1_g = nn.Tensor(1, 128)
    L2_g = nn.Tensor(1, 10)

    layer1: L1_seed @ L1_g
    layer2: L2_seed @ L2_g

    # model layer
    layer1 = nn.Tensor(784, 128); trainable = False
    layer2 = nn.Tensor(128, 10);  trainable = False
    
    L1_s   L2_s
      v     v
    L1_g   L2_g
      v     v
 x -> L1 -> L2 -> y

forward pass:
  side chains -> main model
"""


class AutoML:
    def __init__(self):
        pass
