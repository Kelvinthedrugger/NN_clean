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

"""
# inputs: 
#   layers and the shape
#   (optional) hyperparams: learning_rate, batchsize
"""


class AutoML:
    def __init__(self, layers):
        # layer: shape of layers
        # seed -> generator layer -> the layer (output)
        model = [[] for _ in range(len(layers))]
        for i in range(len(layers)):
            assert isinstance(layers[i], tuple)
            model[i].append(layer_init(layers[i][0], 1))
            model[i].append(Tensor(1, layers[i][0]))
        self.model = model  # the production layer

    def forward_layer(self):
        """forward pass to generate weights"""
        for model in self.model:
            layer = Tensor(model[0] @ model[1])
            layer.trainable = False
            model.append(layer)

    def forward(self, x):
        """actual forward pass on the dataset"""
        # forget about activation function right now
        # so set the learning rate lower
        for i in range(len(self.model)):
            # last layer is the layer in the main model
            layer = self.model[i][-1]
            layer.forward = x
            x = x @ layer.weight

    def backward(self, bpass):
        """actual backprop on the dataset"""
        # no update in weights of the main model via backprop!
        for i in range(len(self.model)):
            layer = self.model[-i][-1]  # it's backprop
            layer.grad = layer.forward.T @ bpass
            bpass = bpass @ (layer.weight.T)

    def backward_layer(self):
        """backprop to generate weights"""
        for model in self.model:
            # ready for optim
            model[1].grad = model[0].T @ model[2].grad
