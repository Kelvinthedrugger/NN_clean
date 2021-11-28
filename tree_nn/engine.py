# helper function
import numpy as np

def layer_init(row, col):
    return np.random.uniform(-1., 1., size=(row, col))/np.sqrt(row*col)

# sth wrong
def postorder(layer,x):
    if len(layer.child) == 0:
        return x

    for m in layer.child:
        postorder(m, x.T @ m.weight)

def preorder(layer,x):
    pass


# layers
class Tensor:
    # abstraction: to be inherited
    def __init__(self,h=1,w=1,weight=None):
        if weight is None:
            self.weight = layer_init(h,w)
        else:
            self.weight = weight

        # previous layers, don't need parent since we use tree traversals
        self.child = []

class Linear(Tensor):
    def __init__(self,h=1,w=1,weight=None):
        super().__init__()

# forward pass: postorder
# backprop: preorder
# instead of saving forward passes to each layer
# , we calculate the matrices every time

# Loss:: root of the model, also a node of tree
class Lossfn:
    # abstraction: to be inherited
    def __init__(self):
        self.child = []

    def forward(self,x):
        # idk to design input layer as a Tensor object or not
        return postorder(self,x)



if __name__ == "__main__":
    pass
