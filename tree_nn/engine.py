# helper function
import numpy as np

def layer_init(row, col):
    return np.random.uniform(-1., 1., size=(row, col))/np.sqrt(row*col)

# use recursion
# forward pass: postorder
# backprop: preorder
# instead of saving forward passes to each layer
# , we calculate the matrices every time

def postorder(grad,fpass,layer,lr): # void
    d_weight = fpass.T @ grad
    grad = grad @ (layer.weight.T)
    layer.weight -= lr*d_weight

# layers
class Tensor:
    # abstraction: to be inherited
    # every object as Tensor object?
    def __init__(self,h=1,w=1,weight=None):
        if weight is None:
            self.weight = layer_init(h,w)
        else:
            self.weight = weight

        # previous layers, don't need parent since we use tree traversals
        self.child = []

def backward(fpass,matrix,bpass,lr,lossfn):
    if matrix.link == None: # lossfn
        bpass = lossfn()
    d_weight = fpass.T @ bpass
    matrix.weight -= lr * d_weight
    bpass = bpass @ (matrix.weight.T)


def forward(inputs,matrix,output,lr=1e-4,lossfn=None):
    # model as tree (degenerate to linkedlist sometimes)

    # last node of model: lossfn
    if matrix.link == None:
        # last layer: forward pass are all done
        # receive gradient from loss function
        backward(inputs,matrix,output,lr,lossfn)
        return
    # output: result and receive gradient
    output = inputs @ matrix.weight
    forward(output,matrix.link,output,lr,lossfn)
    # execute after forward pass are done and not the last layer
    backward(inputs,matrix,output,lr,lossfn)


def traverse():
    # use recursion: max stack depth: 1000 in python
    # re-calculate: don't save forward array for backward? 
    # loop from layer1 to layer n: (recursion)
    #    output = None
    #    forward(input,matrix,output)
    #    # backward(matrix,output)
    pass

def test():
    np.random.seed(1337)
    in1 = Tensor(1,2) # col vector .T
    lr = 1e-3
    w1 = Tensor(2,3)
    w2 = Tensor(3,2)
    g = Tensor(1,2)
    print(w1.weight,"\n\n", w2.weight,"\n\n")
    f1 = in1.weight @ w1.weight
    b1 = g.weight @ (w2.weight.T)
    # as layers
    postorder(g.weight,f1,w2,lr)
    postorder(b1,in1.weight,w1,lr)
    print(w1.weight,"\n\n", w2.weight)
    # as a model:: modify postorder() to traverse()


class Linear(Tensor):
    def __init__(self,h=1,w=1,weight=None):
        super().__init__()
"""
# Loss:: root of the model, also a node of tree
class Lossfn:
    # abstraction: to be inherited
    def __init__(self):
        self.child = []

    def forward(self,x):
        pass
"""
if __name__ == "__main__":
    test()
