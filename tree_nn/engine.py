# helper function
import numpy as np

def layer_init(row, col):
    return np.random.uniform(-1., 1., size=(row, col))/np.sqrt(row*col).astype(np.float32)

# layers
class Tensor:
    # abstraction: to be inherited
    # every object as Tensor object, not just layers but also lossfn
    def __init__(self):

        # prototype of child[]
        self.link = None
        
        # previous layers, don't need parent since we use tree traversals
        self.child = []

class Linear(Tensor):
    def __init__(self,h=1,w=1,weight=None):
        super().__init__()

        if weight is None:
            self.weight = layer_init(h,w)
        else:
            self.weight = weight
        
        # activation function
        self.act = None

        self.trainable = True

def mse(yhat,y):
    loss = np.square(yhat-y) # scalar
    grad = 2*np.subtract(yhat,y)#.mean(axis=1) # vector
    return loss.mean(), grad

def relu(x):
    fpass = np.maximum(x,0)
    bpass = (fpass > 0).astype(np.float32)
    return fpass, bpass

# model as tree (degenerate to linkedlist sometimes), lossfn: root node
# output: result(init: label) and receive gradient
# instead of saving forward passes to each layer
# , we calculate the matrices every time

def train(inputs,output,layer,lossfn=None,lr=1e-4):

    # change to child when scale up
    if layer is None: 
        # return gradient from loss function
        _, output = lossfn(inputs,output)
        return

    # activation
    if layer.act is not None:
        fpass, bpass = layer.act(inputs @ layer.weight)
    else:
        fpass = inputs @ layer.weight
        bpass = 1 # np.eye(fpass.shape[0],fpass.shape[1])

    # forward pass, count epoch
    train(fpass,output,layer.link,lossfn,lr)

    # gradient descent & update weight
    if layer.trainable:
        layer.weight -= lr * (fpass).T @ np.multiply(output,bpass)

    # backprop
    output = output @ (layer.weight.T)

def test():
    w1 = Linear(5,5)
    w1.act = relu
    x1 = np.array([1,2,3,2,1],dtype=np.float32)
    y1 = np.array([0,0,1,0,0],dtype=np.float32)
    
    for i in range(100):
        train(x1,y1,w1,mse,lr=1e-4)
        if i % 20 == 9:
            print("yhat: ",x1 @ w1.weight,end=" ")
            print("norm: %.4f, loss: %.4f" % (w1.weight.sum(),mse(x1 @ w1.weight, y1)[0].mean()))

"""
# Loss:: root of the model, also a node of tree
class Lossfn:
    # abstraction: to be inherited
    def __init__(self):
        self.child = []

    def forward(self,x):
        pass
"""
class LSTM(Linear):
    pass

if __name__ == "__main__":
    test()
