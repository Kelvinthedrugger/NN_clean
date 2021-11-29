# helper function
import numpy as np

def layer_init(row, col):
    return np.random.uniform(-1., 1., size=(row, col))/np.sqrt(row*col)

# instead of saving forward passes to each layer
# , we calculate the matrices every time

# layers
class Tensor:
    # abstraction: to be inherited
    # every object as Tensor object?
    def __init__(self,h=1,w=1,weight=None):
        if weight is None:
            self.weight = layer_init(h,w)
        else:
            self.weight = weight

        # prototype of child[]
        self.link = None
        
        # previous layers, don't need parent since we use tree traversals
        self.child = []

def mse(yhat,y):
    loss = np.square(yhat-y) # scalar
    grad = 2*np.subtract(yhat,y)#.mean(axis=1) # vector
    return loss.mean(), grad

def train(inputs,matrix,output,lossfn=None,lr=1e-4):
    # model as tree (degenerate to linkedlist sometimes), lossfn: root node
    # output: result(init: label) and receive gradient
    if matrix is None: # change to child when scale up
        # return gradient from loss function
        _, output = lossfn(inputs,output)
        return
    # forward pass, count epoch
    train(inputs @matrix.weight,matrix.link,output,lossfn,lr)

    # backprop and gradient descent 
    matrix.weight -= lr * (inputs @ matrix.weight).T @ output
    output = output @ (matrix.weight.T)

def test2():
    # worked
    x = np.array([3,1,2,4,2],dtype=np.float32)
    y = np.array([0,1,0,0,0],dtype=np.float32)
    mat = Tensor(5,5)
    for i in range(20):
        train(x,mat,y,mse,lr=1e-5)
        #print("epoch: ",i+1, norm: ",mat.weight.sum(),)
    print("x:\n",x,"\n\nxhat:\n",x @ mat.weight,"\n\ny: ",y)

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
    test2()
