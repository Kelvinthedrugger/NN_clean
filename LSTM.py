from module import Tensor
import numpy as np

# https://zhuanlan.zhihu.com/p/32085405
# slides fron Hung-yi Lee of NTU

"""
i: input gate
f: forget gate
o: output gate
z = tanh(W @ (x(t),h(t-1)))
z(k) = sigmoid(W(i) @ (x(t),h(t-1))), k = i or f or o

c(t) = z(f) x c(t-1) + z(i) x z
h(t) = z(o) x tanh(c(t))
y(t) = sigmoid(W'h(t))

x: elementwise multiplication, must have same shape
"""


class LSTM:
    def __init__(self):
        pass
