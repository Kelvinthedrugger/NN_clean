# lr finder, turned out to be bunch of boilerplates(shoot
from matplotlib.pyplot import plot, show, title, legend, xlabel, ylabel
import matplotlib.pyplot as plt
from nn.module import Loss, Optimizer
from nn.topo import ReLU, Linear
import numpy as np

def lr_example():
    # on mnist
    from fetch_it import mnist
    x_train, y_train, x_test, y_test = mnist()

    layer3 = Linear(784, 128)
    act = ReLU()
    layer4 = Linear(128, 10)

    layer4(act(layer3))
    lossfn = Loss().mse

    # attention! bs that's too small would lead to 0 encountering in runtime
    batch_size = 128
    mnist_loss = {"loss": [], "val_loss": []}
    mnist_acc = {"acc": [], "val_acc": []}


    # lr finder:
    #  create another empty list
    #  append the loss correspondingly
    #  return the learning rate as the loss differs the most

    lrs = []
    # for epoch in range(1):
    lr = 1e-7
    ln = len(x_train)
    lnt = len(x_test)
    for _ in range(ln//batch_size):
        optim = Optimizer(learning_rate=lr).SGD
        # only need to grab a batch
        samp = np.random.randint(0, ln, size=batch_size)
        X = x_train[samp].reshape((-1, 28*28))
        Y = y_train[samp]
        out = layer4.forwards(X)
        lossess, grad = lossfn(Y, out)

        loss_mean = lossess.mean()

        if len(mnist_loss["loss"]) > 0 and loss_mean > mnist_loss["loss"][0]:
            print("loss being too large, exit")
            break

        # back propagation
        layer4.backwards(grad, optim)

        # val loss
        ss = np.random.randint(0, lnt, size=batch_size)
        outf = layer4.forwards(x_test[ss].reshape((-1, 28*28)))
        val_loss, _ = lossfn(y_test[ss], outf)

        # record
        mnist_loss["loss"].append(loss_mean)
        mnist_loss["val_loss"].append(val_loss.mean())

        mnist_acc["acc"].append((Y == out.argmax(axis=1)).mean())
        mnist_acc["val_acc"].append((y_test[ss] == outf.argmax(axis=1)).mean())

        lrs.append(lr)

        # increment lr
        lr *= 1.2


    print("loss: %.4f, val_loss: %.4f" %
          (mnist_loss["loss"][-1], mnist_loss["val_loss"][-1]))
    # print("accs: %.2f, val_accs: %.2f" %
    #      (mnist_acc["acc"][-1]*1e2, mnist_acc["val_acc"][-1]*1e2))
    # we can do subplots but i'm lazy
    plt.subplot(1, 2, 1)
    plot(mnist_loss["loss"])
    plot(mnist_loss["val_loss"])
    # plot(mnist_acc["acc"])
    # plot(mnist_acc["val_acc"])
    title("Mnist dataset")
    #legend(["loss", "val_loss", "accuracy", "val_accuracy"])
    legend(["loss", "val_loss"])
    xlabel("num of batched data")
    # ylabel("loss")
    plt.subplot(1, 2, 2)
    plot(lrs)
    title("learning rate")
    show()

    return mnist_loss, mnist_acc, lrs
