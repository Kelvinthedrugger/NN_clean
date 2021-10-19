

if __name__ == "__main__":
    from module import layer_init, Loss, Optimizer
    import numpy as np
    from model_topo import ReLU, Layer

    # on mnist
    from fetch_it import mnist
    x_train, y_train, x_test, y_test = mnist()
    """
    layer3 = Layer(784, 128)
    act = ReLU()
    layer4 = Layer(128, 10)
    """
    layer3 = Layer(weight=layer_init(784,128))
    act = ReLU()
    layer4 = Layer(weight=layer_init(128,10))

    layer4(act(layer3))

    lossfn = Loss().crossentropy
    optim = Optimizer(learning_rate=1e-4).SGD

    batch_size = 128
    mnist_loss = {"loss": [], "val_loss": []}
    from time import time
    start = time()
    for epoch in range(2):
        for _ in range(0, len(x_train), batch_size):
            samp = np.random.randint(0, len(x_train), size=batch_size)
            X = x_train[samp].reshape((-1, 28*28))
            Y = y_train[samp]
            out = layer4.forwards(X)
            lossess, grad = lossfn(Y, out)
            layer4.backwards(grad, optim)

            # val loss
            ss = np.random.randint(0, len(x_test), size=batch_size)
            outf = layer4.forwards(x_test[ss].reshape((-1, 28*28)))
            val_loss, _ = lossfn(y_test[ss], outf)

            mnist_loss["loss"].append(lossess.mean())
            mnist_loss["val_loss"].append(val_loss.mean())
    end = time()

    from matplotlib.pyplot import plot, show, title, legend, xlabel, ylabel
    print("time spent: %.4f" % (end-start))
    print("loss: %.4f, val_loss: %.4f" %
          (mnist_loss["loss"][-1], mnist_loss["val_loss"][-1]))
    plot(mnist_loss["loss"])
    plot(mnist_loss["val_loss"])
    title("Mnist dataset")
    legend(["loss", "val_loss"])
    xlabel("num of batched data")
    ylabel("loss")
    show()
