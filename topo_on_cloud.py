# this file can be run directly on google colab since we don't use __init__ file
if __name__ == "__main__":
    from nn.module import Loss, Optimizer
    from nn.topo import ReLU, Layer
    import numpy as np

    # on mnist
    from fetch_it import mnist
    x_train, y_train, x_test, y_test = mnist()

    layer3 = Layer(784, 128)
    act = ReLU()
    layer4 = Layer(128, 10)

    # build model
    # __call__(self,layer) in Layer probably causes bugs if we change layer4(..) to model = layer4(..) 
    layer4(act(layer3))

    lossfn = Loss().crossentropy
    optim = Optimizer(learning_rate=1e-4).SGD

    # attention! bs that's too small would lead to 0 encountering in runtime
    batch_size = 256 
    mnist_loss = {"loss": [], "val_loss": []}
    mnist_acc = {"acc": [], "val_acc": []}

    from tqdm import tqdm
    from time import time
    start = time()
    for epoch in range(2):
        print("epoch %d" % (epoch + 1))
        for _ in tqdm(range(0, len(x_train), batch_size)):
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

            # record
            mnist_loss["loss"].append(lossess.mean())
            mnist_loss["val_loss"].append(val_loss.mean())

            mnist_acc["acc"].append((Y == out.argmax(axis=1)).mean())
            mnist_acc["val_acc"].append((y_test[ss] == outf.argmax(axis=1)).mean())

    end = time()

    from matplotlib.pyplot import plot, show, title, legend, xlabel, ylabel

    print("time spent: %.4f" % (end-start))
    print("loss: %.4f, val_loss: %.4f" %
          (mnist_loss["loss"][-1], mnist_loss["val_loss"][-1]))
    print("accs: %.2f, val_accs: %.2f" %
          (mnist_acc["acc"][-1]*1e2, mnist_acc["val_acc"][-1]*1e2))
    # we can do subplots but i'm lazy
    plot(mnist_loss["loss"])
    plot(mnist_loss["val_loss"])
    plot(mnist_acc["acc"])
    plot(mnist_acc["val_acc"])
    title("Mnist dataset")
    legend(["loss", "val_loss", "accuracy", "val_accuracy"])
    xlabel("num of batched data")
    #ylabel("loss")
    show()

