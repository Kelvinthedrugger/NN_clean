A much cleaner neural network from scratch using numpy, for practicing & mood changing

    python test.py

Best performance: 

 loss: 0.0434, accuracy: 0.9297

 time spent: 4.4415 sec (2 epochs)

 test accuracy: 0.9917

 test time: 0.0364 sec


To see how complex topo works on random & mnist dataset, checkout:

   ./nn/topo.py 

Benchmark:

 time spent: 1.7370 (2 epochs)

 loss: 0.0667, val_loss: 0.0316


#### TODO

    add activation (matters for numerical stability): done

    add concat model feature (Adder in electronics): working on complex topo
        -> (fake) AutoML: tuning problem: done

    add Convolution: 96%, backprop done

    add LSTM -> add RL -> Real AutoML

    add model.save() method: 70%, should use similar method as pickle (I mean dict)

#### Progress

```bash
# NN_clean works on colab!

!git clone https://github.com/Kelvinthedrugger/NN_clean.git
```
Example usage in notebook: done; see topo_on_cloud.py

