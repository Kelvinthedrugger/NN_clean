A much more cleaner neural network from scratch using numpy, for practicing & mood changing

    python test.py

Best performance: 

 loss: 0.0434, accuracy: 0.9297

 time spent: 4.4415 sec (2 epochs)

 test accuracy: 0.9917

 test time: 0.0364 sec


To see how complex topo works on random & mnist dataset, directly run:

    python model_topo.py

Result:

 time spent: 1.7370 (2 epochs)

 loss: 0.0667, val_loss: 0.0316


# TODO

    add activation (matters for numerical stability): done

    add concat model feature (Adder in electronics): working on complex topo
        -> (fake) AutoML: tuning problem: done

    add convolution: 50%

    add LSTM -> add RL -> Real AutoML

    add model.save() method: 70%

    working on enabling usage on online notebook
