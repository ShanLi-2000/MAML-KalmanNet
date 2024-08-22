# MAML-KalmanNet

## Link to paper

[None](https://github.com/ShanLi-2000/MAML-KalmanNet)

## Running code

Since the code is not fully adapted to cuda, it is recommended to set use_cuda = False when generating data and then use cuda to train model.

There are main files simulating the UCM system and the UZH FPV systems respectively.
We have saved the trained model in Model/model_name/basenet.pt

* UCM (linear model or non-linear model)

```
python3 main_linear.py
python3 main_nonlinear.py
```

* Lorzen Attractor (matched model, decimation, mismatched model)

```
python3 main_UZH.py
```

## Running plot loss code

### UCM system

- linear_plot_loss.py: plot Figure 3(a) in the paper.
- semi-supervised_compared.py: plot Figure 3(b) in the paper.

### Lorzen Attractor

- UZH_plot_trajectory.py: plot Figure 6 in the paper.

## Introduction to other files

* filter.py

The specific computational process of MAML-KalmanNet and EKF

* meta.py

The computational process of MAML-KalmanNet including MSG for the first half of the training epochs and sandard MAML update method for another half of training epoches.

* state_dict_learner.py

Contain neural network settings.

* Data/model_name
Contain dataset for trainin/testing.

* Model/model_name
Contain the trained neural network state dictionary.

* Simulations/model_name/modelname_syntheticNShot.py

Contain model settings: x_dim, y_dim, f/F, h/H, Q and R. 

* Simulations/model_name/other_files

Plot actual loss between MAML-KalmanNet and other algorithms.

