# MAML-KalmanNet

## Link to Paper

[MAML-KalmanNet: A Neural Network-Assisted Kalman Filter Based on Model-Agnostic Meta-Learning](https://ieeexplore.ieee.org/document/10883047)

## Introduction to the Code

In the paper, the generation of AAL data is **task-specific**. However, in the code implementation, to simplify the process of generating AAL data, we use **two nested loops** to select $ q_2 $ and $ r_2 $ from the list $ \Upsilon $. The rationale behind this approach is:

> Although this method will generate some similar tasks, the probability of sampling highly similar tasks during any single inner-loop update process is extremely low, ensuring sufficient task diversity for meta-training.

This ensures that the generated tasks are diverse enough for effective meta-training, even though some tasks may appear similar.

## Running Code

Since the code is not fully adapted to CUDA, it is recommended to set `use_cuda = False` when generating data and then use CUDA to train the model.

There are main files simulating the UCM system and the UZH FPV systems, respectively. We have saved the trained model in `Model/model_name/basenet.pt`.

### UCM (Linear Model or Non-linear Model)

```
python3 main_linear.py
python3 main_nonlinear.py
```


### Lorenz Attractor (Matched Model, Decimation, Mismatched Model)

```
python3 main_UZH.py
```

## Running Plot Loss Code

### UCM System

- `linear_plot_loss.py`: Plot Figure 3(a) in the paper.
- `semi-supervised_compared.py`: Plot Figure 3(b) in the paper.

### Lorenz Attractor

- `UZH_plot_trajectory.py`: Plot Figure 6 in the paper.

## Introduction to Other Files

### `filter.py`

The specific computational process of MAML-KalmanNet and EKF.

### `meta.py`

The computational process of MAML-KalmanNet, including MSG for the first half of the training epochs and the standard MAML update method for the second half of the training epochs.

### `state_dict_learner.py`

Contains neural network settings.

### `Data/model_name`

Contains datasets for training and testing.

### `Model/model_name`

Contains the trained neural network state dictionary.

### `Simulations/model_name/modelname_syntheticNShot.py`

Contains model settings: `x_dim`, `y_dim`, `f/F`, `h/H`, `Q`, and `R`.

### `Simulations/model_name/other_files`

Plots the actual loss between MAML-KalmanNet and other algorithms.
