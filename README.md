# Introduction

This is a repo dedicated for a RL research paper about agent indicator method. It has been forked from [rl-baseline3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo) project, and modified to conduct experiments about agent indicators used in RL.

# How to use

The scripts are mainly located at [agent-indicator](https://github.com/SonSang/rl-baselines3-zoo/tree/agent-indicator) branch. The script to conduct the hyperparameter optimization is [indicator_opt.py](https://github.com/SonSang/rl-baselines3-zoo/blob/8669d675ae4f4f4328125ab0e9d47c8ded92c7f0/indicator_opt.py). Following comman can be used to run the script.

```
$ python indicator_opt.py --algo {PPO or DQN} --env {env name} --n-timesteps {number of timesteps} --n-trials {number of set of hyperparameters to test with} --n-evaluations {number of evaluations} --sampler {sampler type(Optuna)} --pruner {pruner type(Optuna)}
```

## Agent Indicators

There are four agent indicators implemented in this repo: Inversion, Inversion with Replacement, Geometric, Binary. 

* Inversion: Invert the observation of certain types of agents and add it as a channel to the original observation. For agents that do not need inversion, duplicate original observation.
* Inversion with Replacement: The same as Inversion, but the inverted observation (or the same duplicate observation for agents that do not need it) is used in place of the original observation.
* Geometric: Add an additional channel with alternating geometric checkered pattern for different types of agents.
* Binary: Add additional channels, each of which is entirely black or white based on the type of an agent.

To see the implementation detail, please refer [indicator_utils.py](https://github.com/SonSang/rl-baselines3-zoo/blob/8669d675ae4f4f4328125ab0e9d47c8ded92c7f0/indicator_util.py).

## Evaluations

Hyperparameters that gave best results were trained again for multiple times for fair evaluation. The retraining and evaluation scipt is [indicator_eval_params.py](https://github.com/SonSang/rl-baselines3-zoo/blob/8669d675ae4f4f4328125ab0e9d47c8ded92c7f0/indicator_eval_params.py).

# Results

All the experimental results that used for paper are stored in [indicator_hyperparameters](https://github.com/SonSang/rl-baselines3-zoo/tree/8669d675ae4f4f4328125ab0e9d47c8ded92c7f0/indicator_hyperparameters) folder.
