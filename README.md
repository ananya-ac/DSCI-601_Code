# Readme

Python version 3.11.9

Requirements mentioned in requirements.txt

Data Collection/Preparation:
The data is obtained by the agent (the RL model) interacting with the environment (OpenAI's gymnasium library provides good abstractions for this).

The data generation occurs in the following steps:

1) The agent is provided with an initial state by the environment.
2) The agent chooses an action based on this state.
3) The environment uses this action to step to the next state whilst also providing a reward for this next-state transition.
4) This forms one trajectory tuple of (state, action, reward, next_state), which is the atomic datapoint used in training the agent.
5) These tuples are stored in lists till the process hits the update frequency. Then these lists are used to create a pytorch TensorDataset to facilitate easy training for the model.

Model Training:
The Dataset created in the data generating process is fed to the model using a PyTorch DataLoader while the optimization uses the Trainer class from PyTorch Lightning.
PyTorch Lightning is a wrapper around PyTorch with a performant training loop that helps in avoiding training issues such as Out of Memory errors.
There is a training call-back which saves the best model (guaged by the best average reward).
The hyperparameter tuning is done using the library Optuna, which performs bayesian optimization using the  Tree‑structured Parzen Estimator (TPE) sampler.

Results
The agent is evaluated by making it play 2 episodes in the same environment on which it was trained. The results of this game are averaged and videos for both the runs are stored in the video folder.
