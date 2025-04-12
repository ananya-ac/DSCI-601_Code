import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 42


# Dimensions for inputs and outputs for Value-Distribution Estimater via INN
ndim_obs = 8
ndim_act = 4
ndim_rew = 1
ndim_z = 13
ndim_tot = 14


#gym
env_name = "CartPole-v1"


#Training 
num_epochs = 20
actor_updates = 1
critic_updates = 5
num_samples = 100
actor_lr = 1e-4
critic_lr = 2e-5
max_grad_norm = 1
critic_batch_size = 128
actor_batch_size = 128
update_frequency = 10000
total_timesteps = 1000000
normalize_advantages = True
max_episode_steps = None

#logging
log_freq = 1

#eval
eval_freq = 5000
eval_episodes = 5

#rl hparams
gamma = 0.99
gae_lambda = 0.95
entropy_coef = 0.01
value_loss_coef = 0.5


#model
hidden_dim = 128
flow_dim = 8
num_flow_layers = 8
num_heads = 4

#Optimizer
lr = 1e-4
l2_reg = 2e-5
betas = (0.8, 0.9)
eps = 1e-6

#model
num_layers = 8

#checkpoint
checkpoint_frequency = 10000

