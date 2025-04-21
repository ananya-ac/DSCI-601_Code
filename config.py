import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 42




#gym
env_name = "CartPole-v1"
#env_name = "Pendulum-v1"
reward_scalar = 8.0

#Training 
num_epochs = 8
actor_updates = 1
critic_updates = 1
num_samples = 300
#actor_lr = 0.000481201144669996
#critic_lr = 0.0005084521171404696
critic_lr  = 0.0003
actor_lr = 0.0001
critic_batch_size = 256
actor_batch_size = 256
update_frequency = 2048
total_timesteps = 50000
k = 5
normalize_advantages = True
max_episode_steps = 500

#logging
log_freq = 100

#eval
eval_freq = 5000
eval_episodes = 2

#rl hparams
gamma = 0.99
gae_lambda = 0.95
entropy_coef = 0.01
value_loss_coef = 0.5
epsilon = 0.3
alpha = 0.05 # coeffecient for running average calculationw
#model
hidden_dim = 256
hidden_dim_subnet = 256
flow_dim = 6
num_flow_layers = 4
num_heads = 4

#Optimizer
betas = (0.9, 0.999)
eps = 1e-8
max_grad_norm = 0.5


#checkpoint
checkpoint_frequency = 10000

