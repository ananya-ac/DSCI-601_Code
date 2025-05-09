import optuna
from train import train_model
from eval import eval_model 
import gymnasium as gym
import config
import json
import os

def main():
    
    # Create environment to get state and action dimensions
    env = gym.make(config.env_name)
    state_dim = env.observation_space.shape[0]
    continuous = isinstance(env.action_space, gym.spaces.Box)
    
    if continuous:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n
    
    env.close()
    
    
  
   
    # Update parameters to reflect hyperparameters to tune
    params = {
        'env_name': config.env_name,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'hidden_dim': config.hidden_dim,
        'flow_dim': config.flow_dim,
        'num_flow_layers': config.num_flow_layers,
        'actor_lr': config.actor_lr,  # Use the suggested value
        'critic_lr': config.critic_lr,  # Use the suggested value
        'gamma': config.gamma,
        'continuous': continuous,
        'gae_lambda': config.gae_lambda,
        'entropy_coef': config.entropy_coef,
        'value_loss_coef': config.value_loss_coef,
        'max_grad_norm': config.max_grad_norm,
        'num_heads': config.num_heads,
        'critic_batch_size': config.critic_batch_size,
        'total_timesteps': config.total_timesteps,
        'update_frequency': config.update_frequency,  # Use the suggested value
        'num_epochs': config.num_epochs,  # Use the suggested value
        'batch_size': config.actor_batch_size,
        'log_frequency': config.log_freq,
        'max_episode_steps': config.max_episode_steps,
        'eval_frequency': config.eval_freq,
        'num_eval_episodes': config.eval_episodes,
        'checkpoint_frequency': config.checkpoint_frequency,
        'normalize_advantages': config.normalize_advantages,
        'critic_updates_per_actor_update': config.critic_updates,
        'seed': config.seed  # Use different seeds for different trials
    }
    
    # Train the model and get the average reward
    train_model(**params)
    eval_model(checkpoint_dir="checkpoints/", env_name=config.env_name, num_episodes=config.eval_episodes)
    

if __name__ == "__main__":
    main()
    