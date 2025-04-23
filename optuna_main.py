import optuna
from train import train_model
from eval import eval_model 
import gymnasium as gym
import config
import json
import os

def objective(trial):
    """
    Objective function for Optuna to optimize actor_lr and critic_lr
    based on the average reward.
    
    Args:
        trial: An Optuna trial object
        
    Returns:
        float: The average reward achieved with the hyperparameters
    """
    # Create environment to get state and action dimensions
    env = gym.make(config.env_name)
    state_dim = env.observation_space.shape[0]
    continuous = isinstance(env.action_space, gym.spaces.Box)
    
    if continuous:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n
    
    env.close()
    
    # Hyperparameters to tune
    actor_lr = trial.suggest_float('actor_lr', 5e-5, 5e-4, log=True)
    critic_lr = trial.suggest_float('critic_lr', 1e-4, 1e-2, log=True)
    update_frequency = trial.suggest_int('update_frequency', 2048, 4096, step=512)
    num_epochs = trial.suggest_int('num_epochs', 1, 6)
    # Print current trial values
    print(f"Trial {trial.number}: actor_lr={actor_lr:.6f}, critic_lr={critic_lr:.6f}, update_frequency={update_frequency}, num_epochs={num_epochs}")
    
  
   
    # Update parameters to reflect hyperparameters to tune
    params = {
        'env_name': config.env_name,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'hidden_dim': config.hidden_dim,
        'flow_dim': config.flow_dim,
        'num_flow_layers': config.num_flow_layers,
        'actor_lr': actor_lr,  # Use the suggested value
        'critic_lr': critic_lr,  # Use the suggested value
        'gamma': config.gamma,
        'continuous': continuous,
        'gae_lambda': config.gae_lambda,
        'entropy_coef': config.entropy_coef,
        'value_loss_coef': config.value_loss_coef,
        'max_grad_norm': config.max_grad_norm,
        'num_heads': config.num_heads,
        'critic_batch_size': config.critic_batch_size,
        'total_timesteps': config.total_timesteps,
        'update_frequency': update_frequency,  # Use the suggested value
        'num_epochs': num_epochs,  # Use the suggested value
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
    avg_reward = train_model(**params)
    
    return avg_reward

def save_best_params(study, save_path="best_params_1.json"):
    """
    Save the best parameters from the study to a JSON file.
    
    Args:
        study: The Optuna study object
        save_path: Path to save the JSON file
    """
    best_params = study.best_params
    best_value = study.best_value
    
    # Create a dictionary with best parameters and the achieved value
    params_dict = {
        "best_params": best_params,
        "best_avg_reward": best_value,
        "env_name": config.env_name
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    # Save to JSON file
    with open(save_path, 'w') as f:
        json.dump(params_dict, f, indent=4)
    
    print(f"Best parameters saved to {save_path}")

def main():
    """
    Main function to run the Optuna optimization.
    """
    # Create a study object and optimize the objective function
    study = optuna.create_study(
    study_name="rl_hyperparameter_optimization",
    storage="sqlite:///study.db",
    load_if_exists=True,
    direction="maximize")   
    study.optimize(objective, n_trials=150)  # Adjust n_trials as needed
    

    # Print the best parameters and results
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (avg_reward): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Save the best parameters
    save_best_params(study)
    
    # You can also save the entire study for later analysis
    # optuna.save_study(study=study, storage="sqlite:///study.db")
    
if __name__ == "__main__":
    main()
    