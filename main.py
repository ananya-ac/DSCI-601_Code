from train import train_model
from eval import eval_model
import numpy as np
import torch
import config 
import gymnasium as gym



def parse_vector(vector_data):
    """Parse vector data that might be stored as a string"""
    if isinstance(vector_data, str):
        if '[' in vector_data:
            vector_data = vector_data.strip('[]')
            if ',' in vector_data:
                values = [float(val.strip()) for val in vector_data.split(',') if val.strip()]
            else:
                values = [float(val) for val in vector_data.split() if val]
        else:
            values = [float(val) for val in vector_data.split() if val]
        return np.array(values)
    return vector_data

if __name__ == "__main__":
    
     
    
    # Create environment to get state and action dimensions
    env = gym.make(config.env_name)
    state_dim = env.observation_space.shape[0]
    continuous = isinstance(env.action_space, gym.spaces.Box)
    
    if continuous:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n
    
    env.close()
    
    print(f"Training on {config.env_name}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Continuous actions: {continuous}")
    
    # Hyperparameters
    params = {
        'env_name': config.env_name,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'hidden_dim': config.hidden_dim,
        'flow_dim': config.flow_dim,
        'num_flow_layers': config.num_flow_layers,
        'actor_lr': config.actor_lr,
        'critic_lr': config.critic_lr,
        'gamma': config.gamma,
        'continuous': continuous,
        'gae_lambda': config.gae_lambda,
        'entropy_coef': config.entropy_coef,
        'value_loss_coef': config.value_loss_coef,
        'max_grad_norm': config.max_grad_norm,
        'num_heads': config.num_heads,
        'critic_batch_size': config.critic_batch_size,
        'total_timesteps': config.total_timesteps ,
        'update_frequency': config.update_frequency,
        'num_epochs': config.num_epochs,
        'batch_size': config.actor_batch_size,
        'log_frequency': config.log_freq,
        'max_episode_steps': config.max_episode_steps,
        'eval_frequency': config.eval_freq,
        'num_eval_episodes': config.eval_episodes,
        'checkpoint_frequency': config.checkpoint_frequency,
        'normalize_advantages': config.normalize_advantages,
        'critic_updates_per_actor_update': config.critic_updates,
        'seed': config.seed
    }
    
    
    # Train the model
    model = train_model(**params)
    
    rewards = eval_model(checkpoint_dir='checkpoints',env_name=config.env_name)
    # Evaluate the model - examine distributions for different state-action pairs
    
    # Calculate returns if not already done
    # if 'returns' not in df_r.columns:
    #     df_returns = calculate_returns(df_r)
    # else:
    #     df_returns = df_r
    
    # # Select interesting points in the trajectory
    # points_to_visualize = {
    #     'start': 0,                                  # First state-action in the dataset
    #     'mid': min(50, len(df_r) - 1),               # Mid-episode state-action
    #     'high_return': df_returns['returns'].idxmax(),  # State-action with highest return
    #     'low_return': df_returns['returns'].idxmin()    # State-action with lowest return
    # }
    
    # # Terminal states are also interesting
    # terminal_idxs = df_r[df_r['dones']].index
    # if len(terminal_idxs) > 0:
    #     points_to_visualize['near_terminal'] = terminal_idxs[0] - 1  # State right before terminal
    
    # # Process each state-action pair for visualization
    # for point_name, idx in points_to_visualize.items():
    #     if idx is None or idx >= len(df_r):
    #         continue
            
    #     # Get state and action at this point
    #     state_data = parse_vector(df_r['s'].iloc[idx])
    #     action_data = parse_vector(df_r['a'].iloc[idx])
        
    #     # Convert to tensors
    #     state_tensor = torch.tensor(state_data, dtype=torch.float32).unsqueeze(0)
    #     action_tensor = torch.tensor(action_data, dtype=torch.float32).unsqueeze(0)
        
    #     # Get the return value if available (use calculated returns from df_returns if exists)
    #     if idx < len(df_returns):
    #         return_value = df_returns['returns'].iloc[idx]
    #     else:
    #         return_value = None
        
    #     # Plot the value distribution
    #     plot = plot_value_distribution(
    #         model=model,
    #         state=state_tensor,
    #         action=action_tensor,
    #         return_value=return_value,
    #         num_samples=5000,  # More samples for a smoother distribution
    #         bins=100,
    #         title=f"Value Distribution for {point_name.capitalize()} (idx={idx})"
    #     )
        
    #     plt.savefig(f"value_distribution_{point_name}.png")
    #     if return_value is not None:
    #         print(f"Saved value distribution plot for {point_name} (idx={idx}, return={return_value:.2f})")
    #     else:
    #         print(f"Saved value distribution plot for {point_name} (idx={idx})")
    
    # # Compare actual returns with predicted distributions
    # # Calculate returns
    # df_returns = calculate_returns(df_r)
    
    # # Select a reference state-action pair
    # reference_idx = 0
    # reference_state = parse_vector(df_r['s'].iloc[reference_idx])
    # reference_action = parse_vector(df_r['a'].iloc[reference_idx])
    
    # # Convert to tensors
    # reference_state_tensor = torch.tensor(reference_state, dtype=torch.float32).unsqueeze(0)
    # reference_action_tensor = torch.tensor(reference_action, dtype=torch.float32).unsqueeze(0)
    
    # # Get model predictions
    # with torch.no_grad():
    #     predictions = model.sample_value_distribution(reference_state_tensor, reference_action_tensor, num_samples=5000)
    #     predicted_samples = predictions['samples'].squeeze().cpu().numpy()
    
    # # Plot comparison
    # plt.figure(figsize=(10, 6))
    # plt.hist(predicted_samples, bins=100, alpha=0.5, label='Predicted Distribution')
    # plt.axvline(predictions['mean'].item(), color='r', linestyle='--', label=f'Predicted Mean')
    
    # # Find the actual return for this state-action
    # actual_return = df_returns.iloc[reference_idx]['returns']
    # plt.axvline(actual_return, color='g', linestyle='-', label=f'Actual Return')
    
    # plt.title(f"Predicted vs Actual Return for Reference State-Action")
    # plt.xlabel('Return')
    # plt.ylabel('Density')
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.savefig("predicted_vs_actual_returns.png")
    # print("Saved comparison of predicted vs actual returns")
    
    # # Additional analysis: Compare distributions for same state but different actions
    # # Find a state that appears with different actions
    # # This is a simplified approach - in practice you'd use a more sophisticated state similarity measure
    
    # # For demonstration, we'll pick a reference state and compare different actions
    # reference_state_idx = 0
    # reference_state = parse_vector(df_r['s'].iloc[reference_state_idx])
    # reference_state_tensor = torch.tensor(reference_state, dtype=torch.float32).unsqueeze(0)
    
    # # Find state-action pairs with different actions
    # # For simplicity, let's pick 3 different points in the trajectory with distinct actions
    # action_idxs = [0]  # Start with the first action
    
    # # Try to find actions that are meaningfully different
    # for i in range(1, len(df_r)):
    #     current_action = parse_vector(df_r['a'].iloc[i])
    #     different_enough = True
        
    #     for idx in action_idxs:
    #         prev_action = parse_vector(df_r['a'].iloc[idx])
    #         # Check if actions are significantly different
    #         if np.allclose(current_action, prev_action, atol=0.2):
    #             different_enough = False
    #             break
                
    #     if different_enough and len(action_idxs) < 3:
    #         action_idxs.append(i)
            
    #     if len(action_idxs) >= 3:
    #         break
    
    # # If we didn't find enough different actions, just use some points from the trajectory
    # while len(action_idxs) < 3:
    #     next_idx = action_idxs[-1] + 10
    #     if next_idx < len(df_r):
    #         action_idxs.append(next_idx)
    #     else:
    #         break
    
    # # Collect actions and returns
    # actions = []
    # returns = []
    
    # for idx in action_idxs:
    #     action_data = parse_vector(df_r['a'].iloc[idx])
    #     action_tensor = torch.tensor(action_data, dtype=torch.float32).unsqueeze(0)
    #     actions.append(action_tensor)
        
    #     if idx < len(df_returns):
    #         return_value = df_returns['returns'].iloc[idx]
    #         returns.append(return_value)
    #     else:
    #         returns.append(None)
    
    # # Use the action comparison plot function
    # plot = plot_action_comparison(
    #     model=model,
    #     state=reference_state_tensor,
    #     actions=actions,
    #     returns=returns,
    #     num_samples=1000,
    #     bins=50,
    #     title=f"Value Distributions for Different Actions (State from idx={reference_state_idx})"
    # )
    
    # plt.savefig("value_distributions_different_actions.png")
    # print("Saved comparison of value distributions for different actions")