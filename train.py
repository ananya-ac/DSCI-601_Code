import gym
import torch
import numpy as np
from collections import deque
import pytorch_lightning as pl
from lightning.pytorch.loggers import WandbLogger
from tqdm import tqdm
import random
from torch.utils.data import TensorDataset, DataLoader
from model import ActorCriticWithFlow

def train_model(
    env_name,
    state_dim,
    action_dim,
    hidden_dim,
    flow_dim,
    num_flow_layers,
    actor_lr,
    critic_lr,
    gamma,
    continuous,
    gae_lambda,
    entropy_coef,
    value_loss_coef,
    max_grad_norm,
    num_heads,
    critic_batch_size,  # Optional specific batch size for critic updates
    total_timesteps,
    update_frequency,  # Number of steps to collect before updating
    num_epochs,  # Number of times to update on the same batch
    batch_size,  # Mini-batch size for updates
    log_frequency,  # How often to log metrics
    max_episode_steps,  # Max steps per episode, None to use env default
    eval_frequency,  # How often to run evaluation episodes
    num_eval_episodes,  # Number of episodes for evaluation
    checkpoint_frequency,# How often to save a checkpoint
    normalize_advantages,
    critic_updates_per_actor_update,
    seed
):
    """
    Train an ActorCriticWithFlow model online using an OpenAI Gym environment.
    Now with proper mini-batch updates using the batch_size parameter.
    """
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Create environments
    env = gym.make(env_name)
    
    if max_episode_steps is not None:
        env._max_episode_steps = max_episode_steps
        
    # Create the ActorCriticWithFlow model
    model = ActorCriticWithFlow(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        flow_dim=flow_dim,
        num_flow_layers=num_flow_layers,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        gamma=gamma,
        continuous=continuous,
        gae_lambda=gae_lambda,
        entropy_coef=entropy_coef,
        value_loss_coef=value_loss_coef,
        max_grad_norm=max_grad_norm,
        normalize_advantages = normalize_advantages,
        num_heads=num_heads,
        critic_updates_per_actor_update=critic_updates_per_actor_update
        
    )
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath='./checkpoints/',
        filename='ac_flow_online_{step}',
        save_top_k=5,
        monitor='eval_reward',
        mode='max'
    )
    # Setup logger
    logger = WandbLogger(project='dist_rl_online')
    logger.watch(model, log='all', log_freq=100)
    
    # trainer = pl.Trainer(
    #     max_epochs=num_epochs,
    #     logger=logger,
    #     callbacks=[checkpoint_callback],
    #     enable_progress_bar=True,
    #     enable_model_summary=True,
    #     accelerator="auto"
    # )
    
    # Storage for collected experiences
    states = []
    actions = []
    rewards = []
    dones = []
    next_states = []
    
    # Training loop
    state, _ = env.reset(seed=seed)
    episode_reward = 0
    episode_length = 0
    episodes_completed = 0
    
    # Metrics tracking
    all_episode_rewards = []
    recent_rewards = deque(maxlen=100)
    
    pbar = tqdm(total=total_timesteps, desc="Training")
    timestep = 0
    
    # Initialize normalization statistics with some defaults
    # These will be updated based on collected returns
    model.critic.return_mean.data = torch.tensor(0.0, dtype=torch.float32)
    model.critic.return_std.data = torch.tensor(1.0, dtype=torch.float32)
    
    while timestep < total_timesteps:
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Select action
        with torch.no_grad():
            action = model.act(state_tensor, deterministic=False)
            if continuous:
                action_np = action.squeeze().cpu().numpy()
            else:
                action_np = action.item()
        
        # Take step in environment
        next_state, reward, terminated, truncated, _ = env.step(action_np)
        done = terminated or truncated
        
        # Store transition
        states.append(state)
        actions.append(action_np)
        rewards.append(reward)
        dones.append(done)
        next_states.append(next_state)
        
        # Update for next iteration
        state = next_state
        episode_reward += reward
        episode_length += 1
        timestep += 1
        
        # Handle episode termination
        if done:
            state, _ = env.reset()
            episodes_completed += 1
            recent_rewards.append(episode_reward)
            all_episode_rewards.append(episode_reward)
            
            # Log episode metrics
            if episodes_completed % log_frequency == 0:
                avg_reward = np.mean(recent_rewards) if recent_rewards else episode_reward
                tqdm.write(f"Episode {episodes_completed}, Reward: {episode_reward:.2f}, Avg Reward (100 ep): {avg_reward:.2f}")
                logger.log_metrics({
                    'env/episode_reward': episode_reward,
                    'env/avg_reward_100': avg_reward,
                    'env/episode_length': episode_length,
                    'env/episodes_completed': episodes_completed,
                    'env/timestep': timestep
                })
            
            episode_reward = 0
            episode_length = 0
        
        # Update model when enough data is collected
        if len(states) >= update_frequency or timestep >= total_timesteps:
            tqdm.write(f"Updating model at timestep {timestep} with {len(states)} experiences...")
            
            # Update return normalization statistics
            # all_rewards = np.array(rewards)
            # model.critic.return_mean.data = torch.tensor(np.mean(all_rewards), dtype=torch.float32)
            # model.critic.return_std.data = torch.tensor(np.std(all_rewards) if np.std(all_rewards) > 1e-8 else 1.0, dtype=torch.float32)
            
            # Convert experiences to tensors
            states_tensor = torch.FloatTensor(np.array(states))
            if continuous:
                actions_tensor = torch.FloatTensor(np.array(actions))
            else:
                actions_tensor = torch.LongTensor(np.array(actions))
            rewards_tensor = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
            dones_tensor = torch.FloatTensor(np.array(dones)).unsqueeze(1)
            next_states_tensor = torch.FloatTensor(np.array(next_states))
            
            # Create batch
            full_batch = {
                'states': states_tensor,
                'actions': actions_tensor,
                'rewards': rewards_tensor,
                'dones': dones_tensor,
                'next_states': next_states_tensor
            }
            
            # Process the full batch to calculate advantages and returns
            with torch.no_grad():
                model._prepare_training_data(full_batch)
            
            # Get the processed data
            flat_states = model.critic_batch['states']
            flat_actions = model.critic_batch['actions']
            advantages = model.advantages
            returns = model.critic_returns
            
            returns_np = returns.detach().cpu().numpy()

            # Update return statistics based on calculated returns, not raw rewards
            model.critic.return_mean.data = torch.tensor(np.mean(returns_np), dtype=torch.float32)
            model.critic.return_std.data = torch.tensor(np.std(returns_np) if np.std(returns_np) > 1e-8 else 1.0, dtype=torch.float32)
            
            # Create a dataset and data loader for this batch of experiences
            dataset = TensorDataset(flat_states, flat_actions, advantages, returns)
            # actor_data_loader = DataLoader(
            #     dataset, 
            #     batch_size=batch_size,
            #     shuffle=True,
            #     drop_last=False
            # )
            
            # And another for critic with the critic batch size
            # critic_data_loader = DataLoader(
            #     dataset, 
            #     batch_size=critic_batch_size,
            #     shuffle=True,
            #     drop_last=False
            # )
            data_loader = DataLoader(
                dataset, 
                batch_size=critic_batch_size,
                shuffle=True,
                drop_last=False
            )
            
            # Build train_dataloaders dictionary for each optimizer
            # train_dataloaders = {
            #     0: actor_data_loader,  # For actor optimizer
            #     1: critic_data_loader  # For critic optimizer
            # }
            
            trainer = pl.Trainer(
                        max_epochs=num_epochs,
                        logger=logger,
                        callbacks=[checkpoint_callback],
                        enable_progress_bar=True,
                        enable_model_summary=True,
                        accelerator="auto"
                    )
            
            
            # Train the model using Lightning Trainer with separate data loaders for each optimizer
            trainer.fit(model, train_dataloaders=data_loader)
            
            # Clear experience buffer after training
            states = []
            actions = []
            rewards = []
            dones = []
            next_states = []
            
            
            # Save checkpoint if needed
            if timestep % checkpoint_frequency < update_frequency:
                checkpoint_path = f"./checkpoints/ac_flow_online_step_{timestep}.ckpt"
                torch.save(model.state_dict(), checkpoint_path)
                tqdm.write(f"Saved checkpoint at step {timestep}: {checkpoint_path}")
        
        # Update progress bar
        pbar.update(1)

    # Final checkpoint
    final_checkpoint_path = f"./checkpoints/ac_flow_online_final.ckpt"
    torch.save(model.state_dict(), final_checkpoint_path)
    tqdm.write(f"Saved final checkpoint: {final_checkpoint_path}")
    
    # Close environments
    env.close()
    
    return model