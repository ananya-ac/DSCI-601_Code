import os
import gym
import torch
import numpy as np
import time
from datetime import datetime
from model import ActorCriticWithFlow  
import config

# Check if gym.wrappers is available for video recording
try:
    from gym.wrappers import RecordVideo
    HAS_RECORD_VIDEO = True
except ImportError:
    try:
        from gym.wrappers.monitoring.video_recorder import VideoRecorder
        HAS_RECORD_VIDEO = True
    except ImportError:
        HAS_RECORD_VIDEO = False

def eval_model(checkpoint_dir, env_name, num_episodes=2, render=True, record_video=True, video_dir="./videos"):
    """
    Evaluates a saved model on an OpenAI Gym environment.
    
    Args:
        checkpoint_dir (str): Directory containing model checkpoints
        env_name (str): Name of the OpenAI Gym environment to evaluate on
        num_episodes (int): Number of episodes to run (default: 2)
        render (bool): Whether to render the environment (default: True)
        
    Returns:
        list: List of episode rewards
    """
    # Find the best model in the checkpoint directory
    best_model_path = os.path.join(checkpoint_dir, "best_model_reward.ckpt")
    
    
    print(f"Loading model from {best_model_path}")
    
    # Create the environment with proper rendering
    try:
        # First try with render_mode parameter (Gym 0.26.0+)
        env = gym.make(env_name, render_mode="rgb_array" if render else None)
    except TypeError:
        # Fall back to older Gym versions
        env = gym.make(env_name)
        if render:
            try:
                env.render(mode='human')
            except Exception as e:
                print(f"Warning: Render failed with error: {e}")
                print("Will try rendering during the episode loop")
    
    # Set up video recording if requested
    if record_video and HAS_RECORD_VIDEO:
        try:
            # Make sure video directory exists
            os.makedirs(video_dir, exist_ok=True)
            
            # Try the newer gym RecordVideo wrapper first
            if hasattr(gym.wrappers, 'RecordVideo'):
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                video_path = os.path.join(video_dir, f"{env_name}-{timestamp}")
                env = gym.wrappers.RecordVideo(
                    env, 
                    video_path,
                    episode_trigger=lambda episode_id: True  # Record all episodes
                )
                print(f"Recording videos to {video_path}")
            # Fall back to older VideoRecorder
            else:
                video_recorder = None
                video_path = os.path.join(video_dir, f"{env_name}-{time.strftime('%Y%m%d-%H%M%S')}.mp4")
                print(f"Recording video to {video_path}")
                video_recorder = gym.wrappers.monitoring.video_recorder.VideoRecorder(
                    env, path=video_path
                )
                # Store recorder in function scope
                eval_model.video_recorder = video_recorder
        except Exception as e:
            print(f"Failed to set up video recording: {e}")
            record_video = False
    
    # Get environment dimensions
    state, _ = env.reset()
    state_dim = len(state) if hasattr(state, "__len__") else 1
    num_actions = env.action_space.n if isinstance(env.action_space, gym.spaces.Discrete) else env.action_space.shape[0]
    
    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n
        continuous = False
    else:
        action_dim = env.action_space.shape[0]
        continuous = True
    
    # Create model with same architecture as during training
    # Note: You'll need to ensure these hyperparameters match the ones used during training
    # If you have the hyperparameters saved somewhere, you should load them instead
    model = ActorCriticWithFlow(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=config.hidden_dim,  
        flow_dim=config.flow_dim,     
        num_flow_layers=config.num_flow_layers, 
        actor_lr=config.actor_lr,   
        critic_lr=config.critic_lr,  
        gamma=config.gamma,      
        continuous=continuous,
        gae_lambda=config.gae_lambda, 
        entropy_coef=config.entropy_coef,  
        value_loss_coef=config.value_loss_coef, 
        max_grad_norm=config.max_grad_norm,   
        num_heads=config.num_heads,         
        normalize_advantages=config.normalize_advantages,  
        critic_updates_per_actor_update=config.critic_updates,
        num_actions = num_actions
    )
    
    # Load the saved model weights
    try:
        # Try standard loading first
        model.load_state_dict(torch.load(best_model_path))
    except RuntimeError as e:
        # Handle specific tensor shape mismatches
        if "size mismatch for critic.return_mean" in str(e) or "size mismatch for critic.return_std" in str(e):
            print("Detected shape mismatch in normalization parameters. Attempting to fix...")
            
            # Load checkpoint with shape mismatch handling
            checkpoint = torch.load(best_model_path)
            model_dict = model.state_dict()
            
            # Fix shape mismatches
            for key in list(checkpoint.keys()):
                if key in model_dict:
                    if checkpoint[key].shape != model_dict[key].shape:
                        print(f"Fixing shape mismatch for {key}: {checkpoint[key].shape} vs {model_dict[key].shape}")
                        
                        # Handle scalar vs 1D tensor conversion
                        if checkpoint[key].numel() == 1 and model_dict[key].numel() == 1:
                            checkpoint[key] = checkpoint[key].view(model_dict[key].shape)
                        elif model_dict[key].numel() == 1 and checkpoint[key].numel() == 1:
                            # Get the scalar value and reshape it
                            value = checkpoint[key].item()
                            checkpoint[key] = torch.tensor([value], dtype=model_dict[key].dtype)
                
            # Load the adjusted state dict
            model.load_state_dict(checkpoint, strict=False)
            print("Model loaded with shape adjustments")
        else:
            # If it's a different error, re-raise it
            raise e

    model.eval()  # Set to evaluation mode
    
    # Run evaluation episodes
    episode_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        step = 0
        
        print(f"Starting episode {episode+1}/{num_episodes}")
        
        while not done:
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Select action (deterministic for evaluation)
            with torch.no_grad():
                action = model.act(state_tensor, deterministic=True)
                if continuous:
                    action_np = action.squeeze().cpu().numpy()
                else:
                    action_np = action.item()
            
            # Take step in environment
            next_state, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated
            
            # Update for next iteration
            state = next_state
            episode_reward += reward
            step += 1
            
            # Handle rendering based on Gym version
            if render:
                try:
                    # For older versions of Gym that need explicit render calls
                    if not hasattr(env, 'render_mode') or env.render_mode is None:
                        env.render()
                except Exception as e:
                    if step == 0:  # Only print the warning once per episode
                        print(f"Warning: Rendering failed with error: {e}")
        
        print(f"Episode {episode+1} finished with reward {episode_reward:.2f} in {step} steps")
        episode_rewards.append(episode_reward)
    
    # Close environment
    env.close()
    
    # Print summary
    avg_reward = np.mean(episode_rewards)
    print(f"\nEvaluation complete over {num_episodes} episodes")
    print(f"Average reward: {avg_reward:.2f}")
    
    return episode_rewards