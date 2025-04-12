import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import pandas as pd
import config as c


def get_rollout_df():
    
# Initialize rollouts storage with episode_id to track the order
    rollouts = {
        'episode_id': [],
        's': [],
        'a': [],
        'r': [],
        's_prime': [],
        'dones': []
    }

    vec_env = make_vec_env("LunarLander-v2", n_envs=4)
    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=100)
    obs = vec_env.reset()
    episode_id = 0  # Start with episode 0
    
    while episode_id < c.NUM_TRAINING_ROLLOUTS_EPISODES:
        action, _states = model.predict(obs)
        n_obs, rewards, dones, info = vec_env.step(action)
        
        # Collect data for each environment and timestep
        for idx in range(len(dones)):
            rollouts['episode_id'].append(episode_id)
            rollouts['s'].append(obs[idx])
            rollouts['a'].append(action[idx])
            rollouts['s_prime'].append(n_obs[idx])
            rollouts['r'].append(rewards[idx])
            rollouts['dones'].append(dones[idx])
        
        # For each timestep, record the episode_id
        for idx in range(len(dones)):
            if dones[idx]:  # If this environment finished the episode
                episode_id += 1  # Increment the episode ID for the next one
                
        obs = n_obs

    # Create DataFrame
    df_rollouts = pd.DataFrame(rollouts)
    df_rollouts = df_rollouts.sort_values(by='episode_id').reset_index(drop=True)
    df_rollouts.to_csv('rollouts.csv')
    return df_rollouts