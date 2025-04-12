import torch
from torch.utils import data
import numpy as np
import pandas as pd


class RolloutDataset(data.Dataset):
    """Dataset for rollout data with calculated returns."""
    
    def __init__(self, states, actions, returns):
        """
        Initialize dataset.
        
        Args:
            states: List of state vectors
            returns: List of corresponding returns
        """
        self.states = torch.tensor(np.array(states), dtype=torch.float32)
        self.actions = torch.tensor(np.array(actions), dtype=torch.float32)
        self.returns = torch.tensor(np.array(returns), dtype=torch.float32).unsqueeze(1)
        
        
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return {
            'state': self.states[idx],
            'action': self.actions[idx],
            'return': self.returns[idx]
        }


def calculate_returns(df, gamma=0.99, type='mc'):
    """
    Calculate returns based on different methods: Monte Carlo or Temporal Difference.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The rollout dataframe with columns: 'episode_id', 's', 'a', 'r', 's_prime', 'dones'
    gamma : float, optional
        Discount factor for future rewards, default is 0.99
    type : str, optional
        Method to calculate returns: 'mc' (Monte Carlo) or 'td' (1-step Temporal Difference),
        default is 'mc'
    
    Returns:
    --------
    pandas.DataFrame
        The original dataframe with an additional 'returns' column
    """
    # Create a copy of the dataframe to avoid modifying the original
    df_returns = df.copy()
    
    # Initialize the returns column
    df_returns['returns'] = 0.0  # Column to store returns
    
    # Find terminal states
    terminal_state_idxs = df_returns[df_returns['dones']].index
    
    # Process each episode
    for i in range(len(terminal_state_idxs)):
        # Determine episode start and end indices
        end_idx = terminal_state_idxs[i]
        
        if i == 0:
            start_idx = 0
        else:
            start_idx = terminal_state_idxs[i-1] + 1
        
        # Get episode data
        episode_data = df_returns.iloc[start_idx:end_idx+1]
        
        if type == 'mc':
            # Monte Carlo returns (going backwards)
            G = 0  # Initialize cumulative return
            for j in range(len(episode_data) - 1, -1, -1):
                idx = episode_data.index[j]
                reward = episode_data.iloc[j]['r']
                G = reward + gamma * G
                df_returns.at[idx, 'returns'] = G
                
        elif type == 'td':
            # 1-step Temporal Difference returns
            for j in range(len(episode_data) - 1):
                idx = episode_data.index[j]
                next_idx = episode_data.index[j+1]
                reward = episode_data.iloc[j]['r']
                
                # For the last state in the episode
                if j == len(episode_data) - 2:
                    next_return = 0  # Terminal state has zero return
                else:
                    next_return = df_returns.at[next_idx, 'returns']
                
                # Update the return for the current state
                df_returns.at[idx, 'returns'] = reward + gamma * next_return
            
            # Handle the terminal state
            terminal_idx = episode_data.index[-1]
            terminal_reward = episode_data.iloc[-1]['r']
            df_returns.at[terminal_idx, 'returns'] = terminal_reward
            

    
    # Process the last episode if there are transitions after the last terminal state
    if terminal_state_idxs[-1] < len(df_returns) - 1:
        start_idx = terminal_state_idxs[-1] + 1
        end_idx = len(df_returns) - 1
        
        # Get episode data
        episode_data = df_returns.iloc[start_idx:end_idx+1]
        
        if type == 'mc':
            # Monte Carlo returns (going backwards)
            G = 0  # Initialize cumulative return
            for j in range(len(episode_data) - 1, -1, -1):
                idx = episode_data.index[j]
                reward = episode_data.iloc[j]['r']
                G = reward + gamma * G
                df_returns.at[idx, 'returns'] = G
                
        elif type == 'td':
            # 1-step Temporal Difference returns
            for j in range(len(episode_data) - 1):
                idx = episode_data.index[j]
                next_idx = episode_data.index[j+1]
                reward = episode_data.iloc[j]['r']
                
                # For the last state in the truncated episode
                if j == len(episode_data) - 2:
                    next_return = 0  # Assume zero return for the final state
                else:
                    next_return = df_returns.at[next_idx, 'returns']
                
                # Update the return for the current state
                df_returns.at[idx, 'returns'] = reward + gamma * next_return
            
            # Handle the last state of the truncated episode
            last_idx = episode_data.index[-1]
            last_reward = episode_data.iloc[-1]['r']
            df_returns.at[last_idx, 'returns'] = last_reward
            

    
    return df_returns



def get_state_action_returns(df_returns):
    """
    Extract state-return pairs from the dataframe to use for training value distribution models.
    
    Parameters:
    -----------
    df_returns : pandas.DataFrame
        The dataframe with calculated returns
    
    Returns:
    --------
    tuple
        (states, returns) where states is a list of state vectors and returns is a list of corresponding return values
    """
    # Convert string representation of state vectors to actual numpy arrays
    states = []
    for state_str in df_returns['s']:
        # The state is stored as a string representation of a list
        if isinstance(state_str, str):
            # Parse the string representation into a list of floats
            # This handles different formats of string representation
            if '[' in state_str:
                # Format: "[value1 value2 value3]" or "[value1, value2, value3]"
                state_str = state_str.strip('[]')
                # Handle both space and comma-separated values
                if ',' in state_str:
                    state_values = [float(val.strip()) for val in state_str.split(',') if val.strip()]
                else:
                    state_values = [float(val) for val in state_str.split() if val]
            else:
                # Format: "value1 value2 value3"
                state_values = [float(val) for val in state_str.split() if val]
            
            states.append(np.array(state_values))
        else:
            # Already a list or array
            states.append(np.array(state_str))
    
    returns = df_returns['returns'].values
    actions = df_returns['a'].values
    
    return states, actions, returns



def create_rollout_dataset(df, gamma=0.99):
    """
    Create a RolloutDataset from a dataframe of rollouts.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The rollout dataframe with columns: 'episode_id', 's', 'a', 'r', 's_prime', 'dones'
    gamma : float, optional
        Discount factor for future rewards, default is 0.99
    
    Returns:
    --------
    RolloutDataset
        A PyTorch dataset containing state-return pairs
    """
    
    
    # Calculate returns for the dataframe
    df_with_returns = calculate_returns(df, gamma)
    
    # Extract state-return pairs
    states, actions, returns = get_state_action_returns(df_with_returns)
    
    # Create and return the RolloutDataset
    return RolloutDataset(states, actions, returns)