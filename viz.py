import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_value_distribution(model, state, action, return_value=None, num_samples=1000, bins=50, title="Value Distribution"):
    """
    Plot the value distribution for a given state-action pair.
    
    Args:
        model: Trained ConditionalNormalizingFlow model
        state: State tensor or numpy array
        action: Action tensor or numpy array
        return_value: Actual return value from data (optional, for display only)
        num_samples: Number of samples to draw
        bins: Number of histogram bins
        title: Plot title base
    """
    
    # Convert inputs to tensors if needed
    if not isinstance(state, torch.Tensor):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    if not isinstance(action, torch.Tensor):
        action = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
    
    # Ensure model is in evaluation mode
    model.eval()
    
    # Sample from the value distribution
    with torch.no_grad():
        result = model.sample_value_distribution(state, action, num_samples)
    
    # Get samples and statistics
    samples = result['samples'].squeeze().cpu().numpy()
    mean = result['mean'].item()
    std = result['std'].item()
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.hist(samples, bins=bins, density=True, alpha=0.7)
    plt.axvline(mean, color='r', linestyle='--', label=f'Mean: {mean:.2f}')
    plt.axvline(mean + std, color='g', linestyle='--', label=f'Mean + Std: {mean+std:.2f}')
    plt.axvline(mean - std, color='g', linestyle='--', label=f'Mean - Std: {mean-std:.2f}')
    
    # Include return value in title if provided
    if return_value is not None:
        if isinstance(return_value, torch.Tensor):
            return_value = return_value.item()
        full_title = f"{title} | Actual Return: {return_value:.2f} | Predicted Return: {mean:.2f}"
    else:
        full_title = f"{title} | Predicted Return: {mean:.2f}"
    
    plt.title(full_title)
    plt.xlabel('Return')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()

def plot_comparison_distributions(model, states, actions, returns, labels=None, num_samples=1000, bins=50, title="Value Distributions Comparison"):
    """
    Plot multiple value distributions for different state-action pairs.
    
    Args:
        model: Trained ConditionalNormalizingFlow model
        states: List of state tensors or numpy arrays
        actions: List of action tensors or numpy arrays
        returns: List of actual return values (optional, for display)
        labels: List of labels for the legend
        num_samples: Number of samples to draw for each distribution
        bins: Number of histogram bins
        title: Plot title
    """
    
    # Ensure model is in evaluation mode
    model.eval()
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Generate color map
    colors = plt.cm.viridis(np.linspace(0, 1, len(states)))
    
    for i, (state, action, return_value) in enumerate(zip(states, actions, returns)):
        # Convert inputs to tensors if needed
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        
        # Sample from the value distribution
        with torch.no_grad():
            result = model.sample_value_distribution(state, action, num_samples)
        
        # Get samples and statistics
        samples = result['samples'].squeeze().cpu().numpy()
        mean = result['mean'].item()
        
        # Create label with return information
        if isinstance(return_value, torch.Tensor):
            return_value = return_value.item()
            
        label = labels[i] if labels and i < len(labels) else f"Distribution {i+1}"
        label = f"{label} (Actual Return: {return_value:.2f}, Predicted: {mean:.2f})"
        
        # Plot histogram and mean line
        plt.hist(samples, bins=bins, density=True, alpha=0.4, color=colors[i], label=label)
        plt.axvline(mean, color=colors[i], linestyle='--')
        
        # Plot actual return value
        plt.axvline(return_value, color=colors[i], linestyle='-', linewidth=2)
    
    plt.title(title)
    plt.xlabel('Return')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()

def plot_action_comparison(model, state, actions, returns=None, num_samples=1000, bins=50, title="Value Distributions for Different Actions"):
    """
    Plot value distributions for the same state with different actions.
    
    Args:
        model: Trained ConditionalNormalizingFlow model
        state: State tensor or numpy array
        actions: List of action tensors or numpy arrays
        returns: List of actual return values (optional)
        num_samples: Number of samples to draw for each distribution
        bins: Number of histogram bins
        title: Plot title
    """
    
    # Ensure model is in evaluation mode
    model.eval()
    
    # Convert state to tensor if needed
    if not isinstance(state, torch.Tensor):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    
    # Repeat state for each action
    states = [state] * len(actions)
    
    # If returns not provided, use None for each action
    if returns is None:
        returns = [None] * len(actions)
    
    # Generate labels
    labels = [f"Action {i+1}" for i in range(len(actions))]
    
    return plot_comparison_distributions(model, states, actions, returns, labels, num_samples, bins, title)
