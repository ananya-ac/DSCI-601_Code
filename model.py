import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import pytorch_lightning as pl
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
from data import create_rollout_dataset
import pdb
from geomloss import SamplesLoss
from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet, ConditionNode
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger
from torch.distributions import Normal, Categorical
import config
import gymnasium as gym
from tqdm import tqdm
from collections import deque


def NLL(z:torch.Tensor, det_log_j:torch.Tensor) -> torch.Tensor:
    """
    Negative log-likelihood loss for the normalizing flow.
    
    Args:
        z: Latent variable samples
        det_log_j: Log determinant of the Jacobian
        
    Returns:
        Negative log-likelihood loss
    """
    return torch.mean(torch.sum(0.5 * z**2, dim=-1) - det_log_j)


def subnet_fc(c_in:int, c_out:int) -> nn.Module:
    """
    Subnet for the coupling blocks in the normalizing flow.
    
    Args:
        c_in: Input dimension
        c_out: Output dimension
        
    Returns:
        Neural network module
    """
    return nn.Sequential(
        nn.Linear(c_in, 512), 
        nn.ReLU(),
        nn.Linear(512, c_out)
    )


def build_normalizing_flow(input_dim:int, condition_dim:int, num_layers:int=8) -> ReversibleGraphNet:
    """
    Build a normalizing flow model using FrEIA with explicit conditioning.
    
    Args:
        input_dim: Dimension of the input data (returns)
        condition_dim: Dimension of the conditioning data (state encoding)
        num_layers: Number of coupling layers
        
    Returns:
        ReversibleGraphNet model with conditioning
    """
    
    # Create the nodes for the graph
    nodes = [InputNode(input_dim, name='input')]
    
    # Add a condition node for the state encoding
    cond_node = ConditionNode(condition_dim, name='condition')
    
    for k in range(num_layers):
        nodes.append(Node(
            nodes[-1],
            GLOWCouplingBlock,
            {'subnet_constructor': subnet_fc, 'clamp': 2.0},
            conditions=cond_node,  # Connect the condition node to this layer
            name=f'coupling_{k}'
        ))
        nodes.append(Node(
            nodes[-1],
            PermuteRandom,
            {'seed': k},
            name=f'permute_{k}'
        ))
    
    nodes.append(OutputNode(nodes[-1], name='output'))
    
    # Create the reversible graph with the nodes
    model = ReversibleGraphNet(nodes + [cond_node], verbose=False)
    
    # Initialize parameters
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    for param in trainable_parameters:
        param.data = 0.05 * torch.randn_like(param)
    
    return model


class SelfAttentionEncoder(nn.Module):
    """Self-attention based state-action encoder"""
    def __init__(self, state_dim:int, action_dim:int, hidden_dim:int, num_heads:int, continuous:bool, num_actions:int=2):
        super().__init__()
        
        # Input embeddings for state and action features
        self.state_embedding = nn.Linear(state_dim, hidden_dim)
        self.action_embedding = nn.Linear(action_dim, hidden_dim)
        
        # Self-attention layer
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Output projection
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        if continuous:
            self.action_embedding = nn.Linear(action_dim, hidden_dim)
        else:
            # Embedding layer for discrete actions
            self.action_embedding = nn.Embedding(num_actions, hidden_dim)
    
    def forward(self, state:torch.Tensor, action:torch.Tensor) -> torch.Tensor:
        batch_size = state.shape[0]
        
        # Ensure action has correct dimensions
        # if action.dim() == 1:
        #     action = action.unsqueeze(1)
            
        # Embed state features
        state_emb = self.state_embedding(state).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        # Embed action
        action_emb = self.action_embedding(action).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        # Concatenate state and action embeddings
        # This treats them as a set of tokens to apply self-attention 
        combined = torch.cat([state_emb, action_emb], dim=1)  # [batch_size, 2, hidden_dim]
        # Apply self-attention
        attn_output, _ = self.self_attention(combined, combined, combined)
        
        # Apply layer normalization with residual connection
        normalized = self.layer_norm(attn_output + combined)
        
        # Pool the attended features (mean pooling)
        pooled = torch.mean(normalized, dim=1)  # [batch_size, hidden_dim]
        
        # Final encoding
        encoding = self.output_layer(pooled)
        
        return encoding

class ConditionalNormalizingFlow(pl.LightningModule):
    def __init__(self, state_dim:int, action_dim:int, hidden_dim:int, flow_dim:int, num_layers:int, 
                 learning_rate:float, normalize_returns:bool, num_heads:int,continuous:bool):
        super().__init__()
        self.save_hyperparameters()
        
        # Replace sequential encoder with self-attention encoder
        self.state_action_encoder = SelfAttentionEncoder(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            continuous=continuous
        )
        
        # Return projection layers
        self.return_proj = nn.Linear(1, flow_dim)
        self.inv_return_proj = nn.Linear(flow_dim, 1)
        
        # Build the normalizing flow with explicit conditioning
        self.flow = build_normalizing_flow(flow_dim, hidden_dim, num_layers)
        
        self.flow_dim = flow_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.normalize_returns = normalize_returns
        self.return_mean = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.return_std = nn.Parameter(torch.ones(1), requires_grad=False)
    
    def forward(self, state, action, return_value, num_samples=0, rev=False):
        """
        Forward pass with state-action conditioning.
        
        Args:
            state: State tensor [batch_size, state_dim]
            action: Action tensor [batch_size, action_dim]
            return_value: Optional return values to evaluate [batch_size, 1]
            num_samples: Number of samples to generate per state-action pair
            rev: Whether to run the flow in reverse (for sampling)
            
        Returns:
            Dictionary containing samples, log-likelihood, etc.
        """
        batch_size = state.shape[0]
        
        # if action.dim() == 1 and self.action_dim > 1:
        #     action = action.unsqueeze(1)
        
        # Get self-attention encoding of state-action pair
        encoding = self.state_action_encoder(state, action)
        
        if not rev:
            # Forward direction: Return → Latent
            if return_value is None:
                raise ValueError("Return value must be provided for forward pass")
            if self.normalize_returns:
                return_value = (return_value - self.return_mean) / self.return_std
            # Project return to flow dimension
            return_emb = self.return_proj(return_value)
            
            # Forward through the flow with explicit conditioning
            z, log_det_J = self.flow([return_emb], c=[encoding], rev=False, jac=True)
            
            return {
                'z': z,
                'log_det_J': log_det_J,
                'nll': NLL(z, log_det_J)
            }
        else:
            # Reverse direction: Latent → Return (sampling)
            # Generate samples from base distribution
            z = torch.randn(batch_size * num_samples, self.flow_dim, device=state.device)
            
            # Expand encoding for multiple samples per state-action pair
            if num_samples > 1:
                # Repeat encoding for each sample
                encoding = encoding.repeat_interleave(num_samples, dim=0)
            
            # Transform through the flow in reverse with explicit conditioning
            samples_emb, _ = self.flow([z], c=[encoding], rev=True, jac=False)
            
            # Project back to scalar returns
            samples = self.inv_return_proj(samples_emb)
            
            if self.normalize_returns:
                samples = samples * self.return_std + self.return_mean
            # Reshape to [batch_size, num_samples]
            samples = samples.view(batch_size, num_samples)
            
            return {
                'samples': samples,
                'mean': samples.mean(dim=1, keepdim=True),
                'std': samples.std(dim=1, keepdim=True)
            }
    
    def training_step(self, batch, batch_idx):
        """
        Training step for state-action conditional model.
        
        Args:
            batch: Dictionary containing 'state', 'action', and 'return'
            batch_idx: Batch index
            
        Returns:
            Loss value
        """
        state = batch['state']
        action = batch['action']
        return_value = batch['return']
        
        # Forward pass through the flow
        output = self(state, action, return_value)
        
        # Calculate negative log-likelihood loss
        loss = output['nll']
        
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        
        Args:
            batch: Dictionary containing 'state', 'action', and 'return'
            batch_idx: Batch index
        """
        state = batch['state']
        action = batch['action']
        return_value = batch['return']
        
        # Forward pass
        output = self(state, action, return_value)
        
        # Calculate loss
        loss = output['nll']
        
        self.log('val_loss', loss)
    
    def configure_optimizers(self):
        """
        Configure optimizer.
        
        Returns:
            Optimizer
        """
        # Get all trainable parameters
        trainable_parameters = [p for p in self.parameters() if p.requires_grad]
        
        optimizer = torch.optim.Adam(
            trainable_parameters, 
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0
        )
        
        return optimizer
    
    def sample_value_distribution(self, state, action, num_samples):
        """
        Sample from the value distribution for a given state-action pair.
        
        Args:
            state: State tensor [batch_size, state_dim]
            action: Action tensor [batch_size, action_dim]
            num_samples: Number of samples to generate
            
        Returns:
            Dictionary with samples and statistics
        """
        with torch.no_grad():
            return self(state, action, num_samples=num_samples, rev=True)


class ActorNetwork(nn.Module):
    """
    Actor network for continuous or discrete action spaces.
    """
    def __init__(self, state_dim, action_dim, hidden_dim, continuous):
        super().__init__()
        self.continuous = continuous
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        if continuous:
            # For continuous actions, output mean and log_std
            self.mean = nn.Linear(hidden_dim, action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            # For discrete actions, output action logits
            self.action_head = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        if self.continuous:
            mean = self.mean(x)
            std = self.log_std.exp()
            return mean, std
        else:
            action_logits = self.action_head(x)
            return action_logits
    
    def get_action(self, state, deterministic):
        if self.continuous:
            mean, std = self(state)
            if deterministic:
                return mean
            else:
                dist = Normal(mean, std)
                action = dist.sample()
                return action, dist.log_prob(action)
        else:
            logits = self(state)
            if deterministic:
                return torch.argmax(logits, dim=1)
            else:
                dist = Categorical(logits=logits)
                action = dist.sample()
                return action, dist.log_prob(action)


class ActorCriticWithFlow(pl.LightningModule):
    """
    Actor-Critic model using a ConditionalNormalizingFlow as the critic.
    """
    def __init__(
        self, 
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
        normalize_advantages,
        num_heads,
        critic_updates_per_actor_update
        
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Actor network
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim, continuous)
        
        # Critic network (ConditionalNormalizingFlow)
        self.critic = ConditionalNormalizingFlow(
            state_dim=state_dim,
            action_dim=action_dim,  # For discrete actions, action is just an index
            hidden_dim=hidden_dim,
            flow_dim=flow_dim,
            num_layers=num_flow_layers,
            learning_rate=critic_lr,
            normalize_returns=normalize_advantages,
            num_heads=num_heads,
            continuous=continuous
        )
        
        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.normalize_advantages = normalize_advantages
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.continuous = continuous
        self.automatic_optimization = False
        self.critic_updates_per_actor_update = critic_updates_per_actor_update
        self.update_counter = 0
        
        self.critic_batch = None
        self.critic_returns = None
        self.advantages = None
        
        
    def get_value(self, states, actions):
        """
        Get value estimate (mean of distribution) for state-action pairs.
        """
        
        with torch.no_grad():
            output = self.critic(states, actions, rev=True, return_value = None, num_samples=config.num_samples)
            values = output['mean']
        
        return values
    
    def compute_gae(self, rewards, values, next_values, dones):
        """
        Compute Generalized Advantage Estimation.
        """
        deltas = rewards + self.gamma * next_values * (1 - dones) - values
        advantages = torch.zeros_like(rewards)
        
        gae = 0
        for t in reversed(range(len(rewards))):
            gae = deltas[t] + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            
        # Compute returns for critic training
        returns = advantages + values
        
        # Normalize advantages
        if self.normalize_advantages and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
        return advantages, returns
    
        

       
    def on_train_start(self):
        # Initialize step counters
        self.critic_step_counter = 0
        self.actor_step_counter = 0
    
    def training_step(self, batch, batch_idx, optimizer_idx=0):
        """
        Training step using actor-critic with GAE.
        
        batch: Dictionary containing episodes of states, actions, rewards, dones, next_states
        optimizer_idx: Index of the optimizer to use (0 for actor, 1 for critic)
        """
        # First, prepare the data if this is a new batch
        if self.critic_batch is None or batch_idx % (self.hparams.critic_updates_per_step + 1) == 0:
            self._prepare_training_data(batch)
        
        # Depending on the optimizer index, train either actor or critic
        if optimizer_idx == 0:
            # Actor update
            return self._actor_training_step()
        else:
            # Critic update
            return self._critic_training_step()
        
    def training_step(self, batch, batch_idx):
    # Get optimizers
    
        actor_optimizer, critic_optimizer = self.optimizers()
        
        # Unpack the batch
        states, actions, advantages, returns = batch
        
        
        # Store the current batch data
        self.advantages = advantages
        self.critic_returns = returns
        self.critic_batch = {'states': states, 'actions': actions}
        
        # Always update critic
        critic_optimizer.zero_grad()
        critic_loss = self._critic_training_step()
        self.manual_backward(critic_loss)
        critic_optimizer.step()
        self.log('critic/loss', critic_loss, prog_bar=True)
        self.log('critic/step', float(self.critic_step_counter), prog_bar=True)
        
        
        # Update actor less frequently
        update_actor = (self.update_counter % self.critic_updates_per_actor_update == 0)
        
        if update_actor:
            actor_optimizer.zero_grad()
            actor_loss = self._actor_training_step()
            self.manual_backward(actor_loss)
            actor_optimizer.step()
            self.log('actor/loss', actor_loss, prog_bar=True)
            self.log('actor/step', float(self.actor_step_counter), prog_bar=True)
        
        
        # Increment update counter
        self.update_counter += 1    

    
    def _prepare_training_data(self, batch):
        """Prepare training data for both actor and critic."""
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        dones = batch['dones']
        next_states = batch['next_states']
        # Flatten batch dimensions for processing
        # flat_states = states.reshape(-1, states.shape[-1])
        # flat_actions = actions.reshape(-1, actions.shape[-1] if actions.dim() > 2 else 1)
        # flat_next_states = next_states.reshape(-1, next_states.shape[-1])
        # flat_rewards = rewards.reshape(-1, 1)
        # flat_dones = dones.reshape(-1, 1)
        
        # Get current values
        values = self.get_value(states, actions)
        
        # For next states, we need to get the action from our policy first
        if self.continuous:
            next_actions, _ = self.actor.get_action(next_states, deterministic=False)
        else:
            next_actions, _ = self.actor.get_action(next_states, deterministic=False)
            
        # Get next values
        next_values = self.get_value(next_states, next_actions)
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(
            rewards, values, next_values, dones
        )
        
        # Store the data for actor and critic training
        self.critic_batch = {'states': states, 'actions': actions}
        self.critic_returns = returns
        self.advantages = advantages
    
    def _actor_training_step(self):
        """Train the actor using the advantages."""
        flat_states = self.critic_batch['states']
        flat_actions = self.critic_batch['actions']
        advantages = self.advantages
        
        # Actor loss
        if self.continuous:
            action_mean, action_std = self.actor(flat_states)
            dist = Normal(action_mean, action_std)
            log_probs = dist.log_prob(flat_actions.squeeze(-1)).sum(dim=-1, keepdim=True)
            entropy = dist.entropy().sum(dim=-1, keepdim=True)
        else:
            action_logits = self.actor(flat_states)
            dist = Categorical(logits=action_logits)
            log_probs = dist.log_prob(flat_actions.squeeze(-1)).unsqueeze(-1)
            entropy = dist.entropy().unsqueeze(-1)
        
        # Policy gradient loss
        actor_loss = -(log_probs * advantages).mean()
        
        # Entropy bonus for exploration
        entropy_loss = -self.entropy_coef * entropy.mean()
        
        # Total actor loss
        total_actor_loss = actor_loss + entropy_loss
        
        # Log metrics
        
        # Increment actor step counter
        self.actor_step_counter += 1
        
        return total_actor_loss
    
    def _critic_training_step(self):
        """Train the critic using the returns."""
        flat_states = self.critic_batch['states']
        flat_actions = self.critic_batch['actions']
        returns = self.critic_returns
        
        # Sample a subset of the batch for critic update if batch_size is specified
        if hasattr(self.hparams, 'critic_batch_size') and self.hparams.critic_batch_size and self.hparams.critic_batch_size < len(flat_states):
            indices = torch.randperm(len(flat_states))[:self.hparams.critic_batch_size]
            critic_states = flat_states[indices]
            critic_actions = flat_actions[indices]
            critic_returns = returns[indices]
        else:
            critic_states = flat_states
            critic_actions = flat_actions
            critic_returns = returns
            
        # Forward pass through critic
        critic_output = self.critic(critic_states, critic_actions, critic_returns)
        critic_loss = self.value_loss_coef * critic_output['nll'].mean()
        
        # Log metrics
        
        # Increment critic step counter
        self.critic_step_counter += 1
        
        return critic_loss
    
    
    
    def configure_optimizers(self):
        """
        Configure separate optimizers for actor and critic with custom frequency.
        """
        actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        
        # Critic has its own optimizer in the ConditionalNormalizingFlow class
        critic_optimizer = self.critic.configure_optimizers()
        
        # Return optimizers with a frequency dictionaries to control how often they're used
        # The actor optimizer will be called once every `critic_updates_per_step + 1` calls
        # The critic optimizer will be called `critic_updates_per_step` times for every actor update
        return [
            {"optimizer": actor_optimizer, "frequency": config.actor_updates},
            {"optimizer": critic_optimizer, "frequency": config.critic_updates}
        ]
    
    def act(self, state, deterministic=False):
        """
        Select an action given a state.
        """
        with torch.no_grad():
            action, _ = self.actor.get_action(state, deterministic)
            return action




