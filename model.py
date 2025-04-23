import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import pytorch_lightning as pl
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet, ConditionNode
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom, AllInOneBlock
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import config
import copy
import wandb
from geomloss import SamplesLoss
import pdb

wasserstein_loss = SamplesLoss(loss = "gaussian")
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
        nn.Linear(c_in, config.hidden_dim_subnet), 
        nn.ReLU(),
        nn.Linear(config.hidden_dim_subnet, c_out)
    )


def build_normalizing_flow(input_dim: int,
                            condition_dim: int,
                            num_layers: int
                           ) -> ReversibleGraphNet:
        """
        Build a Glow-style normalizing flow using alternating GLOWCouplingBlock and
        PermuteRandom layers, with explicit conditioning on a state encoding.

        Args:
            input_dim: Dimension of the input (flow_dim).
            condition_dim: Dimension of the conditioning vector.
            num_layers: Number of coupling-permutation pairs.
            hidden_dim_subnet: Hidden dimension for the subnetworks.
            clamp: Affine clamping value for coupling blocks.

        Returns:
            A ReversibleGraphNet implementing the flow.
        """
        # Build graph nodes
        nodes = [InputNode(input_dim, name='input')]
        cond_node = ConditionNode(condition_dim, name='condition')

        for k in range(num_layers):
            # Coupling block
            nodes.append(Node(
                nodes[-1],
                GLOWCouplingBlock,
                {
                    'subnet_constructor': subnet_fc,
                    'clamp': 2.0
                },
                conditions=cond_node,
                name=f'coupling_{k}'
            ))
            # Permutation
            nodes.append(Node(
                nodes[-1],
                PermuteRandom,
                {'seed': k},
                name=f'permute_{k}'
            ))

        # Output node
        nodes.append(OutputNode(nodes[-1], name='output'))

        # Instantiate the reversible graph
        model = ReversibleGraphNet(
            nodes + [cond_node],
            verbose=False
        )

        # Xavier-type initialization
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)

        return model

# class SelfAttentionEncoder(nn.Module):
#     """Self-attention based state-action encoder"""
#     def __init__(self, state_dim:int, action_dim:int, hidden_dim:int, num_heads:int, continuous:bool, num_actions:int):
#         super().__init__()
        
#         # Input embeddings for state and action features
#         self.state_embedding = nn.Linear(state_dim, hidden_dim)
        
#         # Self-attention layer
#         self.self_attention = nn.MultiheadAttention(
#             embed_dim=hidden_dim,
#             num_heads=num_heads,
#             batch_first=True
#         )
        
#         # Output projection
#         self.layer_norm = nn.LayerNorm(hidden_dim)
#         self.output_layer = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim)
#         )
#         if continuous:
#             self.action_embedding = nn.Linear(action_dim, hidden_dim)
#         else:
#             # Embedding layer for discrete actions
#             self.action_embedding = nn.Embedding(num_actions, hidden_dim)
    
#     def forward(self, state:torch.Tensor, action:torch.Tensor) -> torch.Tensor:
        
#         # Ensure action has correct dimensions
#         # if action.dim() == 1:
#         #     action = action.unsqueeze(1)
#         batch_size = state.shape[0]
#         action = action.view(batch_size, -1)    
#         # Embed state features
#         state_emb = self.state_embedding(state).unsqueeze(1)  # [batch_size, 1, hidden_dim]
#         # Embed action
#         action_emb = self.action_embedding(action).unsqueeze(1)  # [batch_size, 1, hidden_dim]
#         # Concatenate state and action embeddings
#         # This treats them as a set of tokens to apply self-attention 
#         combined = torch.cat([state_emb, action_emb], dim=1)  # [batch_size, 2, hidden_dim]
#         # Apply self-attention
#         attn_output, _ = self.self_attention(combined, combined, combined)
        
#         # Apply layer normalization with residual connection
#         normalized = self.layer_norm(attn_output + combined)
        
#         # Pool the attended features (mean pooling)
#         pooled = torch.mean(normalized, dim=1)  # [batch_size, hidden_dim]
        
#         # Final encoding
#         encoding = self.output_layer(pooled)
        
#         return 

import torch
import torch.nn as nn

class SimpleEncoder(nn.Module):
    """Simple state-action encoder with LayerNorm"""
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        continuous: bool,
        num_actions: int
    ):
        super().__init__()
        
        # State embedding with LayerNorm
        self.state_embedding = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Action embedding (continuous or discrete) with LayerNorm
        if continuous:
            self.action_embedding = nn.Sequential(
                nn.Linear(action_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            )
        else:
            # For embeddings, apply LayerNorm after lookup
            self.action_embedding = nn.Sequential(
                nn.Embedding(num_actions, hidden_dim),
                nn.LayerNorm(hidden_dim)
            )
        
        # Fusion network with LayerNorm after each Linear
        self.fusion_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Output projection with LayerNorm
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.continuous = continuous    

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # state: (batch, state_dim)
        # action: (batch, action_dim) if continuous, else (batch,) long tensor
        s = self.state_embedding(state)
        a = self.action_embedding(action)
        x = torch.cat([s, a], dim=-1)
        x = self.fusion_network(x)
        return self.output_layer(x)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        batch_size = state.shape[0]
        action = action.view(batch_size, -1)
        
        # Embed state features
        state_emb = self.state_embedding(state)  # [batch_size, hidden_dim]
        
        # Embed action
        if not self.continuous:
            # For discrete actions
            action_emb = self.action_embedding(action.long().squeeze(-1))
        else:
            # For continuous actions
            action_emb = self.action_embedding(action)
        # Concatenate state and action embeddings
        combined = torch.cat([state_emb, action_emb], dim=1)  # [batch_size, 2*hidden_dim]
        
        # Apply fusion network to learn interactions between state and action
        fused = self.fusion_network(combined)  # [batch_size, hidden_dim]
        
        # Final encoding
        encoding = self.output_layer(fused)
        
        return encoding

# class SimpleEncoder(nn.Module):
#     def __init__(self, state_dim, hidden_dim):
#         super().__init__()
        
#         self.state_encoder = nn.Sequential(
#             nn.Linear(state_dim, hidden_dim),
#             nn.LeakyReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.LeakyReLU(),
#             nn.Linear(hidden_dim, hidden_dim)
#         )
    
#     def forward(self, state):
#         # Encode state only
#         state_encoding = self.state_encoder(state)
#         return state_encoding

class ConditionalNormalizingFlow(nn.Module):
    def __init__(self, state_dim:int, action_dim:int, hidden_dim:int, flow_dim:int, num_layers:int, 
                 learning_rate:float, normalize_returns:bool, num_heads:int,continuous:bool, num_actions):
        super().__init__()
        #self.save_hyperparameters()
        
        # Replace sequential encoder with self-attention encoder
        # self.state_action_encoder = SelfAttentionEncoder(
        #     state_dim=state_dim,
        #     action_dim=action_dim,
        #     hidden_dim=hidden_dim,
        #     num_heads=num_heads,
        #     continuous=continuous,
        #     num_actions=num_actions
        # )
        
        self.state_action_encoder = SimpleEncoder(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            continuous=continuous,
            num_actions=num_actions
        )
        # self.state_action_encoder = SimpleEncoder(
        #     state_dim=state_dim,
        #     hidden_dim=hidden_dim,
            
        # )
        
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
    
    def forward(self, state, action, return_value, num_samples=None, rev=False):
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
        #encoding = self.state_action_encoder(state)
        
        if not rev:
            # Forward direction: Return → Latent
            if return_value is None:
                raise ValueError("Return value must be provided for forward pass in the forward direction")
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
    
    # def training_step(self, batch, batch_idx):
    #     """
    #     Training step for state-action conditional model.
        
    #     Args:
    #         batch: Dictionary containing 'state', 'action', and 'return'
    #         batch_idx: Batch index
            
    #     Returns:
    #         Loss value
    #     """
    #     state = batch['state']
    #     action = batch['action']
    #     return_value = batch['return']
        
    #     # Forward pass through the flow
    #     output = self(state, action, return_value)
        
    #     # Calculate negative log-likelihood loss
    #     loss = output['nll']
        
    #     return loss
    
        
        
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
            betas=config.betas,
            eps=config.eps,
            
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
        
        # hidden layers + LayerNorm
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        if continuous:
            # state-dependent mean & std heads
            self.mean_head    = nn.Linear(hidden_dim, action_dim)
            self.log_std_head = nn.Linear(hidden_dim, action_dim)
        else:
            # discrete: logits head
            self.action_head = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        # LayerNorm before activation
        x = self.fc1(state)
        x = self.ln1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)
        
        if self.continuous:
            mean    = self.mean_head(x)
            log_std = self.log_std_head(x)
            std     = torch.exp(log_std)
            return mean, std
        else:
            logits = self.action_head(x)
            return logits
    
    def get_action(self, state, deterministic):
        # if self.continuous:
        #     mean, std = self(state)
        #     if deterministic:
        #         return mean
        #     else:
        #         dist = Normal(mean, std)
        #         action = dist.sample()
        #         return action, dist.log_prob(action)
        if self.continuous:
            mean, std = self(state)
            if deterministic:
                return torch.tanh(mean) * 2.0, None
            
            dist       = Normal(mean, std)
            raw_action = dist.rsample()                              # reparameterized sample
            logp       = dist.log_prob(raw_action).sum(-1, keepdim=True)
            
            action     = torch.tanh(raw_action)
            # Jacobian correction term: log |d(tanh)/d(raw_action)| = -log(1 - tanh^2)
            logp      -= torch.log(1 - action.pow(2) + 1e-6).sum(-1, keepdim=True)
            
            scaled     = action * 2.0
            return scaled, logp
        
        else:
            logits = self(state)
            if deterministic:
                a = torch.argmax(logits, dim=1)
                return a, None
            dist   = Categorical(logits=logits)
            action = dist.sample().unsqueeze(-1)
            logp   = dist.log_prob(action.squeeze(-1)).unsqueeze(-1)
            return action, logp


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
        critic_updates_per_actor_update,
        num_actions
        
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
            continuous=continuous,
            num_actions=num_actions
        )
        
        #self.critic = MLPCritic(state_dim, action_dim, hidden_dim, continuous)
        
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
        self._old_synced = False   
        self.critic_batch = None
        self.critic_returns = None
        self.advantages = None
        
        self.old_actor = copy.deepcopy(self.actor)

        
        
    
    def compute_gae(self, rewards, values, next_values, dones):
        """
        Compute Generalized Advantage Estimation.
        """
        
        if rewards.dim() > 2:
            rewards = rewards.squeeze(-1)
            dones = dones.squeeze(-1)
        deltas = rewards + self.gamma * next_values * (1 - dones) - values
        advantages = torch.zeros_like(rewards)
        
        
        gae = 0
        for t in reversed(range(len(rewards))):
            
            gae = deltas[t] + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            if dones[t]:
                gae = 0
            
        
        # Normalize advantages
        if self.normalize_advantages and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
        return advantages
    
    
    
    def compute_temporal_difference_return(self, rewards, next_values, dones, k):
        
        done_indices = torch.where(dones[:k] == 1)[0]
    
        if len(done_indices) == 0:
            # No terminations in first k steps
            effective_k = k
            terminated = False
        else:
            # Include the done-step reward
            effective_k = done_indices[0] + 1
            terminated = True
        
        # Compute discounted sum of rewards
        gammas = torch.pow(self.gamma, torch.arange(effective_k, device=rewards.device))
        td_return = torch.sum(gammas * rewards[:effective_k])
        # Add bootstrap value if we didn't terminate within k steps
        if not terminated and k < len(next_values):
            td_return += ((self.gamma ** effective_k) * next_values[effective_k - 1]).item()
        
        return td_return

    def td_returns(self, rewards, next_values, dones, k):
        """
        Compute Temporal Difference returns.
        """
        returns = torch.zeros_like(rewards)
        for t in range(len(rewards)):
            returns[t] = self.compute_temporal_difference_return(rewards[t:], next_values[t:], dones[t:], k)
        return returns  
    
    def on_train_start(self):
        # Initialize step counters
        self.critic_step_counter = 0
        self.actor_step_counter = 0
    
        
        
        
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
        self.clip_gradients(critic_optimizer, gradient_clip_val=config.max_grad_norm, gradient_clip_algorithm="norm")
        critic_optimizer.step()
        
        
        # Update actor less frequently
        update_actor = (self.update_counter % self.critic_updates_per_actor_update == 0)
        
        if update_actor:
            if not self._old_synced:
                self.old_actor.load_state_dict(self.actor.state_dict())
                self._old_synced = True
            
            actor_optimizer.zero_grad()
            actor_loss = self._actor_training_step()
            self.manual_backward(actor_loss)
            self.clip_gradients(actor_optimizer, gradient_clip_val=config.max_grad_norm, gradient_clip_algorithm="norm")
            actor_optimizer.step()
            
        
        # Increment update counter
        self.update_counter += 1    
    
    def on_train_epoch_end(self):
        # called once at the end of the epoch — now clear the flag
        self._old_synced = False

    
    def prepare_training_data(self, batch):
        """Prepare training data for both actor and critic."""
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        dones = batch['dones']
        next_states = batch['next_states']
        if states.dim() > 2:
            states = states.squeeze(-1)
            
        if actions.dim() > 2:
            actions = actions.squeeze(-1)
        if rewards.dim() > 2:
            rewards = rewards.squeeze(-1)
        if dones.dim() > 2:
            dones = dones.squeeze(-1)
        if next_states.dim() > 2:
            next_states = next_states.squeeze(-1)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        next_states = next_states.to(self.device)
        
        # Get current values
        _, values = self.get_critic_value(states, actions=actions)
    
        # Get next state values (average over multiple sampled actions)
        next_values = self.get_critic_value(next_states, actions=None, num_samples=5)        

        # Compute advantages and returns
        advantages = self.compute_gae(
            rewards, values, next_values, dones
        )
        returns = self.td_returns(
            rewards, next_values, dones, config.k
        )
        # Store the data for actor and critic training
        self.critic_batch = {'states': states, 'actions': actions}
        self.critic_returns = returns
        self.advantages = advantages
        self.critic.return_mean.data = returns.mean()
        self.critic.return_std.data = returns.std() + 1e-6
    
    
    
    def _actor_training_step(self):
        """Train the actor using PPO-style trust region optimization."""
        flat_states = self.critic_batch['states']
        flat_actions = self.critic_batch['actions']
        advantages = self.advantages
        
            
        # Compute old log probabilities using the old policy network
        with torch.no_grad():
            if self.continuous:
                old_mean, old_std = self.old_actor(flat_states)
                old_dist = Normal(old_mean, old_std)
                
                # Convert bounded actions back to raw actions
                tanh_a     = flat_actions / 2.0
                raw_actions = torch.atanh(tanh_a.clamp(-0.99999, 0.99999))
                
                # Calculate log probability of raw actions
                old_log_probs = old_dist.log_prob(raw_actions).sum(dim=-1, keepdim=True)
                
                # Apply correction for the change of variables (tanh transformation)
                old_log_probs -= torch.log(1 - torch.tanh(raw_actions).pow(2) + 1e-6).sum(dim=-1, keepdim=True)
            else:
                old_logits = self.old_actor(flat_states)
                old_dist = Categorical(logits=old_logits)
                old_log_probs = old_dist.log_prob(flat_actions.squeeze(-1)).unsqueeze(-1)
                
            # Store raw actions for reuse in the next steps
            if self.continuous:
                self.raw_actions = raw_actions
        
        # Get current policy distribution and log probabilities
        if self.continuous:
            action_mean, action_std = self.actor(flat_states)
            dist = Normal(action_mean, action_std)
            
            # Use the previously computed raw actions
            log_probs = dist.log_prob(self.raw_actions).sum(dim=-1, keepdim=True)
            
            # Apply correction for the change of variables (tanh transformation)
            tanh_actions = flat_actions / 2.0  # Scale back to [-1, 1]
            log_probs -= torch.log(1 - tanh_actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
            
            # Calculate entropy for exploration bonus
            entropy = dist.entropy().sum(dim=-1, keepdim=True)
            
            # Calculate KL divergence for monitoring
            with torch.no_grad():
                old_mean, old_std = self.old_actor(flat_states)
                kl_div = torch.mean(
                    torch.sum(
                        0.5 * ((action_mean - old_mean) / old_std)**2 + 
                        0.5 * (action_std**2 / old_std**2) - 
                        0.5 - 
                        torch.log(action_std / old_std),
                        dim=-1
                    )
                )
        else:
            action_logits = self.actor(flat_states)
            dist = Categorical(logits=action_logits)
            log_probs = dist.log_prob(flat_actions.squeeze(-1)).unsqueeze(-1)
            entropy = dist.entropy().unsqueeze(-1)
            
            # Calculate KL divergence
            with torch.no_grad():
                old_logits = self.old_actor(flat_states)
                kl_div = torch.mean(
                    torch.sum(
                        F.softmax(old_logits, dim=-1) * 
                        (F.log_softmax(old_logits, dim=-1) - F.log_softmax(action_logits, dim=-1)),
                        dim=-1
                    )
                )
        
        # Clamp the log probabilities to avoid numerical issues
        old_log_probs = torch.clamp(old_log_probs, min=-10, max=10)
        log_probs = torch.clamp(log_probs, min=-10, max=10)
        # Compute probability ratio between new and old policies
        ratio = torch.exp(log_probs - old_log_probs)
        
        # Clamp the ratio for PPO-style trust region
        clipped_ratio = torch.clamp(ratio, 1.0 - config.epsilon, 1.0 + config.epsilon)
        
        # Compute surrogate objectives
        surr1 = ratio * advantages
        surr2 = clipped_ratio * advantages
        
        # Take the minimum to implement the pessimistic bound
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Calculate entropy loss (encourage exploration)
        entropy_loss = -self.entropy_coef * entropy.mean()
        
        # Combine losses
        total_actor_loss = actor_loss + entropy_loss + kl_div
        
        self.actor_step_counter += 1
        
        # Log metrics
        wandb.log({
            'actor/advantage': advantages.mean().item(),
            'actor/log_probs': log_probs.mean().item(),
            'actor/old_log_probs': old_log_probs.mean().item(),
            'actor/ratio': ratio.mean().item(),
            'actor/policy_loss': actor_loss.item(),
            'actor/entropy_loss': entropy_loss.item(),
            'actor/kl_divergence': kl_div.item(),
            'actor/step': int(self.actor_step_counter),
            'actor/clipped_ratio': clipped_ratio.mean().item(),
        })
        
        # Increment actor step counter
        
            
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
        
        samples = self.critic(critic_states, critic_actions, return_value=None, num_samples=10, rev=True)['mean']
        mse_loss = F.mse_loss(samples, critic_returns)
        critic_loss += mse_loss
        
        
        # critic_output = self.critic(critic_states, critic_actions)
        # critic_loss = self.value_loss_coef * critic_output.mean()
        
        wandb.log({
            'critic/loss': critic_loss.item(),
            'critic/step': float(self.critic_step_counter),
            
            
            })
      
        # Increment critic step counter
        self.critic_step_counter += 1
        
        return critic_loss
    
    def reset_batch(self):
        """Reset the batch data."""
        self.critic_batch = None
        self.critic_returns = None
        self.advantages = None
    
    
    
    def configure_optimizers(self):
        """
        Configure separate optimizers for actor and critic with custom frequency.
        """
        actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr, betas = config.betas, eps = config.eps)
        
        critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr, betas=config.betas, eps=config.eps)

        
        # Return optimizers with a frequency dictionaries to control how often they're used
        # The actor optimizer will be called once every `critic_updates_per_step + 1` calls
        # The critic optimizer will be called `critic_updates_per_step` times for every actor update
        return [
            {"optimizer": actor_optimizer},
            {"optimizer": critic_optimizer}
        ]
        
        
    def get_critic_value(self, states, actions=None, num_samples=20):
        """
        Get value estimates from the state-action conditioned critic.
        
        Args:
            states: Tensor of states [batch_size, state_dim]
            actions: Optional tensor of actions [batch_size, action_dim]. 
                    If None, will sample actions from the current policy.
            num_samples: Number of action samples to use for averaging when actions aren't provided
                        
        Returns:
            If actions is provided: Tuple of (q_values, v_values)
            If actions is None: v_values only
        """
        batch_size = states.shape[0]
        
        # Case 1: Actions are provided - return both Q(s,a) and V(s)
        if actions is not None:
            with torch.no_grad():
                # Calculate Q(s,a) for the provided actions
                output = self.critic(states, actions, rev=True, return_value=None, num_samples=config.num_samples)
                q_values = output['mean']
                
                # Calculate V(s) by sampling actions and averaging
                v_values = []
                for _ in range(num_samples):
                    # Sample actions from the current policy
                    sampled_actions, _ = self.actor.get_action(states, deterministic=False)
                    output = self.critic(states, sampled_actions, rev=True, return_value=None, num_samples=config.num_samples)
                    v_values.append(output['mean'])
                
                # Average over all sampled actions to get V(s)
                v_values = torch.stack(v_values, dim=0).mean(dim=0)
                
                return q_values, v_values
        
        # Case 2: Actions not provided - compute V(s) only
        else:
            all_values = []
            with torch.no_grad():
                for _ in range(num_samples):
                    # Sample actions from the current policy
                    sampled_actions, _ = self.actor.get_action(states, deterministic=False)
                    output = self.critic(states, sampled_actions, rev=True, return_value=None, num_samples=config.num_samples)
                    all_values.append(output['mean'])
                
                # Average over all sampled actions
                v_values = torch.stack(all_values, dim=0).mean(dim=0)
                
            return v_values
        
    def act(self, state, deterministic=False):
        """
        Select an action given a state.
        """
        with torch.no_grad():
            action, _ = self.actor.get_action(state, deterministic)
            return action




class MLPCritic(nn.Module):
    """
    Simple MLP-based critic that directly estimates state-action values
    """
    def __init__(self, state_dim, action_dim, hidden_dim, continuous):
        super().__init__()
        
        self.continuous = continuous
        
        # Input layers for state and action
        self.state_encoder = nn.Linear(state_dim, hidden_dim // 2)
        
        if continuous:
            self.action_encoder = nn.Linear(action_dim, hidden_dim // 2)
        else:
            # Embedding for discrete actions
            self.action_encoder = nn.Embedding(action_dim, hidden_dim // 2)
        
        # Value prediction layers
        self.value_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state, action):
        """
        Forward pass to compute value for a state-action pair
        
        Args:
            state: State tensor [batch_size, state_dim]
            action: Action tensor [batch_size, action_dim] or [batch_size] for discrete
            
        Returns:
            Estimated value for the state-action pair
        """
        batch_size = state.shape[0]
        
        # Encode state
        state_features = F.relu(self.state_encoder(state))
        
        # Encode action based on type
        if self.continuous:
            action = action.view(batch_size, -1)
            action_features = F.relu(self.action_encoder(action))
        else:
            action = action.view(batch_size).long()
            action_features = F.relu(self.action_encoder(action))
        
        # Concatenate state and action features
        combined = torch.cat([state_features, action_features], dim=1)
        
        # Compute value estimate
        value = self.value_network(combined)
        
        return value