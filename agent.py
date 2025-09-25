from model import ActorNetwork,  ConditionalNormalizingFlow, TwinFlowCritic, ConditionalNSF
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.distributions import Categorical, Normal
import copy
import config
import wandb
import numpy as np


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
        critic_updates_per_actor_update,
        num_actions,
        critic_type='nspline_flow' # or 'nspline_flow' or 'stochastic_nf' or 'conditional_flow' 
        
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Actor network
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim, continuous)
        
        if critic_type == 'conditional_flow':
        # Critic network (ConditionalNormalizingFlow)
            self.critic = ConditionalNormalizingFlow(
                state_dim=state_dim,
                action_dim=action_dim,  # For discrete actions, action is just an index
                hidden_dim=hidden_dim,
                flow_dim=flow_dim,
                num_layers=num_flow_layers,
                learning_rate=critic_lr,
                normalize_returns=normalize_advantages,
                continuous=continuous,
                num_actions=num_actions
        )
        
        
        elif critic_type == 'nspline_flow':
            self.critic = ConditionalNSF(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                flow_dim=flow_dim,
                num_layers=num_flow_layers,
                learning_rate=critic_lr,
                normalize_returns=normalize_advantages,
                continuous=continuous,
                num_actions=num_actions,
                nsf_kind="coupling",          # or "autoregressive"
                nsf_hidden_units=128,
                nsf_hidden_layers=2,
                nsf_num_bins=8,
                mh_every_k=0                  # set >0 to enable MH blocks
            )

        elif critic_type == 'stochastic_nf':
            self.critic = ConditionalNSF(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                flow_dim=flow_dim,
                num_layers=num_flow_layers,
                learning_rate=critic_lr,
                normalize_returns=normalize_advantages,
                continuous=continuous,
                num_actions=num_actions,
                nsf_kind="coupling",
                mh_every_k=2                  # e.g., insert MH after every 2 spline blocks
            )

        else:
            raise ValueError(f"Unknown critic type: {critic_type}")
        
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
        
    
    
    def calculate_monte_carlo_returns(self, rewards: torch.Tensor, dones: torch.Tensor, gamma: float = None) -> torch.Tensor:
        """
        Compute Monte Carlo returns for a batch of trajectories.
        
        Args:
            rewards: Tensor of shape [batch_size, trajectory_len] — reward sequences.
            dones: Tensor of shape [batch_size, trajectory_len] — done flags (1 if terminal).
            gamma: Discount factor. If None, uses self.gamma.
        
        Returns:
            returns: Tensor of shape [batch_size, trajectory_len] — MC returns from each time step.
        """
        if gamma is None:
            gamma = self.gamma

        batch_size, T = rewards.shape
        returns = torch.zeros_like(rewards)

        for b in range(batch_size):
            G = 0
            for t in reversed(range(T)):
                if dones[b, t]:
                    G = 0  # reset on terminal state
                G = rewards[b, t] + gamma * G
                returns[b, t] = G

        return returns      
            
        
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
        q_vals, v_vals = self.get_critic_value(states=states, actions=actions)
    
        # Get next state values (average over multiple sampled actions)
        next_values = self.get_critic_value(states=next_states, actions=None, num_samples=config.num_samples)   
        
        # Compute advantages and returns
        advantages = self.compute_gae(
            rewards, v_vals, next_values, dones
        )
        # returns = self.td_returns(
        #     rewards, next_values, dones, config.k
        # )
        returns = self.calculate_monte_carlo_returns(rewards=rewards,dones=dones, gamma=self.gamma)
        
        
        # Store the data for actor and critic training
        self.critic_batch = {'states': states, 'actions': actions}
        self.advantages = advantages
        # self.critic_returns = (returns - self.critic.return_mean) / self.critic.return_std
        # Update running mean and std using a running average
        self.critic.return_mean.data = (1 - config.alpha) * self.critic.return_mean.data + config.alpha * returns.mean()
        self.critic.return_std.data = (1 - config.alpha) * self.critic.return_std.data + config.alpha * returns.std()
        
        self.critic_returns = (returns - self.critic.return_mean) / (self.critic.return_std.data + 1e-6)
    
    
    
    
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
        critic_loss = critic_output['nll'].mean()
        
        samples = self.critic(critic_states, critic_actions, return_value=None, num_samples=config.num_samples, rev=True)['mean']
        
        huber_loss = F.smooth_l1_loss(samples, critic_returns)
        critic_loss += huber_loss
        # critic_output = self.critic(critic_states, critic_actions)
        # critic_loss = self.value_loss_coef * critic_output.mean()
        
        # wandb.log({
        #     'critic/loss': critic_loss.item(),
        #     'critic/step': float(self.critic_step_counter),
        #     'critic/huber': huber_loss.item(),
        #     'critic/nll_loss': critic_output['nll'].mean().item()
        # })
      
        # Increment critic step counter
        self.critic_step_counter += 1
        
        return critic_loss * self.value_loss_coef
    
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





class SACWithFlow(pl.LightningModule):
    """
    Soft Actor-Critic with twin flow-based critics.
    Expects batches: (states, actions, rewards, next_states, dones)
    """
    def __init__(self,
                 state_dim, action_dim, hidden_dim,
                 flow_dim, num_flow_layers,
                 actor_lr, critic_lr, alpha_lr,
                 gamma, tau,
                 continuous, num_actions,
                 target_entropy=None,
                 normalize_returns=True):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        # Actor (tanh-squashed Gaussian already implemented in your code)
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim, continuous)

        # Critics (twin)
        critic_kwargs = dict(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            flow_dim=flow_dim,
            num_layers=num_flow_layers,
            lr=critic_lr,
            normalize_returns=normalize_returns,
            continuous=continuous,
            num_actions=num_actions
        )
        self.critics = TwinFlowCritic(**critic_kwargs)
        self.target_critics = TwinFlowCritic(**critic_kwargs)
        self._hard_update_targets()

        # Entropy temperature (learnable)
        if target_entropy is None:
            # Standard heuristic: -|A|
            target_entropy = -float(action_dim)
        self.log_alpha = torch.nn.Parameter(torch.zeros(1))
        self.target_entropy = target_entropy

        self.gamma = gamma
        self.tau = tau
        self.continuous = continuous

    def _hard_update_targets(self):
        self.target_critics.load_state_dict(self.critics.state_dict())

    @torch.no_grad()
    def _soft_update_targets(self):
        for targ, src in zip(self.target_critics.parameters(), self.critics.parameters()):
            if targ.dtype!=torch.int64 and src.dtype!=torch.int64:
                targ.data.mul_(1 - self.tau).add_(self.tau * src.data)
                
    def configure_optimizers(self):
        actor_opt  = torch.optim.Adam(self.actor.parameters(),   lr=self.hparams.actor_lr,  betas=config.betas, eps=config.eps)
        critic_opt = torch.optim.Adam(self.critics.parameters(), lr=self.hparams.critic_lr, betas=config.betas, eps=config.eps)
        alpha_opt  = torch.optim.Adam([self.log_alpha],          lr=self.hparams.alpha_lr,  betas=config.betas, eps=config.eps)
        return [actor_opt, critic_opt, alpha_opt]

    def _sample_action_and_logp(self, states):
        # Reuse your ActorNetwork forward & get_action logic
        mean, std = self.actor(states)
        dist = Normal(mean, std)
        raw = dist.rsample()
        logp = dist.log_prob(raw).sum(-1, keepdim=True)
        a = torch.tanh(raw)
        logp -= torch.log(1 - a.pow(2) + 1e-6).sum(-1, keepdim=True)
        a = a * 2.0  # same scaling used in your actor
        return a, logp
    
    @torch.no_grad()
    def act(self, state, deterministic: bool = False):
        """
        SAC action selector to match ActorCriticWithFlow's interface.
        - Accepts a single state tensor [obs_dim] or a batch [B, obs_dim].
        - If deterministic: uses tanh(mean)
          Else: samples with reparameterization, then tanh-squashes.
        - Returns a torch.Tensor on the same device.
        """
        # Accept numpy -> tensor quietly
        if isinstance(state, np.ndarray):
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device if hasattr(self, "device") else None)
        elif not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32, device=self.device if hasattr(self, "device") else None)

        # Ensure 2D [B, obs_dim]
        single = False
        if state.dim() == 1:
            state = state.unsqueeze(0)
            single = True

        # Actor outputs mean & std for a Gaussian before tanh
        
        mean, std = self.actor(state)  # expect [B, act_dim] each
        
            

        if deterministic:
            raw = mean
        else:
            dist = torch.distributions.Normal(mean, std)
            raw = dist.rsample()  # reparameterized sample

        # Tanh squashing to keep actions in [-1, 1]
        action = torch.tanh(raw)

        # Optional scale/bias if you use larger action bounds (e.g., [-2, 2])
        action_scale = getattr(self, "action_scale", 1.0)
        action_bias  = getattr(self, "action_bias", 0.0)
        action = action * action_scale + action_bias

        # Return [act_dim] for single state to match your discrete actor's API
        if single:
            action = action.squeeze(0)

        return action


    def training_step(self, batch, batch_idx):
        states, actions, rewards, next_states, dones = batch
        
        if states.dim() > 2:      states = states.squeeze(-1)
        if actions.dim() > 2:     actions = actions.squeeze(-1)
        if rewards.dim() > 2:     rewards = rewards.squeeze(-1)
        if dones.dim() > 2:       dones = dones.squeeze(-1)
        if next_states.dim() > 2: next_states = next_states.squeeze(-1)

        
        # -------- Critic update --------
        # Target: y = r + γ * ( min_i Q_i_target(s', a') - α * log π(a'|s') ) * (1 - done)
        with torch.no_grad():
            next_actions, next_logp = self._sample_action_and_logp(next_states)
            q1_targ, q2_targ = self.target_critics.q_values(next_states, next_actions, num_samples=config.num_samples)
            q_targ_min = torch.min(q1_targ, q2_targ)
            alpha = self.log_alpha.exp()
            y = rewards + self.gamma * (1 - dones) * (q_targ_min - alpha * next_logp)

        
        # Likelihood + Huber toward y for both critics
        
        nll1 = self.critics.q1.nll_to_targets(states, actions, y).mean()
        nll2 = self.critics.q2.nll_to_targets(states, actions, y).mean()
        hub1 = self.critics.q1.huber_to_targets(states, actions, y, num_samples=config.num_samples)
        hub2 = self.critics.q2.huber_to_targets(states, actions, y, num_samples=config.num_samples)
        critic_loss = (nll1 + nll2) + (hub1 + hub2)

        # -------- Actor update --------
        new_actions, logp = self._sample_action_and_logp(states)
        q1_pi, q2_pi = self.critics.q_values(states, new_actions, num_samples=config.num_samples)
        q_pi = torch.min(q1_pi, q2_pi)
        alpha = self.log_alpha.exp()
        actor_loss = (alpha * logp - q_pi).mean()

        # -------- Temperature (alpha) update --------
        alpha_loss = -(self.log_alpha * (self.target_entropy + logp.detach()).mean())

        # Optimize (Lightning: return combined loss; or step manually if preferred)
        actor_opt, critic_opt, alpha_opt = self.optimizers()
        critic_opt.zero_grad(set_to_none=True)
        self.manual_backward(critic_loss)
        self.clip_gradients(critic_opt, gradient_clip_val=config.max_grad_norm, gradient_clip_algorithm="norm")
        critic_opt.step()

        actor_opt.zero_grad(set_to_none=True)
        self.manual_backward(actor_loss)
        self.clip_gradients(actor_opt, gradient_clip_val=config.max_grad_norm, gradient_clip_algorithm="norm")
        actor_opt.step()

        alpha_opt.zero_grad(set_to_none=True)
        self.manual_backward(alpha_loss)
        alpha_opt.step()

        # Target network
        self._soft_update_targets()

        self.log_dict({
            "sac/critic_loss": critic_loss.detach(),
            "sac/actor_loss": actor_loss.detach(),
            "sac/alpha": alpha.detach(),
            "sac/alpha_loss": alpha_loss.detach(),
        }, prog_bar=True)

        # Lightning requires a tensor to be returned
        return critic_loss + actor_loss + alpha_loss