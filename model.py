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
import normflows as nf


import pdb

wasserstein_loss = SamplesLoss()
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
        nn.GELU(),
        nn.Linear(config.hidden_dim_subnet, config.hidden_dim_subnet),
        nn.GELU(),
        nn.Linear(config.hidden_dim_subnet, c_out)
    )


def build_normalizing_flow(input_dim: int,
                            condition_dim: int,
                            num_layers: int,
                            use_conditioning: bool = True
                           ) -> ReversibleGraphNet:
    nodes = [InputNode(input_dim, name='input')]

    if use_conditioning:
        cond_node = ConditionNode(condition_dim, name='condition')
        cond_nodes = [cond_node]
    else:
        cond_nodes = []

    for k in range(num_layers):
        kwargs = {
            'subnet_constructor': subnet_fc,
            'clamp': 2.0
        }

        # Attach condition only if enabled
        conditions = [cond_nodes[0]] if use_conditioning else None

        nodes.append(Node(
            nodes[-1],
            GLOWCouplingBlock,
            kwargs,
            conditions=conditions,
            name=f'coupling_{k}'
        ))

        nodes.append(Node(
            nodes[-1],
            PermuteRandom,
            {'seed': k},
            name=f'permute_{k}'
        ))

    nodes.append(OutputNode(nodes[-1], name='output'))

    model = ReversibleGraphNet(
        nodes + cond_nodes,
        verbose=False
    )
    

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.zeros_(p)

    return model





# def build_normalizing_flow_(input_dim: int,
#                             condition_dim: int,
#                             num_layers: int,
#                             clamp=1.0
#                            ) -> ReversibleGraphNet:
#     """
#     Build a conditional normalizing flow using AllInOneBlock.

#     Args:
#         input_dim: Dimension of the flow input (e.g., return embedding).
#         condition_dim: Dimension of the conditioning vector (e.g., state-action encoding).
#         num_layers: Number of AllInOne blocks.
#         subnet_fc: Function defining the subnet MLP.
#         clamp: Clamping value for affine terms in AllInOneBlock.

#     Returns:
#         A conditional ReversibleGraphNet.
#     """
#     nodes = [InputNode(input_dim, name='input')]
#     cond_node = ConditionNode(condition_dim, name='condition')

#     for k in range(num_layers):
#         # AllInOneBlock (conditionally affine + built-in permutation)
#         nodes.append(Node(
#             nodes[-1],
#             AllInOneBlock,
#             {'subnet_constructor': subnet_fc, 'affine_clamping': clamp},
#             conditions=cond_node,
#             name=f'block_{k}'
#         ))

#     nodes.append(OutputNode(nodes[-1], name='output'))

#     flow = ReversibleGraphNet(nodes + [cond_node], verbose=False)

#     # Initialize weights
#     for p in flow.parameters():
#         if p.dim() > 1:
#             nn.init.xavier_uniform_(p)
#         else:
#             nn.init.zeros_(p)

#     return flow




class SimpleEncoder(nn.Module):
    """Simple state-action encoder with LayerNorm"""
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        out_dim: int,
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


class ConditionalNormalizingFlow(nn.Module):
    def __init__(self, state_dim:int, action_dim:int, hidden_dim:int, flow_dim:int, num_layers:int, 
                 learning_rate:float, normalize_returns:bool, continuous:bool, num_actions:int):
        super().__init__()
        
        self.state_action_encoder = SimpleEncoder(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            out_dim=flow_dim,
            continuous=continuous,
            num_actions=num_actions
        )
        
        
        
        

        
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
        
        
        encoding = self.state_action_encoder(state, action)
        
        if not rev:
            # Forward direction: Return → Latent
            
            #wandb.log({"return_value:" : return_value[0].item()})
            
            if return_value is None:
                raise ValueError("Return value must be provided for pass in the forward direction")
            # if self.normalize_returns:
            #     return_value = (return_value - self.return_mean) / self.return_std
            zeros = torch.randn(batch_size, self.flow_dim - 1, device=state.device) * config.zero_scale # 1 is for return dim
            return_padded = torch.cat([return_value, zeros], dim=-1)
            # Forward through the flow with explicit conditioning
            
            
            z, log_det_J = self.flow(return_padded, c=encoding, rev=False, jac=True)
            
            
            
            
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
            samples_emb, _ = self.flow(z, c=encoding, rev=True, jac=False)
            
            samples = samples_emb[:, 0:1]  # First dimension corresponds to return
            if self.normalize_returns:
                samples = samples * self.return_std + self.return_mean
            #Reshape to [batch_size, num_samples]
            samples = samples.view(batch_size, num_samples)
            
            return {
                'samples': samples,
                'mean': samples.mean(dim=1, keepdim=True),
                'std': samples.std(dim=1, keepdim=True)
            }
    
    
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
                return torch.tanh(mean), None
            
            dist       = Normal(mean, std)
            raw_action = dist.rsample()                              # reparameterized sample
            logp       = dist.log_prob(raw_action).sum(-1, keepdim=True)
            
            action     = torch.tanh(raw_action)
            # Jacobian correction term: log |d(tanh)/d(raw_action)| = -log(1 - tanh^2)
            logp      -= torch.log(1 - action.pow(2) + 1e-6).sum(-1, keepdim=True)
            
            scaled     = action
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
    
    
    
class FlowQCritic(torch.nn.Module):
    """
    Wraps ConditionalNormalizingFlow to behave like a scalar Q(s,a) estimator.
    We train it to match scalar targets y (SAC TD targets) and use its
    reverse pass mean as the Q estimate.
    """
    def __init__(self, state_dim, action_dim, hidden_dim, flow_dim, num_layers,
                 lr, normalize_returns, continuous, num_actions):
        super().__init__()
        self.flow = ConditionalNormalizingFlow(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            flow_dim=flow_dim,
            num_layers=num_layers,
            learning_rate=lr,
            normalize_returns=normalize_returns,
            continuous=continuous,
            num_actions=num_actions
        )

    @torch.no_grad()
    def q_value(self, states, actions, num_samples=20):
        # Use the mean of the learned return/value distribution as Q estimate
        out = self.flow(states, actions, return_value=None, num_samples=num_samples, rev=True)
        return out['mean']  # [B,1]

    def nll_to_targets(self, states, actions, targets):
        """
        Likelihood term that pulls the flow toward scalar targets.
        """
        out = self.flow(states, actions, return_value=targets, rev=False)
        return out['nll']  # [B]

    def huber_to_targets(self, states, actions, targets, num_samples=20):
        with torch.no_grad():
            mean_pred = self.q_value(states, actions, num_samples=num_samples)
        return F.smooth_l1_loss(mean_pred, targets)

class TwinFlowCritic(torch.nn.Module):
    """
    Standard SAC twin Q critics, but each Q is a flow-based critic.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.q1 = FlowQCritic(**kwargs)
        self.q2 = FlowQCritic(**kwargs)

    @torch.no_grad()
    def q_values(self, states, actions, num_samples=20):
        return self.q1.q_value(states, actions, num_samples), self.q2.q_value(states, actions, num_samples)


# --- NEW: Conditional NSFs with optional Metropolis–Hastings blocks ---
import importlib

class ConditionalNSF(nn.Module):
    """
    Conditional value-distribution critic built with normflows:
      - Neural Spline Flows (Rational Quadratic)
      - Optional stochastic Metropolis–Hastings blocks
    Matches the API of ConditionalNormalizingFlow for easy swapping.
    """
    def __init__(self,
                 state_dim:int, action_dim:int, hidden_dim:int,
                 flow_dim:int,               # dimension of the 'flow vector' (first coord is return)
                 num_layers:int,
                 learning_rate:float,
                 normalize_returns:bool,
                 continuous:bool,
                 num_actions:int,
                 *,
                 nsf_kind:str = "autoregressive",  # "coupling" or "autoregressive"
                 nsf_hidden_units:int = 64,
                 nsf_hidden_layers:int = 2,
                 nsf_num_bins:int = 6,
                 mh_every_k:int = 0,  
                 tail_bound = 6,# 0 => no MH; otherwise insert MH after every k flow blocks
                 base_trainable:bool = False):
        super().__init__()

        # Reuse your (s,a)->context encoder
        self.state_action_encoder = SimpleEncoder(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            continuous=continuous,
            num_actions=num_actions
        )

        
        self.flow_dim       = flow_dim
        self.hidden_dim     = hidden_dim
        self.learning_rate  = learning_rate
        self.state_dim      = state_dim
        self.action_dim     = action_dim
        self.normalize_returns = normalize_returns
        self.return_mean    = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.return_std     = nn.Parameter(torch.ones(1),  requires_grad=False)

        # --- Build conditional NSF chain ---
        flows = []
        for i in range(num_layers):
            if nsf_kind.lower().startswith("auto"):
                # Autoregressive NSF (conditional)
                flows += [nf.flows.AutoregressiveRationalQuadraticSpline(
                    in_channels=flow_dim,
                    hidden_layers=nsf_hidden_layers,
                    hidden_units=nsf_hidden_units,
                    num_bins=nsf_num_bins,
                    num_context_channels=hidden_dim,
                    tail_bound=tail_bound
                    
                )]
            else:
                # Coupling NSF (conditional)
                flows += [nf.flows.CoupledRationalQuadraticSpline(
                    num_input_channels=flow_dim,
                    num_blocks=nsf_hidden_layers,
                    num_hidden_channels=nsf_hidden_units,
                    num_bins=nsf_num_bins,
                    num_context_channels=hidden_dim,
                    tail_bound=tail_bound   
                )]

            # learned linear permutation (like 1x1 conv / LU)
            flows += [nf.flows.LULinearPermute(flow_dim)]

            # Optional: insert a stochastic MH block every k layers
            if mh_every_k > 0 and (i + 1) % mh_every_k == 0:
                # Some installations expose this as nf.flows.MetropolisHastings
                # If unavailable, this will raise with a clear message.
                if hasattr(nf.flows, "MetropolisHastings"):
                    flows += [nf.flows.MetropolisHastings()]
                else:
                    raise AttributeError(
                        "normflows: MetropolisHastings block not found. "
                        "Update normflows to a version with stochastic flow blocks."
                    )

        # Base distribution (diagonal Gaussian over 'flow_dim')
        q0 = nf.distributions.DiagGaussian(flow_dim, trainable=base_trainable)

        # We don’t know the environment’s true target density, so we use a conditional flow model
        # without a target distribution and train via forward KLD / NLL on observed returns.
        self.nf = nf.ConditionalNormalizingFlow(q0=q0, flows=flows)

    def _pad_return(self, return_value: torch.Tensor, device):
        """Pad the scalar return with noise for the remaining (flow_dim-1) coords."""
        b = return_value.shape[0]
        zeros = torch.randn(b, self.flow_dim - 1, device=device) * config.zero_scale
        return torch.cat([return_value, zeros], dim=-1)

    def forward(self, state, action, return_value=None, num_samples=None, rev:bool=False):
        """
        If rev=False: compute NLL(z) of the observed return under q(r | s,a).
        If rev=True : sample num_samples returns for each (s,a).
        """
        device = state.device
        context = self.state_action_encoder(state, action)  # [B, hidden_dim]

        if not rev:
            if return_value is None:
                raise ValueError("Provide return_value when rev=False")
            # (Optional) normalize observed returns
            # if self.normalize_returns:
            #     return_value = (return_value - self.return_mean) / self.return_std

            x = self._pad_return(return_value, device)
            # normflows uses 'log_prob(x, context)'
            logp = self.nf.log_prob(x, context)
            nll = -logp.mean()
            # For parity with your FrEIA version, return a dict with 'nll'
            return {'z': None, 'log_det_J': None, 'nll': nll}
        else:
            if num_samples is None or num_samples < 1:
                raise ValueError("num_samples must be >= 1 when rev=True")

            B = state.shape[0]
            # Repeat context B*num_samples and sample from conditional flow, then keep the first coord (return)
            context_rep = context.repeat_interleave(num_samples, dim=0)  # [B*num_samples, hidden_dim]
            ###x_samples = self.nf.sample(B * num_samples, context=context_rep)  # [B*num_samples, flow_dim]
            
            x_samples, _ = self.nf.sample(B * num_samples, context=context_rep)  # [B*num_samples, flow_dim]
            
            returns = x_samples[:, :1].view(B, num_samples)

            if self.normalize_returns:
                returns = returns * self.return_std + self.return_mean

            return {
                'samples': returns,
                'mean': returns.mean(dim=1, keepdim=True),
                'std':  returns.std(dim=1, keepdim=True)
            }

    @torch.no_grad()
    def sample_value_distribution(self, state, action, num_samples:int):
        return self(state, action, num_samples=num_samples, rev=True)
