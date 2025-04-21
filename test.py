import torch
import pytest
import pdb

class DummyAgent:
    def __init__(self, gamma, gae_lambda, normalize_advantages):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages

    def compute_gae(self, rewards, values, next_values, dones):
        deltas = rewards + self.gamma * next_values * (1 - dones) - values
        advantages = torch.zeros_like(rewards)
        
        gae = 0.0
        for t in reversed(range(len(rewards))):
            gae = deltas[t] + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            if dones[t]:
                gae = 0.0

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


@pytest.mark.parametrize("rewards, values, next_values, dones, expected", [
    # Case 1: no dones, gamma=1, lambda=1 -> reversed cumulative sum of rewards
    (
        torch.tensor([1.0, 2.0, 3.0]),
        torch.zeros(3),
        torch.zeros(3),
        torch.zeros(3),
        torch.tensor([6.0, 5.0, 3.0])
    ),
   
])


def test_compute_gae_no_norm(rewards, values, next_values, dones, expected):
    agent = DummyAgent(gamma=1.0, gae_lambda=1.0, normalize_advantages=False)
    adv = agent.compute_gae(rewards, values, next_values, dones)
    assert torch.allclose(adv, expected, atol=1e-6)

def test_compute_gae_with_normalization():
    # simple case: advantages [2, 4, 6] -> mean=4, std=~1.63299 -> normalized to [-1.2247, 0, 1.2247]
    rewards = torch.tensor([1.0, 2.0, 3.0])
    values = torch.zeros(3)
    next_values = torch.zeros(3)
    dones = torch.zeros(3)

    agent = DummyAgent(gamma=1.0, gae_lambda=1.0, normalize_advantages=True)
    adv = agent.compute_gae(rewards, values, next_values, dones)
    
    # Check zero mean and unit std
    assert pytest.approx(adv.mean().item(), abs=1e-6) == 0.0
    assert pytest.approx(adv.std().item(), rel=1e-3) == 1.0

def test_compute_td_return_no_done_with_bootstrap():
    """
    No dones in first k steps → effective_k = k, terminated=False,
    so TD‑return = sum_{i=0..k-1} gamma^i * rewards[i]
                   + gamma^k * next_values[k-1]
    """
    agent = DummyAgent(gamma=0.5, gae_lambda=1.0, normalize_advantages=False)
    rewards     = torch.tensor([1.0, 2.0, 3.0])
    next_values = torch.tensor([10.0, 20.0, 30.0])
    dones       = torch.zeros(3, dtype=torch.int32)
    k = 2

    # 1*1 + 0.5*2 = 2  plus bootstrap: 0.5**2 * next_values[1] = 0.25*20 = 5 → total = 7
    ret = agent.compute_temporal_difference_return(rewards, next_values, dones, k)
    assert pytest.approx(ret, abs=1e-6) == 7

def test_compute_td_return_done_before_k():
    """
    A done at index=1 (within k=3) → effective_k = 1+1 = 2, terminated=True,
    so TD‑return = sum_{i=0..1} gamma^i * rewards[i]  (no bootstrap)
    """
    agent = DummyAgent(gamma=1.0, gae_lambda=1.0, normalize_advantages=False)
    rewards     = torch.tensor([1.0, 2.0, 3.0])
    next_values = torch.tensor([ 5.0,  6.0,  7.0])
    dones       = torch.tensor([0, 1, 0], dtype=torch.int32)
    k = 3

    # gamma=1 => 1*1 + 1*2 = 3
    ret = agent.compute_temporal_difference_return(rewards, next_values, dones, k)
    assert pytest.approx(ret, abs=1e-6) == 3.0

