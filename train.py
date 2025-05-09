import gymnasium as gym
import torch
import numpy as np
from collections import deque
import pytorch_lightning as pl
from lightning.pytorch.loggers import WandbLogger
from tqdm import tqdm
import random
from torch.utils.data import TensorDataset, DataLoader
from model import ActorCriticWithFlow
import os
import wandb
import config
from stable_baselines3.common.buffers import ReplayBuffer

def save_model_on_best_reward(model, recent_rewards, best_avg_reward, checkpoint_dir="./checkpoints/"):
    if not recent_rewards:
        return best_avg_reward
    current_avg = np.mean(recent_rewards)
    if current_avg > best_avg_reward:
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(checkpoint_dir, "best_model_reward.ckpt")
        torch.save(model.state_dict(), path)
        print(f"New best avg reward {current_avg:.2f}, saved to {path}")
        return current_avg
    return best_avg_reward


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
    critic_batch_size,
    total_timesteps,
    update_frequency,
    num_epochs,
    batch_size,
    log_frequency,
    max_episode_steps,
    eval_frequency,
    num_eval_episodes,
    checkpoint_frequency,
    normalize_advantages,
    critic_updates_per_actor_update,
    seed,
    mode='on_policy',  # 'on_policy' or 'off_policy'
    replay_buffer_size=100_000,
):
    # seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = gym.make(env_name)
    if max_episode_steps:
        env._max_episode_steps = max_episode_steps
    num_actions = env.action_space.n if not continuous else env.action_space.shape[0]

    # buffer for off-policy
    buffer = None
    if mode == 'off_policy':
        buffer = ReplayBuffer(
            buffer_size=replay_buffer_size,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=config.device,
        )

    model = ActorCriticWithFlow(
        state_dim, action_dim, hidden_dim,
        flow_dim, num_flow_layers,
        actor_lr, critic_lr,
        gamma, continuous,
        gae_lambda, entropy_coef,
        value_loss_coef, max_grad_norm,
        normalize_advantages, num_heads,
        critic_updates_per_actor_update,
        num_actions
    )

    # wandb
    wandb.init(project='dist_rl_online')
    wandb.config.update(locals())
    best_avg = float('-inf')
    recent_rewards = deque(maxlen=100)

    # on-policy storage
    if mode == 'on_policy':
        states, actions, rewards, dones, next_states = [], [], [], [], []

    state, _ = env.reset(seed=seed)
    episode_reward = 0
    episodes_completed = 0
    timestep = 0
    pbar = tqdm(total=total_timesteps, desc="Training")

    while timestep < total_timesteps:
        model = model.to(config.device)
        st_tensor = torch.FloatTensor(state).to(config.device)
        with torch.no_grad():
            at_tensor = model.act(st_tensor, deterministic=False)
        action = at_tensor.cpu().numpy()
        next_state, reward, term, trunc, _ = env.step(action.item())
        # pos, vel = next_state
        # reward += 0.1 * abs(vel) + 0.5 * pos
        done = term or trunc

        if mode == 'off_policy':
            buffer.add(obs=state, next_obs=next_state, action=action, reward=reward, done=done, infos=[{}])
        else:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            next_states.append(next_state)

        state = next_state
        episode_reward += reward
        timestep += 1

        if done:
            episodes_completed += 1
            recent_rewards.append(episode_reward)
            wandb.log({'env/episode_reward': episode_reward})
            avg100 = np.mean(recent_rewards)
            if episodes_completed % log_frequency == 0:
                tqdm.write(f"Ep {episodes_completed}, avg100 {avg100:.2f}")
                wandb.log({'env/avg_reward_100': avg100, 'env/episodes': episodes_completed, 'env/timestep': timestep})
            best_avg = save_model_on_best_reward(model, recent_rewards, best_avg)
            state, _ = env.reset()
            episode_reward = 0

        # Off-policy update
        if mode == 'off_policy' and buffer.size() >= batch_size:
            batch = buffer.sample(batch_size=batch_size)
            full = {
                'states': batch.observations,
                'actions': batch.actions,
                'rewards': batch.rewards.unsqueeze(1),
                'dones': batch.dones.unsqueeze(1).float(),
                'next_states': batch.next_observations
            }
            with torch.no_grad(): model.prepare_training_data(full)
            fs = model.critic_batch['states']
            fa = model.critic_batch['actions']
            adv = model.advantages
            ret = model.critic_returns
            ds = TensorDataset(fs, fa, adv, ret)
            loader = DataLoader(ds, batch_size=critic_batch_size, shuffle=True)
            trainer = pl.Trainer(max_epochs=num_epochs, accelerator='auto', logger=False)
            trainer.fit(model, train_dataloaders=loader)

        # On-policy update
        if mode == 'on_policy' and len(states) >= update_frequency:
            st_t = torch.FloatTensor(np.array(states))
            if continuous:
                at_t = torch.FloatTensor(np.array(actions))
            else:
                at_t = torch.LongTensor(np.array(actions))
            rw_t = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
            dn_t = torch.FloatTensor(np.array(dones)).unsqueeze(1)
            nxt_t = torch.FloatTensor(np.array(next_states))
            full = {'states': st_t, 'actions': at_t, 'rewards': rw_t, 'dones': dn_t, 'next_states': nxt_t}
            with torch.no_grad(): model.prepare_training_data(full)
            fs = model.critic_batch['states']
            fa = model.critic_batch['actions']
            adv = model.advantages
            ret = model.critic_returns
            ds = TensorDataset(fs, fa, adv, ret)
            loader = DataLoader(ds, batch_size=critic_batch_size, shuffle=True)
            trainer = pl.Trainer(max_epochs=num_epochs, accelerator='auto', logger=False)
            trainer.fit(model, train_dataloaders=loader)
            # clear
            states, actions, rewards, dones, next_states = [], [], [], [], []

        pbar.update(1)

    # final save
    os.makedirs("./checkpoints/final/", exist_ok=True)
    fp = os.path.join("./checkpoints/final/", f"final_model_{total_timesteps}.ckpt")
    torch.save(model.state_dict(), fp)
    tqdm.write(f"Done. Saved to {fp}")
    env.close()
    wandb.finish()
    return best_avg + avg100
