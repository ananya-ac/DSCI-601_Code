import os
import random
from collections import deque

import gymnasium as gym
import numpy as np
import pytorch_lightning as pl
import torch
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import config
from agent import ActorCriticWithFlow, SACWithFlow
import wandb


def _save_model_on_best_reward(model, recent_rewards, best_avg_reward, checkpoint_dir="./checkpoints/"):
    if not recent_rewards:
        return best_avg_reward
    current_avg = float(np.mean(recent_rewards))
    if current_avg > best_avg_reward:
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(checkpoint_dir, "best_model_reward.ckpt")
        torch.save(model.state_dict(), path)
        print(f"New best avg reward {current_avg:.2f}, saved to {path}")
        return current_avg
    return best_avg_reward


def _is_continuous(space):
    return space.dtype.name != "int64"
    

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
    continuous,                   # kept for signature compatibility; env decides
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
    replay_buffer_size=100_000,   # off-policy (SAC)
    tau=0.005,                    # SAC target smoothing
    alpha_lr=3e-4,                # SAC temperature lr
):
    # -----------------------
    # Seeding & Env
    # -----------------------
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = gym.make(env_name)
    if max_episode_steps:
        try:
            env._max_episode_steps = max_episode_steps
        except Exception:
            pass

    # Decide strictly from env
    is_cont = _is_continuous(env.action_space)
    if is_cont:
        action_dim = env.action_space.shape[0]
        num_actions = 0
        algo = "sac"               # continuous -> off-policy SAC
        mode = "off_policy"
    else:
        num_actions = env.action_space.n
        action_dim = 1             # not used in PPO path
        algo = "ppo_flow"          # discrete -> on-policy ActorCriticWithFlow
        mode = "on_policy"

    # -----------------------
    # Instantiate model (NO helper)
    # -----------------------
    if algo == "sac":
        model = SACWithFlow(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            flow_dim=flow_dim,
            num_flow_layers=num_flow_layers,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            alpha_lr=alpha_lr,
            gamma=gamma,
            tau=tau,
            continuous=True,     # enforced for SAC
            num_actions=0,       # unused in continuous path
            target_entropy=None, # defaults to -|A|
            normalize_returns=True,
        )
    else:
        model = ActorCriticWithFlow(
            state_dim=state_dim,
            action_dim=num_actions,   # discrete branch uses num_actions
            hidden_dim=hidden_dim,
            flow_dim=flow_dim,
            num_flow_layers=num_flow_layers,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            gamma=gamma,
            continuous=False,         # enforced for PPO path
            gae_lambda=gae_lambda,
            entropy_coef=entropy_coef,
            value_loss_coef=value_loss_coef,
            max_grad_norm=max_grad_norm,
            normalize_advantages=normalize_advantages,
            critic_updates_per_actor_update=critic_updates_per_actor_update,
            num_actions=num_actions,
        )

    # -----------------------
    # Logging
    # -----------------------
    wandb.init(project="dist_rl_online")
    wandb.config.update({
        "env": env_name,
        "algo": algo,
        "mode": mode,
        "seed": seed,
        "hidden_dim": hidden_dim,
        "flow_dim": flow_dim,
        "num_flow_layers": num_flow_layers,
        "actor_lr": actor_lr,
        "critic_lr": critic_lr,
        "gamma": gamma,
        "tau": tau,
        "alpha_lr": alpha_lr,
        "update_frequency": update_frequency,
        "critic_batch_size": critic_batch_size,
    })

    best_avg = float("-inf")
    recent_rewards = deque(maxlen=100)

    # Storage
    if mode == "on_policy":
        roll_states, roll_actions, roll_rewards, roll_dones, roll_next = [], [], [], [], []
        buffer = None
    else:
        buffer = ReplayBuffer(
            buffer_size=replay_buffer_size,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=config.device,
        )

    # -----------------------
    # Loop
    # -----------------------
    state, _ = env.reset(seed=seed)
    episode_reward = 0.0
    episodes_completed = 0
    timestep = 0
    pbar = tqdm(total=total_timesteps, desc=f"Training ({algo})")

    model = model.to(config.device)

    while timestep < total_timesteps:
        st = torch.as_tensor(state, dtype=torch.float32, device=config.device)

        with torch.no_grad():
            # Both models should expose .act(state, deterministic=False)
            at = model.act(st, deterministic=False)

        if algo == "sac":
            action = at.cpu().numpy()  # continuous np.array
            next_state, reward, term, trunc, _ = env.step(action)
        else:
            action = at.item() if torch.is_tensor(at) else int(at)
            next_state, reward, term, trunc, _ = env.step(action)

        done = bool(term or trunc)

        # Store
        if mode == "off_policy":
            buffer.add(
                obs=state,
                next_obs=next_state,
                action=action,
                reward=reward,
                done=done,
                infos=[{}],
            )
        else:
            roll_states.append(state)
            roll_actions.append(action)
            roll_rewards.append(reward)
            roll_dones.append(done)
            roll_next.append(next_state)

        # Bookkeeping
        state = next_state
        episode_reward += reward
        timestep += 1
        pbar.update(1)

        # Episode end
        if done:
            episodes_completed += 1
            recent_rewards.append(episode_reward)
            wandb.log({"env/episode_reward": episode_reward, "env/timestep": timestep})
            avg100 = float(np.mean(recent_rewards))
            if episodes_completed % max(1, log_frequency) == 0:
                tqdm.write(f"[{algo}] Ep {episodes_completed}, avg100 {avg100:.2f}")
                wandb.log({"env/avg_reward_100": avg100, "env/episodes": episodes_completed})
            best_avg = _save_model_on_best_reward(model, recent_rewards, best_avg)
            state, _ = env.reset()
            episode_reward = 0.0

        # ===========================
        # UPDATES
        # ===========================

        # OFF-POLICY (SAC)
        if algo == "sac":
            if buffer.size() >= critic_batch_size and (timestep % max(1, update_frequency) == 0):
                batch = buffer.sample(batch_size=max(critic_batch_size, 2048))
                ds = TensorDataset(
                    batch.observations,
                    batch.actions,
                    batch.rewards.unsqueeze(1),
                    batch.next_observations,
                    batch.dones.unsqueeze(1).float(),
                )
                loader = DataLoader(ds, batch_size=critic_batch_size, shuffle=True, drop_last=True)
                trainer = pl.Trainer(max_epochs=num_epochs, accelerator="auto", logger=False)
                trainer.fit(model, train_dataloaders=loader)
                

        # ON-POLICY (PPO-style with flow critic)
        if algo == "ppo_flow":
            if len(roll_states) >= update_frequency:
                st_t = torch.as_tensor(np.array(roll_states), dtype=torch.float32)
                at_t = torch.as_tensor(np.array(roll_actions), dtype=torch.long)
                rw_t = torch.as_tensor(np.array(roll_rewards), dtype=torch.float32).unsqueeze(1)
                dn_t = torch.as_tensor(np.array(roll_dones), dtype=torch.float32).unsqueeze(1)
                nx_t = torch.as_tensor(np.array(roll_next), dtype=torch.float32)

                full = {"states": st_t, "actions": at_t, "rewards": rw_t, "dones": dn_t, "next_states": nx_t}
                with torch.no_grad():
                    model.prepare_training_data(full)

                fs = model.critic_batch["states"]
                fa = model.critic_batch["actions"]
                adv = model.advantages
                ret = model.critic_returns

                ds = TensorDataset(fs, fa, adv, ret)
                loader = DataLoader(ds, batch_size=critic_batch_size, shuffle=True, drop_last=True)
                trainer = pl.Trainer(max_epochs=num_epochs, accelerator="auto", logger=False)
                trainer.fit(model, train_dataloaders=loader)

                roll_states, roll_actions, roll_rewards, roll_dones, roll_next = [], [], [], [], []
        model = model.to(config.device)

    # -----------------------
    # Final save & close
    # -----------------------
    os.makedirs("./checkpoints/final/", exist_ok=True)
    fp = os.path.join("./checkpoints/final/", f"final_model_{total_timesteps}.ckpt")
    torch.save(model.state_dict(), fp)
    tqdm.write(f"Done. Saved to {fp}")
    env.close()
    wandb.finish()

    return float(np.mean(recent_rewards)) if len(recent_rewards) > 0 else -np.inf
