import gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class LivePlotCallback(BaseCallback):
    """
    Custom callback for plotting live training rewards.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []

    def _on_training_start(self) -> None:
        # Enable interactive mode and set up the plot
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Training Rewards")
        self.ax.set_xlabel("Episodes")
        self.ax.set_ylabel("Reward")

    def _on_step(self) -> bool:
        # `infos` is a list of dicts, one per environment
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                # Episode finished, extract reward
                self.episode_rewards.append(info['episode']['r'])
                # Update the plot
                self.ax.clear()
                self.ax.plot(self.episode_rewards)
                self.ax.set_title("Training Rewards")
                self.ax.set_xlabel("Episodes")
                self.ax.set_ylabel("Reward")
                plt.pause(0.001)
        return True


def main():
    # Create the training environment (no rendering)
    train_env = gym.make('CartPole-v1')

    # Instantiate the PPO model with an MLP policy and tensorboard logging
    model = PPO(
        'MlpPolicy',
        train_env,
        verbose=1,
        tensorboard_log='./ppo_cartpole_tensorboard/'
    )

    # Train the model with live plotting callback
    live_plot = LivePlotCallback()
    model.learn(
        total_timesteps=50000,
        callback=live_plot,
        tb_log_name='PPO_CartPole'
    )

    # Save the trained model
    model.save('ppo_cartpole')
    train_env.close()

    # Evaluation environment (with rendering)
    eval_env = gym.make('CartPole-v1', render_mode='human')
    obs, _info = eval_env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        result = eval_env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result
        if done:
            obs, _info = eval_env.reset()
    eval_env.close()


if __name__ == '__main__':
    main()
