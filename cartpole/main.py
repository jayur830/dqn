import os
import gym

from rl.dqn import DQN
from cartpole.model import agent_model


def on_episode_end(episode, reward, info):
    print(f"episode {episode + 1}: \033[91mDONE\033[0m")


def on_step_end():
    env.render()


if __name__ == "__main__":
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    episodes = 1000000
    replay_buffer_size = 100

    env = gym.make("CartPole-v1")

    dqn = DQN(
        env=env,
        model=agent_model(),
        replay_buffer_size=replay_buffer_size)
    dqn.fit(
        episodes=episodes,
        batch_size=100,
        on_step_end=on_step_end,
        on_episode_end=on_episode_end)
