import os
import random
import gym

from rl.dqn import DQN
from cartpole.agent import CartPoleAgent
from cartpole.model import agent_model


def on_episode_end(episode, reward, info):
    print(f"episode {episode + 1}: \033[91mDONE")


if __name__ == "__main__":
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    episodes = 1000000
    replay_buffer_size = 100

    agent = CartPoleAgent(
        model=agent_model(),
        e_greedy_fn=lambda epsilon: max(epsilon - 0.01 * random.randint(0, 10), 0.5))

    dqn = DQN(
        env=gym.make("CartPole-v1"),
        agent=agent,
        replay_buffer_size=replay_buffer_size)
    dqn.learn(
        episodes=episodes,
        buffer_sample_size=100,
        on_episode_end=on_episode_end)
