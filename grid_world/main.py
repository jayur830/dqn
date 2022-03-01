import os
import numpy as np
import random

from collections import deque
from rl.dqn import DQN
from grid_world.env import GridWorldEnvironment
from grid_world.agent import GridWorldAgent
from grid_world.model import agent_model
from grid_world.commons import reward_win, grid_world_width, grid_world_height

n_wins = 100
win_counts = deque(maxlen=n_wins)


def on_episode_end(episode, reward, info):
    win_counts.append(reward == reward_win)
    color = ""
    if info["status"] == "LOSE":
        color = "\033[91m"
    elif info["status"] == "WIN":
        color = "\033[94m"
    print(f"episode {episode + 1}: {color}{info['status']}, reward: {reward}\033[0m,\t\trate of wins for recent {n_wins} episodes: {int(round(np.sum(win_counts) / n_wins * 100))}%")


if __name__ == "__main__":
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    episodes = 1000000
    replay_buffer_size = 100

    env = GridWorldEnvironment(init_state=np.zeros(shape=(grid_world_width, grid_world_height)))
    agent = GridWorldAgent(
        model=agent_model(),
        e_greedy_fn=lambda epsilon: max(epsilon - 0.01 * random.randint(0, 10), 0.5))

    dqn = DQN(
        env=env,
        agent=agent,
        replay_buffer_size=replay_buffer_size)
    dqn.learn(
        episodes=episodes,
        buffer_sample_size=100,
        on_episode_end=on_episode_end)
