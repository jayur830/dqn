import os
import numpy as np

from collections import deque
from grid_world.commons import reward_win, grid_world_width, grid_world_height
from grid_world.env import GridWorldEnvironment
from grid_world.model import agent_model
from rl.dqn import DQN

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

    env = GridWorldEnvironment(init_state=np.zeros(shape=(grid_world_width, grid_world_height, 1)))

    dqn = DQN(
        env=env,
        model=agent_model(),
        replay_buffer_size=replay_buffer_size)
    dqn.fit(
        episodes=episodes,
        batch_size=100,
        on_episode_end=on_episode_end)
