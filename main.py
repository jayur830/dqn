import os
import numpy as np
import random

from collections import deque
from rl.dqn import DQN
from tictactoe.env import TicTacToeEnvironment
from tictactoe.agent import TicTacToeAgent
from tictactoe.model import agent_model
from tictactoe.commons import reward_reset, reward_draw, reward_lose, reward_win

n_wins = 100
win_counts = deque(maxlen=n_wins)


def on_episode_end(episode, reward, info):
    win_counts.append(reward == reward_win)
    color = ""
    if info["status"] == "RESET":
        color = "\033[93m"
    elif info["status"] == "DRAW":
        color = "\033[92m"
    elif info["status"] == "LOSE":
        color = "\033[91m"
    elif info["status"] == "WIN":
        color = "\033[94m"
    print(f"episode {episode + 1}: {color}{info['status']}, reward: {reward}\033[0m,\t\trate of wins for recent {n_wins} episodes: {int(round(np.sum(win_counts) / n_wins * 100))}%")


if __name__ == "__main__":
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    episodes = 1000000
    replay_buffer_size = 1000

    env = TicTacToeEnvironment(init_state=np.asarray([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]))
    agent = TicTacToeAgent(
        model=agent_model(),
        e_greedy_fn=lambda epsilon: max(epsilon - 0.01 * random.randint(0, 1), 0.1))

    dqn = DQN(
        env=env,
        agent=agent,
        replay_buffer_size=replay_buffer_size)
    dqn.learn(
        episodes=episodes,
        on_episode_end=on_episode_end)
