import os
import numpy as np
import cv2
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
    print(f"episode {episode}: {color}{info['status']}, reward: {reward}\033[0m,\t\trate of wins for recent {n_wins} episodes: {int(round(np.sum(win_counts) / n_wins * 100))}%")


def on_step_end(state, action, reward, next_state, done, info):
    img = np.zeros(shape=(300, 300, 3), dtype=np.uint8)
    img = cv2.line(img, pt1=(0, 100), pt2=(300, 100), color=(200, 200, 200))
    img = cv2.line(img, pt1=(0, 200), pt2=(300, 200), color=(200, 200, 200))
    img = cv2.line(img, pt1=(100, 0), pt2=(100, 300), color=(200, 200, 200))
    img = cv2.line(img, pt1=(200, 0), pt2=(200, 300), color=(200, 200, 200))
    next_state = next_state.reshape(next_state.shape[:-1])

    for i in range(next_state.shape[0]):
        for j in range(next_state.shape[1]):
            if next_state[i, j] == 1.:
                img = cv2.circle(img, center=(i * 100 + 50, j * 100 + 50), radius=30, color=(0, 0, 255), thickness=2)
            elif next_state[i, j] == -1.:
                img = cv2.line(img, pt1=(i * 100 + 20, j * 100 + 20), pt2=(i * 100 + 80, j * 100 + 80), color=(0, 255, 0), thickness=2)
                img = cv2.line(img, pt1=(i * 100 + 80, j * 100 + 20), pt2=(i * 100 + 20, j * 100 + 80), color=(0, 255, 0), thickness=2)

    cv2.imshow("Tic Tac Toe", img)
    cv2.waitKey(1)


def action_mask(state, q_output):
    indexes = np.where(state.reshape(-1) != 0)[0]
    q_output = q_output.reshape(-1)
    q_output = (q_output - np.min(q_output)) / (np.max(q_output) - np.min(q_output))
    q_output[indexes] = 0
    return q_output.argmax() if indexes.shape[0] > 0 else -1


if __name__ == "__main__":
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    episodes = 1000000
    replay_buffer_size = 10000

    env = TicTacToeEnvironment(init_state=np.zeros(shape=(3, 3, 1)))

    dqn = DQN(
        env=env,
        model=agent_model(),
        replay_buffer_size=replay_buffer_size)
    dqn.fit(
        episodes=episodes,
        batch_size=64,
        action_mask=action_mask,
        on_episode_end=on_episode_end,
        # on_step_end=on_step_end,
        # checkpoint_path="checkpoint/tictactoe_agent_{episode}_{reward:.1f}.h5",
        # checkpoint_freq=100
    )
