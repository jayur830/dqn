import os
import tensorflow as tf
import numpy as np
import cv2
import random

from collections import deque
from rl.dqn import DQN
from tictactoe.env import TicTacToeEnvironment
from tictactoe.model import agent_model
from tictactoe.commons import reward_reset, reward_draw, reward_lose, reward_win, agent, enemy, empty

n_wins, win_rate = 100, 0
win_counts = deque(maxlen=n_wins)


def on_episode_end(episode, reward, info):
    global win_rate
    win_counts.append(info["status"] == "WIN")
    win_rate = int(round(np.sum(win_counts) / len(win_counts) * 100))
    color = ""
    if info["status"] == "RESET":
        color = "\033[93m"
    elif info["status"] == "DRAW":
        color = "\033[92m"
    elif info["status"] == "LOSE":
        color = "\033[91m"
    elif info["status"] == "WIN":
        color = "\033[94m"
    print(f"episode {episode}: {color}{info['status']}, reward: {round(reward * 10) / 10}\033[0m,\t\trate of wins for recent {n_wins} episodes: {win_rate}%")


def on_step_end(state, action, reward, next_state, done, info):
    if win_rate >= 50:
        cell_size = 100
        img = np.zeros(
            shape=(3 * cell_size, 3 * cell_size, 3),
            dtype=np.uint8)
        img = cv2.line(
            img=img,
            pt1=(0, cell_size),
            pt2=(3 * cell_size, cell_size),
            color=(200, 200, 200))
        img = cv2.line(
            img=img,
            pt1=(0, 2 * cell_size),
            pt2=(3 * cell_size, 2 * cell_size),
            color=(200, 200, 200))
        img = cv2.line(
            img=img,
            pt1=(cell_size, 0),
            pt2=(cell_size, 3 * cell_size),
            color=(200, 200, 200))
        img = cv2.line(
            img=img,
            pt1=(2 * cell_size, 0),
            pt2=(2 * cell_size, 3 * cell_size),
            color=(200, 200, 200))
        next_state = next_state.reshape(next_state.shape[:-1])

        for i in range(next_state.shape[0]):
            for j in range(next_state.shape[1]):
                if next_state[i, j] == agent:
                    img = cv2.circle(
                        img=img,
                        center=(int(i * cell_size + cell_size / 2), int(j * cell_size + cell_size / 2)),
                        radius=int(cell_size * .3),
                        color=(0, 0, 255),
                        thickness=2)
                elif next_state[i, j] == enemy:
                    img = cv2.line(
                        img=img,
                        pt1=(int(i * cell_size + cell_size * .2), int(j * cell_size + cell_size * .2)),
                        pt2=(int(i * cell_size + cell_size * .8), int(j * cell_size + cell_size * .8)),
                        color=(0, 255, 0),
                        thickness=2)
                    img = cv2.line(
                        img=img,
                        pt1=(int(i * cell_size + cell_size * .8), int(j * cell_size + cell_size * .2)),
                        pt2=(int(i * cell_size + cell_size * .2), int(j * cell_size + cell_size * .8)),
                        color=(0, 255, 0),
                        thickness=2)

        cv2.imshow("Tic Tac Toe", img)
        cv2.waitKey(1)


def action_mask(state, q_output):
    q_output = tf.reshape(q_output, [-1])
    q_output = (q_output - tf.reduce_min(q_output)) / (tf.reduce_max(q_output) - tf.reduce_min(q_output))
    q_output = tf.where(np.reshape(state, -1) != empty, 0, q_output)
    return tf.argmax(q_output)


if __name__ == "__main__":
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    episodes = 100000
    replay_buffer_size = 1000

    env = TicTacToeEnvironment(init_state=np.ones(shape=(3, 3, 1)) * empty)

    dqn = DQN(
        env=env,
        model=agent_model(),
        replay_buffer_size=replay_buffer_size)
    dqn.fit(
        episodes=episodes,
        batch_size=256,
        action_mask=action_mask,
        target_update_freq=256,
        on_episode_end=on_episode_end,
        # on_step_end=on_step_end,
        # checkpoint_path="checkpoint/tictactoe_agent_{episode}_{reward:.1f}.h5",
        # checkpoint_freq=100
    )

    dqn.save("tictactoe.h5")
