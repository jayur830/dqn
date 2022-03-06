import os
import tensorflow as tf
import numpy as np
import cv2

from collections import deque
from rl.dqn import DQN
from gomoku.env import GomokuEnvironment
from gomoku.model import agent_model
from gomoku.commons import gomoku_size, reward_win, black, white, empty

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
    if win_rate >= 0:
        cell_size = 50
        img = np.zeros(shape=(cell_size * (gomoku_size + 1), cell_size * (gomoku_size + 1), 3))
        img[:, :, 0], img[:, :, 1], img[:, :, 2] = 68 / 255., 166 / 255., 229 / 255.

        for i in range(1, gomoku_size + 1):
            img = cv2.line(
                img=img,
                pt1=(cell_size, cell_size * i),
                pt2=(cell_size * gomoku_size, cell_size * i),
                color=(50 / 255., 50 / 255., 50 / 255.),
                thickness=1)
        for i in range(1, gomoku_size + 1):
            img = cv2.line(
                img=img,
                pt1=(cell_size * i, cell_size),
                pt2=(cell_size * i, cell_size * gomoku_size),
                color=(50 / 255., 50 / 255., 50 / 255.),
                thickness=1)

        black_indexes = np.transpose(np.where(next_state == black)) + 1
        white_indexes = np.transpose(np.where(next_state == white)) + 1

        for i in range(black_indexes.shape[0]):
            img = cv2.circle(
                img=img,
                center=(cell_size * black_indexes[i, 0], cell_size * black_indexes[i, 1]),
                radius=int(cell_size * .45),
                color=(0, 0, 0),
                thickness=-1)
        for i in range(white_indexes.shape[0]):
            img = cv2.circle(
                img=img,
                center=(cell_size * white_indexes[i, 0], cell_size * white_indexes[i, 1]),
                radius=int(cell_size * .45),
                color=(1., 1., 1.),
                thickness=-1)

        cv2.imshow("Gomoku", img)
        cv2.waitKey(1)


def action_mask(state, q_output):
    q_output = tf.reshape(q_output, [-1])
    q_output = (q_output - tf.reduce_min(q_output)) / (tf.reduce_max(q_output) - tf.reduce_min(q_output))
    q_output = tf.where(np.reshape(state, -1) != empty, 0, q_output)
    return tf.argmax(q_output)


if __name__ == "__main__":
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    episodes = 100
    replay_buffer_size = 10000

    env = GomokuEnvironment(init_state=np.ones(shape=(gomoku_size, gomoku_size, 1)) * empty)

    dqn = DQN(
        env=env,
        model=agent_model(),
        replay_buffer_size=replay_buffer_size)
    dqn.fit(
        episodes=episodes,
        batch_size=64,
        action_mask=action_mask,
        target_update_freq=512,
        on_episode_end=on_episode_end,
        on_step_end=on_step_end,
        # checkpoint_path="checkpoint/gomoku_agent_{episode}_{reward:.1f}.h5",
        # checkpoint_freq=100
    )

    dqn.save("gomoku.h5")
