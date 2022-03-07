import os
import tensorflow as tf
import numpy as np
import cv2

from collections import deque
from grid_world.commons import reward_win, grid_world_width, grid_world_height
from grid_world.env import GridWorldEnvironment
from grid_world.model import agent_model
from rl.dqn import DQN

n_wins = 100
win_counts = deque(maxlen=n_wins)


def on_episode_end(episode, reward, loss, info):
    win_counts.append(info["status"] == "WIN")
    color = ""
    if info["status"] == "LOSE":
        color = "\033[91m"
    elif info["status"] == "WIN":
        color = "\033[94m"
    print(f"episode {episode}: {color}{info['status']}, reward: {reward}\033[0m,\t\trate of wins for recent {n_wins} episodes: {int(round(np.sum(win_counts) / len(win_counts) * 100))}%, loss: {tf.reduce_mean(loss):.4f}")


def on_step_end(state, action, reward, next_state, done, info):
    # if int(round(np.sum(win_counts) / n_wins * 100)) >= 50:
    cell_size = 80
    img = np.zeros(shape=(cell_size * grid_world_height, cell_size * grid_world_width, 3), dtype=np.uint8)
    for i in range(1, grid_world_height):
        img = cv2.line(
            img=img,
            pt1=(0, cell_size * i),
            pt2=(cell_size * grid_world_width, cell_size * i),
            color=(200, 200, 200))
    for j in range(1, grid_world_width):
        img = cv2.line(
            img=img,
            pt1=(cell_size * j, 0),
            pt2=(cell_size * j, cell_size * grid_world_height),
            color=(200, 200, 200))

    if 1. in next_state:
        agent_point = np.transpose(np.where(next_state == 1.))[0][:-1]
        img = cv2.circle(
            img=img,
            center=(cell_size * agent_point[1] + cell_size // 2, cell_size * agent_point[0] + cell_size // 2),
            radius=int(cell_size * .4),
            color=(0, 0, 255),
            thickness=-1)
    if 2. in next_state:
        goal_point = np.transpose(np.where(next_state == 2.))[0][:-1]
        img = cv2.rectangle(
            img=img,
            pt1=(cell_size * goal_point[1], cell_size * goal_point[0]),
            pt2=(cell_size * goal_point[1] + cell_size, cell_size * goal_point[0] + cell_size),
            color=(255, 0, 0),
            thickness=-1)
    if -1. in next_state:
        obstacle_points = np.transpose(np.where(next_state == -1.))[:, :-1]
        for i in range(obstacle_points.shape[0]):
            img = cv2.line(
                img=img,
                pt1=(int(cell_size * obstacle_points[i, 1] + cell_size * .2), int(cell_size * obstacle_points[i, 0] + cell_size * .2)),
                pt2=(int(cell_size * obstacle_points[i, 1] + cell_size * .8), int(cell_size * obstacle_points[i, 0] + cell_size * .8)),
                color=(0, 255, 0),
                thickness=2)
            img = cv2.line(
                img=img,
                pt1=(int(cell_size * obstacle_points[i, 1] + cell_size * .2), int(cell_size * obstacle_points[i, 0] + cell_size * .8)),
                pt2=(int(cell_size * obstacle_points[i, 1] + cell_size * .8), int(cell_size * obstacle_points[i, 0] + cell_size * .2)),
                color=(0, 255, 0),
                thickness=2)

    cv2.imshow("Grid World", img)
    cv2.waitKey(1)


if __name__ == "__main__":
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    episodes = 1000000
    replay_buffer_size = 1000

    env = GridWorldEnvironment(init_state=np.zeros(shape=(grid_world_width, grid_world_height, 1)))

    dqn = DQN(
        env=env,
        model=agent_model(),
        replay_buffer_size=replay_buffer_size)
    dqn.fit(
        episodes=episodes,
        batch_size=32,
        on_step_end=on_step_end,
        on_episode_end=on_episode_end)
