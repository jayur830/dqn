import os
import numpy as np
import cv2

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


def on_step_end(state, action, reward, next_state, done, info):
    img = np.zeros(shape=(100 * grid_world_height, 100 * grid_world_width, 3), dtype=np.uint8)
    for i in range(1, grid_world_height):
        img = cv2.line(img, pt1=(0, 100 * i), pt2=(100 * grid_world_width, 100 * i), color=(200, 200, 200))
    for j in range(1, grid_world_width):
        img = cv2.line(img, pt1=(100 * j, 0), pt2=(100 * j, 100 * grid_world_height), color=(200, 200, 200))

    if 1. in next_state:
        agent_point = np.transpose(np.where(next_state == 1.))[0][:-1]
        img = cv2.circle(img, center=(100 * agent_point[1] + 50, 100 * agent_point[0] + 50), radius=40, color=(0, 0, 255), thickness=-1)

    if 2. in next_state:
        goal_point = np.transpose(np.where(next_state == 2.))[0][:-1]
        img = cv2.rectangle(img, pt1=(100 * goal_point[1], 100 * goal_point[0]), pt2=(100 * goal_point[1] + 100, 100 * goal_point[0] + 100), color=(255, 0, 0), thickness=-1)

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
        batch_size=100,
        on_step_end=on_step_end,
        on_episode_end=on_episode_end)
