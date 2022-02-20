import numpy as np
import random

from rl.dqn import DQN
from tictactoe.env import TicTacToeEnvironment
from tictactoe.agent import TicTacToeAgent
from tictactoe.model import agent_model
from tictactoe.commons import reward_reset, reward_draw, reward_lose, reward_win


def on_episode_end(episode, reward):
    if reward == reward_reset:
        print(f"episode {episode + 1}: \033[93mRESET, reward: {reward}\033[0m")
    elif reward == reward_draw:
        print(f"episode {episode + 1}: \033[92mDRAW, reward: {reward}\033[0m")
    elif reward == reward_lose:
        print(f"episode {episode + 1}: \033[91mLOSE, reward: {reward}\033[0m")
    elif reward == reward_win:
        print(f"episode {episode + 1}: \033[94mWIN, reward: {reward}\033[0m")


if __name__ == "__main__":
    episodes = 1000000
    replay_buffer_size = 500
    update_freq = 50

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
        update_freq=update_freq,
        on_episode_end=on_episode_end)
