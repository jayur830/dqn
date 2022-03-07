import os
import gym
import tensorflow as tf

from rl.dqn import DQN
from cartpole.model import agent_model


def on_episode_end(episode, reward, loss, info):
    print(f"episode {episode + 1}: \033[91mDONE\033[0m, loss: {tf.reduce_mean(loss):.4f}")


def on_step_end(state, action, reward, next_state, done, info):
    env.render()


if __name__ == "__main__":
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    episodes = 1000000
    replay_buffer_size = 1000

    env = gym.make("CartPole-v1")

    dqn = DQN(
        env=env,
        model=agent_model(),
        replay_buffer_size=replay_buffer_size)
    dqn.fit(
        episodes=episodes,
        batch_size=64,
        target_update_freq=128,
        tau=1.,
        on_step_end=on_step_end,
        on_episode_end=on_episode_end)
