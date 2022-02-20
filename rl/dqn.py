import numpy as np

from rl.env import Environment
from rl.agent import Agent
from rl.replay_buffer import ReplayBuffer
from typing import Callable, Any


class DQN:
    def __init__(self, env: Environment = None, agent: Agent = None, replay_buffer_size: int = 1000):
        self.__env = env
        self.__agent = agent
        self.__replay_buffer = ReplayBuffer(maxlen=replay_buffer_size)
        self.__replay_buffer_size = replay_buffer_size

    def learn(self,
              episodes: int,
              update_freq: int,
              discount_factor: float = .9,
              on_episode_end: Callable[[Any, Any], Any] = None):
        step, reward = 0, 0
        for episode in range(episodes):
            self.__env.reset()
            while not self.__env.done():
                step += 1
                state = self.__env.state()
                q, action = self.__agent.predict(state.reshape((1,) + state.shape + (1,)))
                reward, next_state = self.__env.step(action)
                self.__replay_buffer.put(state, action, reward, next_state)
                if step >= self.__replay_buffer_size:
                    states, actions, rewards, next_states = self.__replay_buffer.sample()
                    q, _ = self.__agent.predict(
                        np.vstack([states.reshape(states.shape + (1,)), next_states.reshape(next_states.shape + (1,))]))
                    q_values, next_q_values = q[:states.shape[0]], q[states.shape[0]:]
                    indexes = np.arange(len(self.__replay_buffer))
                    q_values[indexes, actions] = rewards + (1 - self.__env.done()) * discount_factor * np.max(next_q_values, axis=1)
                    self.__agent.train(
                        x=states,
                        y=q_values)
                    if step % update_freq == 0:
                        self.__agent.update_target_model()
            if on_episode_end is not None and callable(on_episode_end):
                on_episode_end(episode, reward)
