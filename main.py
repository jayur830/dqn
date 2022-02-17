import tensorflow as tf
import numpy as np

from typing import Callable
from collections import deque
from rl.env import Environment
from rl.agent import Agent
from model import agent_model


class Action:
    pass


def mask(state: np.ndarray, agent_output: np.ndarray):
    state = state.reshape(state.shape[:-1])
    q = agent_output.reshape(agent_output.shape[1:-1])
    indexes = np.transpose(np.where(state != 0))
    for i in range(indexes.shape[0]):
        q[indexes[i][0], indexes[i][1]] = 0
    return q


"""sudo code

<pre>
    episodes = 1000
    replay_buffer_size = 2000
    replay_buffer = deque(maxlen=replay_buffer_size)
    
    env = Environment()
    agent = Agent()
    
    step = 0
    for episode in episodes:
        env.reset()
        while not env.done():
            state = env.state
            _, action = agent.predict(state)
            next_state, reward = env.step(action)
            replay_buffer.append((state, action, reward, next_state))
            step += 1
        if step >= replay_buffer_size:
            states, actions, rewards, next_states = np.transpose(np.asarray(replay_buffer))
            next_q_value, _ = agent.predict(next_states)
            target_q_values = rewards + discount * max(next_q_values)
            all_q_values, _ = agent.predict(states)
            q_values = sum(all_q_values * mask(actions))
            agent.train(
                x=target_q_values,
                y=q_values,
                loss=mean_squared_error)
</pre>
"""


if __name__ == "__main__":
    episodes = 1000
    replay_buffer_size = 2000
    replay_buffer = deque(maxlen=replay_buffer_size)

    env = Environment(init_state=np.asarray([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]))
    agent = Agent(model=agent_model())
    discount = 0.1

    step = 0
    for episode in range(episodes):
        env.reset()
        while not env.done():
            state = env.state
            _, action = agent.predict(state)
            reward, next_state = env.step(action)
            replay_buffer.append((state, action, reward, next_state))
            step += 1
            print(replay_buffer[-1])
            input()
        if step >= replay_buffer_size:
            states, actions, rewards, next_states = np.transpose(np.asarray(replay_buffer))
            next_q_value, _ = agent.predict(next_states)
            target_q_values = rewards + discount * np.max(next_q_value)
            all_q_values, _ = agent.predict(states)
            q_values = np.sum(all_q_values * mask(actions))
            agent.train(
                x=target_q_values,
                y=q_values)
