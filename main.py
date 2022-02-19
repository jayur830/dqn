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


def experiences(replay_buffer: deque):
    states, actions, rewards, next_states = \
        np.zeros(shape=(0, 3, 3)), \
        np.asarray([]), \
        np.asarray([]), \
        np.zeros(shape=(0, 3, 3))
    for experience in replay_buffer:
        states = np.concatenate([states, experience[0].reshape((1,) + experience[0].shape)])
        actions = np.append(actions, experience[1])
        rewards = np.append(rewards, experience[2])
        next_states = np.concatenate([next_states, experience[3].reshape((1,) + experience[3].shape)])
    return states, actions, rewards, next_states


if __name__ == "__main__":
    episodes = 10000
    replay_buffer_size = 200
    replay_buffer = deque(maxlen=replay_buffer_size)

    env = Environment(init_state=np.asarray([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]))
    agent = Agent(model=agent_model())
    discount_factor = 0.9

    step, reward = 0, 0
    for episode in range(episodes):
        env.reset()
        while not env.done():
            step += 1
            state = env.state()
            _, action = agent.predict(state.reshape((1,) + state.shape + (1,)))
            reward, next_state = env.step(action)
            replay_buffer.append((state, action, reward, next_state))
            if step >= replay_buffer_size:
                states, _, _, next_states = experiences(replay_buffer)
                q_values, _ = agent.predict(states.reshape(states.shape + (1,)))
                next_q_values, _ = agent.predict(next_states.reshape(next_states.shape + (1,)))
                for i, (s, a, r, ns) in enumerate(replay_buffer):
                    q_values[i][a] = r + (0 if env.done() else discount_factor * np.max(next_q_values[i]))
                agent.train(
                    x=states,
                    y=q_values)
                if step % 100 == 0:
                    agent.update_target_model()
        if reward == -100:
            print(f"episode {episode + 1}: RESET")
        elif reward == 0:
            print(f"episode {episode + 1}: DRAW")
        elif reward == -10:
            print(f"episode {episode + 1}: LOSE")
        elif reward > 0:
            print(f"episode {episode + 1}: WIN")
