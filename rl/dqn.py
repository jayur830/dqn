import tensorflow as tf
import numpy as np

from rl.env import Environment
from rl.replay_buffer import ReplayBuffer
from typing import Callable, Any


class DQN:
    def __init__(self, env: Environment, model: tf.keras.models.Model, replay_buffer_size: int = 1000, epsilon: float = 1., epsilon_decay: float = .999):
        self.__env = env
        self.__target_model = model
        self.__q_model = tf.keras.models.clone_model(model)
        self.__q_model.set_weights(model.get_weights())
        self.__q_model.compile(
            optimizer=model.optimizer,
            loss=model.loss)
        self.__replay_buffer = ReplayBuffer(maxlen=replay_buffer_size)
        self.__replay_buffer_size = replay_buffer_size
        self.__epsilon = epsilon
        self.__epsilon_decay = epsilon_decay

    def fit(self,
            episodes: int,
            batch_size: int,
            gamma: float = .9,
            action_mask: Callable = None,
            on_step_end: Callable[[np.ndarray, int, float, np.ndarray, bool, Any], Any] = None,
            on_episode_end: Callable[[Any, Any, Any], Any] = None,
            checkpoint_path: str = None,
            checkpoint_freq: int = 100):
        step, reward, info = 0, 0, None
        acc_rewards = 0
        for episode in range(episodes):
            state = self.__env.reset()
            while True:
                step += 1
                self.__epsilon = max(self.__epsilon * self.__epsilon_decay, 1e-2)
                if np.random.rand() <= self.__epsilon:
                    action = np.random.randint(self.__q_model.output_shape[-1])
                else:
                    action = np.argmax(self.__q_model(state.reshape((1,) + state.shape)))
                next_state, reward, done, info = self.__env.step(action)
                acc_rewards += reward
                self.__replay_buffer.put(state.reshape((1,) + state.shape), action, reward, next_state.reshape((1,) + state.shape), done)
                state = next_state
                if step >= self.__replay_buffer_size:
                    states, actions, rewards, next_states, dones = self.__replay_buffer.sample(batch_size)
                    with tf.GradientTape() as tape:
                        q_target = rewards + (1 - dones) * gamma * tf.reduce_max(self.__target_model(next_states), axis=1, keepdims=True)
                        q_values = tf.reduce_sum(self.__q_model(states) * tf.one_hot(tf.cast(tf.reshape(actions, [-1]), tf.int32), self.__q_model.output_shape[-1]), axis=1, keepdims=True)
                        self.__q_model.optimizer.apply_gradients(zip(tape.gradient(self.__q_model.loss(q_values, q_target), self.__q_model.trainable_weights), self.__q_model.trainable_weights))
                if on_step_end is not None:
                    on_step_end(state, action, reward, next_state, done, info)
                if done:
                    self.__target_model.set_weights(self.__q_model.get_weights())
                    break
            if on_episode_end is not None and callable(on_episode_end):
                on_episode_end(episode, reward, info)
            if checkpoint_path is not None and (episode + 1) % checkpoint_freq == 0:
                self.__target_model.save(checkpoint_path.format(episode=episode, reward=acc_rewards))
