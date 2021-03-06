import tensorflow as tf
import numpy as np

from collections import deque
from typing import Callable, Any, Union
from gym import Env
from rl.env import Environment
from rl.replay_buffer import ReplayBuffer
from rl.utils import randint


def set_target_weights(i: int, weights: list, q_weights: list, target_weights: list, tau: float):
    weights.append(q_weights[i] * tau + target_weights[i] * (1 - tau))
    i += 1
    return i


class DQN:
    def __init__(self, env: Union[Environment, Env], model: tf.keras.models.Model, replay_buffer_size: int = 1000, epsilon: float = 1.):
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

    def fit(self,
            episodes: int,
            batch_size: int,
            gamma: float = .9,
            tau: float = 1e-2,
            epsilon_decay: float = .99,
            e_greedy_threshold: float = 0.1,
            target_update_freq: int = 128,
            action_mask: Callable[[np.ndarray, np.ndarray], int] = None,
            on_step_end: Callable[[np.ndarray, int, float, np.ndarray, bool, Any], Any] = None,
            on_episode_end: Callable[[int, float, Any], Any] = None,
            checkpoint_path: str = None,
            checkpoint_freq: int = 100):
        step, reward, info = 0, 0, None
        acc_rewards = 0
        for episode in range(episodes):
            state = self.__env.reset()
            loss = 0
            while True:
                step += 1
                self.__epsilon = tf.maximum(self.__epsilon * epsilon_decay, e_greedy_threshold)
                if tf.random.uniform(shape=(1,))[0] <= self.__epsilon:
                    if action_mask is None:
                        action = randint(self.__q_model.output_shape[-1])
                    else:
                        rand_indexes = np.random.choice(self.__q_model.output_shape[-1], size=self.__q_model.output_shape[-1], replace=False) + 0.5
                        masked_indexes = tf.reshape(action_mask(tf.reshape(state, tf.concat([[1], tf.shape(state)], axis=0)), tf.reshape(rand_indexes, tf.concat([[1], tf.shape(rand_indexes)], axis=0))), [-1])
                        action = tf.argmax(masked_indexes)
                else:
                    q_output = self.__q_model(tf.reshape(state, tf.concat([[1], tf.shape(state)], axis=0)))
                    if action_mask is not None:
                        q_output = action_mask(tf.reshape(state, tf.concat([[1], tf.shape(state)], axis=0)), q_output)
                    action = tf.argmax(q_output, axis=1)[0]
                next_state, reward, done, info = self.__env.step(int(action))
                acc_rewards += reward
                self.__replay_buffer.put(np.reshape(state, (1,) + state.shape), action, reward, np.reshape(next_state, (1,) + next_state.shape), done)
                state = next_state
                if on_step_end is not None:
                    on_step_end(state, action, reward, next_state, done, info)
                if done:
                    break
                if len(self.__replay_buffer) >= batch_size:
                    states, actions, rewards, next_states, dones = self.__replay_buffer.sample(batch_size)
                    with tf.GradientTape() as tape:
                        next_q_values = self.__target_model(next_states)
                        if action_mask is not None:
                            next_q_values = action_mask(next_states, next_q_values)
                        q_target = rewards + (1 - dones) * gamma * tf.reduce_max(next_q_values, axis=1, keepdims=True)
                        # q_target = rewards + (1 - dones) * gamma * tf.math.log(tf.reduce_sum(tf.exp(next_q_values), axis=1))
                        q_values = tf.reduce_sum(self.__q_model(states) * tf.one_hot(tf.cast(tf.reshape(actions, [-1]), tf.int32), self.__q_model.output_shape[-1]), axis=1, keepdims=True)
                        loss = self.__q_model.loss(q_values, q_target)
                        self.__q_model.optimizer.apply_gradients(zip(tape.gradient(loss, self.__q_model.trainable_weights), self.__q_model.trainable_weights))
                if episode % target_update_freq == 0:
                    weights = []
                    q_weights = self.__q_model.get_weights()
                    target_weights = self.__target_model.get_weights()
                    tf.while_loop(
                        lambda i: tf.less(i, len(q_weights)),
                        lambda i: (set_target_weights(i, weights, q_weights, target_weights, tau),),
                        [tf.constant(0)])
                    self.__target_model.set_weights(weights)
            if on_episode_end is not None and callable(on_episode_end):
                on_episode_end(episode + 1, reward, loss, info)
            if checkpoint_path is not None and (episode + 1) % checkpoint_freq == 0:
                self.__target_model.save(checkpoint_path.format(episode=episode + 1, reward=acc_rewards))

    def save(self, filepath: str = "model.h5"):
        self.__target_model.save(filepath)

    def load(self, filepath: str):
        self.__target_model = tf.keras.models.load_model(filepath)
        self.__q_model = tf.keras.models.clone_model(self.__target_model)
        self.__q_model.set_weights(self.__target_model.get_weights())
        self.__q_model.compile(
            optimizer=self.__target_model.optimizer,
            loss=self.__target_model.loss)
