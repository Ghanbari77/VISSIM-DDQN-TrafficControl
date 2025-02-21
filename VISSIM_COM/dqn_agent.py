import numpy as np
import tensorflow as tf
from collections import deque
import random
from config import learning_rate, gamma

def build_q_network(state_size):
    """Builds a Q-network model using Keras."""
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(state_size,)),
        tf.keras.layers.Dense(200, activation='relu'),
        tf.keras.layers.Dense(200, activation='relu'),
        tf.keras.layers.Dense(200, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=15000)
        self.q_network_1 = build_q_network(state_size)
        self.q_network_2 = build_q_network(state_size)

    def select_action(self, state, epsilon, possible_actions):
        """Epsilon-greedy action selection."""
        if np.random.rand() <= epsilon:
            return random.choice(possible_actions)
        
        q_values_1 = self.q_network_1.predict(state, verbose=0)
        q_values_2 = self.q_network_2.predict(state, verbose=0)
        return possible_actions[np.argmax(q_values_1 + q_values_2)]

    def train(self, batch_size):
        """Trains the Q-network using experience replay."""
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state in minibatch:
            target = reward + gamma * np.max(self.q_network_2.predict(next_state, verbose=0))
            self.q_network_1.fit(state, target, epochs=1, verbose=0)
