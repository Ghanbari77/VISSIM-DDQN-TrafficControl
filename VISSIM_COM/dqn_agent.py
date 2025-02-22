import numpy as np
import tensorflow as tf
import random
import copy
from config import INTERSECTIONS, DEEP_LEARNING_PARAMS
from replay_memory import ReplayMemory
from utils import flatten_state, flatten_list  # corrected module name

# --- Access Deep Learning Parameters ---
state_size = DEEP_LEARNING_PARAMS["state_sizes"]["total"]
action_size = DEEP_LEARNING_PARAMS["action_size"]
action_values = DEEP_LEARNING_PARAMS["action_values"]
learning_rate = DEEP_LEARNING_PARAMS["learning_rate"]
batch_size = DEEP_LEARNING_PARAMS["batch_size"]
max_memory_size = DEEP_LEARNING_PARAMS["max_memory_size"]
gamma = DEEP_LEARNING_PARAMS["gamma"]

def build_q_network():
    """
    Builds a Q-network model using TensorFlow's Keras API.
    The model consists of four hidden layers with ReLU activations.
    """
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

# --- Access Exploration Parameters ---
epsilon = DEEP_LEARNING_PARAMS["exploration"]["epsilon"]
epsilon_decay = DEEP_LEARNING_PARAMS["exploration"]["epsilon_decay"]
epsilon_min = DEEP_LEARNING_PARAMS["exploration"]["epsilon_min"]

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayMemory(max_size=max_memory_size)
        self.q_network_1 = build_q_network()
        self.q_network_2 = build_q_network()

    def select_action(self, state, epsilon, all_modified_actions):
        """
        Selects an action using the epsilon-greedy policy.
        """
        if np.random.rand() <= epsilon:
            return all_modified_actions[np.random.randint(len(all_modified_actions))]
        else:
            flattened_states = []
            for action in all_modified_actions:
                modified_state = copy.deepcopy(state)
                # Apply the action to the appropriate part of the state (e.g., Dixie_Shawson)
                modified_state[0][0] = np.add(modified_state[0][0], action)
    
                # Flatten each sub-state (assumed order: [Dixie_Shawson, Dixie_Britannia, Dixie_401])
                flat_shawson = flatten_state(modified_state[0])
                flat_britannia = flatten_state(modified_state[1])
                flat_401 = flatten_state(modified_state[2])
                combined_flat_state = np.concatenate([flat_shawson, flat_britannia, flat_401])
                flattened_states.append(combined_flat_state)
    
            flattened_states_batch = np.array(flattened_states).reshape(-1, self.state_size)
    
            # Get Q-values from both networks
            q_values_1 = self.q_network_1.predict(flattened_states_batch, verbose=0).flatten()
            q_values_2 = self.q_network_2.predict(flattened_states_batch, verbose=0).flatten()
    
            # Sum Q-values for Double DQN
            q_values_sum = q_values_1 + q_values_2
            best_action_index = np.argmax(q_values_sum)
            return all_modified_actions[best_action_index]

    def train(self):
        """
        Trains the Q-network using Double DQN (DDQN) update.
        """
        # Check if there are enough experiences in memory
        if len(self.memory.memory) < batch_size:
            return
    
        # Sample a batch from memory
        minibatch = self.memory.sample_batch(batch_size)
    
        flattened_states = []
        target_Q1 = []
        target_Q2 = []
    
        for experience in minibatch:
            # Experience is expected to be (state, action, reward, next_state)
            state, action, reward, next_state = experience
            modified_state = copy.deepcopy(state)
            modified_state[0][0] = np.add(modified_state[0][0], action)
    
            flat_shawson = flatten_state(modified_state[0])
            flat_britannia = flatten_state(modified_state[1])
            flat_401 = flatten_state(modified_state[2])
            combined_flat_state = np.concatenate([flat_shawson, flat_britannia, flat_401])
            flattened_states.append(combined_flat_state)
    
            # Generate possible next states by applying each possible action
            flattened_next_states = []
            for possible_action in action_values:
                modified_next_state = copy.deepcopy(next_state)
                modified_next_state[0][0] = np.add(modified_next_state[0][0], possible_action)
    
                flat_next_shawson = flatten_state(modified_next_state[0])
                flat_next_britannia = flatten_state(modified_next_state[1])
                flat_next_401 = flatten_state(modified_next_state[2])
                combined_flat_next_state = np.concatenate([flat_next_shawson, flat_next_britannia, flat_next_401])
                flattened_next_states.append(combined_flat_next_state)
    
            flattened_next_states_batch = np.array(flattened_next_states).reshape(-1, self.state_size)
            next_q_values_1 = self.q_network_1.predict(flattened_next_states_batch, verbose=0).flatten()
            next_q_values_2 = self.q_network_2.predict(flattened_next_states_batch, verbose=0).flatten()
    
            # Use Double DQN: network1 selects the best action while network2 evaluates it, and vice versa.
            best_action_index_1 = np.argmax(next_q_values_1)
            best_action_index_2 = np.argmax(next_q_values_2)
            next_q2_value_based_q1 = next_q_values_2[best_action_index_1]
            next_q1_value_based_q2 = next_q_values_1[best_action_index_2]
    
            target_Q1.append(reward + gamma * next_q2_value_based_q1)
            target_Q2.append(reward + gamma * next_q1_value_based_q2)
    
        flattened_states_batch = np.array(flattened_states).reshape(-1, self.state_size)
        target_Q1_batch = np.array(target_Q1)
        target_Q2_batch = np.array(target_Q2)
    
        # Alternate between updating network 1 and network 2
        if np.random.rand() < 0.5:
            self.q_network_1.fit(flattened_states_batch, target_Q1_batch, epochs=1, verbose=0)
        else:
            self.q_network_2.fit(flattened_states_batch, target_Q2_batch, epochs=1, verbose=0)
