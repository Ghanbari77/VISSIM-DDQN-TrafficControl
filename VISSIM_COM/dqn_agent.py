import numpy as np
import tensorflow as tf
from collections import deque
import random
from config import INTERSECTIONS, DEEP_LEARNING_PARAMS
from replay_memory import ReplayMemory

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

        Parameters:
        -----------
        state : list
            The current state representation.
        epsilon : float
            The exploration rate.
        all_modified_actions : np.ndarray
            Array of all possible actions.

        Returns:
        --------
        np.ndarray
            The selected action.
        """
        if np.random.rand(1) <= epsilon:
            return all_modified_actions[np.random.randint(len(all_modified_actions))]

        else:
            flattened_state = []
            for action in all_modified_actions:
                modified_state = copy.deepcopy(state)
                # Apply action to green time
                modified_state[0][0] = np.add(modified_state[0][0], action)

                # Flatten the state
                modified_state[0] = flatten_state(modified_state[0])  # Dixie_Shawson
                modified_state[1] = flatten_state(modified_state[1])  # Dixie_Britannia
                modified_state[2] = flatten_state(modified_state[2])  # Dixie_401
                flattened_state.append(np.concatenate([modified_state[0], modified_state[1], modified_state[2]]))

            # Convert to NumPy array for batch processing
            flattened_states_batch = np.array(flattened_state).reshape(-1, self.state_size)

            # Predict Q-values from both target networks
            q_values_1_batch = self.q_network_1.predict(flattened_states_batch, verbose=0)
            q_values_2_batch = self.q_network_2.predict(flattened_states_batch, verbose=0)

            # Flatten Q-values
            q_values_1 = q_values_1_batch.flatten()
            q_values_2 = q_values_2_batch.flatten()

            # Sum Q-values for Double DQN
            q_values_sum = q_values_1 + q_values_2
            best_action_index = np.argmax(q_values_sum)

            # Select action with maximum Q-value
            return all_modified_actions[best_action_index]

    def train(self):
            """
            Trains the Q-network using Double DQN (DDQN) update.
            """
            if len(self.memory) < batch_size:
                return
    
            # Sample a batch from memory
            minibatch = self.memory.sample_experiences(batch_size - 2)
            # Add the last two experience to the batch
            minibatch.append(self.memory[-1])
            minibatch.append(self.memory[-2])
    
            flattened_states, target_Q1, target_Q2 = [], [], []
    
            for state, action, reward, next_state, _ in minibatch:
                modified_state = copy.deepcopy(state)
                modified_state[0][0] = np.add(modified_state[0][0], action)
    
                modified_state[0] = flatten_state(modified_state[0])  # Dixie_Shawson
                modified_state[1] = flatten_state(modified_state[1])  # Dixie_Britannia
                modified_state[2] = flatten_state(modified_state[2])  # Dixie_401
                flattened_states.append(np.concatenate([modified_state[0], modified_state[1], modified_state[2]]))
    
                flattened_next_state = []
                for next_action in minibatch:  # Using minibatch to generate possible next states
                    modified_next_state = copy.deepcopy(next_state)
                    modified_next_state[0][0] = np.add(modified_next_state[0][0], next_action[1])  # Apply action
    
                    modified_next_state[0] = flatten_state(modified_next_state[0])
                    modified_next_state[1] = flatten_state(modified_next_state[1])
                    modified_next_state[2] = flatten_state(modified_next_state[2])
                    flattened_next_state.append(np.concatenate([modified_next_state[0], modified_next_state[1], modified_next_state[2]]))
    
                flattened_next_states_batch = np.array(flattened_next_state).reshape(-1, self.state_size)
                next_q_values_1_batch = self.q_network_1.predict(flattened_next_states_batch, verbose=0)
                next_q_values_2_batch = self.q_network_2.predict(flattened_next_states_batch, verbose=0)
    
                next_q_values_1 = next_q_values_1_batch.flatten()
                next_q_values_2 = next_q_values_2_batch.flatten()
    
                next_action_index1 = np.argmax(next_q_values_1)
                next_action_index2 = np.argmax(next_q_values_2)
    
                next_q2_value_based_q1 = next_q_values_2[next_action_index1]
                next_q1_value_based_q2 = next_q_values_1[next_action_index2]
    
                target_Q1.append(reward + gamma * next_q2_value_based_q1)
                target_Q2.append(reward + gamma * next_q1_value_based_q2)
    
            flattened_states_batch = np.array(flattened_states).reshape(-1, self.state_size)
            target_Q1_batch = np.array(target_Q1)
            target_Q2_batch = np.array(target_Q2)
    
            if np.random.rand() < 0.5:
                self.q_network_1.fit(flattened_states_batch, target_Q1_batch, epochs=1, verbose=0)
            else:
                self.q_network_2.fit(flattened_states_batch, target_Q2_batch, epochs=1, verbose=0)
