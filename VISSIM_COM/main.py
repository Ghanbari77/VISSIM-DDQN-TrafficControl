import os
import win32com.client as com
import random
from scipy.optimize import linprog
import numpy as np
import tensorflow as tf
from collections import deque
import pandas as pd
import copy


def Ring_Barrier(splits):
    """
    Uses linear programming to adjust ring-barrier signal timings based on given minimum
    split times (green intervals). The objective is to minimize a weighted sum of green
    times under two equality constraints that ensure certain movements have balanced durations.

    Parameters
    ----------
    splits : list of int or float
        A list of 8 values representing minimum split times for the following movements:
        [0] Min_Split_NBL
        [1] Min_Split_SB
        [2] Min_Split_SBL
        [3] Min_Split_NB
        [4] Min_Split_EBL
        [5] Min_Split_WB
        [6] Min_Split_WBL
        [7] Min_Split_EB
        If any entry is 0, that movement is forced to remain at 0 seconds.
        Otherwise, the movement is bounded below by the provided value with no upper bound.

    Returns
    -------
    result : OptimizeResult
        The result object returned by scipy.optimize.linprog, which contains information
        such as the optimized split times, solver status, etc.
    """
    # Objective function coefficients: these weights correspond to each movement in 'splits'.
    # The goal is to minimize the sum c[0]*x0 + c[1]*x1 + ... + c[7]*x7.
    c = [2, 1, 2, 1, 2, 1, 2, 1]

    # Build the bounds for each movement:
    #   If split == 0, that movement is completely disabled (bounded between 0 and 0).
    #   Otherwise, it's bounded below by split[i] with no upper limit.
    bounds = []
    for split in splits:
        if split == 0:
            bounds.append((0, 0))  # Fix this movement's green time to 0
        else:
            bounds.append((split, None))  # Lower bound = given split, upper bound = None (no limit)

    # Equality constraints:
    #   1) NBL + SB == SBL + NB
    #   2) EBL + WB == WBL + EB
    # This ensures that certain opposing phases are balanced in total duration.
    A_eq = [
        [1, 1, -1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, -1, -1]
    ]
    b_eq = [0, 0]  # Each equality constraint must sum to zero when rearranged

    # Solve the linear programming problem using the 'highs' solver
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    return result


# --- VISSIM Integration ---

# Create a COM object to interface with the Vissim software
Vissim = com.Dispatch("Vissim.Vissim")

# Specify the paths for the Vissim network file (.inpx) and layout file (.layx)
filename = "C:/Users/qanba.TRANSLAB/Desktop/VISSIM/Vissim.inpx"
layoutfile = "C:/Users/qanba.TRANSLAB/Desktop/VISSIM/Vissim.layx"

# Load the Vissim network
# Note: 'flag_read_additionally' indicates whether to merge the new network with an open network
Filename = os.path.join(os.path.abspath(os.getcwd()), filename)
flag_read_additionally = False
Vissim.LoadNet(Filename, flag_read_additionally)

# Load the associated layout file (which defines how the network is visually displayed)
Vissim.LoadLayout(layoutfile)

# Access the network object and its signal controllers
net = Vissim.Net
scs = net.SignalControllers


def Dixie_401_Signal(Sim_time, offset, Min_green_SB_NB, Min_green_WB, End_of_Last_Cycle_Time):
    """
    Controls the traffic signal states for the Dixie & 401 intersection using Vissim SignalControllers.

    Parameters
    ----------
    Sim_time : int or float
        The current simulation time in seconds.
    offset : int or float
        Offset time to synchronize this signal with other intersections.
    Min_green_SB_NB : int or float
        The minimum green duration (in seconds) for the Southbound/Northbound (SB_NB) approach.
    Min_green_WB : int or float
        The minimum green duration (in seconds) for the Westbound (WB) approach.
    End_of_Last_Cycle_Time : int or float
        The simulation time at which the previous cycle ended. If 0, it indicates the first cycle.

    Returns
    -------
    End_of_Last_Cycle_Time : int or float
        Updated simulation time at which the current cycle ends.
    Local_time : int or float
        Time within the current cycle (i.e., how many seconds have elapsed since the last cycle ended).

    Notes
    -----
    - SigState values correspond to:
        1 -> Red,
        3 -> Green,
        4 -> Yellow.
    - If End_of_Last_Cycle_Time is zero, we apply an offset correction at the beginning of the simulation.
    """
    # Retrieve Dixie and 401 signal controller from the global SignalControllers (scs).
    Dixie_401 = scs.ItemByKey(1)

    # Access individual signal groups for southbound/northbound (Key=1) and westbound (Key=2).
    SB_NB = Dixie_401.SGs.ItemByKey(1)
    WB = Dixie_401.SGs.ItemByKey(2)

    # Initialize both approaches to Red.
    SB_NB.SetAttValue("SigState", 1)
    WB.SetAttValue("SigState", 1)

    # --- Handle first cycle offset ---
    # If this is the very first cycle (End_of_Last_Cycle_Time == 0), adjust the current simulation time to incorporate the offset.
    if End_of_Last_Cycle_Time == 0:
        Sim_time = Sim_time + offset - 1

    # --- Compute Local_time (time since last cycle started) ---
    # If we are exactly at the end of the last cycle, reset local time to zero.
    # Otherwise, local time is how many seconds have passed since the last cycle ended.
    if Sim_time == End_of_Last_Cycle_Time and Sim_time != 0:
        Local_time = 0
    else:
        Local_time = Sim_time - End_of_Last_Cycle_Time

    # --- Set Signal States by Time Intervals ---
    # Below conditions define when SB_NB or WB signals turn green or yellow based on the local time and minimum green durations.

    # Red clearance time of WB = 3 seconds
    if 3 <= Local_time < Min_green_SB_NB + 3:
        SB_NB.SetAttValue("SigState", 3)  # Green
    # Yellow time of SB_NB = 4 seconds
    elif Min_green_SB_NB + 3 <= Local_time < Min_green_SB_NB + 7:
        SB_NB.SetAttValue("SigState", 4)  # Yellow
    # Red clearance time of SB_NB = 4 seconds
    elif Min_green_SB_NB + 11 <= Local_time < (Min_green_SB_NB + Min_green_WB + 11):
        WB.SetAttValue("SigState", 3)  # Green
    # Yellow time of WB = 4 seconds
    elif (Min_green_SB_NB + Min_green_WB + 11) <= Local_time < (Min_green_SB_NB + Min_green_WB + 15):
        WB.SetAttValue("SigState", 4)  # Yellow

    # --- End of Cycle Update ---
    # If Local_time matches the end of both SB_NB and WB green+yellow intervals, the cycle ends.
    # Then, update End_of_Last_Cycle_Time to mark the new end of cycle.
    if Local_time == (Min_green_SB_NB + Min_green_WB + 14):
        # For the first cycle, reverse the initial offset adjustment once it ends.
        if End_of_Last_Cycle_Time == 0:
            Sim_time = Sim_time - offset + 1
        End_of_Last_Cycle_Time = Sim_time + 1

    return End_of_Last_Cycle_Time, Local_time


def Dixie_Shawson_Signal(
    Sim_time,
    Last_Barrier_Time,
    Ring_1,
    RBC_Function_Result,
    offset,
    Initial_Splits,
    End_of_Last_Cycle_Time,
    change
):
    """
    Controls signal phases at the Dixie-Shawson intersection based on a ring-barrier approach.
    Updates signal group (SG) states by considering detector presence, timing offsets,
    ring-phase transitions, and optimized split results from the Ring_Barrier function.

    Parameters
    ----------
    Sim_time : float
        Current simulation time (in seconds).
    Last_Barrier_Time : float
        The simulation time when the previous ring barrier finished.
    Ring_1 : bool
        A flag indicating which ring is currently active: True if Ring 1, False if Ring 2.
    RBC_Function_Result : OptimizeResult
        The result of the ring-barrier calculation (from Ring_Barrier), containing
        optimized split times in RBC_Function_Result.x.
    offset : float
        An offset for synchronizing the start time of this signal (applied if first cycle).
    Initial_Splits : list of float
        List of initial green split times for up to 8 signal phases or movements.
    End_of_Last_Cycle_Time : float
        The recorded simulation time when the last ring cycle began (used for logging or analysis).
    change : bool
        Indicates whether splits have been updated and a new ring-barrier calculation
        needs to be performed mid-cycle.

    Returns
    -------
    Last_Barrier_Time : float
        Updated simulation time when the last ring barrier finishes.
    Ring_1 : bool
        Flipped if the cycle transitions from Ring 1 to Ring 2 or vice versa.
    RBC_Function_Result : OptimizeResult
        Updated ring-barrier result.
    End_of_Last_Cycle_Time : float
        Updated cycle start time.
    current_phase : list of int
        A list of length 8 indicating which phases are currently green (1) or off (0).
    Local_time : float
         Time within the current ring (i.e., how many seconds have elapsed since the last ring ended).
    """

    # Make a local copy of Initial_Splits so we don't mutate the original list.
    splits = Initial_Splits.copy()

    # Retrieve the signal controller for the Dixie-Shawson intersection (Controller Key=2).
    Dixie_Shawson = scs.ItemByKey(2)

    # Access the individual signal groups (SGs) controlling each movement (SG keys 1–8).
    NBL = Dixie_Shawson.SGs.ItemByKey(1)
    SB  = Dixie_Shawson.SGs.ItemByKey(2)
    WB  = Dixie_Shawson.SGs.ItemByKey(4)
    SBL = Dixie_Shawson.SGs.ItemByKey(5)
    NB  = Dixie_Shawson.SGs.ItemByKey(6)
    WBL = Dixie_Shawson.SGs.ItemByKey(7)
    EB  = Dixie_Shawson.SGs.ItemByKey(8)

    # Initialize all signals to Red (SigState=1).
    NBL.SetAttValue("SigState", 1)
    SB.SetAttValue("SigState", 1)
    WB.SetAttValue("SigState", 1)
    SBL.SetAttValue("SigState", 1)
    NB.SetAttValue("SigState", 1)
    WBL.SetAttValue("SigState", 1)
    EB.SetAttValue("SigState", 1)

    # current_phase[i] = 1 if movement i is currently green; otherwise 0.
    current_phase = [0] * 8

    # --- Adjust for first cycle offset ---
    # If no barrier cycle has occurred yet (End_of_Last_Barrier_Time == 0), apply the offset.
    if Last_Barrier_Time == 0:
        End_of_Last_Cycle_Time = End_of_Last_Cycle_Time + offset - 1

    # --- Cycle boundary checks ---
    # If the simulation time matches the end of the last barrier, check detectors to see if
    # any approach has zero vehicles. If so, set its split to 0, then recalculate RBC_Function_Result.
    if Sim_time == Last_Barrier_Time and Sim_time != 0:
        if net.Detectors.ItemByKey(50).AttValue("Presence") == 0:
            splits[0] = 0   # NBL movement
        if net.Detectors.ItemByKey(51).AttValue("Presence") == 0:
            splits[6] = 0   # WBL movement
        if net.Detectors.ItemByKey(53).AttValue("Presence") == 0:
            splits[2] = 0   # SBL movement

        # Recompute optimized splits based on any zeros now inserted.
        RBC_Function_Result = Ring_Barrier(splits)

        # Since we are at the start of a new ring, local time resets to 0.
        Local_time = 0

        # If we are entering Ring 1, change the start time of the new cycle.
        if Ring_1:
            End_of_Last_Cycle_Time = Sim_time


    else:
        # If a new ring haven't been started, find how long we've been in the current ring.
        Local_time = Sim_time - Last_Barrier_Time

    # --- Mid-cycle split changes ---
    # If 'change' is True, the RBC_Function_Result might need recalculation again here.
    if change:
        for i in range(8):
            if RBC_Function_Result.x[i] == 0:
                splits[i] = 0
        RBC_Function_Result = Ring_Barrier(splits)

    # --- Ring 1 Logic ---
    # Movements: NBL, SB, SBL, NB (indices in RBC_Function_Result.x are 0, 1, 2, 3).
    if Ring_1:
        # Safety/truncation: Ensure after changes in the mid-cycle the Local_time doesn't exceed the current ring time.
        if Local_time > RBC_Function_Result.x[0] + RBC_Function_Result.x[1] - 1:
            Local_time = RBC_Function_Result.x[0] + RBC_Function_Result.x[1] - 1

        # NBL: Green then Yellow
        # Yellow time of NBL = 3
        # Red Clearance time of NBL = 0
        if Local_time < RBC_Function_Result.x[0] - 3:
            NBL.SetAttValue("SigState", 3)  # Green
            current_phase[0] = 1
        elif RBC_Function_Result.x[0] - 3 <= Local_time < RBC_Function_Result.x[0]:
            NBL.SetAttValue("SigState", 4)  # Yellow

        # SB: Green then Yellow
        # Yellow time of SB = 4
        # Red Clearance time of SB = 3
        if RBC_Function_Result.x[0] <= Local_time < (RBC_Function_Result.x[0] + RBC_Function_Result.x[1] - 7):
            SB.SetAttValue("SigState", 3)   # Green
            current_phase[1] = 1
        elif (RBC_Function_Result.x[0] + RBC_Function_Result.x[1] - 7) <= Local_time < (RBC_Function_Result.x[0] + RBC_Function_Result.x[1] - 3):
            SB.SetAttValue("SigState", 4)   # Yellow

        # SBL: Green then Yellow
        # Yellow time of SBL = 3
        # Red Clearance time of SBL = 0
        if Local_time < RBC_Function_Result.x[2] - 3:
            SBL.SetAttValue("SigState", 3)  # Green
            current_phase[2] = 1
        elif RBC_Function_Result.x[2] - 3 <= Local_time < RBC_Function_Result.x[2]:
            SBL.SetAttValue("SigState", 4)  # Yellow

        # NB: Green then Yellow
        # Yellow time of NB = 4
        # Red Clearance time of NB = 3
        if RBC_Function_Result.x[2] <= Local_time < (RBC_Function_Result.x[2] + RBC_Function_Result.x[3] - 7):
            NB.SetAttValue("SigState", 3)   # Green
            current_phase[3] = 1
        elif (RBC_Function_Result.x[2] + RBC_Function_Result.x[3] - 7) <= Local_time < (RBC_Function_Result.x[2] + RBC_Function_Result.x[3] - 3):
            NB.SetAttValue("SigState", 4)   # Yellow

        # If we hit the end of Ring 1’s cycle, set a new barrier time and switch to Ring 2.
        if Local_time == (RBC_Function_Result.x[0] + RBC_Function_Result.x[1] - 1):
            Last_Barrier_Time = Sim_time + 1
            Ring_1 = False

    # --- Ring 2 Logic ---
    # Movements: (WB, WBL, EB). Indices in RBC_Function_Result.x are 5, 6, 7.
    else:
        # Safety/truncation: Ensure after changes in the mid-cycle the Local_time doesn't exceed the current ring time.
        if Local_time > RBC_Function_Result.x[5] - 1:
            Local_time = RBC_Function_Result.x[5] - 1

        # WB: Green then Yellow
        # Yellow time of WB = 4
        # Red Clearance time of WB = 4
        if Local_time < (RBC_Function_Result.x[5] - 8):
            WB.SetAttValue("SigState", 3)
            current_phase[5] = 1
        elif (RBC_Function_Result.x[5] - 8) <= Local_time < (RBC_Function_Result.x[5] - 4):
            WB.SetAttValue("SigState", 4)

        # WBL: Green then Yellow
        # Yellow time of WBL = 3
        # Red Clearance time of WBL = 2
        if Local_time < (RBC_Function_Result.x[6] - 5):
            WBL.SetAttValue("SigState", 3)
            current_phase[6] = 1
        elif (RBC_Function_Result.x[6] - 5) <= Local_time < (RBC_Function_Result.x[6] - 2):
            WBL.SetAttValue("SigState", 4)

        # EB: Green then Yellow
        # Yellow time of EB = 4
        # Red Clearance time of EB = 4
        if RBC_Function_Result.x[6] <= Local_time < (RBC_Function_Result.x[6] + RBC_Function_Result.x[7] - 8):
            EB.SetAttValue("SigState", 3)
            current_phase[7] = 1
        elif (RBC_Function_Result.x[6] + RBC_Function_Result.x[7] - 8) <= Local_time < (RBC_Function_Result.x[6] + RBC_Function_Result.x[7] - 4):
            EB.SetAttValue("SigState", 4)

        # If we hit the end of Ring 2’s cycle, update barrier time and return to Ring 1.
        if Local_time == RBC_Function_Result.x[5] - 1:
            Last_Barrier_Time = Sim_time + 1
            Ring_1 = True

    return (
        Last_Barrier_Time,
        Ring_1,
        RBC_Function_Result,
        End_of_Last_Cycle_Time,
        current_phase,
        Local_time
    )

# --- Deep Learning Hyperparameters ---
state_size_Dixie_Shawson = 57
state_size_Dixie_401 = 33
state_size_Dixie_Britannia = 57
state_size = state_size_Dixie_Shawson + state_size_Dixie_401 + state_size_Dixie_Britannia

action_size = 7
action_values = [-5, 0, 5]

learning_rate = 0.001
batch_size = 64
max_memory_size = 15000
gamma = 0.99 # Discount factor for future rewards

# Exploration Parameters
epsilon = 1
epsilon_decay = 0.9995
epsilon_min = 0.01



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


# Create two Q-network models for Double DQN (DDQN)
q_network_1 = build_q_network()
q_network_2 = build_q_network()


# Initialize experience replay memory
memory_Dixie_Shawson = deque(maxlen=max_memory_size)


def store_experience(memory, state, action, reward, next_state, episode_Number):
    """Stores an experience tuple in the replay memory."""
    memory.append((state, action, reward, next_state, episode_Number))



def flatten_state(state):
    """ Flattens the state representation for neural network input. """
    # Concatenate all sub-vectors from state[0], state[1], and state[2]
    return np.concatenate(
        [state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7], state[8], state[9], state[10],
         state[11]])  # state[0] = split time of different phases; state[1] = current phase; state[2] = local time; State[3] = queue length of different phases; state[4:8] = flow; state[8:12] = Truck Percentage


def flatten_list(nested_list):
    """Flattens a nested list into a single list."""
    flat_list = []
    for element in nested_list:
        if isinstance(element, list):
            flat_list.extend(flatten_list(element))
        else:
            flat_list.append(element)
    return flat_list



def generate_all_actions():
    """Generates all possible actions based on action size and values."""
    return np.array(np.meshgrid(*[action_values] * action_size)).T.reshape(-1, action_size)


def select_action(state, epsilon, all_modified_actions):
    """ Selects an action using the epsilon-greedy policy. """
    if np.random.rand(1) <= epsilon:
        return all_modified_actions[np.random.randint(len(all_modified_actions))]

    else:
        flattened_state = []
        q_values_sum_batch = []
        for action in all_modified_actions:
            modified_state = copy.deepcopy(state)
            # action impact on green time
            modified_state[0][0] = np.add(modified_state[0][0], action)

            # Flatten the state
            modified_state[0] = flatten_state(modified_state[0])  # Dixie_Shawson
            modified_state[1] = flatten_state(modified_state[1])  # Dixie_Britannia
            modified_state[2] = flatten_state(modified_state[2])  # Sixie_401
            flattened_state.append(np.concatenate([modified_state[0], modified_state[1], modified_state[2]]))

        # Convert the list to a NumPy array of shape (batch_size, state_size)
        flattened_states_batch = np.array(flattened_state).reshape(-1, state_size)

        # Batch predict Q-values from both target networks
        q_values_1_batch = q_network_1.predict(flattened_states_batch, verbose=0)
        q_values_2_batch = q_network_2.predict(flattened_states_batch, verbose=0)

        # Extract the Q-values for each modified states
        # Since the model output is typically of shape (batch_size, 1), we need to flatten it
        q_values_1 = q_values_1_batch.flatten()
        q_values_2 = q_values_2_batch.flatten()

        q_values_sum = q_values_1 + q_values_2
        best_action_index = np.argmax(q_values_sum)

        # Select the action with the maximum combined Q-value
        return all_modified_actions[best_action_index]



def train_q_network():
    """Trains the Q-network using Double DQN (DDQN) update."""
    # Sample a batch from memory
    minibatch = random.sample(memory_Dixie_Shawson, batch_size - 2)
    minibatch.append(memory_Dixie_Shawson[-1])
    minibatch.append(memory_Dixie_Shawson[-2])

    # process the batch
    flattened_states = []
    target_Q1 = []
    target_Q2 = []
    for state, action, reward, next_state, episode in minibatch:

        # Create a modified state by applying the action to the state
        modified_state = copy.deepcopy(state)

        modified_state[0][0] = np.add(modified_state[0][0], action)

        # Flatten the state

        modified_state[0] = flatten_state(modified_state[0])  # Dixie_Shawson
        modified_state[1] = flatten_state(modified_state[1])  # Dixie_Britannia
        modified_state[2] = flatten_state(modified_state[2])  # Dixie_401
        flattened_states.append(np.concatenate([modified_state[0], modified_state[1], modified_state[2]]))

        # Target Q-value
        flattened_next_state = []
        for next_action in all_modified_actions:
            modified_next_state = copy.deepcopy(next_state)
            # action impact on green time
            modified_next_state[0][0] = np.add(modified_next_state[0][0], next_action)

            # Flatten the state
            modified_next_state[0] = flatten_state(modified_next_state[0])  # Dixie_Shawson
            modified_next_state[1] = flatten_state(modified_next_state[1])  # Dixie_Britannia
            modified_next_state[2] = flatten_state(modified_next_state[2])  # Sixie_401
            flattened_next_state.append(np.concatenate([modified_next_state[0], modified_next_state[1], modified_next_state[2]]))

        # Convert the list to a NumPy array of shape (batch_size, state_size)
        flattened_next_states_batch = np.array(flattened_next_state).reshape(-1, state_size)
        next_q_values_1_batch = q_network_1.predict(flattened_next_states_batch, verbose=0)
        next_q_values_2_batch = q_network_2.predict(flattened_next_states_batch, verbose=0)

        next_q_values_1 = next_q_values_1_batch.flatten()
        next_q_values_2 = next_q_values_2_batch.flatten()

        next_action_index1 = np.argmax(next_q_values_1)
        next_action_index2 = np.argmax(next_q_values_2)

        next_q2_value_based_q1 = next_q_values_2[next_action_index1]
        next_q1_value_based_q2 = next_q_values_1[next_action_index2]
        # Calculate the target Q-value
        target_for_Q1 = reward
        target_for_Q2 = reward
        target_for_Q1 += gamma * next_q2_value_based_q1
        target_for_Q2 += gamma * next_q1_value_based_q2

        target_Q1.append(target_for_Q1)
        target_Q2.append(target_for_Q2)

    # Convert the list to a NumPy array of shape (batch_size, state_size)
    flattened_states_batch = np.array(flattened_states).reshape(-1, state_size)
    target_Q1_batch = np.array(target_Q1)
    target_Q2_batch = np.array(target_Q2)

    # Update either q_network_1 or q_network_2 with 50% probability
    if np.random.rand() < 0.5:
        q_network_1.fit(flattened_states_batch, target_Q1_batch, epochs=1, verbose=0)
    else:
        q_network_2.fit(flattened_states_batch, target_Q2_batch, epochs=1, verbose=0)



def save_data_to_csv(filename, data_records, append=False):
    """Saves collected data to a CSV file."""
    # Define column names: states, actions, reward
    column_names = [f'state_{i}' for i in range(state_size)] + \
                   [f'action_{i}' for i in range(8)] + \
                   ['reward'] + \
                   ['episode']

    # Convert the collected records to a DataFrame
    df = pd.DataFrame([data_records], columns=column_names)

    # Save to CSV
    if append:
        # Append to the existing CSV file
        try:
            df_existing = pd.read_csv(filename)
            df = pd.concat([df_existing, df], ignore_index=True)
        except FileNotFoundError:
            # If the file doesn't exist, just save the current DataFrame
            pass

    df.to_csv(filename, index=False)

# Main Loop for Training
Num_episodes = 200

# load the weights from the last run
#q_network_1.load_weights(f"C:/Users/qanba.TRANSLAB/Desktop/Weights/Result14/q_network_1_episode_{153}_step_{60}.weights.h5")
#q_network_2.load_weights(f"C:/Users/qanba.TRANSLAB/Desktop/Weights/Result14/q_network_2_episode_{153}_step_{60}.weights.h5")



# Start Episodes
for episode in range(Num_episodes):
    # Start simulation
    time_step = 0

    all_states_Dixie_Shawson = []
    all_states_Dixie_Britannia = []
    all_states_Dixie_401 = []

    final_state_Dixie_Shawson_batch = []

    all_actions = []
    VehDelay = []
    TruckDelay = []
    VehDelay_160_Second = []
    TruckDelay_160_Second = []

    Neighbour1Delay = []
    Neighbour2Delay = []
    Neighbour1_160_Delay = []
    Neighbour2_160_Delay = []

    VehInput_Dixie_Shawson = []
    VehInput_Dixie_Britannia = []
    VehInput_Dixie_401 = []

    TruckPercentage_Dixie_Shawson = []
    TruckPercentage_Dixie_Britannia = []
    TruckPercentage_Dixie_401 = []

    # Dixie & 401
    offset_Dixie_401 = 129
    Min_green_SB_NB_Dixie_401 = 100
    Min_green_WB_Dixie_401 = 45
    End_of_Last_Cycle_Time_Dixie_401 = 0
    split_Dixie_401 = [Min_green_SB_NB_Dixie_401, Min_green_WB_Dixie_401]

    # Dixie & Shawson
    End_of_Last_Barrier_Time_Dixie_Shawson = 0
    offset_Dixie_Shawson = 1
    Start_time_Dixie_Shawson = 0
    spilits_Dixie_Shawson = [14, 87, 10, 87, 0, 52, 31, 23]
    result_Dixie_Shawson = Ring_Barrier(spilits_Dixie_Shawson)
    Ring1_Dixie_Shawson = True
    if offset_Dixie_Shawson >= result_Dixie_Shawson.x[0] + result_Dixie_Shawson.x[1]:
        Ring1_Dixie_Shawson = False
        offset_Dixie_Shawson = offset_Dixie_Shawson - result_Dixie_Shawson.x[0] - result_Dixie_Shawson.x[1]
    change_Dixie_Shawson = False

    # Dixie & Britannia
    End_of_Last_Barrier_Time_Dixie_Britannia = 0
    offset_Dixie_Britannia = 1
    Start_time_Dixie_Britannia = 0
    spilits_Dixie_Britannia = [14, 82, 11, 85, 0, 58, 21, 43]
    result_Dixie_Britannia = Ring_Barrier(spilits_Dixie_Britannia)
    Ring1_Dixie_Britannia = True
    if offset_Dixie_Britannia >= result_Dixie_Britannia.x[0] + result_Dixie_Britannia.x[1]:
        Ring1_Dixie_Britannia = False
        offset_Dixie_Britannia = offset_Dixie_Britannia - result_Dixie_Britannia.x[0] - result_Dixie_Britannia.x[1]

    # Britannia & Shawson
    End_of_Last_Cycle_Britannia_Shawson = 0
    offset_Britannia_Shawson = 1
    Start_time_Britannia_Shawson = 0
    Ring1_Britannia_Shawson = True
    phase_switch_Britannia_Shawson = False
    Min_green_EW_Britannia_Shawson = 42
    Split_EW_Britannia_Shawson = 49
    Max_green_EW_Britannia_Shawson = 25
    Min_green_NS_Britannia_Shawson = 26
    Split_NS_Britannia_Shawson = 33
    if offset_Britannia_Shawson >= Min_green_EW_Britannia_Shawson:
        Ring1_Britannia_Shawson = False

    # Create action matrix and modify to have the zero value in the same index of split vector
    all_possible_actions = generate_all_actions()

    zero_indices = [i for i, value in enumerate(spilits_Dixie_Shawson) if value == 0]

    # Modify each action to match the size of spilits_Dixie_Shawson, inserting zeros at the appropriate indices
    all_modified_actions = []

    for action in all_possible_actions:
        modified_action = list(action)
        for index in zero_indices:
            modified_action.insert(index, 0)
        all_modified_actions.append(modified_action)
    # Convert modified_actions to a NumPy array
    all_modified_actions = np.array(all_modified_actions)

    for i in range(1, 5701):
        Sim_time = i  # simulation second [s]
        Vissim.Simulation.SetAttValue('SimBreakAt', Sim_time)
        Vissim.Simulation.RunContinuous()  # start the simulation until SimBreakAt

        # End_of_Last_Cycle_Time_Dixie_401, current_phase_Dixie_401, Local_time_Dixie_401  = Dixie_401_Signal(Sim_time, offset_Dixie_401, Min_green_SB_NB_Dixie_401, Min_green_WB_Dixie_401, End_of_Last_Cycle_Time_Dixie_401)
        End_of_Last_Barrier_Time_Dixie_Shawson, Ring1_Dixie_Shawson, result_Dixie_Shawson, Start_time_Dixie_Shawson, current_phase_Dixie_Shawson, Local_time_Dixie_Shawson = Dixie_Shawson_Signal(
            Sim_time, End_of_Last_Barrier_Time_Dixie_Shawson, Ring1_Dixie_Shawson, result_Dixie_Shawson,
            offset_Dixie_Shawson, spilits_Dixie_Shawson, Start_time_Dixie_Shawson, change_Dixie_Shawson)
        # End_of_Last_Barrier_Time_Dixie_Britannia, Ring1_Dixie_Britannia, result_Dixie_Britannia, Start_time_Dixie_Britannia, current_phase_Dixie_Britannia, Local_time_Dixie_Britannia = Dixie_Britannia_Signal(Sim_time, End_of_Last_Barrier_Time_Dixie_Britannia, Ring1_Dixie_Britannia, result_Dixie_Britannia, offset_Dixie_Britannia, spilits_Dixie_Britannia, Start_time_Dixie_Britannia)

        # Observe Vehicle Flow and Truck Percentage
        if i % 20 == 0:
            F = []
            T = []
            for j in range(2, 6):
                Flow = net.DataCollectionMeasurements.ItemByKey(j).AttValue('Vehs(Current,Last,All)')
                Truck = net.DataCollectionMeasurements.ItemByKey(j).AttValue('Vehs(Current,Last,20)')
                if type(Flow) != int:
                    Flow = 0
                if type(Truck) != int:
                    Truck = 0
                F.append(Flow)
                T.append(Truck)
            VehInput_Dixie_Shawson.append(F)
            TruckPercentage_Dixie_Shawson.append(T)

            F = []
            T = []
            for j in range(6, 9):
                Flow = net.DataCollectionMeasurements.ItemByKey(j).AttValue('Vehs(Current,Last,All)')
                Truck = net.DataCollectionMeasurements.ItemByKey(j).AttValue('Vehs(Current,Last,20)')
                if type(Flow) != int:
                    Flow = 0
                if type(Truck) != int:
                    Truck = 0
                F.append(Flow)
                T.append(Truck)
            VehInput_Dixie_401.append(F)
            TruckPercentage_Dixie_401.append(T)

            F = []
            T = []
            for j in range(9, 13):
                Flow = net.DataCollectionMeasurements.ItemByKey(j).AttValue('Vehs(Current,Last,All)')
                Truck = net.DataCollectionMeasurements.ItemByKey(j).AttValue('Vehs(Current,Last,20)')
                if type(Flow) != int:
                    Flow = 0
                if type(Truck) != int:
                    Truck = 0
                F.append(Flow)
                T.append(Truck)
            VehInput_Dixie_Britannia.append(F)
            TruckPercentage_Dixie_Britannia.append(T)

        # Training Part
        if i % 80 == 0:  # defininfg warm-up and time steps
            # Observe State
            time_step += 1

            State_Dixie_Shawson = []
            State_Dixie_Britannia = []
            State_Dixie_401 = []

            final_state_Dixie_Shawson = []

            # Take the signal time data of neighbour intersections
            # Dixie Britannia
            current_phase_Dixie_Britannia = []
            for j in range(1, 9):
                if j == 3:
                    current_phase_Dixie_Shawson.append(0)
                else:
                    if scs.ItemByKey(3).SGs.ItemByKey(j).AttValue('SigState') == 'GREEN':
                        current_phase_Dixie_Shawson.append(1)
                    else:
                        current_phase_Dixie_Shawson.append(0)

            Local_time_Dixie_Britannia = scs.ItemByKey(3).SGs.ItemByKey(2).AttValue('tSigState')

            # Dixie 401
            current_phase_Dixie_401 = []
            for j in range(1, 3):
                if scs.ItemByKey(1).SGs.ItemByKey(j).AttValue('SigState') == 'GREEN':
                    current_phase_Dixie_Shawson.append(1)
                else:
                    current_phase_Dixie_Shawson.append(0)

            Local_time_Dixie_401 = scs.ItemByKey(1).SGs.ItemByKey(1).AttValue('tSigState')

            State_Dixie_Shawson.append(spilits_Dixie_Shawson)
            State_Dixie_Shawson.append(current_phase_Dixie_Shawson)
            State_Dixie_Shawson.append([Local_time_Dixie_Shawson])

            State_Dixie_Britannia.append(spilits_Dixie_Britannia)
            State_Dixie_Britannia.append(current_phase_Dixie_Britannia)
            State_Dixie_Britannia.append([Local_time_Dixie_Britannia])

            State_Dixie_401.append(split_Dixie_401)
            State_Dixie_401.append(current_phase_Dixie_401)
            State_Dixie_401.append([Local_time_Dixie_401])

            Queue_Length_Dixie_Shawson = []
            Queue_Length_Dixie_Britannia = []
            Queue_Length_Dixie_401 = []

            for j in range(1, 9):
                QueueCounter = net.QueueCounters.ItemByKey(j).AttValue('QLen(Current,Last)')
                if type(QueueCounter) != float:
                    QueueCounter = 0
                Queue_Length_Dixie_Shawson.append(QueueCounter)
            State_Dixie_Shawson.append(Queue_Length_Dixie_Shawson)

            for j in range(13, 21):
                QueueCounter = net.QueueCounters.ItemByKey(j).AttValue('QLen(Current,Last)')
                if type(QueueCounter) != float:
                    QueueCounter = 0
                Queue_Length_Dixie_Britannia.append(QueueCounter)
            State_Dixie_Britannia.append(Queue_Length_Dixie_Britannia)

            for j in range(9, 13):
                QueueCounter = net.QueueCounters.ItemByKey(j).AttValue('QLen(Current,Last)')
                if type(QueueCounter) != float:
                    QueueCounter = 0
                Queue_Length_Dixie_401.append(QueueCounter)
            State_Dixie_401.append(Queue_Length_Dixie_401)

            State_Dixie_Shawson.append(VehInput_Dixie_Shawson[-1])
            State_Dixie_Shawson.append(VehInput_Dixie_Shawson[-2])
            State_Dixie_Shawson.append(VehInput_Dixie_Shawson[-3])
            State_Dixie_Shawson.append(VehInput_Dixie_Shawson[-4])
            State_Dixie_Shawson.append(TruckPercentage_Dixie_Shawson[-1])
            State_Dixie_Shawson.append(TruckPercentage_Dixie_Shawson[-2])
            State_Dixie_Shawson.append(TruckPercentage_Dixie_Shawson[-3])
            State_Dixie_Shawson.append(TruckPercentage_Dixie_Shawson[-4])

            State_Dixie_Britannia.append(VehInput_Dixie_Britannia[-1])
            State_Dixie_Britannia.append(VehInput_Dixie_Britannia[-2])
            State_Dixie_Britannia.append(VehInput_Dixie_Britannia[-3])
            State_Dixie_Britannia.append(VehInput_Dixie_Britannia[-4])
            State_Dixie_Britannia.append(TruckPercentage_Dixie_Britannia[-1])
            State_Dixie_Britannia.append(TruckPercentage_Dixie_Britannia[-2])
            State_Dixie_Britannia.append(TruckPercentage_Dixie_Britannia[-3])
            State_Dixie_Britannia.append(TruckPercentage_Dixie_Britannia[-4])

            State_Dixie_401.append(VehInput_Dixie_401[-1])
            State_Dixie_401.append(VehInput_Dixie_401[-2])
            State_Dixie_401.append(VehInput_Dixie_401[-3])
            State_Dixie_401.append(VehInput_Dixie_401[-4])
            State_Dixie_401.append(TruckPercentage_Dixie_401[-1])
            State_Dixie_401.append(TruckPercentage_Dixie_401[-2])
            State_Dixie_401.append(TruckPercentage_Dixie_401[-3])
            State_Dixie_401.append(TruckPercentage_Dixie_401[-4])

            all_states_Dixie_Shawson.append(State_Dixie_Shawson)
            all_states_Dixie_Britannia.append(State_Dixie_Britannia)
            all_states_Dixie_401.append(State_Dixie_401)

            final_state_Dixie_Shawson = [State_Dixie_Shawson, State_Dixie_Britannia, State_Dixie_401]
            final_state_Dixie_Shawson_batch.append(final_state_Dixie_Shawson)

            # calculate action
            current_action = select_action(final_state_Dixie_Shawson, epsilon, all_modified_actions)

            next_split_time = np.add(final_state_Dixie_Shawson[0][0], current_action)

            # Check requirements
            # 7 second green as a min green for each phase
            for index in range(0, 8):
                if index == 0 or index == 2 or index == 4:
                    if next_split_time[index] != 0 and next_split_time[index] < 10:
                        next_split_time[index] = 10

                if index == 6:  # WBL
                    if next_split_time[index] < 12:
                        next_split_time[index] = 12

                if index == 1 or index == 3 or index == 5 or index == 7:
                    if next_split_time[index] != 0 and next_split_time[index] < 15:
                        next_split_time[index] = 15
            # Restriction for cycle time
            next_cycle_time = sum(Ring_Barrier(next_split_time).x)
            if next_cycle_time < 120 * 2 or next_cycle_time > 170 * 2:
                next_split_time = final_state_Dixie_Shawson[0][0]

            modified_current_action = np.subtract(next_split_time, final_state_Dixie_Shawson[0][0])

            all_actions.append(modified_current_action.tolist())

            # Apply Action
            spilits_Dixie_Shawson = list(next_split_time)
            change_Dixie_Shawson = True

            # Observe Reward
            current_delay = net.Nodes.ItemByKey(2).TotRes.AttValue('VehDelay(Current,Last,All)')
            truck_delay = net.Nodes.ItemByKey(2).TotRes.AttValue('VehDelay(Current,Last,20)')
            if type(current_delay) != float:
                current_delay = 0
            if type(truck_delay) != float:
                truck_delay = 0

            VehDelay.append(current_delay)
            TruckDelay.append(truck_delay)


            #Observe Neighbour Delays
            neighbour1_delay = net.Nodes.ItemByKey(1).TotRes.AttValue('VehDelay(Current,Last,All)')  # 401 & Dixie intersection
            neighbour2_delay = net.Nodes.ItemByKey(3).TotRes.AttValue('VehDelay(Current,Last,All)')  # Dixie & Britannia intersection
            if type(neighbour1_delay) != float:
                neighbour1_delay = 0
            if type(neighbour2_delay) != float:
                neighbour2_delay = 0

            Neighbour1Delay.append(neighbour1_delay)
            Neighbour2Delay.append(neighbour2_delay)



            if time_step >= 3:
                reward_last_2_steps = -VehDelay[time_step - 1] - VehDelay[time_step - 2]

                # store the experience of last step
                store_experience(memory_Dixie_Shawson, final_state_Dixie_Shawson_batch[time_step - 3],
                                 all_actions[time_step - 3], reward_last_2_steps,
                                 final_state_Dixie_Shawson_batch[time_step - 2], episode)
                #Save Data
                Record_Dixie_Shawson = [final_state_Dixie_Shawson_batch[time_step - 3],
                                 all_actions[time_step - 3], reward_last_2_steps, episode]
                Record_Dixie_Shawson = flatten_list(Record_Dixie_Shawson)
                save_data_to_csv('C:/Users/qanba.TRANSLAB/Desktop/Result14.csv',Record_Dixie_Shawson, append = True)

            #train NN
            if episode > 0:
                train_q_network()
                #Decay exploration rate
                if epsilon > epsilon_min:
                    epsilon *= epsilon_decay

        #Save Q-network weights each 10 steps
        if time_step % 20 == 0 and time_step > 0:
            q_network_1.save_weights(f"C:/Users/qanba.TRANSLAB/Desktop/Weights/Result14/q_network_1_episode_{episode}_step_{time_step}.weights.h5")
            q_network_2.save_weights(f"C:/Users/qanba.TRANSLAB/Desktop/Weights/Result14/q_network_2_episode_{episode}_step_{time_step}.weights.h5")

        if time_step % 2 == 0 and time_step > 0:
            VehDelay_160_Second.append(VehDelay[-1] + VehDelay[-2])
            TruckDelay_160_Second.append(TruckDelay[-1] + TruckDelay[-2])
            Neighbour1_160_Delay.append(Neighbour1Delay[-1] + Neighbour1Delay[-2])
            Neighbour2_160_Delay.append(Neighbour2Delay[-1] + Neighbour2Delay[-2])

    # remove part of memory
    if episode > 9:
        for i in range(70):
            del memory_Dixie_Shawson[i]

    # Print Episode Delay
    print(f"Network Delay of Episode {episode} = {Vissim.Net.VehicleNetworkPerformanceMeasurement.AttValue('DelayAvg(Current,Last,All)')}")
    print(f"Delay average of Dixie & Shawson = {sum(VehDelay_160_Second) / len(VehDelay_160_Second)}")
    print(f"Truck delay average of Dixie & Shawson = {sum(TruckDelay_160_Second) / len(TruckDelay_160_Second)}")
    print(f"Delay average of Dixie & 401 = {sum(Neighbour1_160_Delay) / len(Neighbour1_160_Delay)}")
    print(f"Delay average of Dixie & Britannia = {sum(Neighbour2_160_Delay) / len(Neighbour2_160_Delay)}")
