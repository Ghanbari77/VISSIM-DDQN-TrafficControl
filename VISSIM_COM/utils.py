import pandas as pd
import numpy as np
from config import DEEP_LEARNING_PARAMS

def save_data_to_csv(filename, data_records, append=False):
    """Saves collected data to a CSV file."""
    column_names = [f'state_{i}' for i in range(len(data_records[0]))] + ['reward', 'episode']
    df = pd.DataFrame([data_records], columns=column_names)

    if append:
        try:
            existing_df = pd.read_csv(filename)
            df = pd.concat([existing_df, df], ignore_index=True)
        except FileNotFoundError:
            pass

    df.to_csv(filename, index=False)


def flatten_state(state):
    """ 
    Flattens the state representation for neural network input.
    
    Parameters:
    -----------
    state : list of lists
        Contains sub-vectors representing different traffic parameters.

    Returns:
    --------
    np.ndarray
        Flattened 1D NumPy array.
    """
    return np.concatenate(
        [state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7], state[8], state[9], state[10],
         state[11]]
    )  # state[0] = split time, state[1] = current phase, state[2] = local time, state[3] = queue length, state[4:8] = flow, state[8:12] = Truck Percentage

def flatten_list(nested_list):
    """ 
    Recursively flattens a nested list into a single list.

    Parameters:
    -----------
    nested_list : list
        A potentially nested list of elements.

    Returns:
    --------
    list
        Flattened list of all elements.
    """
    flat_list = []
    for element in nested_list:
        if isinstance(element, list):
            flat_list.extend(flatten_list(element))
        else:
            flat_list.append(element)
    return flat_list

action_size = DEEP_LEARNING_PARAMS["action_size"]
action_values = DEEP_LEARNING_PARAMS["action_values"]

def generate_all_actions():
    """Generates all possible actions based on action size and values."""
    return np.array(np.meshgrid(*[action_values] * action_size)).T.reshape(-1, action_size)
