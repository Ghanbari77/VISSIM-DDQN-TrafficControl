import pandas as pd
import numpy as np

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
