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
