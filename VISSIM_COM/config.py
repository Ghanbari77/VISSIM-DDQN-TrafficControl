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

# Main Loop for Training
Num_episodes = 200


#Traffic Signal Control Parameters  
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
