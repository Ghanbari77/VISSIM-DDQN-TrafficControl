from Ring_Barrier import Ring_Barrier 

# --- Deep Learning Hyperparameters ---
DEEP_LEARNING_PARAMS = {
    "state_sizes": {
        "Dixie_Shawson": 57,
        "Dixie_401": 33,
        "Dixie_Britannia": 57,
        "total": 57 + 33 + 57
    },
    "action_size": 7,
    "action_values": [-5, 0, 5],
    "learning_rate": 0.001,
    "batch_size": 64,
    "max_memory_size": 15000,
    "gamma": 0.99,  # Discount factor for future rewards
    "exploration": {
        "epsilon": 1,
        "epsilon_decay": 0.9995,
        "epsilon_min": 0.01
    }
}

# Main Loop for Training
Num_episodes = 200


# Intersection configurations stored in a dictionary
INTERSECTIONS = {
    "Dixie_401": {
        "offset_Dixie_401": 129,
        "Min_green_SB_NB_Dixie_401": 100,
        "Min_green_WB_Dixie_401": 45,
        "End_of_Last_Cycle_Time_Dixie_401": 0,
        "split_Dixie_401": [100, 45],  # Uses Min_green_SB_NB_Dixie_401 and Min_green_WB_Dixie_401
    },

    "Dixie_Shawson": {
        "End_of_Last_Barrier_Time_Dixie_Shawson": 0,
        "offset_Dixie_Shawson": 1,
        "Start_time_Dixie_Shawson": 0,
        "splits_Dixie_Shawson": [14, 87, 10, 87, 0, 52, 31, 23],
        "result_Dixie_Shawson": None,  # Placeholder for Ring_Barrier result
        "Ring1_Dixie_Shawson": True,
        "change_Dixie_Shawson": False
    },

    "Dixie_Britannia": {
        "End_of_Last_Barrier_Time_Dixie_Britannia": 0,
        "offset_Dixie_Britannia": 1,
        "Start_time_Dixie_Britannia": 0,
        "splits_Dixie_Britannia": [14, 82, 11, 85, 0, 58, 21, 43],
        "result_Dixie_Britannia": None,  # Placeholder for Ring_Barrier result
        "Ring1_Dixie_Britannia": True
    }
}

# Compute Ring_Barrier results and update dictionary
INTERSECTIONS["Dixie_Shawson"]["result_Dixie_Shawson"] = Ring_Barrier(INTERSECTIONS["Dixie_Shawson"]["splits_Dixie_Shawson"])
INTERSECTIONS["Dixie_Britannia"]["result_Dixie_Britannia"] = Ring_Barrier(INTERSECTIONS["Dixie_Britannia"]["splits_Dixie_Britannia"])

# Adjust offset based on conditions
if INTERSECTIONS["Dixie_Shawson"]["offset_Dixie_Shawson"] >= INTERSECTIONS["Dixie_Shawson"]["result_Dixie_Shawson"].x[0] + INTERSECTIONS["Dixie_Shawson"]["result_Dixie_Shawson"].x[1]:
    INTERSECTIONS["Dixie_Shawson"]["Ring1_Dixie_Shawson"] = False
    INTERSECTIONS["Dixie_Shawson"]["offset_Dixie_Shawson"] -= INTERSECTIONS["Dixie_Shawson"]["result_Dixie_Shawson"].x[0] + INTERSECTIONS["Dixie_Shawson"]["result_Dixie_Shawson"].x[1]

if INTERSECTIONS["Dixie_Britannia"]["offset_Dixie_Britannia"] >= INTERSECTIONS["Dixie_Britannia"]["result_Dixie_Britannia"].x[0] + INTERSECTIONS["Dixie_Britannia"]["result_Dixie_Britannia"].x[1]:
    INTERSECTIONS["Dixie_Britannia"]["Ring1_Dixie_Britannia"] = False
    INTERSECTIONS["Dixie_Britannia"]["offset_Dixie_Britannia"] -= INTERSECTIONS["Dixie_Britannia"]["result_Dixie_Britannia"].x[0] + INTERSECTIONS["Dixie_Britannia"]["result_Dixie_Britannia"].x[1]

