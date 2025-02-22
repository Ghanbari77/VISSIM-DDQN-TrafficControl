import os
import numpy as np
import random
from config import INTERSECTIONS, Num_episodes, DEEP_LEARNING_PARAMS
from simulation import load_vissim_network
from fixed_time_controller import init_fixed_time_controller
from actuated_controller import Dixie_Shawson_Signal
from dqn_agent import DQNAgent
from utils import generate_all_actions, save_data_to_csv, flatten_list

def main():
    # --- Simulation & Controller Initialization ---
    # Paths to your VISSIM network and layout files.
    network_file = "path/to/network.inpx"
    layout_file = "path/to/layout.layx"
    
    # Initialize VISSIM and obtain network and signal controllers.
    Vissim, net, scs = init_fixed_time_controller(network_file, layout_file)
    if Vissim is None:
        print("Failed to initialize VISSIM network. Exiting.")
        return

    # --- DQN Agent Initialization ---
    state_size = DEEP_LEARNING_PARAMS["state_sizes"]["total"]
    action_size = DEEP_LEARNING_PARAMS["action_size"]
    agent = DQNAgent(state_size, action_size)



    # --- Main Training Loop ---
    for episode in range(Num_episodes):

        # --- Action Space Preparation ---
        all_possible_actions = generate_all_actions()
        # For the Dixie-Shawson intersection, use the configured splits (note the key 'spilits_Dixie_Shawson' may be a typo)
        splits_shawson = INTERSECTIONS["Dixie_Shawson"].get("spilits_Dixie_Shawson", [])
        zero_indices = [i for i, value in enumerate(splits_shawson) if value == 0]
        all_modified_actions = []
        for action in all_possible_actions:
            mod_action = list(action)
            for index in zero_indices:
                mod_action.insert(index, 0)
            all_modified_actions.append(mod_action)
        all_modified_actions = np.array(all_modified_actions)
    
        # --- Data Containers & Initial Parameters ---
        VehDelay = []
        TruckDelay = []
        final_state_batch = []
        all_actions = []

        # Initialize vehicle flow and truck percentage containers for each intersection
        VehInput_Dixie_Shawson = []
        TruckPercentage_Dixie_Shawson = []
        VehInput_Dixie_401 = []
        TruckPercentage_Dixie_401 = []
        VehInput_Dixie_Britannia = []
        TruckPercentage_Dixie_Britannia = []

        # For state observation from neighbouring intersections:
        all_states_Dixie_Shawson = []
        all_states_Dixie_Britannia = []
        all_states_Dixie_401 = []
        final_state_Dixie_Shawson_batch = []
        
    
        # Dixie_Shawson parameters from config
        dixie_shawson_config = INTERSECTIONS["Dixie_Shawson"]
        offset_shawson = dixie_shawson_config["offset_Dixie_Shawson"]
        splits_shawson_initial = dixie_shawson_config.get("spilits_Dixie_Shawson", [])
        change_shawson = dixie_shawson_config.get("change_Dixie_Shawson", False)
        # Use the precomputed Ring_Barrier result as the initial value
        RBC_Function_Result = dixie_shawson_config["result_Dixie_Shawson"]
        End_of_Last_Cycle_Time_shawson = dixie_shawson_config.get("Start_time_Dixie_Shawson", 0)
        # For the actuated controller, maintain barrier time and ring status
        Last_Barrier_Time = 0.0
        Ring_1 = True
            
        time_step = 0

        # Run simulation for a predefined duration (e.g. 5700 seconds)
        for i in range(1, 5701):
            Sim_time = i  # Current simulation second

            # Set simulation break time and run until that time.
            Vissim.Simulation.SetAttValue('SimBreakAt', Sim_time)
            Vissim.Simulation.RunContinuous()


            # --- Actuated Control for Dixie_Shawson ---
            # This function updates signal states based on detectors, offsets, and ring logic.
            (Last_Barrier_Time, Ring_1, RBC_Function_Result,
             End_of_Last_Cycle_Time_shawson, current_phase, Local_time_shawson) = Dixie_Shawson_Signal(
                net, scs, Sim_time, Last_Barrier_Time, Ring_1, RBC_Function_Result,
                offset_shawson, splits_shawson_initial, End_of_Last_Cycle_Time_shawson, change_shawson
            )

            # --- Data Collection & State Construction ---
            # Here you should build your state representation.
            state_shawson = [splits_shawson_initial, current_phase, [Local_time_shawson]]
            # For Dixie_401, we assume a dummy state representation.
            state_401 = [split_Dixie_401, [1] * len(split_Dixie_401), [Local_time_401]]
            # Combine states (if using both intersections as input; adjust dimensions as needed)
            final_state = [state_shawson, state_401]
            final_state_batch.append(final_state)

            # --- Data Collection: Vehicle Flow and Truck Percentage ---
            if i % 20 == 0:
                # Dixie_Shawson Data Collection
                F = []
                T = []
                for j in range(2, 6):
                    Flow = net.DataCollectionMeasurements.ItemByKey(j).AttValue('Vehs(Current,Last,All)')
                    Truck = net.DataCollectionMeasurements.ItemByKey(j).AttValue('Vehs(Current,Last,20)') #20: Truck
                    if type(Flow) != int:
                        Flow = 0
                    if type(Truck) != int:
                        Truck = 0
                    F.append(Flow)
                    T.append(Truck)
                VehInput_Dixie_Shawson.append(F)
                TruckPercentage_Dixie_Shawson.append(T)

                # Dixie_401 Data Collection
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

                # Dixie_Britannia Data Collection
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
            
            
            # --- Additional State Observation Every 80 Seconds ---
            if i % 80 == 0:
                time_step += 1

                # Initialize state lists for each intersection.
                State_Dixie_Shawson = []
                State_Dixie_Britannia = []
                State_Dixie_401 = []

                final_state_Dixie_Shawson = []

                # --- Observe Neighbour Intersection Signals ---
                # For Dixie-Britannia: (Assuming Controller Key=3)
                current_phase_Dixie_Britannia = []
                for j in range(1, 9):
                    if scs.ItemByKey(3).SGs.ItemByKey(j).AttValue('SigState') == 'GREEN':
                        current_phase_Dixie_Britannia.append(1)
                    else:
                        current_phase_Dixie_Britannia.append(0)
                Local_time_Dixie_Britannia = scs.ItemByKey(3).SGs.ItemByKey(2).AttValue('tSigState')

                # For Dixie 401: (Assuming Controller Key=1)
                current_phase_Dixie_401 = []
                for j in range(1, 3):
                    if scs.ItemByKey(1).SGs.ItemByKey(j).AttValue('SigState') == 'GREEN':
                        current_phase_Dixie_401.append(1)
                    else:
                        current_phase_Dixie_401.append(0)
                Local_time_Dixie_401 = scs.ItemByKey(1).SGs.ItemByKey(1).AttValue('tSigState')

                # Append Traffic Signal States
                # Here, we assume "splits_shawson" is equivalent to the configured spilits_Dixie_Shawson.
                State_Dixie_Shawson.append(splits_shawson)
                State_Dixie_Shawson.append(current_phase)
                State_Dixie_Shawson.append([Local_time_shawson])

                State_Dixie_Britannia.append(splits_britannia)
                State_Dixie_Britannia.append(current_phase_Dixie_Britannia)
                State_Dixie_Britannia.append([Local_time_Dixie_Britannia])
                State_Dixie_401.append(split_Dixie_401)
                State_Dixie_401.append(current_phase_Dixie_401)
                State_Dixie_401.append([Local_time_Dixie_401])

                # --- Queue Length Observations and Append to the State Space ---
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

                # --- Append Historical Flow & Truck Data ---
                # Dixie-Shawson: Append the last 4 measurements as the observation time is each 20 seconds and the time step is each 80 seconds.
                State_Dixie_Shawson.append(VehInput_Dixie_Shawson[-1])
                State_Dixie_Shawson.append(VehInput_Dixie_Shawson[-2])
                State_Dixie_Shawson.append(VehInput_Dixie_Shawson[-3])
                State_Dixie_Shawson.append(VehInput_Dixie_Shawson[-4])
                State_Dixie_Shawson.append(TruckPercentage_Dixie_Shawson[-1])
                State_Dixie_Shawson.append(TruckPercentage_Dixie_Shawson[-2])
                State_Dixie_Shawson.append(TruckPercentage_Dixie_Shawson[-3])
                State_Dixie_Shawson.append(TruckPercentage_Dixie_Shawson[-4])

                # Dixie-Britannia: Append the last 4 measurements.
                State_Dixie_Britannia.append(VehInput_Dixie_Britannia[-1])
                State_Dixie_Britannia.append(VehInput_Dixie_Britannia[-2])
                State_Dixie_Britannia.append(VehInput_Dixie_Britannia[-3])
                State_Dixie_Britannia.append(VehInput_Dixie_Britannia[-4])
                State_Dixie_Britannia.append(TruckPercentage_Dixie_Britannia[-1])
                State_Dixie_Britannia.append(TruckPercentage_Dixie_Britannia[-2])
                State_Dixie_Britannia.append(TruckPercentage_Dixie_Britannia[-3])
                State_Dixie_Britannia.append(TruckPercentage_Dixie_Britannia[-4])

                # Dixie 401: Append the last 4 measurements.
                State_Dixie_401.append(VehInput_Dixie_401[-1])
                State_Dixie_401.append(VehInput_Dixie_401[-2])
                State_Dixie_401.append(VehInput_Dixie_401[-3])
                State_Dixie_401.append(VehInput_Dixie_401[-4])
                State_Dixie_401.append(TruckPercentage_Dixie_401[-1])
                State_Dixie_401.append(TruckPercentage_Dixie_401[-2])
                State_Dixie_401.append(TruckPercentage_Dixie_401[-3])
                State_Dixie_401.append(TruckPercentage_Dixie_401[-4])

                # Store states for each intersection.
                all_states_Dixie_Shawson.append(State_Dixie_Shawson)
                all_states_Dixie_Britannia.append(State_Dixie_Britannia)
                all_states_Dixie_401.append(State_Dixie_401)

                # Combine into a final state batch for Dixie-Shawson.
                final_state_Dixie_Shawson = [State_Dixie_Shawson, State_Dixie_Britannia, State_Dixie_401]
                final_state_Dixie_Shawson_batch.append(final_state_Dixie_Shawson)


                
                # --- Agent Action & Training Every 80 Seconds ---

            
                # Select an action using the agent's epsilon-greedy policy.
                action = agent.select_action(final_state, epsilon=agent.epsilon, all_modified_actions=all_modified_actions)
                next_splits = np.add(splits_shawson, action)

                # Validate cycle time using Ring_Barrier optimization. Cycle time should be between 120 to 170 seconds.
                from Ring_Barrier import Ring_Barrier
                cycle_time = sum(Ring_Barrier(next_splits).x)
                if cycle_time < 240 or cycle_time > 340:
                    next_splits = splits_shawson

                modified_action = np.subtract(next_splits, splits_shawson)
                all_actions.append(modified_action.tolist())
                #Apply Action
                splits_shawson = list(next_splits)
                change_shawson = True

                # --- Reward Observation ---
                current_delay = net.Nodes.ItemByKey(2).TotRes.AttValue('VehDelay(Current,Last,All)')
                truck_delay = net.Nodes.ItemByKey(2).TotRes.AttValue('VehDelay(Current,Last,20)')
                if not isinstance(current_delay, (int, float)):
                    current_delay = 0
                if not isinstance(truck_delay, (int, float)):
                    truck_delay = 0
                VehDelay.append(current_delay)
                TruckDelay.append(truck_delay)

                if time_step >= 3:
                    reward_last_2_steps = -VehDelay[time_step - 1] - VehDelay[time_step - 2]
                    
                    # Store experience
                    agent.memory.store_experience(final_state_Dixie_Shawson_batch[time_step -3], all_action[time_step -3], 
                                                  reward_last_2_steps, final_state_Dixie_Shawson_batch[time_step -2])
            
                    #Save Data
                    Record_Dixie_Shawson = flatten_list([final_state_Dixie_Shawson_batch[time_step -3], all_action[time_step -3], 
                                                  reward_last_2_steps])
                    save_data_to_csv('File_Name', Record_Dixie_Shawson)

            #Train the agent
            if episode > 0:
                agent.train()
                if agent.epsilon > agent.epsilon_min:
                    agent.epsilon *= agent.epsilon_decay

            # --- Periodic Saving of Weights ---
            if time_step > 0 and time_step % 20 == 0:
                agent.q_network_1.save_weights(f"File_Name/Weights/q_network_1_ep{episode}_step{time_step}.weights.h5")
                agent.q_network_2.save_weights(f"File_Name/Weights/q_network_2_ep{episode}_step{time_step}.weights.h5")

        # --- Episode Summary ---
        avg_delay = sum(VehDelay) / len(VehDelay) if VehDelay else 0
        avg_truck_delay = sum(TruckDelay) / len(TruckDelay) if TruckDelay else 0
        print(f"Episode {episode} completed. Avg Delay: {avg_delay}, Avg Truck Delay: {avg_truck_delay}")

    # --- End of Simulation ---
    agent.q_network_1.save_weights("./Weights/final_q_network_1.weights.h5")
    agent.q_network_2.save_weights("./Weights/final_q_network_2.weights.h5")
    print("Simulation completed.")

if __name__ == '__main__':
    main()
