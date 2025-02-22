import numpy as np
import random
from config import INTERSECTIONS, Num_episodes
from utils import generate_all_actions, save_data_to_csv, flatten_list
from Ring_Barrier import Ring_Barrier
from simulation import load_vissim_network
from fixed_time_controller import Dixie_401_Signal
from actuated_controller import Dixie_Shawson_Signal
from dqn_agent import DQNAgent


def main():
    # --- Initialization ---
    # Create or load the VISSIM network instance
    Vissim = load_vissim_network('path/to/network.inpx', 'path/to/layout.layx')
    
    # Instantiate your DQN agent
    agent = DQNAgent(state_size=INTERSECTIONS["state_sizes"]["total"],
                     action_size=INTERSELECTIONS["action_size"]) 
    
    
    # --- Main Training Loop ---
    for episode in range(Num_episodes):
        # Initialize data storage
        VehDelay = []
        TruckDelay = []
        VehDelay_160_Second = []
        TruckDelay_160_Second = []
        Neighbour1Delay = []
        Neighbour2Delay = []
        Neighbour1_160_Delay = []
        Neighbour2_160_Delay = []
        
        # Initialize input lists for vehicle flow and truck percentage
        VehInput_Dixie_Shawson = []
        TruckPercentage_Dixie_Shawson = []
        VehInput_Dixie_401 = []
        TruckPercentage_Dixie_401 = []
        VehInput_Dixie_Britannia = []
        TruckPercentage_Dixie_Britannia = []
        
        # State and action storage for training
        all_states_Dixie_Shawson = []
        all_states_Dixie_Britannia = []
        all_states_Dixie_401 = []
        final_state_Dixie_Shawson_batch = []
        all_actions = []

        # Retrieve intersection parameters from config
        dixie_shawson_config = INTERSECTIONS["Dixie_Shawson"]
        offset_Dixie_Shawson = dixie_shawson["offset_Dixie_Shawson"]
        Start_time_Dixie_Shawson = dixie_shawson["Start_time_Dixie_Shawson"]
        Ring1_Dixie_Shawson = dixie_shawson["Ring1_Dixie_Shawson"]
        change_Dixie_Shawson = dixie_shawson["change_Dixie_Shawson"]
        result_Dixie_Shawson = dixie_shawson["result_Dixie_Shawson"]
        splits_Dixie_Shawson = dixie_shawson.get("spilits_Dixie_Shawson", [])

        # Create action matrix and modify actions based on zero indices in the splits vector
        all_possible_actions = generate_all_actions()
        zero_indices = [i for i, value in enumerate(splits_Dixie_Shawson) if value == 0]
        
        all_modified_actions = []
        for action in all_possible_actions:
            modified_action = list(action)
            for index in zero_indices:
                modified_action.insert(index, 0)
            all_modified_actions.append(modified_action)
        all_modified_actions = np.array(all_modified_actions)



        time_step = 0
        
        # Simulation loop
        for i in range(1, 5701):
            Sim_time = i  # simulation second [s]
            
            # Set simulation break time and run simulation step
            Vissim.Simulation.SetAttValue('SimBreakAt', Sim_time)
            Vissim.Simulation.RunContinuous()
            
            # Call your signal function to update intersection states.
            End_of_Last_Barrier_Time_Dixie_Shawson, Ring1_Dixie_Shawson, result_Dixie_Shawson, \
                Start_time_Dixie_Shawson, current_phase_Dixie_Shawson, Local_time_Dixie_Shawson = \
                     Dixie_Shawson_Signal(Sim_time, ..., splits_Dixie_Shawson, ...)
            
            # --- Data Collection ---
            # Every 20 seconds, collect vehicle flow and truck percentage data.
            if i % 20 == 0:
                # F, T = collect_flow_and_truck_data(net, start_key=2, end_key=5)
                # VehInput_Dixie_Shawson.append(F)
                # TruckPercentage_Dixie_Shawson.append(T)
                pass  # Replace with actual implementation
            
            # --- Training and State Observation ---
            if i % 80 == 0:
                time_step += 1
                
                # Build state representations for each intersection.
                # Ensure that the order of elements (e.g., splits, current phases, local times, queue lengths, etc.)
                # matches what your DQN agent expects.
                State_Dixie_Shawson = [
                    splits_Dixie_Shawson,
                    # current_phase_Dixie_Shawson,
                    # [Local_time_Dixie_Shawson],
                    # ... append other state components
                ]
                State_Dixie_Britannia = [
                    # Define similarly
                ]
                State_Dixie_401 = [
                    split_Dixie_401,
                    # current_phase_Dixie_401,
                    # [Local_time_Dixie_401],
                    # ... append other state components
                ]
                
                # Optionally, include queue lengths and historical input data.
                # Append states to global lists for training.
                all_states_Dixie_Shawson.append(State_Dixie_Shawson)
                all_states_Dixie_Britannia.append(State_Dixie_Britannia)
                all_states_Dixie_401.append(State_Dixie_401)
                
                # Combine states for agent input (if your agent uses a combined state)
                final_state = [State_Dixie_Shawson, State_Dixie_Britannia, State_Dixie_401]
                final_state_Dixie_Shawson_batch.append(final_state)
                
                # --- Action Selection and Application ---
                # Use the DQN agent to select an action.
                current_action = agent.select_action(final_state, epsilon=agent.epsilon, all_modified_actions=all_modified_actions)
                # Calculate new splits based on action
                next_split_time = np.add(final_state[0][0], current_action)
                
                # Enforce minimum green requirements
                for index in range(8):
                    if index in [0, 2, 4]:
                        if next_split_time[index] != 0 and next_split_time[index] < 10:
                            next_split_time[index] = 10
                    if index == 6 and next_split_time[index] < 12:
                        next_split_time[index] = 12
                    if index in [1, 3, 5, 7]:
                        if next_split_time[index] != 0 and next_split_time[index] < 15:
                            next_split_time[index] = 15
                
                # Validate the overall cycle time using Ring_Barrier optimization
                next_cycle_time = sum(Ring_Barrier(next_split_time).x)
                if next_cycle_time < 240 or next_cycle_time > 340:
                    next_split_time = final_state[0][0]  # fallback to previous splits
                
                modified_current_action = np.subtract(next_split_time, final_state[0][0])
                all_actions.append(modified_current_action.tolist())
                
                # Apply the new splits to the simulation (update your intersection config)
                splits_Dixie_Shawson = list(next_split_time)
                change_Dixie_Shawson = True
                
                # --- Reward Observation ---
                # Query delays from the simulation (ensure 'net' is defined and connected)
                # Example:
                # current_delay = net.Nodes.ItemByKey(2).TotRes.AttValue('VehDelay(Current,Last,All)')
                # truck_delay = net.Nodes.ItemByKey(2).TotRes.AttValue('VehDelay(Current,Last,20)')
                # For now, use dummy values if needed.
                current_delay = 0  # Replace with actual value
                truck_delay = 0    # Replace with actual value
                VehDelay.append(current_delay)
                TruckDelay.append(truck_delay)
                
                # Observe neighbour delays similarly.
                neighbour1_delay = 0  # Replace with actual value
                neighbour2_delay = 0  # Replace with actual value
                Neighbour1Delay.append(neighbour1_delay)
                Neighbour2Delay.append(neighbour2_delay)
                
                # --- Store Experience and Train ---
                if time_step >= 3:
                    reward_last_2_steps = -VehDelay[time_step - 1] - VehDelay[time_step - 2]
                    # Store the experience in the replay memory.
                    # (Define a proper function or use agent.memory.store_experience)
                    # Example:
                    # agent.memory.store_experience(final_state_Dixie_Shawson_batch[time_step - 3],
                    #                               all_actions[time_step - 3],
                    #                               reward_last_2_steps,
                    #                               final_state_Dixie_Shawson_batch[time_step - 2])
                    # Save data to CSV
                    Record = flatten_list([final_state_Dixie_Shawson_batch[time_step - 3],
                                             all_actions[time_step - 3],
                                             reward_last_2_steps,
                                             episode])
                    save_data_to_csv('C:/Users/qanba.TRANSLAB/Desktop/Result14.csv', Record, append=True)
                
                # Train the DQN agent
                if episode > 0:
                    agent.train()
                    # Decay epsilon (or update within the agent)
                    if agent.epsilon > agent.epsilon_min:
                        agent.epsilon *= agent.epsilon_decay
            
            # --- Periodically Save Weights and Aggregate Delay Data ---
            if time_step > 0 and time_step % 20 == 0:
                agent.q_network_1.save_weights(f"C:/Users/qanba.TRANSLAB/Desktop/Weights/Result14/q_network_1_episode_{episode}_step_{time_step}.weights.h5")
                agent.q_network_2.save_weights(f"C:/Users/qanba.TRANSLAB/Desktop/Weights/Result14/q_network_2_episode_{episode}_step_{time_step}.weights.h5")
            if time_step > 0 and time_step % 2 == 0:
                VehDelay_160_Second.append(VehDelay[-1] + VehDelay[-2])
                TruckDelay_160_Second.append(TruckDelay[-1] + TruckDelay[-2])
                Neighbour1_160_Delay.append(Neighbour1Delay[-1] + Neighbour1Delay[-2])
                Neighbour2_160_Delay.append(Neighbour2Delay[-1] + Neighbour2Delay[-2])
        
        # Optionally prune replay memory after several episodes
        if episode > 9:
            # Example: remove the oldest 70 experiences from agent.memory
            for _ in range(70):
                if len(agent.memory.memory) > 0:
                    agent.memory.memory.popleft()
        
        # --- Episode Summary ---
        # Print simulation performance metrics
        # Example:
        # network_delay = Vissim.Net.VehicleNetworkPerformanceMeasurement.AttValue('DelayAvg(Current,Last,All)')
        # print(f"Network Delay of Episode {episode} = {network_delay}")
        print(f"Episode {episode} completed.")
        # Additional prints for average delays can be added here.
    
if __name__ == '__main__':
    main()
