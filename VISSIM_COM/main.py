from config import INTERSECTIONS, Num_episodes
from utils import generate_all_actions





# Start Episodes
for episode in range(Num_episodes):
    # Start simulation
    time_step = 0

   # Initialize storage for state, actions, and delays
    state_data = {
        "Dixie_Shawson": [],
        "Dixie_Britannia": [],
        "Dixie_401": [],
        "final_state_Dixie_Shawson_batch": [],
    }
    
    action_data = {
        "all_actions": []
    }
    
    delay_data = {
        "VehDelay": [],
        "TruckDelay": [],
        "VehDelay_160_Second": [],
        "TruckDelay_160_Second": [],
        "Neighbour1Delay": [],
        "Neighbour2Delay": [],
        "Neighbour1_160_Delay": [],
        "Neighbour2_160_Delay": [],
    }
    
    input_data = {
        "VehInput": {
            "Dixie_Shawson": [],
            "Dixie_Britannia": [],
            "Dixie_401": []
        },
        "TruckPercentage": {
            "Dixie_Shawson": [],
            "Dixie_Britannia": [],
            "Dixie_401": []
        }
    }

    # Access Dixie & 401 parameters
    dixie_401_config = INTERSECTIONS["Dixie_401"]
    offset_Dixie_401 = dixie_401_config["offset_Dixie_401"]
    Min_green_SB_NB_Dixie_401 = dixie_401_config["Min_green_SB_NB_Dixie_401"]
    Min_green_WB_Dixie_401 = dixie_401_config["Min_green_WB_Dixie_401"]
    End_of_Last_Cycle_Time_Dixie_401 = dixie_401_config["End_of_Last_Cycle_Time_Dixie_401"]
    split_Dixie_401 = dixie_401_config["split_Dixie_401"]
    
    # Access Dixie & Shawson parameters
    dixie_shawson_config = INTERSECTIONS["Dixie_Shawson"]
    offset_Dixie_Shawson = dixie_shawson_config["offset_Dixie_Shawson"]
    Start_time_Dixie_Shawson = dixie_shawson_config["Start_time_Dixie_Shawson"]
    Ring1_Dixie_Shawson = dixie_shawson_config["Ring1_Dixie_Shawson"]
    change_Dixie_Shawson = dixie_shawson_config["change_Dixie_Shawson"]
    result_Dixie_Shawson = dixie_shawson_config["result_Dixie_Shawson"]

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
