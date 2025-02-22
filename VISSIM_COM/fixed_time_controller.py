import os
from simulation import load_vissim_network

def init_fixed_time_controller(network_file, layoutfile):
    """
    Initializes the fixed time controller by loading the layout and returning the
    network and signal controllers objects.

    Parameters
    ----------
    Vissim : COM object
        The Vissim instance.
    layoutfile : str
        Path to the layout file.

    Returns
    -------
    net : COM object
        The Vissim network object.
    scs : COM object
        The collection of signal controllers from the network.
    """
    try:
        Vissim = load_vissim_network(network_file, layoutfile)
        net = Vissim.Net
        scs = net.SignalControllers
        return Vissim, net, scs
    except Exception as e:
        print(f"Error initializing fixed time controller: {e}")
        return None, None, None


def Dixie_401_Signal(scs, Sim_time, offset, Min_green_SB_NB, Min_green_WB, End_of_Last_Cycle_Time):
    """
    Controls the traffic signal states for the Dixie & 401 intersection using Vissim's SignalControllers.

    Parameters
    ----------
    scs : COM object
        The signal controllers collection from Vissim.Net.
    Sim_time : int or float
        The current simulation time in seconds.
    offset : int or float
        Offset time to synchronize this signal with other intersections.
    Min_green_SB_NB : int or float
        The minimum green duration (in seconds) for the Southbound/Northbound (SB_NB) approach.
    Min_green_WB : int or float
        The minimum green duration (in seconds) for the Westbound (WB) approach.
    End_of_Last_Cycle_Time : int or float
        The simulation time at which the previous cycle ended. A value of 0 indicates the first cycle.

    Returns
    -------
    End_of_Last_Cycle_Time : int or float
        Updated simulation time marking the end of the current cycle.
    Local_time : int or float or None
        Time within the current cycle (seconds elapsed since the last cycle ended). Returns None if an error occurs.
    """
    try:
        # Retrieve the signal controller for Dixie & 401 (assumed key=1)
        Dixie_401 = scs.ItemByKey(1)
    
        # Access individual signal groups:
        # SB_NB (Key=1) for Southbound/Northbound and WB (Key=2) for Westbound.
        SB_NB = Dixie_401.SGs.ItemByKey(1)
        WB = Dixie_401.SGs.ItemByKey(2)
    
        # Initialize both approaches to Red.
        SB_NB.SetAttValue("SigState", 1)
        WB.SetAttValue("SigState", 1)
    
        # --- Handle First Cycle Offset ---
        if End_of_Last_Cycle_Time == 0:
            Sim_time = Sim_time + offset - 1
    
        # --- Compute Local Time ---
        # If Sim_time exactly equals End_of_Last_Cycle_Time (and is nonzero), reset local time to zero.
        Local_time = 0 if (Sim_time == End_of_Last_Cycle_Time and Sim_time != 0) else Sim_time - End_of_Last_Cycle_Time
    
        # --- Set Signal States Based on Time Intervals ---
        # For SB_NB: Allow green after a 3-second clearance.
        if 3 <= Local_time < Min_green_SB_NB + 3:
            SB_NB.SetAttValue("SigState", 3)  # Green
        elif Min_green_SB_NB + 3 <= Local_time < Min_green_SB_NB + 7:
            SB_NB.SetAttValue("SigState", 4)  # Yellow
        # For WB: Begin green after SB_NB’s interval plus a 4-second clearance.
        elif Min_green_SB_NB + 11 <= Local_time < (Min_green_SB_NB + Min_green_WB + 11):
            WB.SetAttValue("SigState", 3)  # Green
        elif (Min_green_SB_NB + Min_green_WB + 11) <= Local_time < (Min_green_SB_NB + Min_green_WB + 15):
            WB.SetAttValue("SigState", 4)  # Yellow
    
        # --- End of Cycle Update ---
        # When local time reaches the end of both phases' green+yellow intervals,
        # update the cycle end time.
        if Local_time == (Min_green_SB_NB + Min_green_WB + 14):
            if End_of_Last_Cycle_Time == 0:
                Sim_time = Sim_time - offset + 1
            End_of_Last_Cycle_Time = Sim_time + 1
    
        return End_of_Last_Cycle_Time, Local_time
    except Exception as e:
        print(f"Error in Dixie_401_Signal: {e}")
        return End_of_Last_Cycle_Time, None
