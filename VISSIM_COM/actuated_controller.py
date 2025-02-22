import copy
from Ring_Barrier import Ring_Barrier

def Dixie_Shawson_Signal(net, scs, Sim_time, Last_Barrier_Time, Ring_1, 
                          RBC_Function_Result, offset, Initial_Splits, 
                          End_of_Last_Cycle_Time, change):
    """
    Controls signal phases at the Dixie-Shawson intersection based on a ring-barrier approach.
    Updates signal group (SG) states by considering detector presence, timing offsets, ring-phase
    transitions, and optimized split results from the Ring_Barrier function.

    Parameters
    ----------
    net : COM object
        The VISSIM network object.
    scs : COM object
        The VISSIM SignalControllers collection.
    Sim_time : float
        Current simulation time in seconds.
    Last_Barrier_Time : float
        Simulation time when the previous ring barrier finished.
    Ring_1 : bool
        Flag indicating whether Ring 1 is active (True) or Ring 2 (False).
    RBC_Function_Result : OptimizeResult
        The result of the ring-barrier calculation, with optimized splits in RBC_Function_Result.x.
    offset : float
        Offset to synchronize the start time for the signal.
    Initial_Splits : list of float
        Initial green split times for up to 8 signal phases/movements.
    End_of_Last_Cycle_Time : float
        Recorded simulation time when the last ring cycle began.
    change : bool
        Indicates whether splits have been updated mid-cycle, requiring recalculation.

    Returns
    -------
    Last_Barrier_Time : float
        Updated simulation time when the last ring barrier finished.
    Ring_1 : bool
        Updated flag for the active ring.
    RBC_Function_Result : OptimizeResult
        Updated ring-barrier calculation result.
    End_of_Last_Cycle_Time : float
        Updated cycle start time.
    current_phase : list of int
        A list of length 8 indicating current phase status (1 for green, 0 otherwise).
    Local_time : float
        Time elapsed within the current ring (seconds since the last barrier ended).
    """
    try:
        # Make a local copy of the initial splits so as not to modify the original list.
        splits = Initial_Splits.copy()

        # Retrieve the Dixie-Shawson signal controller (assumed Controller Key=2)
        Dixie_Shawson = scs.ItemByKey(2)

        # Access individual signal groups for the various movements.
        # SG keys: 1: NBL, 2: SB, 4: WB, 5: SBL, 6: NB, 7: WBL, 8: EB.
        NBL = Dixie_Shawson.SGs.ItemByKey(1)
        SB  = Dixie_Shawson.SGs.ItemByKey(2)
        WB  = Dixie_Shawson.SGs.ItemByKey(4)
        SBL = Dixie_Shawson.SGs.ItemByKey(5)
        NB  = Dixie_Shawson.SGs.ItemByKey(6)
        WBL = Dixie_Shawson.SGs.ItemByKey(7)
        EB  = Dixie_Shawson.SGs.ItemByKey(8)

        # Initialize all signals to Red (SigState = 1).
        for sg in [NBL, SB, WB, SBL, NB, WBL, EB]:
            sg.SetAttValue("SigState", 1)

        # Initialize current phase indicator (length 8)
        current_phase = [0] * 8

        # --- Adjust for first cycle offset ---
        if Last_Barrier_Time == 0:
            End_of_Last_Cycle_Time = End_of_Last_Cycle_Time + offset - 1

        # --- Cycle Boundary Checks ---
        if Sim_time == Last_Barrier_Time and Sim_time != 0:
            # Check detector presence to determine if a movement should be disabled (split set to 0)
            if net.Detectors.ItemByKey(50).AttValue("Presence") == 0:
                splits[0] = 0   # NBL movement
            if net.Detectors.ItemByKey(51).AttValue("Presence") == 0:
                splits[6] = 0   # WBL movement
            if net.Detectors.ItemByKey(53).AttValue("Presence") == 0:
                splits[2] = 0   # SBL movement

            # Recalculate optimized splits with any zeros inserted.
            RBC_Function_Result = Ring_Barrier(splits)
            Local_time = 0  # Reset local time at the start of a new ring.
            
            # Update cycle start time if entering Ring 1.
            if Ring_1:
                End_of_Last_Cycle_Time = Sim_time
        else:
            # If not at a cycle boundary, compute elapsed time in the current ring.
            Local_time = Sim_time - Last_Barrier_Time

        # --- Mid-Cycle Split Changes ---
        if change:
            for i in range(8):
                if RBC_Function_Result.x[i] == 0:
                    splits[i] = 0
            RBC_Function_Result = Ring_Barrier(splits)

        # --- Ring 1 Logic ---
        if Ring_1:
            # Safety: Ensure Local_time does not exceed the duration of Ring 1.
            ring1_duration = RBC_Function_Result.x[0] + RBC_Function_Result.x[1]
            if Local_time > ring1_duration - 1:
                Local_time = ring1_duration - 1

            # NBL: Green then Yellow (yellow period = 3 sec)
            if Local_time < RBC_Function_Result.x[0] - 3:
                NBL.SetAttValue("SigState", 3)  # Green
                current_phase[0] = 1
            elif RBC_Function_Result.x[0] - 3 <= Local_time < RBC_Function_Result.x[0]:
                NBL.SetAttValue("SigState", 4)  # Yellow

            # SB: Green then Yellow (yellow period = 4 sec; red clearance = 3 sec)
            if RBC_Function_Result.x[0] <= Local_time < (RBC_Function_Result.x[0] + RBC_Function_Result.x[1] - 7):
                SB.SetAttValue("SigState", 3)  # Green
                current_phase[1] = 1
            elif (RBC_Function_Result.x[0] + RBC_Function_Result.x[1] - 7) <= Local_time < (RBC_Function_Result.x[0] + RBC_Function_Result.x[1] - 3):
                SB.SetAttValue("SigState", 4)  # Yellow

            # SBL: Green then Yellow (yellow period = 3 sec)
            if Local_time < RBC_Function_Result.x[2] - 3:
                SBL.SetAttValue("SigState", 3)  # Green
                current_phase[2] = 1
            elif RBC_Function_Result.x[2] - 3 <= Local_time < RBC_Function_Result.x[2]:
                SBL.SetAttValue("SigState", 4)  # Yellow

            # NB: Green then Yellow (yellow period = 4 sec; red clearance = 3 sec)
            if RBC_Function_Result.x[2] <= Local_time < (RBC_Function_Result.x[2] + RBC_Function_Result.x[3] - 7):
                NB.SetAttValue("SigState", 3)  # Green
                current_phase[3] = 1
            elif (RBC_Function_Result.x[2] + RBC_Function_Result.x[3] - 7) <= Local_time < (RBC_Function_Result.x[2] + RBC_Function_Result.x[3] - 3):
                NB.SetAttValue("SigState", 4)  # Yellow

            # At the end of Ring 1’s cycle, update barrier time and switch to Ring 2.
            if Local_time == ring1_duration - 1:
                Last_Barrier_Time = Sim_time + 1
                Ring_1 = False

        # --- Ring 2 Logic ---
        else:
            # Safety: Ensure Local_time does not exceed Ring 2's duration.
            if Local_time > RBC_Function_Result.x[5] - 1:
                Local_time = RBC_Function_Result.x[5] - 1

            # WB: Green then Yellow (yellow period = 4 sec; red clearance = 4 sec)
            if Local_time < (RBC_Function_Result.x[5] - 8):
                WB.SetAttValue("SigState", 3)
                current_phase[5] = 1
            elif (RBC_Function_Result.x[5] - 8) <= Local_time < (RBC_Function_Result.x[5] - 4):
                WB.SetAttValue("SigState", 4)

            # WBL: Green then Yellow (yellow period = 3 sec; red clearance = 2 sec)
            if Local_time < (RBC_Function_Result.x[6] - 5):
                WBL.SetAttValue("SigState", 3)
                current_phase[6] = 1
            elif (RBC_Function_Result.x[6] - 5) <= Local_time < (RBC_Function_Result.x[6] - 2):
                WBL.SetAttValue("SigState", 4)

            # EB: Green then Yellow (yellow period = 4 sec; red clearance = 4 sec)
            if RBC_Function_Result.x[6] <= Local_time < (RBC_Function_Result.x[6] + RBC_Function_Result.x[7] - 8):
                EB.SetAttValue("SigState", 3)
                current_phase[7] = 1
            elif (RBC_Function_Result.x[6] + RBC_Function_Result.x[7] - 8) <= Local_time < (RBC_Function_Result.x[6] + RBC_Function_Result.x[7] - 4):
                EB.SetAttValue("SigState", 4)

            # At the end of Ring 2’s cycle, update barrier time and return to Ring 1.
            if Local_time == RBC_Function_Result.x[5] - 1:
                Last_Barrier_Time = Sim_time + 1
                Ring_1 = True

        return Last_Barrier_Time, Ring_1, RBC_Function_Result, End_of_Last_Cycle_Time, current_phase, Local_time

    except Exception as e:
        print(f"Error in Dixie_Shawson_Signal: {e}")
        # Return current values if an error occurs (or choose appropriate fallbacks)
        return Last_Barrier_Time, Ring_1, RBC_Function_Result, End_of_Last_Cycle_Time, [0]*8, 0
