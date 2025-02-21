import os
import win32com.client as com

def load_vissim_network(filename, layoutfile):
    """Loads the VISSIM network and layout."""
    Vissim = com.Dispatch("Vissim.Vissim")
    Vissim.LoadNet(os.path.abspath(filename), False)
    Vissim.LoadLayout(layoutfile)
    return Vissim
