import os
import win32com.client as com

def load_vissim_network(filename, layoutfile):
    """Loads the VISSIM network and layout."""
    try:
        Vissim = com.Dispatch("Vissim.Vissim")
    except Exception as e:
        print(f"Error creating VISSIM instance: {e}")
        return None

    try:
        network_path = os.path.abspath(filename)
        Vissim.LoadNet(network_path, False)
    except Exception as e:
        print(f"Error loading VISSIM network from {network_path}: {e}")
        return None

    try:
        layout_path = os.path.abspath(layoutfile)
        Vissim.LoadLayout(layout_path)
    except Exception as e:
        print(f"Error loading VISSIM layout from {layout_path}: {e}")
        return None

    return Vissim
