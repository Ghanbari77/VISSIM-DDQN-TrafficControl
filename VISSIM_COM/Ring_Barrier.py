import win32com.client as com
from scipy.optimize import linprog

def Ring_Barrier(splits):
    """ Uses linear programming to optimize signal timings based on minimum green times. """
    c = [2, 1, 2, 1, 2, 1, 2, 1]
    bounds = [(split, None) if split != 0 else (0, 0) for split in splits]
    
    A_eq = [
        [1, 1, -1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, -1, -1]
    ]
    b_eq = [0, 0]

    return linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
