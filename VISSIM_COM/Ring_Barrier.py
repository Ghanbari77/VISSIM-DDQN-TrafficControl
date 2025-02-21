from scipy.optimize import linprog

def Ring_Barrier(splits):
    """
    Uses linear programming to adjust ring-barrier signal timings based on given minimum
    split times (green intervals). The objective is to minimize a weighted sum of green
    times under two equality constraints that ensure certain movements have balanced durations.

    Parameters
    ----------
    splits: list of int or float
        A list of 8 values representing minimum split times for the following movements:
        [0] Min_Split_NBL
        [1] Min_Split_SB
        [2] Min_Split_SBL
        [3] Min_Split_NB
        [4] Min_Split_EBL
        [5] Min_Split_WB
        [6] Min_Split_WBL
        [7] Min_Split_EB
        If any entry is 0, that movement is forced to remain at 0 seconds.
        Otherwise, the movement is bounded below by the provided value with no upper bound.

    Returns
    -------
    result : OptimizeResult
        The result object returned by scipy.optimize.linprog, which contains information
        such as the optimized split times, solver status, etc.
    """
    # Objective function coefficients: these weights correspond to each movement in 'splits'.
    # The goal is to minimize the sum c[0]*x0 + c[1]*x1 + ... + c[7]*x7.
    c = [2, 1, 2, 1, 2, 1, 2, 1]

    # Build the bounds for each movement:
    #   If split == 0, that movement is completely disabled (bounded between 0 and 0).
    #   Otherwise, it's bounded below by split[i] with no upper limit.
    bounds = []
    for split in splits:
        if split == 0:
            bounds.append((0, 0))  # Fix this movement's green time to 0
        else:
            bounds.append((split, None))  # Lower bound = given split, upper bound = None (no limit)

    # Equality constraints:
    #   1) NBL + SB == SBL + NB
    #   2) EBL + WB == WBL + EB
    # This ensures that certain opposing phases are balanced in total duration.
    A_eq = [
        [1, 1, -1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, -1, -1]
    ]
    b_eq = [0, 0]  # Each equality constraint must sum to zero when rearranged

    # Solve the linear programming problem using the 'highs' solver
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    return result
