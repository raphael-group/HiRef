import numpy as np
from scipy.optimize import linprog

def solve_ot_lp(p, q, C):
    """
    Solve the unregularized discrete OT problem _exactly_ via linear programming.
    
    Parameters
    ----------
    p : np.ndarray of shape (N,)
        Source distribution (nonnegative, summing to 1).
    q : np.ndarray of shape (M,)
        Target distribution (nonnegative, summing to 1).
    C : np.ndarray of shape (N, M)
        Cost matrix, C[i, j] = cost of transporting from i-th source to j-th target.
    
    Returns
    -------
    gamma : np.ndarray of shape (N, M)
        Optimal transport plan.
    min_cost : float
        Optimal transportation cost (objective value).
    """
    # Flatten the cost matrix
    N, M = C.shape
    c = C.reshape(N*M)
    
    # Build equality constraints: A_eq x = b_eq
    # 1) Sum_j gamma[i,j] = p[i], for each i in [0..N-1]
    # 2) Sum_i gamma[i,j] = q[j], for each j in [0..M-1]

    A_eq = []
    b_eq = []

    # (1) Row constraints
    for i in range(N):
        row = np.zeros(N*M)
        for j in range(M):
            row[i*M + j] = 1.0
        A_eq.append(row)
        b_eq.append(p[i])

    # (2) Column constraints
    for j in range(M):
        row = np.zeros(N*M)
        for i in range(N):
            row[i*M + j] = 1.0
        A_eq.append(row)
        b_eq.append(q[j])

    A_eq = np.array(A_eq)
    b_eq = np.array(b_eq)
    
    # Bounds: gamma[i,j] >= 0, no upper limit
    bounds = [(0, None)] * (N*M)
    
    # Solve with the "highs" method (fast modern solver in SciPy)
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if not res.success:
        raise RuntimeError(f"LP did not converge: {res.message}")
    
    # Reshape solution x back to the transport plan Gamma
    gamma = res.x.reshape(N, M)
    min_cost = res.fun  # c^T x
    
    return gamma, min_cost