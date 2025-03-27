import numpy as np
import scipy
from scipy.optimize import linprog
from tqdm import tqdm
import jax
import jax.numpy as jnp
from ott.geometry.geometry import Geometry
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

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

@jax.jit
def sinkhorn_batch(xs, ys, p=1):
    
    # Pre-compile a jitted instance
    cost_mat = jnp.linalg.norm(xs[:, None, :] - ys[None, :, :], axis=-1) ** p
    geom = Geometry(cost_mat)
    ot_problem = linear_problem.LinearProblem(geom)
    solver = sinkhorn.Sinkhorn()
    ot_solution = solver(ot_problem)
    gamma = ot_solution.matrix
    batch_cost = jnp.sum(cost_mat * gamma)
    
    return batch_cost


def minibatch_sinkhorn_ot_without_replacement(X, Y, batch_size, p=1):
    """
    Compute mini-batch OT using entropic regularization (Sinkhorn via ott-jax) without replacement.
    Implicit coupling corresponds to definition 6 of 
    Parameters:
      X: np.array, shape (n, d) - source samples.
      Y: np.array, shape (n, d) - target samples.
      batch_size: int - number of samples in each mini-batch.
      p: float - power for the cost (default is 1, but cost is computed as squared Euclidean).
    
    Returns:
      transport cost over the mini-batches.
    """
    
    n = X.shape[0]
    assert X.shape[0] == Y.shape[0], "X and Y must have the same number of points."

    batch_size = min(n, batch_size)
    
    # Create a random permutation for batching without replacement.
    perm = np.random.permutation(n)
    batches = [(perm[i:i+batch_size], perm[i:i+batch_size])
               for i in range(0, n, batch_size)]
    
    total_cost = 0.0
    num_batches = len(batches)
    
    for idx_src, idx_tgt in tqdm(batches, desc="Mini-batch Sinkhorn"):
        # Convert the mini-batch data to jax.numpy arrays.
        xs = jnp.array(X[idx_src])
        ys = jnp.array(Y[idx_tgt])
        
        # Use the precompiled Sinkhorn function.
        batch_cost = float(sinkhorn_batch(xs, ys), p = p)
        total_cost += batch_cost

    # Return the average cost across batches.
    return total_cost / num_batches


