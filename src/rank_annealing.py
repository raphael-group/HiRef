import operator
import functools
from functools import reduce
import torch

"""
Package for computing an optimal rank-annealing schedule for Hierarchical Refinement
"""

def optimal_rank_schedule(n, hierarchy_depth=6, max_Q=int(2**10), max_rank=16):
    """
    
    A function to compute the optimal rank-scheduler of refinement.
    
    Parameters
    ----------
    n: int
        Size of the input dataset -- cannot be a prime number
    hierarchy_depth: int
        Maximal permissible depth of the multi-scale hierarchy
    max_Q: int
        Maximal rank at terminal base case (before reducing the \leq max_Q rank coupling to a 1-1 alignment)
    max_rank: int
        Maximal rank at the intermediate steps of the rank-schedule
        
    """

    # Factoring out the max factor
    Q = max_factor_lX(n, max_Q)
    ndivQ = int(n / Q)

    # Compute partial rank schedule up to Q
    min_value, rank_schedule = min_sum_partial_products_with_factors(ndivQ, hierarchy_depth, max_rank)
    rank_schedule.sort()
    rank_schedule.append(Q)
    rank_schedule = [x for x in rank_schedule if x != 1]
    
    print(f'Optimized rank-annealing schedule: { rank_schedule }')
    
    assert functools.reduce(operator.mul, rank_schedule) == n, "Error! Rank-schedule does not factorize n!"
    
    return rank_schedule

def factors(n):
    # Return list of all factors of an integer
    return set(reduce(
        list.__add__,
        ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

def max_factor_lX(n, max_X):
    # Find max factor of n , such that max_factor \leq max_X
    factor_lst = factors(n)
    max_factor = 0
    for factor in factor_lst:
        if factor > max_factor and factor < max_X:
            max_factor = factor
    return max_factor

def min_sum_partial_products_with_factors(n, k, C):
    """
    Dynamic program to compute the rank-schedule, subject to a constraint of intermediates being \leq C

    Parameters
    ----------
    n: int
        The dataset size to be factored into a rank-scheduler. Assumed to be non-prime.
    k: int
        The depth of the hierarchy.
    C: int
        A constraint on the maximal intermediate rank across the hierarchy.
    
    """
    INF = float('inf')
    
    dp = [[INF]*(k+1) for _ in range(n+1)]
    choice = [[-1]*(k+1) for _ in range(n+1)]
    
    for d in range(1, n+1):
        if d <= C:
            dp[d][1] = d
            choice[d][1] = d
    
    for t in range(2, k+1):
        for d in range(1, n+1):
            if dp[d][t-1] == INF and t > 1:
                pass
            
            for r in range(1, min(C,d)+1):
                if d % r == 0:
                    candidate = r + r * dp[d // r][t-1]
                    if candidate < dp[d][t]:
                        dp[d][t] = candidate
                        choice[d][t] = r
    
    
    if dp[n][k] == INF:
        return None, []
    
    factors = []
    d_cur, t_cur = n, k
    
    while t_cur > 0:
        r_cur = choice[d_cur][t_cur]
        factors.append(r_cur)
        d_cur //= r_cur
        t_cur -= 1
    
    return dp[n][k], factors

