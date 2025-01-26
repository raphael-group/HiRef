import operator
import functools
from functools import reduce
import torch

"""
Package for computing an optimal rank-annealing schedule for Hierarchical Refinement
"""

def min_sum_partial_products_with_factors(n, k, C):
    INF = float('inf')
    
    # dp[d][t] = minimal sum of partial products 
    #            if we want exactly t factors (each ≤ C) with product d.
    dp = [[INF]*(k+1) for _ in range(n+1)]
    
    # choice[d][t] = which first factor 'r' gave the optimal dp[d][t].
    choice = [[-1]*(k+1) for _ in range(n+1)]
    
    # Base case: t=1
    for d in range(1, n+1):
        if d <= C:
            dp[d][1] = d
            choice[d][1] = d  # the only factor is d itself
    
    # Fill dp for t = 2 .. k
    for t in range(2, k+1):
        for d in range(1, n+1):
            if dp[d][t-1] == INF and t > 1:
                # if dp[d][t-1] is already INF, no point continuing,
                # but we still need to check all possible r because
                # the product changes from d/r.
                pass
            # Try all possible r ≤ C that divide d
            for r in range(1, min(C,d)+1):
                if d % r == 0:
                    candidate = r + r * dp[d // r][t-1]
                    if candidate < dp[d][t]:
                        dp[d][t] = candidate
                        choice[d][t] = r
    
    # If dp[n][k] is INF, no valid factorization exists
    if dp[n][k] == INF:
        return None, []
    
    # Reconstruct the chosen factors from 'choice'
    factors = []
    d_cur, t_cur = n, k
    
    while t_cur > 0:
        r_cur = choice[d_cur][t_cur]
        factors.append(r_cur)
        d_cur //= r_cur
        t_cur -= 1
    
    return dp[n][k], factors

def factors(n):
    # Return list of all factors of an integer
    return set(reduce(
        list.__add__,
        ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

def max_factor_lX(n, max_X):
    factor_lst = factors(n)
    max_factor = 0
    for factor in factor_lst:
        if factor > max_factor and factor < max_X:
            max_factor = factor
    return max_factor

def optimal_rank_schedule(n, hierarchy_depth=6, max_Q=int(2**10), max_rank=16):
    Q = max_factor_lX(n, max_Q)
    ndivQ = int(n / Q)
    min_value, rank_schedule = min_sum_partial_products_with_factors(ndivQ, hierarchy_depth, max_rank)
    rank_schedule.sort()
    rank_schedule.append(Q)
    rank_schedule = [x for x in rank_schedule if x != 1]
    
    print(f'Optimized rank-annealing schedule: { rank_schedule }')
    
    assert functools.reduce(operator.mul, rank_schedule) == n, "Error! Rank-schedule does not factorize n!"
    '''
    RS = torch.tensor(rank_schedule)[:-1]
    print(RS)
    print(torch.cumprod(RS))
    LR_runs = torch.cumsum(torch.cumprod(RS)).sum() + 1
    print(f'Runs required before terminal Q-size case: {LR_runs}')
    '''
    return rank_schedule
    