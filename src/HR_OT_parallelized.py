import FRLC
from FRLC import FRLC_opt, FRLC_LR_opt
import util
import torch
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from typing import List, Callable, Union
import time

"""

Parallelized version (not object oriented for now)

"""


def HROT_worker_subproblem(args):
    
    C_XY, idxX, idxY, solver, i, rank_level, rank_schedule, base_rank, clustering_type, plot_clusterings, N = args
    
    # Parallelization not GPU optimized currently (but can be, of course)
    device='cpu'
    
    if len(idxX) <= base_rank or len(idxY) <= base_rank:
                # Return tuple of base-rank sized index sets (e.g. (x,T(x)) for base_rank=1)
                return [ ( idxX, idxY ) ]
    
    try:
        # Solve a low-rank OT sub-problem with black-box solver
        Q, R, diagG, errs = FRLC_opt(C_XY,
                                   gamma=30,
                                   r = rank_level,
                                   max_iter= 120,
                                   device=device,
                                   min_iter = 40, 
                                   max_inneriters_balanced=100, 
                                   max_inneriters_relaxed=100,
                                   diagonalize_return=True,
                                   printCost=False, tau_in=100000)
        
    except Exception as e:
        print(f'Solver failed with error: {e}')
    
    if plot_clusterings:
        plt.imshow(Q.cpu().numpy())
        plt.show()
        
        plt.imshow(R.cpu().numpy())
        plt.show()
        
        # to print inner-marginal and verify it is balanced, may print: torch.diag(diagG) 
    
    # TODO: case where C_XY itself is low-rank
    
    # Compute cluster-assignments for next level
    # Q: Shape ( len(idxX) x r_level )
    # R: Shape ( len(idxY) x r_level )

    # Next level cluster capacity
    capacity = int( N / torch.prod(rank_schedule[0:i+1]) )

    new_coclusters = []
    
    if clustering_type == 'soft':
        
        idx_seenX = torch.arange(Q.shape[0]).to(device)
        idx_seenY = torch.arange(R.shape[0]).to(device)
        
        for z in range(rank_level):
            
            # Greedy assignment : ensure one _exactly_ fills the capacity of each cluster
            topk_values, topk_indices_X = torch.topk( Q[idx_seenX][:,z], k=capacity )
            idxX_z = idxX[idx_seenX[topk_indices_X]]
            topk_values, topk_indices_Y = torch.topk( R[idx_seenY][:,z], k=capacity )
            idxY_z = idxY[idx_seenY[topk_indices_Y]]
            
            new_coclusters.append(( idxX_z, idxY_z ))
            
            #print(f'Length of assignment sets X: {len(topk_indices_X)}, Y: {len(topk_indices_Y)} [=capacity]')
            idx_seenX = idx_seenX[~torch.isin(idx_seenX, idx_seenX[topk_indices_X])]
            idx_seenY = idx_seenY[~torch.isin(idx_seenY, idx_seenY[topk_indices_Y])]
    
    else:
        zX = torch.argmax(Q, axis=1) # X-assignments
        zY = torch.argmax(R, axis=1) # Y-assignments
        
        for z in range(rank_level):
            
            idxX_z = idxX[zX == z]
            idxY_z = idxY[zY == z]

            assert len(idxX_z) == len(idxY_z) == capacity, \
                                "Assertion failed! Not a hard-clustering function, or point sets of unequal size!"
            
            new_coclusters.append(( idxX_z, idxY_z ))

    return new_coclusters


def hierarchical_refinement_parallelized(C, rank_schedule, solver, base_rank=1, \
                            clustering_type='soft', plot_clusterings=False, return_as_coupling=False, \
                                         num_processes=None):
    '''
    A wrapper for solving HR-OT using node parallelization (cpu)
    
    Parameters
    ----------
    X: (n,d) torch.tensor
        First dataset of n points in R^d
    Y: (n,d) torch.tensor
        Second dataset of n points in R^d
    rank_schedule: list
        List of ranks to use at each level of hierarchical refinement
        which factorizes n:
                    rank_schedule[0]*...*rank_schedule[-1] = n
    solver: function
        Black-box solver for low-rank OT instance
    base_rank: int (default: 1)
        Base-case rank at which to output minimal set of tuples (x, T(x))
    clustering_type: str
        If the low-rank OT returns an approximate "soft" clustering, 
        forces it to be hard one for cluster sizes to remain Monge-consistent.
        Otherwise, takes arg-max.
    plot_clusterings: bool
        If true, plots a heatmap of the Q and R clusterings.
    return_as_coupling: bool
        If true, returns a coupling matrix. Else, returns a sparse list of points with their Monge mapping.
    num_processes: int
        The number of nodes (CPUs) on which to run.
    
     Returns
    -------
    gamma : torch.tensor of shape (N, N)
        Optimal transport plan.
    (OR)
    F_t : list (size N)
        List of (x, T(x)) tuples
    '''
    
    # Assuming strictly CPU Parallelization (todo: extend to gpu)
    device = 'cpu'
    
    mp.set_start_method("spawn", force=True)
    C.share_memory_()
    
    F_t = [(torch.arange(C.shape[0]).to(device), torch.arange(C.shape[1]).to(device))]
    N = C.shape[0]
    
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    with mp.Pool(processes = num_processes) as pool: # , initializer=init_worker_process, initargs=(C,)
        
        for i, rank_level in enumerate(rank_schedule):
            # Iterate over ranks in the scheduler
            F_tp1 = []
    
            subproblem_args = []
            # Building args as input for mp
            for (idxX, idxY) in F_t:
                
                submat = torch.index_select(C, 0, idxX)
                C_XY = torch.index_select(submat, 1, idxY)
                subproblem_args.append(
                    ( C_XY, 
                     idxX, 
                     idxY, 
                     solver, 
                     i, 
                     rank_level, 
                     rank_schedule, 
                     base_rank, 
                     clustering_type, 
                     plot_clusterings, 
                     N )
                )
            
            print(f'Mapping subproblems')
            
            level_results = pool.map( HROT_worker_subproblem, subproblem_args )
            
            for co_cluster_lst in level_results:
                F_tp1.extend(co_cluster_lst)
            
            F_t = F_tp1

    if return_as_coupling is False:
        # Returns set of x,T(x) tuples
        return F_t

    else:
        # Returns a sparse coupling
        return compute_coupling_from_Ft(F_t, C.shape)

