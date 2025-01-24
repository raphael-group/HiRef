import FRLC
from FRLC import FRLC_opt, FRLC_LR_opt
import util
import torch
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from typing import List, Callable, Union

class HierarchicalRefinementOT:
    """
    A class to perform Hierarchical OT refinement with optional (CPU) parallelization.
    
    Attributes
    ----------
    C : torch.tensor
        The cost matrix of shape (N, N), currently assumed square for hierarchical OT.
        Can represent general user-defined costs.
    rank_schedule : list
        The list of ranks for each hierarchical level -- i.e. the rank-annealing schedule.
    solver : callable
        A low-rank OT solver that takes a cost submatrix and returns Q, R, diagG, errs.
    device : str
        The device ('cpu' or 'cuda') to be used for computations.
    base_rank : int
        Base-case rank at which to stop subdividing clusters.
    clustering_type : str
        'soft' or 'hard'. Determines how cluster assignments are computed after each OT solve.
    plot_clusterings : bool
        Whether to plot the Q and R matrices at each step for debugging.
    parallel : bool
        Whether to execute each subproblem at a level in parallel.
    num_processes : int or None
        Number of worker processes to spawn (if `parallel=True`). Defaults to `None` which uses `mp.cpu_count()`.
    """
    
    def __init__(self,
                C: torch.Tensor,
                 rank_schedule: List[int],
                 solver: Callable = FRLC_opt,
                 device: str = 'cpu',
                 base_rank: int = 1,
                 clustering_type: str = 'soft',
                 plot_clusterings: bool = False,
                 parallel: bool = False,
                 num_processes: Union[int, None] = None
                ):
    
        self.C = C.to(device)
        self.rank_schedule = rank_schedule
        self.solver = solver
        self.device = device
        self.base_rank = base_rank
        self.clustering_type = clustering_type
        self.plot_clusterings =  plot_clusterings
        self.parallel = parallel
        self.num_processes = num_processes

        # Point clouds optional attributes
        self.X, self.Y = None, None
        self.N = C.shape[0]
        self.Monge_clusters = None
        
        assert C.shape[0] == C.shape[1], "Currently assume square costs so that |X| = |Y| = N"
    
    @classmethod
    def init_from_point_clouds(cls,
                            X: torch.Tensor,
                            Y: torch.Tensor,
                            rank_schedule: List[int],
                            distance_rank_schedule: Union[List[int], None] = None,
                            solver: Callable = FRLC_LR_opt,
                            device: str = 'cpu',
                            base_rank: int = 1,
                            clustering_type: str = 'soft',
                            plot_clusterings: bool = False,
                            parallel: bool = False,
                            num_processes: Union[int, None] = None):
        r"""
        Constructor for initializing from point clouds.
        
        Attributes
        ----------
        X : torch.tensor
            The point-cloud of shape N for measure \mu
        Y: torch.tensor
            Point cloud of shape N for measure \nu
        distance_rank_schedule: List[int]
            A separate rank-schedule for the low-rank distance matrix being factorized.
        """
        
        obj = cls.__new__(cls)
        
        obj.X = X
        obj.Y = Y
        obj.rank_schedule = rank_schedule
        
        if distance_rank_schedule is None:
            # Default: assume distance rank schedule is identical to rank schedule for coupling.
            obj.distance_rank_schedule = rank_schedule
        else:
            obj.distance_rank_schedule = distance_rank_schedule
        
        obj.solver = solver
        obj.device = device
        obj.base_rank = base_rank
        obj.clustering_type = clustering_type
        obj.plot_clusterings =  plot_clusterings
        obj.parallel = parallel
        obj.num_processes = num_processes
        obj.N = X.shape[0]
        
        # Cost-mat an optional attribute
        obj.C = None
        obj.Monge_clusters = None
        
        assert X.shape[0] == Y.shape[0], "Currently assume square costs so that |X| = |Y| = N"
        
        return obj

    def run(self, return_as_coupling: bool = False):
        """
        Routine to run hierarchical refinement.
        
        Parameters
        ----------
        return_as_coupling : bool
            Whether to return a full coupling matrix (size NxN) 
            or a list of (idxX, idxY) co-clusters / assignments.
        
        Returns
        -------
        list of (idxX, idxY) pairs OR torch.tensor
            If return_as_coupling=False: returns a list of tuples (idxX, idxY) for each co-cluster.
            If return_as_coupling=True: returns a dense coupling matrix of shape (N, N).
        """
        if self.parallel:
            return self._hierarchical_refinement_parallelized(return_as_coupling = return_as_coupling)
        else:
            return self._hierarchical_refinement(return_as_coupling = return_as_coupling)

    def _hierarchical_refinement(self, return_as_coupling: bool = False):
        """
        Single-process (serial) Hierarchical Refinement
        """
            
        F_t = [(torch.arange( self.N , device=self.device), 
                torch.arange( self.N , device=self.device))]
    
        for i, rank_level in enumerate(self.rank_schedule):
            # Iterate over ranks in the scheduler
            F_tp1 = []
            
            for (idxX, idxY) in F_t:
                
                if len(idxX) <=self.base_rank or len(idxY) <= self.base_rank:
                    # Return tuple of base-rank sized index sets (e.g. (x,T(x)) for base_rank=1)
                    F_tp1.append( ( idxX, idxY ) )
                    continue

                if self.C is not None:
                    Q,R = self._solve_prob( idxX, idxY, rank_level)
                else:
                    rank_D = self.distance_rank_schedule[i]
                    Q,R = self._solve_LR_prob( idxX, idxY, rank_level, rank_D )
                
                if self.plot_clusterings:
                    plt.figure(figsize=(12, 5))
                    plt.subplot(1, 2, 1)
                    plt.imshow(Q.detach().cpu().numpy(), aspect='auto', cmap='viridis')
                    plt.title(f"Q Clustering Level {i+1}")
                    plt.colorbar()
    
                    plt.subplot(1, 2, 2)
                    plt.imshow(R.detach().cpu().numpy(), aspect='auto', cmap='viridis')
                    plt.title(f"R Clustering Level {i+1}")
                    plt.colorbar()
                    plt.show()
                
                # Next level cluster capacity
                capacity = int( self.N / torch.prod( torch.Tensor(self.rank_schedule[0:i+1]) ) )
                idx_seenX, idx_seenY = torch.arange(Q.shape[0], device=self.device), \
                                                    torch.arange(R.shape[0], device=self.device)
                
                # Split by hard or soft-clustering
                if self.clustering_type == 'soft':
                    for z in range(rank_level):
                        topk_values, topk_indices_X = torch.topk( Q[idx_seenX][:,z], k=capacity )
                        idxX_z = idxX[idx_seenX[topk_indices_X]]
                        topk_values, topk_indices_Y = torch.topk( R[idx_seenY][:,z], k=capacity )
                        idxY_z = idxY[idx_seenY[topk_indices_Y]]
                        
                        F_tp1.append(( idxX_z, idxY_z ))
                        
                        idx_seenX = idx_seenX[~torch.isin(idx_seenX, idx_seenX[topk_indices_X])]
                        idx_seenY = idx_seenY[~torch.isin(idx_seenY, idx_seenY[topk_indices_Y])]
                
                elif clustering_type == 'hard':
                    zX = torch.argmax(Q, axis=1) # X-assignments
                    zY = torch.argmax(R, axis=1) # Y-assignments
                    
                    for z in range(rank_level):
                        
                        idxX_z = idxX[zX == z]
                        idxY_z = idxY[zY == z]
    
                        assert len(idxX_z) == len(idxY_z) == capacity, \
                                            "Assertion failed! Not a hard-clustering function, or point sets of unequal size!"
                        
                        F_tp1.append((idxX_z, idxY_z))
            
            F_t = F_tp1
        
        self.Monge_clusters = F_t
        
        if return_as_coupling is False:
            return self.Monge_clusters
        else:
            return self._compute_coupling_from_Ft()

    def _hierarchical_refinement_parallelized():
        
        raise NotImplementedError

    def _solve_LR_prob(self, idxX, idxY, rank_level, rankD, eps=0.02):
        """
        Solve problem for low-rank coupling under a low-rank factorization of distance matrix.
        """
        _x0, _x1 = torch.index_select(self.X, 0, idxX), torch.index_select(self.Y, 0, idxY)
        C1, C2 = util.low_rank_distance_factorization(_x0,
                                                      _x1,
                                                      r=rankD,
                                                      eps=eps,
                                                      device=self.device)
        # Normalize appropriately
        c = ( C1.max()**1/2 ) * ( C2.max()**1/2 )
        C1, C2 = C1/c, C2/c
        C_factors = (C1, C2)
        
        A_factors = None
        B_factors = None
        
        # Solve a low-rank OT sub-problem with black-box solver
        Q, R, diagG, errs = self.solver(C_factors, A_factors, B_factors,
                                   gamma=30,
                                   r = rank_level,
                                   max_iter=120,
                                   device=self.device,
                                   min_iter = 40,
                                   max_inneriters_balanced=100,
                                   max_inneriters_relaxed=100,
                                   diagonalize_return=True,
                                   printCost=False, tau_in=100000,
                                       dtype = C1.dtype)
        return Q, R
        
    def _solve_prob(self, idxX, idxY, rank_level):
        """
        Solve problem for low-rank coupling assuming cost sub-matrix.
        """
        
        # Index into sub-cost
        submat = torch.index_select(self.C, 0, idxX)
        C_XY = torch.index_select(submat, 1, idxY)
        
        # Solve a low-rank OT sub-problem with black-box solver
        Q, R, diagG, errs = self.solver(C_XY,
                                   gamma=30,
                                   r = rank_level,
                                   max_iter=120,
                                   device=self.device,
                                   min_iter = 40,
                                   max_inneriters_balanced=100,
                                   max_inneriters_relaxed=100,
                                   diagonalize_return=True,
                                   printCost=False, tau_in=100000,
                                       dtype = C_XY.dtype)
        return Q, R

    def _compute_coupling_from_Ft(self):
        '''
        Returns coupling as a full-rank matrix rather than as a set of (x, T(x)) pairs.
        '''
        size = (self.N, self.N)
        P = torch.zeros(size)
        for pair in self.Monge_clusters:
            idx1, idx2 = pair
            P[idx1, idx2] = 1
        return P / self.N
    
    def compute_OT_cost(self):
        cost = 0
        for clus in self.Monge_clusters:
            idx1, idx2 = clus
            if self.C is not None:
                cost += self.C[idx1, idx2]
            else:
                # Note: LR-factorization for Euclidean dist
                cost += torch.norm(self.X[idx1,:] - self.Y[idx2,:])
        cost = cost / self.N
        return cost
                



"""

Below: parallelized version (not object oriented for now)

"""


def HROT_worker_subproblem(args):
    
    C_XY, idxX, idxY, solver, i, rank_level, rank_schedule, base_rank, clustering_type, plot_clusterings, N = args
    #C_XY = global_cost_matrix[idxX][:, idxY]
    
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
        print(torch.diag(diagG))
        
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
                    ( C_XY, idxX, idxY, solver, i, rank_level, rank_schedule, base_rank, clustering_type, plot_clusterings, N )
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

