import FRLC
from FRLC import FRLC_opt, FRLC_LR_opt
import util
import torch
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from typing import List, Callable, Union
import time

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
        # This is a dummy line -- this init doesn't compute C or its factorization
        self.sq_Euclidean = False
        
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
                            num_processes: Union[int, None] = None,
                              sq_Euclidean = False):
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
        obj.sq_Euclidean = sq_Euclidean
        
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

            if i == len(self.rank_schedule)-1:
                fin_iters = int(self.N / rank_level)
                print(f'Last level, rank chunk-size {rank_level} with {fin_iters} iterations to completion.')
                j = 0
            
            for (idxX, idxY) in F_t:

                if i == len(self.rank_schedule)-1:
                    print(f'{j}/{fin_iters} of final-level iterations to completion')
                    j += 1
                
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

    def _solve_LR_prob(self, idxX, idxY, rank_level, rankD, eps=0.04):
        """
        Solve problem for low-rank coupling under a low-rank factorization of distance matrix.
        """
        _x0, _x1 = torch.index_select(self.X, 0, idxX), torch.index_select(self.Y, 0, idxY)
        print(f'x0 shape: {_x0.shape}, x1 shape: {_x1.shape}, rankD: {rankD}')
        
        if rankD < _x0.shape[0]:
            
            C_factors, A_factors, B_factors = self.get_dist_mats(_x0, _x1, rankD, eps, self.sq_Euclidean )
            
            # Solve a low-rank OT sub-problem with black-box solver
            Q, R, diagG, errs = self.solver(C_factors, A_factors, B_factors,
                                       gamma=30,
                                       r = rank_level,
                                       max_iter=60,
                                       device=self.device,
                                       min_iter = 25,
                                       max_inneriters_balanced=100,
                                       max_inneriters_relaxed=40,
                                       diagonalize_return=True,
                                       printCost=False, tau_in=100000,
                                        dtype = _x0.dtype)
        
        else:
            if self.sq_Euclidean:
                C_XY = torch.cdist(_x0, _x1)**2
            else:
                # normal Euclidean distance otherwise
                C_XY = torch.cdist(_x0, _x1)
            
            Q, R, diagG, errs = FRLC_opt(C_XY,
                                   gamma=30,
                                   r = rank_level,
                                   max_iter=60,
                                   device=self.device,
                                   min_iter = 25,
                                   max_inneriters_balanced=100,
                                   max_inneriters_relaxed=40,
                                   diagonalize_return=True,
                                   printCost=False, tau_in=100000,
                                       dtype = C_XY.dtype)
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
                                   max_iter=50,
                                   device=self.device,
                                   min_iter = 15,
                                   max_inneriters_balanced=100,
                                   max_inneriters_relaxed=40,
                                   diagonalize_return=True,
                                   printCost=False, tau_in=100000,
                                       dtype = C_XY.dtype)
        return Q, R
    
    def _compute_coupling_from_Ft(self):
        """
        Returns coupling as a full-rank matrix rather than as a set of (x, T(x)) pairs.
        """
        size = (self.N, self.N)
        P = torch.zeros(size)
        for pair in self.Monge_clusters:
            idx1, idx2 = pair
            P[idx1, idx2] = 1
        return P / self.N
    
    def compute_OT_cost(self):
        """
        Compute the optimal transport in linear space and time (w/o coupling).
        """
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

    
    def get_dist_mats(self, _x0, _x1, rankD, eps , sq_Euclidean ):
        
        # Wasserstein-only, setting A and B factors to be NoneType
        A_factors = None
        B_factors = None
        
        if sq_Euclidean:
            # Sq Euclidean
            C_factors = compute_lr_sqeuclidean_matrix(_x0, _x1, True)
        else:
            # Standard Euclidean dist
            C_factors = self.ret_normalized_cost(_x0, _x1, rankD, eps)
        
        return C_factors, A_factors, B_factors
    
    def ret_normalized_cost(self, X, Y, rankD, eps):
        
        C1, C2 = util.low_rank_distance_factorization(X,
                                                      Y,
                                                      r=rankD,
                                                      eps=eps,
                                                      device=self.device)
        # Normalize appropriately
        c = ( C1.max()**1/2 ) * ( C2.max()**1/2 )
        C1, C2 = C1/c, C2/c
        C_factors = (C1.to(X.dtype), C2.to(X.dtype))
        
        return C_factors


def compute_lr_sqeuclidean_matrix(X_s,
                                  X_t,
                                  rescale_cost,
                                  device=None,
                                  dtype=None):
    """
    Adapted from "Section 3.5, proposition 1" in Scetbon, M., Cuturi, M., & Peyré, G. (2021).
    """
    dtype, device = X_s.dtype, X_s.device
    
    ns, dim = X_s.shape
    nt, _ = X_t.shape
    
    # First low rank decomposition of the cost matrix (M1)
    # Compute sum of squares for each source sample
    sum_Xs_sq = torch.sum(X_s ** 2, dim=1).reshape(ns, 1)  # Shape: (ns, 1)
    ones_ns = torch.ones((ns, 1), device=device, dtype=dtype)  # Shape: (ns, 1)
    neg_two_Xs = -2 * X_s  # Shape: (ns, dim)
    M1 = torch.cat((sum_Xs_sq, ones_ns, neg_two_Xs), dim=1)  # Shape: (ns, dim + 2)
    
    # Second low rank decomposition of the cost matrix (M2)
    ones_nt = torch.ones((nt, 1), device=device, dtype=dtype)  # Shape: (nt, 1)
    sum_Xt_sq = torch.sum(X_t ** 2, dim=1).reshape(nt, 1)  # Shape: (nt, 1)
    Xt = X_t  # Shape: (nt, dim)
    M2 = torch.cat((ones_nt, sum_Xt_sq, Xt), dim=1)  # Shape: (nt, dim + 2)
    
    if rescale_cost:
        # Compute the maximum value in M1 and M2 for rescaling
        max_M1 = torch.max(M1)
        max_M2 = torch.max(M2)
        
        # Avoid division by zero
        if max_M1 > 0:
            M1 = M1 / torch.sqrt(max_M1)
        if max_M2 > 0:
            M2 = M2 / torch.sqrt(max_M2)
    
    return (M1, M2.T)



