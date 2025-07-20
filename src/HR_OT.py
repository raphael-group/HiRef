# Solver imports
import FRLC
from FRLC import FRLC_opt, FRLC_LR_opt
from LR_mini import LROT_opt, LROT_LR_opt
import util
# Other misc imports
import torch
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from typing import List, Callable, Union, Dict, Any
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
    solver_params: Dict[str, Any], optional
        Additional parameters for the low-rank solver. If None, default values are used.
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
    X, Y : torch.tensor
        The point-clouds for the first dataset (X) and the second dataset (Y)
    N : int
        The size of the dataset.
    Monge_clusters : list (tuples of type torch.float)
        A list containing the Monge-map pairings
    """
    
    def __init__(self,
                C: torch.Tensor,
                 rank_schedule: List[int],
                 solver: Callable = FRLC_opt,
                 solver_params: Union[Dict[str, Any] , None] = None,
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
        
        # Setting parameters to use with the FRLC solver
        default_solver_params = {
            'gamma' : 90,
            'max_iter' : 30,
            'min_iter' : 15,
            'max_inneriters_balanced' : 300,
            'max_inneriters_relaxed' : 300,
            'printCost' : False,
            'tau_in' : 100000
        }
        
        if solver_params is not None:
            default_solver_params.update(solver_params)
        self.solver_params = default_solver_params
        
        assert C.shape[0] == C.shape[1], "Currently assume square costs so that |X| = |Y| = N"
    
    @classmethod
    def init_from_point_clouds(cls,
                            X: torch.Tensor,
                            Y: torch.Tensor,
                            rank_schedule: List[int],
                            distance_rank_schedule: Union[List[int], None] = None,
                            solver: Callable = FRLC_LR_opt,
                            solver_full: Callable = FRLC_opt,
                            solver_params: Union[Dict[str, Any] , None] = None,
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
        solver_full : callable
            A low-rank OT solver that takes a full cost submatrix (not low-rank cost) 
            and returns Q, R, diagG, errs.
        distance_rank_schedule: List[int]
            A separate rank-schedule for the low-rank distance matrix being factorized.
        sq_Euclidean : bool
            If True, assumes squared Euclidean cost. Otherwise, defaults to Euclidean.
            Needed for the point-cloud variant, in order to define a distance metric
            to use for the low-rank approximation of C.
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
        obj.solver_full = solver_full
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
        
        if sq_Euclidean:
            # compute once on the full point clouds
            obj.C_factors_global = compute_lr_sqeuclidean_matrix(
                obj.X, obj.Y, rescale_cost=True
            )
        
        # Setting parameters to use with the FRLC solver
        default_solver_params = {
            'gamma' : 90,
            'max_iter' : 30,
            'min_iter' : 15,
            'max_inneriters_balanced' : 300,
            'max_inneriters_relaxed' : 300,
            'printCost' : False,
            'tau_in' : 100000
        }
        if solver_params is not None:
            default_solver_params.update(solver_params)
        obj.solver_params = default_solver_params
        
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
            """ WARNING: not currently implemented in this class! See HR_OT_parallelized instead! """
            return self._hierarchical_refinement_parallelized(return_as_coupling = return_as_coupling)
        else:
            return self._hierarchical_refinement(return_as_coupling = return_as_coupling)

    def _hierarchical_refinement(self, return_as_coupling: bool = False):
        """
        Single-process (serial) Hierarchical Refinement
        """

        # Define partitions
        F_t = [(torch.arange( self.N , device=self.device), 
                torch.arange( self.N , device=self.device))]
        
        for i, rank_level in enumerate(self.rank_schedule):
            # Iterate over ranks in the scheduler
            F_tp1 = []
            
            print(f'\n--- Rank Level {i + 1} of {len(self.rank_schedule)} ---')
            
            if i == len(self.rank_schedule)-1:
                fin_iters = int(self.N / rank_level)
                print(f'>>> Final Rank Level | Rank Chunk Size: {rank_level} | Remaining Iterations: {fin_iters}')
                j = 0
            
            for (idxX, idxY) in F_t:

                if i == len(self.rank_schedule)-1:
                    
                    print(f'    Base-Level Iteration {j + 1}/{fin_iters} ')
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
                    # If visualizing the Q - R clustering matrices.
                    
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
                capacity = int(self.N) / int(torch.prod(torch.Tensor(self.rank_schedule[0:i+1])))
                capacity = int(capacity)
                
                idx_seenX, idx_seenY = torch.arange(Q.shape[0], device=self.device), \
                                                    torch.arange(R.shape[0], device=self.device)
                
                # Split by hard or soft-clustering
                if self.clustering_type == 'soft':
                    # If using a solver which returns "soft" clusterings, must strictly fill partitions to capacities.
                    
                    for z in range(rank_level):
                        
                        topk_values, topk_indices_X = torch.topk( Q[idx_seenX][:,z], k=capacity )
                        idxX_z = idxX[idx_seenX[topk_indices_X]]
                        topk_values, topk_indices_Y = torch.topk( R[idx_seenY][:,z], k=capacity )
                        idxY_z = idxY[idx_seenY[topk_indices_Y]]
                        
                        F_tp1.append(( idxX_z, idxY_z ))
                        
                        idx_seenX = idx_seenX[~torch.isin(idx_seenX, idx_seenX[topk_indices_X])]
                        idx_seenY = idx_seenY[~torch.isin(idx_seenY, idx_seenY[topk_indices_Y])]
                
                elif self.clustering_type == 'hard':
                    # If using a solver which returns "hard" clusterings, can exactly take argmax.
                    
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
        # In separate file, without low-rank distance matrix for the moment!
        raise NotImplementedError

    def _solve_LR_prob(self, idxX, idxY, rank_level, rankD, eps=0.04):
        
        """
        Solve problem for low-rank coupling under a low-rank factorization of distance matrix.
        """
        
        if rankD < idxX.numel():
            
            C_factors, A_factors, B_factors = self.get_dist_mats(
                            idxX, idxY, rankD, eps, self.sq_Euclidean 
                                    )
            
            # Solve a low-rank OT sub-problem with black-box solver
            Q, R, _, _ = self.solver(
                                    C_factors, A_factors, B_factors,
                                       gamma = self.solver_params['gamma'],
                                       r = rank_level,
                                       max_iter = self.solver_params['max_iter'],
                                       device=self.device,
                                       min_iter = self.solver_params['min_iter'],
                                       max_inneriters_balanced = self.solver_params['max_inneriters_balanced'],
                                       max_inneriters_relaxed = self.solver_params['max_inneriters_relaxed'],
                                       diagonalize_return = True,
                                       printCost = self.solver_params['printCost'],
                                        tau_in = self.solver_params['tau_in'],
                                        dtype = self.X.dtype
                                    )
            return Q, R
            
        else:
            
            _x0, _x1 = torch.index_select(self.X, 0, idxX), torch.index_select(self.Y, 0, idxY)
            
            # Final base instance -- can compute within-cluster costs explicitly
            # (explicit dense cost for tiny size cluster)
            
            C_XY = torch.cdist(_x0, _x1, p=2)**2 if self.sq_Euclidean else torch.cdist(_x0, _x1, p=2)
            
            # LR Solver for full cost
            Q, R, _, _ = self.solver_full(C_XY,
                                   gamma = self.solver_params['gamma'],
                                   r = rank_level,
                                   max_iter = self.solver_params['max_iter'],
                                   device = self.device,
                                   min_iter = self.solver_params['min_iter'],
                                   max_inneriters_balanced = self.solver_params['max_inneriters_balanced'],
                                   max_inneriters_relaxed = self.solver_params['max_inneriters_relaxed'],
                                   diagonalize_return=True,
                                   printCost=self.solver_params['printCost'], tau_in = self.solver_params['tau_in'],
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
                                   gamma = self.solver_params['gamma'],
                                   r = rank_level,
                                   max_iter = self.solver_params['max_iter'],
                                   device=self.device,
                                   min_iter = self.solver_params['min_iter'],
                                   max_inneriters_balanced = self.solver_params['max_inneriters_balanced'],
                                   max_inneriters_relaxed = self.solver_params['max_inneriters_relaxed'],
                                   diagonalize_return=True,
                                   printCost=self.solver_params['printCost'], tau_in = self.solver_params['tau_in'],
                                    dtype = C_XY.dtype)
        return Q, R
    
    def _compute_coupling_from_Ft(self):
        """
        Returns coupling as a full-rank matrix rather than as a set of (x, T(x)) pairs.
        """
        size = (self.N, self.N)
        P = torch.zeros(size)
        
        # Fill sparse coupling with entries
        
        for pair in self.Monge_clusters:
            idx1, idx2 = pair
            P[idx1, idx2] = 1
        # Return, trivially normalized to satisfy standard OT constraints
        return P / self.N
    
    def compute_OT_cost(self):
        """
        Compute the optimal transport in linear space and time (w/o coupling).
        """
        
        cost = 0
        for clus in self.Monge_clusters:
            idx1, idx2 = clus
            
            if self.C is not None:
                # If C saved, index into general cost directly
                cost += self.C[idx1, idx2]
            else:
                # In case point-cloud init used, must directly compute distances between point pairs in X, Y.
                if self.sq_Euclidean:
                    # squared Euclidean case
                    cost += torch.norm(self.X[idx1,:] - self.Y[idx2,:])**2
                else:
                    # normal Euclidean cost
                    cost += torch.norm(self.X[idx1,:] - self.Y[idx2,:])
                    
        # Appropriately normalize the cost
        cost = cost / self.N
        return cost
    
    def get_dist_mats(self, idxX, idxY, rankD, eps , sq_Euclidean ):
        
        # Wasserstein-only, setting A and B factors to be NoneType
        A_factors = None
        B_factors = None
        
        if sq_Euclidean:
            # Sq-Euclidean dist
            
            M1_global, M2T_global = self.C_factors_global
            C_factors = ( M1_global[idxX], 
                         M2T_global[:, idxY] )
            
            # alternatively (slower): 
            # _x0, _x1 = torch.index_select(self.X, 0, idxX), torch.index_select(self.Y, 0, idxY)
            # C_factors = compute_lr_sqeuclidean_matrix(_x0, _x1, True)
        
        else:
            # Standard Euclidean dist
            _x0, _x1 = torch.index_select(self.X, 0, idxX), torch.index_select(self.Y, 0, idxY)
            C_factors = self.ret_normalized_cost(_x0, _x1, 
                                                 rankD, eps)
        
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
    
    A function for computing a low-rank factorization of a squared Euclidean distance matrix.
    
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



