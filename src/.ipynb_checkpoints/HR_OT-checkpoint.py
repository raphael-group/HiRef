import FRLC
from FRLC import FRLC_opt
import torch
import matplotlib.pyplot as plt

def hierarchical_refinement(X, Y, rank_schedule, solver, device='cpu', base_rank=1, \
                            clustering_type='soft', plot_clusterings=True):
    '''
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
    '''
    
    C = torch.cdist(X, Y)
    C = C / torch.max(C)
    
    F_t = [(torch.arange(len(X)).to(device), torch.arange(len(Y)).to(device))]

    for i, rank_level in enumerate(rank_schedule):
        # Iterate over ranks in the scheduler
        F_tp1 = []
        
        for (idxX, idxY) in F_t:
            
            if len(idxX) <= base_rank or len(idxY) <= base_rank:
                # Return tuple of base-rank sized index sets (e.g. (x,T(x)) for base_rank=1)
                F_tp1.append( ( idxX, idxY ) )
                continue

            submat = torch.index_select(C, 0, idxX)
            C_XY = torch.index_select(submat, 1, idxY)
            
            # Solve a low-rank OT sub-problem with black-box solver
            Q, R, diagG, errs = solver(C_XY, tau_in = 1000, tau_out=100, gamma=200, \
                                         r = rank_level, max_iter=200, device=device, min_iter = 25, 
                                               max_inneriters_balanced=100, max_inneriters_relaxed=100, \
                                              diagonalize_return=True, printCost=False)
            if plot_clusterings:
                plt.imshow(Q.cpu().numpy())
                plt.show()
                
                plt.imshow(R.cpu().numpy())
                plt.show()
            
            print( torch.diag(diagG) )
            # TODO: case where C_XY itself is low-rank
            
            # Compute cluster-assignments for next level
            # Q: Shape ( len(idxX) x r_level )
            # R: Shape ( len(idxY) x r_level )
            
            capacity = int( X.shape[0] / torch.prod(rank_schedule[0:i+1]) )
            print(f'Next level cluster capacity: {capacity}')
            
            idx_seenX, idx_seenY = torch.arange(Q.shape[0]).to(device), torch.arange(R.shape[0]).to(device)
            
            if clustering_type == 'soft':
                for z in range(rank_level):
                    # Greedy assignment : ensure one _exactly_ fills the capacity of each cluster
                    
                    topk_values, topk_indices_X = torch.topk( Q[idx_seenX][:,z], k=capacity )
                    idxX_z = idxX[idx_seenX[topk_indices_X]]
                    topk_values, topk_indices_Y = torch.topk( R[idx_seenY][:,z], k=capacity )
                    idxY_z = idxY[idx_seenY[topk_indices_Y]]
                    
                    F_tp1.append(( idxX_z, idxY_z ))
                    
                    print(f'Length of assignment sets X: {len(topk_indices_X)}, Y: {len(topk_indices_Y)} [=capacity]')
                    
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
                    
                    F_tp1.append((idxX_z, idxY_z))
        
        F_t = F_tp1
    
    return F_t