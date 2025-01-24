
import torch
import util
import objective_grad as gd
import matplotlib.pyplot as plt



def FRLC_opt(C, a=None, b=None, A=None, B=None, tau_in = 50, tau_out=50, \
             gamma=90, r = 10, r2=None, max_iter=200, device='cpu', dtype=torch.float64, \
                  semiRelaxedLeft=False, semiRelaxedRight=False, Wasserstein=True, \
                 printCost=True, returnFull=False, FGW=False, alpha=0.0, unbalanced=False, \
                  initialization='Full', init_args = None, full_grad=True, \
                   convergence_criterion=True, tol=1e-5, min_iter = 25, \
                   min_iterGW = 500, max_iterGW = 1000, \
                   max_inneriters_balanced= 300, max_inneriters_relaxed=50, \
                  diagonalize_return=False):
    
    '''
    FRLC: Factor Relaxation with Latent Coupling
    ------Parameters------
    C: torch.tensor (N1 x N2)
        A matrix of pairwise feature distances in space X and space Y (inter-space).
    a: torch.tensor (N1)
        A vector representing marginal one.
    b: torch.tensor (N2)
        A vector representing marginal two.
    A: torch.tensor (N1 x N1)
        A matrix of pairwise distances between points in metric space X.
    B: torch.tensor (N2 x N2)
        A matrix of pairwise distances between points in metric space Y.
    tau_in: float (> 0)
        A scalar which controls the regularity of the inner marginal update path.
    tau_out: float (> 0)
        A scalar which controls the regularization of the outer marginals a and b (e.g. for semi-relaxed or unbalanced OT)
    gamma: float (> 0)
        The mirror descent step-size, a scalar which controls the scaling of gradients
        before being exponentiated into Sinkhorn kernels.
    r: int (> 1)
        A non-negative integer rank, controlling the rank of the FRLC learned OT coupling. 
    max_iter: int
        The maximal number of iterations FRLC will run until convergence.
    device: str
        The device (i.e. 'cpu' or 'cuda') which FRLC runs on.
    dtype: dtype
        The datatype all tensors are stored on (naturally there is a space-accuracy
        tradeoff for low-rank between 32 and 64 bit).
    semiRelaxedLeft: bool
        True if running the left-marginal relaxed low-rank algorithm.
    semiRelaxedRight: bool
        True if running the right-marginal relaxed low-rank algorithm.
    Wasserstein: bool
        True if using the Wasserstein loss <C, P>_F as the objective cost,
        else runs GW if FGW false and FGW if GW true.
    printCost: bool
        True if printing the value of the objective cost at each iteration.
        This can be expensive for large datasets if C is not factored.
    returnFull: bool
        True if returning P_r = Q Lambda R.T, else returns iterates (Q, R, T).
    FGW: bool
        True if running the Fused-Gromov Wasserstein problem, and otherwise false.
    alpha: float
        A balance parameter between the Wasserstein term and
        the Gromov-Wasserstein term of the objective.
    unbalanced: bool
        True if running the unbalanced problem;
        if semiRelaxedLeft/Right and unbalanced False (default) then running the balanced problem.
    initialization: str, 'Full' or 'Rank-2'
        'Full' if sub-couplings initialized to be full-rank, if 'Rank-2' set to a rank-2 initialization.
        We advise setting this to be 'Full'.
    init_args: tuple of 3-tensors
        A tuple of (Q0, R0, T0) for tuple[i] of type tensor
    full_grad: bool
        If True, evaluates gradient with rank-1 perturbations.
        Else if False, omits perturbation terms.
    convergence_criterion: bool
        If True, use the convergence criterion. Else if False, default to running up to max_iters.
    tol: float
        Tolerance used for established when convergence is reached.
    min_iter: int
        The minimum iterations for the algorithm to run for in the Wasserstein case.
    min_iterGW: int
        The minimum number of iterations to run for in the GW case.
    max_iterGW: int
        The maximum number of iterations to run for in the GW case.
    max_inneriters_balanced: int
        The maximum number of inner iterations for the Sinkhorn loop.
    max_inneriters_relaxed: int
        The maximum number of inner iterations for the relaxed and semi-relaxed loops.
    diagonalize_return: bool
        If True, diagonalize the LC-factorization to the form of Scetbon et al '21.
        Else if False, return the LC-factorization.
    '''

    
    N1, N2 = C.size(dim=0), C.size(dim=1)
    k = 0
    stationarity_gap = torch.inf
    
    one_N1 = torch.ones((N1), device=device, dtype=dtype)
    one_N2 = torch.ones((N2), device=device, dtype=dtype)
    
    if a is None:
        a = one_N1 / N1
    if b is None:
        b = one_N2 / N2
    if r2 is None:
        r2 = r

    one_r = torch.ones((r), device=device, dtype=dtype)
    one_r2 = torch.ones((r2), device=device, dtype=dtype)
    
    # Initialize inner marginals to uniform; 
    # generalized to be of differing dimensions to account for non-square latent-coupling.
    gQ = (1/r)*one_r
    gR = (1/r2)*one_r2
    
    full_rank = True if initialization == 'Full' else False
    
    if initialization == 'Full':
        full_rank = True
    elif initialization == 'Rank-2':
        full_rank = False
    else:
        full_rank = True
        print('Initialization must be either "Full" or "Rank-2", defaulting to "Full".')
        
    if init_args is None:
        Q, R, T, Lambda = util.initialize_couplings(a, b, gQ, gR, \
                                                    gamma, full_rank=full_rank, \
                                                device=device, dtype=dtype, \
                                                    max_iter = max_inneriters_balanced)
    else:
        Q, R, T = init_args
        Lambda = torch.diag(1/ (Q.T @ one_N1)) @ T @ torch.diag(1/ (R.T @ one_N2))

    if Wasserstein is False:
        min_iter = min_iterGW
        max_iter = max_iterGW

    '''
    Preparing main loop.
    '''
    errs = []
    grad = torch.inf
    gamma_k = gamma
    Q_prev, R_prev, T_prev = None, None, None
    
    while (k < max_iter and (not convergence_criterion or \
                       (k < min_iter or util.Delta((Q, R, T), (Q_prev, R_prev, T_prev), gamma_k) > tol))):
        
        if convergence_criterion:
            # Set previous iterates to evaluate convergence at the next round
            Q_prev, R_prev, T_prev = Q, R, T
        
        if k % 25 == 0:
            print(f'Iteration: {k}')
        
        gradQ, gradR, gamma_k = gd.compute_grad_A(C, Q, R, Lambda, gamma, semiRelaxedLeft, \
                                               semiRelaxedRight, device, Wasserstein=Wasserstein, \
                                               A=A, B=B, FGW=FGW, alpha=alpha, \
                                                  unbalanced=unbalanced, full_grad=full_grad, dtype=dtype)
        if semiRelaxedLeft:
            
            R = util.logSinkhorn(gradR - (gamma_k**-1)*torch.log(R), b, gR, gamma_k, max_iter = max_inneriters_relaxed, \
                         device=device, dtype=dtype, balanced=False, unbalanced=False, tau=tau_in)
            Q = util.logSinkhorn(gradQ - (gamma_k**-1)*torch.log(Q), a, gQ, gamma_k, max_iter = max_inneriters_relaxed, \
                         device=device, dtype=dtype, balanced=False, unbalanced=True, tau=tau_out, tau2=tau_in)
            
            gQ, gR = Q.T @ one_N1, R.T @ one_N2
            gradT, gamma_T = gd.compute_grad_B(C, Q, R, Lambda, gQ, gR, \
                                               gamma, device, Wasserstein=Wasserstein, \
                                               A=A, B=B, FGW=FGW, alpha=alpha, dtype=dtype)
            
        elif semiRelaxedRight:
            
            Q = util.logSinkhorn(gradQ - (gamma_k**-1)*torch.log(Q), a, gQ, gamma_k, max_iter = max_inneriters_relaxed, \
                         device=device, dtype=dtype, balanced=False, unbalanced=False, tau=tau_in)
            R = util.logSinkhorn(gradR - (gamma_k**-1)*torch.log(R), b, gR, gamma_k, max_iter = max_inneriters_relaxed, \
                         device=device, dtype=dtype, balanced=False, unbalanced=True, tau=tau_out, tau2=tau_in)
            
            gQ, gR = Q.T @ one_N1, R.T @ one_N2
            gradT, gamma_T = gd.compute_grad_B(C, Q, R, Lambda, gQ, gR, \
                                               gamma, device, Wasserstein=Wasserstein, \
                                               A=A, B=B, FGW=FGW, alpha=alpha, dtype=dtype)
            
        elif unbalanced:
            
            Q = util.logSinkhorn(gradQ - (gamma_k**-1)*torch.log(Q), a, gQ, gamma_k, max_iter = max_inneriters_relaxed, \
                         device=device, dtype=dtype, balanced=False, unbalanced=True, tau=tau_out, tau2=tau_in)
            R = util.logSinkhorn(gradR - (gamma_k**-1)*torch.log(R), b, gR, gamma_k, max_iter = max_inneriters_relaxed, \
                         device=device, dtype=dtype, balanced=False, unbalanced=True, tau=tau_out, tau2=tau_in)
            
            gQ, gR = Q.T @ one_N1, R.T @ one_N2
            
            gradT, gamma_T = gd.compute_grad_B(C, Q, R, Lambda, gQ, gR, gamma, \
                                               device, Wasserstein=Wasserstein, \
                                               A=A, B=B, FGW=FGW, alpha=alpha, dtype=dtype)
            
        else:
            
            # Balanced
            Q = util.logSinkhorn(gradQ - (gamma_k**-1)*torch.log(Q), a, gQ, gamma_k, max_iter = max_inneriters_relaxed, \
                         device=device, dtype=dtype, balanced=False, unbalanced=False, tau=tau_in)
            R = util.logSinkhorn(gradR - (gamma_k**-1)*torch.log(R), b, gR, gamma_k, max_iter = max_inneriters_relaxed, \
                         device=device, dtype=dtype, balanced=False, unbalanced=False, tau=tau_in)
            
            gQ, gR = Q.T @ one_N1, R.T @ one_N2
            gradT, gamma_T = gd.compute_grad_B(C, Q, R, Lambda, gQ, gR, gamma, \
                                               device, Wasserstein=Wasserstein, \
                                               A=A, B=B, FGW=FGW, alpha=alpha, dtype=dtype)
        
        T = util.logSinkhorn(gradT - (gamma_T**-1)*torch.log(T), gQ, gR, gamma_T, max_iter = max_inneriters_balanced, \
                         device=device, dtype=dtype, balanced=True, unbalanced=False)
        
        # Inner latent transition-inverse matrix
        Lambda = torch.diag(1/gQ) @ T @ torch.diag(1/gR)
        
        if printCost:
            if Wasserstein:
                #P = Q @ Lambda @ R.T
                cost = torch.trace(( (Q.T @ C) @ R) @ Lambda.T) #torch.sum(C * P)
            else:
                P = Q @ Lambda @ R.T
                M1 = Q.T @ A**2 @ Q
                M2 = R.T @ B**2 @ R
                cost = one_r.T @ M1 @ one_r + one_r.T @ M2 @ one_r -2*torch.trace((A @ P @ B).T @ P)
                #cost = one_N2.T @ P.T @ A**2 @ P @ one_N2 + one_N1.T @ P @ B**2 @ P.T @ one_N1 - 2*torch.trace((A @ P @ B).T @ P)
                if FGW:
                    cost = (1-alpha)*torch.sum(C * P) + alpha*cost
            errs.append(cost.cpu())
            
        k+=1
    if printCost:
        ''' 
        Plotting OT objective value across iterations.
        '''
        plt.plot(range(len(errs)), errs)
        plt.xlabel('Iterations')
        plt.ylabel('OT-Cost')
        plt.show()
        '''
        Plotting latent coupling.
        '''
        plt.imshow(T.cpu())
        plt.show()
    
    if returnFull:
        P = Q @ Lambda @ R.T
        return P, errs
    else:
        
        if diagonalize_return:
            '''
            Diagonalize return to factorization of Scetbon '21
            '''
            Q = Q @ torch.diag(1 / gQ) @ T
            gR = R.T @ one_N2
            T = torch.diag(gR)
            
        return Q, R, T, errs

'''
----------
Below:
Optimization of FRLC supposing the distance matrices C, A, B have been factorized
----------
'''

def FRLC_LR_opt(C_factors, A_factors, B_factors, a=None, b=None, tau_in = 50, tau_out=50, \
                  gamma=90, r = 10, r2=None, max_iter=200, device='cpu', dtype=torch.float64, \
                 printCost=True, returnFull=False, alpha=0.0, \
                  initialization='Full', init_args = None, full_grad=True, \
                   convergence_criterion=True, tol=5e-6, min_iter = 25, \
                   max_inneriters_balanced= 300, max_inneriters_relaxed=50, \
                  diagonalize_return=False):
    '''
    FRLC with a low-rank factorization of the distance matrices (C, A, B) assumed.
    
    *** Currently only implements balanced OT ***
    
    ------Parameters------
    C_factors: tuple of torch.tensor (n x d, d x m)
        A tuple of two tensors representing the factors of C (Wasserstein term).
    A_factors: tuple of torch.tensor (n x d, d x n)
        A tuple of the A factors (GW term).
    B_factors: torch.tensor
        A tuple of the B factors (GW term).
    a: torch.tensor, optional (default=None)
        A vector representing marginal one.
    b: torch.tensor, optional (default=None)
        A vector representing marginal two.
    tau_in: float, optional (default=0.0001)
        The inner marginal regularization parameter.
    tau_out: float, optional (default=75)
        The outer marginal regularization parameter.
    gamma: float, optional (default=90)
        Mirror descent step size.
    r: int, optional (default=10)
        A parameter representing a rank or dimension.
    r2: int, optional (default=None)
        A secondary rank parameter (if None, defaults to square latent coupling)
    max_iter: int, optional (default=200)
        The maximum number of iterations.
    device: str, optional (default='cpu')
        The device to run the computation on ('cpu' or 'cuda').
    dtype: torch.dtype, optional (default=torch.float64)
        The data type of the tensors.
    printCost: bool, optional (default=True)
        Whether to print and plot the cost during computation.
    returnFull: bool, optional (default=False)
        Whether to return the full coupling P. If False, returns (Q,R,T)
    alpha: float, optional (default=0.2)
        A parameter controlling weight to Wasserstein (alpha = 0.0) or GW (alpha = 1.0) terms
    initialization: str, optional (default='Full')
        'Full' if sub-couplings initialized to be full-rank, if 'Rank-2' set to a rank-2 initialization.
        We advise setting this to be 'Full'.
    init_args: dict, optional (default=None)
        Arguments for the initialization method.
    full_grad: bool, optional (default=True)
        If True, evaluates gradient with rank-1 perturbations. Else if False, omits perturbation terms.
    convergence_criterion: bool, optional (default=True)
        If True, use the convergence criterion. Else if False, default to running up to max_iters.
    tol: float, optional (default=5e-6)
        The tolerance for convergence.
    min_iter: int, optional (default=25)
        The minimum number of iterations.
    max_inneriters_balanced: int, optional (default=300)
        The maximum number of inner iterations for balanced OT sub-routines.
    max_inneriters_relaxed: int, optional (default=50)
        The maximum number of inner iterations for relaxed OT sub-routines.
    diagonalize_return: bool, optional (default=False)
         If True, diagonalize the LC-factorization to the form of Scetbon et al '21.
        Else if False, return the LC-factorization.
    '''
    
    N1, N2 = C_factors[0].size(dim=0), C_factors[1].size(dim=1)
    k = 0
    stationarity_gap = torch.inf
    
    one_N1 = torch.ones((N1), device=device, dtype=dtype)
    one_N2 = torch.ones((N2), device=device, dtype=dtype)
    
    if a is None:
        a = one_N1 / N1
    if b is None:
        b = one_N2 / N2
    if r2 is None:
        r2 = r

    one_r = torch.ones((r), device=device, dtype=dtype)
    one_r2 = torch.ones((r2), device=device, dtype=dtype)
    
    # Initialize inner marginals to uniform; 
    # generalized to be of differing dimensions to account for non-square latent-coupling.
    gQ = (1/r)*one_r
    gR = (1/r2)*one_r2
    
    full_rank = True if initialization == 'Full' else False
    
    if initialization == 'Full':
        full_rank = True
    elif initialization == 'Rank-2':
        full_rank = False
    else:
        full_rank = True
        print('Initialization must be either "Full" or "Rank-2", defaulting to "Full".')
        
    if init_args is None:
        Q, R, T, Lambda = util.initialize_couplings(a, b, gQ, gR, \
                                                    gamma, full_rank=full_rank, \
                                                device=device, dtype=dtype, \
                                                    max_iter = max_inneriters_balanced)
    else:
        # Initialize to given factors
        Q, R, T = init_args
        if Q is not None:
            gQ = Q.T @ one_N1
        if R is not None:
            gR =  R.T @ one_N2
        if Q is None or R is None or T is None:
            _Q, _R, _T, Lambda = util.initialize_couplings(a, b, gQ, gR, \
                                                    gamma, full_rank=full_rank, \
                                                device=device, dtype=dtype, \
                                                    max_iter = max_inneriters_balanced)
            if Q is None:
                Q = _Q
            if R is None:
                R = _R
            if T is None:
                T = _T
        
        Lambda = torch.diag(1/ (Q.T @ one_N1)) @ T @ torch.diag(1/ (R.T @ one_N2))
    
    '''
    Preparing main loop.
    '''
    errs = {'total_cost':[], 'W_cost':[], 'GW_cost': []}
    grad = torch.inf
    gamma_k = gamma
    Q_prev, R_prev, T_prev = None, None, None
    
    while (k < max_iter and (not convergence_criterion or \
                       (k < min_iter or util.Delta((Q, R, T), (Q_prev, R_prev, T_prev), gamma_k) > tol))):
        
        if convergence_criterion:
            # Set previous iterates to evaluate convergence at the next round
            Q_prev, R_prev, T_prev = Q, R, T
        
        if k % 25 == 0:
            print(f'Iteration: {k}')
        
        gradQ, gradR, gamma_k = gd.compute_grad_A_LR(C_factors, A_factors, B_factors, Q, R, Lambda, gamma, device, \
                                   alpha=alpha, dtype=dtype, full_grad=full_grad)
        ### Semi-relaxed updates ###
        R = util.logSinkhorn(gradR - (gamma_k**-1)*torch.log(R), b, gR, gamma_k, max_iter = max_inneriters_relaxed, \
                         device=device, dtype=dtype, balanced=False, unbalanced=False, tau=tau_in)
        
        Q = util.logSinkhorn(gradQ - (gamma_k**-1)*torch.log(Q), a, gQ, gamma_k, max_iter = max_inneriters_relaxed, \
                         device=device, dtype=dtype, balanced=False, unbalanced=False, tau=tau_in)
        
        gQ, gR = Q.T @ one_N1, R.T @ one_N2
        
        gradT, gamma_T = gd.compute_grad_B_LR(C_factors, A_factors, B_factors, Q, R, Lambda, gQ, gR, gamma, device, \
                                       alpha=alpha, dtype=dtype)
        
        T = util.logSinkhorn(gradT - (gamma_T**-1)*torch.log(T), gQ, gR, gamma_T, max_iter = max_inneriters_balanced, \
                         device=device, dtype=dtype, balanced=True, unbalanced=False)
        
        # Inner latent transition-inverse matrix
        Lambda = torch.diag(1/gQ) @ T @ torch.diag(1/gR)
        k+=1
        
        if printCost:
            primal_cost = torch.trace(((Q.T @ C_factors[0]) @ (C_factors[1] @ R)) @ Lambda.T)
            errs['W_cost'].append(primal_cost.cpu())
            if A_factors is not None and B_factors is not None:
                X = R @ ((Lambda.T @ ((Q.T @ A_factors[0]) @ (A_factors[1] @ Q)) @ Lambda) @ (R.T @ B_factors[0])) @ B_factors[1]
                GW_cost = - 2 * torch.trace(X) # add these: one_r.T @ M1 @ one_r + one_r.T @ M2 @ one_r
                del X
                A1_tild, A2_tild = util.hadamard_square_lr(A_factors[0], A_factors[1].T, device=device)
                GW_cost += torch.inner(A1_tild.T @ (Q @ one_r), A2_tild.T @ (Q @ one_r))
                del A1_tild, A2_tild
                B1_tild, B2_tild = util.hadamard_square_lr(B_factors[0], B_factors[1].T, device=device)
                GW_cost += torch.inner(B1_tild.T @ (R @ one_r2), B2_tild.T @ (R @ one_r2))
                del B1_tild, B2_tild
                # Update costs
                errs['GW_cost'].append((GW_cost).cpu())
                errs['total_cost'].append(((1-alpha)*primal_cost + alpha*GW_cost).cpu())
            else:
                errs['GW_cost'].append('NA')
                errs['total_cost'].append('NA')
        
    if printCost:
        print(f"Initial Wasserstein cost: {errs['W_cost'][0]}, GW-cost: {errs['GW_cost'][0]}, Total cost: {errs['total_cost'][0]}")
        print(f"Final Wasserstein cost: {errs['W_cost'][-1]}, GW-cost: {errs['GW_cost'][-1]}, Total cost: {errs['total_cost'][-1]}")
        plt.plot(errs['total_cost'])
        plt.show()
    
    if returnFull:
        P = Q @ Lambda @ R.T
        return P, errs
    else:
        if diagonalize_return:
            '''
            Diagonalize return to factorization of Scetbon '21
            '''
            Q = Q @ torch.diag(1 / gQ) @ T
            gR = R.T @ one_N2
            T = torch.diag(gR)
        return Q, R, T, errs



