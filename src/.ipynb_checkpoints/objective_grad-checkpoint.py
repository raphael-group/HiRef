import torch
import util

def compute_grad_A(C, Q, R, Lambda, gamma, \
                   semiRelaxedLeft, semiRelaxedRight, device, \
                   Wasserstein=True, FGW=False, A=None, B=None, \
                   alpha=0.0, unbalanced=False, \
                   dtype=torch.float64, full_grad=True):
    
    '''
    Code for computing the Wasserstein, Gromov-Wasserstein, and fused Gromov-Wasserstein gradients with respect to Q and R.
    These depend on the marginal relaxation of the specific OT problem you want to solve, due to proportionality simplifications.
    
    ------Parameters------
    C: torch.tensor (N1 x N2)
        A matrix of pairwise feature distances in space X and space Y (inter-space).
    Q: torch.tensor (N1 x r)
        The left sub-coupling matrix.
    R: torch.tensor (N2 x r)
        The right sub-coupling matrix.
    Lambda: torch.tensor (r x r)
        The inner transition matrix.
    gamma: float
        The mirror-descent step-size.
    semiRelaxedLeft: bool
        True if relaxing the left marginal.
    semiRelaxedRight: bool
        True if relaxing the right marginal.
    device: str
        The device (i.e. 'cpu' or 'cuda')
    Wasserstein: bool
        True if using the Wasserstein loss <C, P>_F as the objective cost,
        else runs GW if FGW false and FGW if GW true.
    FGW: bool
        True if running the Fused-Gromov Wasserstein problem, and otherwise false.
    A: torch.tensor (N1 x N1)
        Pairwise distance matrix in metric space X.
    B: torch.tensor (N2 x N2)
        Pairwise distance matrix in metric space Y.
    alpha: float
        A balance parameter between the Wasserstein term and
        the Gromov-Wasserstein term of the objective.
    unbalanced: bool
        True if running the unbalanced problem;
        if semiRelaxedLeft/Right and unbalanced False (default) then running the balanced problem.
    '''
    
    r = Lambda.shape[0]
    one_r = torch.ones((r), device=device, dtype=dtype)
    One_rr = torch.outer(one_r, one_r).to(device)
    
    if Wasserstein:
        gradQ, gradR = Wasserstein_Grad(C, Q, R, Lambda, device, \
                   dtype=dtype, full_grad=full_grad)
        
    elif A is not None and B is not None:
        if not semiRelaxedLeft and not semiRelaxedRight and not unbalanced:
            # Balanced gradient (Q1_r = a AND R1_r = b)
            gradQ = - 4 * (A@Q)@Lambda@(R.T@B@R)@Lambda.T
            gradR = - 4 * (B@R@Lambda.T)@(Q.T@A@Q)@Lambda
        elif semiRelaxedRight:
            # Semi-relaxed right marginal gradient (Q1_r = a)
            gradQ = - 4 * (A@Q)@Lambda@(R.T@B@R)@Lambda.T
            gradR = 2*B**2 @ R @ One_rr - 4*(B@R@Lambda.T)@(Q.T@A@Q)@Lambda
        elif semiRelaxedLeft:
            # Semi-relaxed left marginal gradient (R1_r = b)
            gradQ = 2*A**2 @ Q @ One_rr - 4 * (A@Q)@Lambda@(R.T@B@R)@Lambda.T
            gradR = - 4 * (B@R@Lambda.T)@(Q.T@A@Q)@Lambda
        elif unbalanced:
            # Fully unbalanced with no marginal constraints
            gradQ = 2*A**2 @ Q @ One_rr - 4 * (A@Q)@Lambda@(R.T@B@R)@Lambda.T
            gradR = 2*B**2 @ R @ One_rr - 4 * (B@R@Lambda.T)@(Q.T@A@Q)@Lambda

        if full_grad:
            N1, N2 = Q.shape[0], R.shape[0]
            one_N1, one_N2 = torch.ones((N1), device=device, dtype=dtype), torch.ones((N2), device=device, dtype=dtype)
            gQ, gR = Q.T @ one_N1, R.T @ one_N2
            F = (Q@Lambda@R.T)
            MR = Lambda.T @ Q.T @ A @ F @ B @ R @ torch.diag(1/gR)
            MQ = Lambda @ R.T @ B @ F.T @ A @ Q @ torch.diag(1/gQ)
            gradQ += 4*torch.outer(one_N1, torch.diag(MQ))
            gradR += 4*torch.outer(one_N2, torch.diag(MR))
        
        # Readjust cost for FGW problem
        if FGW:
            gradQW, gradRW = Wasserstein_Grad(C, Q, R, Lambda, device, \
                   dtype=dtype, full_grad=full_grad)
            gradQ = (1-alpha)*gradQW + alpha*gradQ
            gradR = (1-alpha)*gradRW + alpha*gradR
    else:
        raise ValueError("---Input either Wasserstein=True or provide distance matrices A and B for GW problem---")
        
    normalizer = torch.max(torch.tensor([torch.max(torch.abs(gradQ)) , torch.max(torch.abs(gradR))]))
    gamma_k = gamma / normalizer
    
    return gradQ, gradR, gamma_k

def compute_grad_B(C, Q, R, Lambda, gQ, gR, gamma, device, Wasserstein=True, \
                   FGW=False, A=None, B=None, alpha=0.0, \
                   dtype=torch.float64):
    '''
    Code for computing the Wasserstein, Gromov-Wasserstein, and fused Gromov-Wasserstein gradients with respect to T.
    
    ------Parameters------
    C: torch.tensor (N1 x N2)
        A matrix of pairwise feature distances in space X and space Y (inter-space).
    Q: torch.tensor (N1 x r)
        The left sub-coupling matrix.
    R: torch.tensor (N2 x r)
        The right sub-coupling matrix.
    Lambda: torch.tensor (r x r)
        The inner transition matrix.
    gQ: torch.tensor (r)
        The inner marginal corresponding to the matrix Q.
    gR: torch.tensor (r)
        The inner marginal corresponding to the matrix R.
    gamma: float
        The mirror-descent step-size.
    device: str
        The device (i.e. 'cpu' or 'cuda')
    Wasserstein: bool
        True if using the Wasserstein loss <C, P>_F as the objective cost,
        else runs GW if FGW false and FGW if GW true.
    FGW: bool
        True if running the Fused-Gromov Wasserstein problem, and otherwise false.
    A: torch.tensor (N1 x N1)
        Pairwise distance matrix in metric space X.
    B: torch.tensor (N2 x N2)
        Pairwise distance matrix in metric space Y.
    alpha: float
        A balance parameter between the Wasserstein term and
        the Gromov-Wasserstein term of the objective.
    '''
    if Wasserstein:
        gradLambda = Q.T @ C @ R
    else:
        gradLambda = -4 * Q.T @ A @ Q @ Lambda @ R.T @ B @ R
        if FGW:
            gradLambda = (1-alpha)*(Q.T @ C @ R) + alpha*gradLambda
    gradT = torch.diag(1/gQ) @ gradLambda @ torch.diag(1/gR) # (mass-reweighted form)
    gamma_T = gamma / torch.max(torch.abs(gradT))
    return gradT, gamma_T

def Wasserstein_Grad(C, Q, R, Lambda, device, \
                   dtype=torch.float64, full_grad=True):
    
    gradQ = (C @ R) @ Lambda.T
    if full_grad:
        # rank-one perturbation
        N1 = Q.shape[0]
        one_N1 = torch.ones((N1), device=device, dtype=dtype)
        gQ = Q.T @ one_N1
        w1 = torch.diag( (gradQ.T @ Q) @ torch.diag(1/gQ) )
        gradQ -= torch.outer(one_N1, w1)
    
    # linear term
    gradR = (C.T @ Q) @ Lambda
    if full_grad:
        # rank-one perturbation
        N2 = R.shape[0]
        one_N2 = torch.ones((N2), device=device, dtype=dtype)
        gR = R.T @ one_N2
        w2 = torch.diag( torch.diag(1/gR) @ (R.T @ gradR) )
        gradR -= torch.outer(one_N2, w2)
    
    return gradQ, gradR

'''
--------------
Code for gradients assuming low-rank distance matrices C, A, B
--------------
'''

def compute_grad_A_LR(C_factors, A_factors, B_factors, Q, R, Lambda, gamma, device, \
                   alpha=0.0, dtype=torch.float64, full_grad=False):
    
    r = Lambda.shape[0]
    one_r = torch.ones((r), device=device, dtype=dtype)
    One_rr = torch.outer(one_r, one_r).to(device)
    N1, N2 = C_factors[0].size(0), C_factors[1].size(1)
    
    if A_factors is not None and B_factors is not None:
        A1, A2 = A_factors[0], A_factors[1]
        B1, B2 = B_factors[0], B_factors[1]
        
        # A*2's low-rank factorization
        A1_tild, A2_tild = util.hadamard_square_lr(A1, A2.T, device=device)
        
        # GW gradients for balanced marginal cases
        gradQ = - 4 * (A1 @ (A2 @ (Q @ Lambda@( (R.T@ B1) @ (B2 @R) )@Lambda.T)) )
        gradR = - 4 * (B1 @ (B2 @ (R @ (Lambda.T@( (Q.T @ A1) @ ( A2 @ Q ))@Lambda)) ) )
    
        one_N1, one_N2 = torch.ones((N1), device=device, dtype=dtype), torch.ones((N2), device=device, dtype=dtype)
        if full_grad:
            # Rank-1 GW perturbation
            N1, N2 = Q.shape[0], R.shape[0]
            gQ, gR = Q.T @ one_N1, R.T @ one_N2
            
            MR = Lambda.T @ ( (Q.T @ A1) @ (A2 @ Q) ) @ Lambda @ ((R.T @ B1) @ (B2 @ R)) @ torch.diag(1/gR)
            MQ = Lambda @ ( (R.T @ B1) @ (B2 @ R) ) @ Lambda.T @ ((Q.T @ A1) @ (A2 @ Q) ) @ torch.diag(1/gQ)
            gradQ += 4*torch.outer(one_N1, torch.diag(MQ))
            gradR += 4*torch.outer(one_N2, torch.diag(MR))
    else:
        gradQ = 0
        gradR = 0
    
    # total gradients -- readjust cost for FGW problem by adding W gradients
    gradQW, gradRW = Wasserstein_Grad_LR(C_factors, Q, R, Lambda, device, \
                                                   dtype=dtype, full_grad=full_grad)
    gradQ = (1-alpha)*gradQW + (alpha/2)*gradQ
    gradR = (1-alpha)*gradRW + (alpha/2)*gradR
    
    normalizer = torch.max(torch.tensor([torch.max(torch.abs(gradQ)) , torch.max(torch.abs(gradR))]))
    gamma_k = gamma / normalizer
    
    return gradQ, gradR, gamma_k

def compute_grad_B_LR(C_factors, A_factors, B_factors, Q, R, Lambda, gQ, gR, gamma, device, \
                   alpha=0.0, dtype=torch.float64):
    
    N1, N2 = C_factors[0].size(0), C_factors[1].size(1)

    if A_factors is not None and B_factors is not None:
        A1, A2 = A_factors[0], A_factors[1]
        B1, B2 = B_factors[0], B_factors[1]
        # GW grad
        gradLambda = -4 * ( (Q.T @ A1) @ (A2 @ Q) ) @ Lambda @ ( (R.T @ B1) @ (B2 @ R) )
        del A1,A2,B1,B2
    else:
        gradLambda = 0
    
    C1, C2 = C_factors[0], C_factors[1]
    # total grad
    gradLambda = (1-alpha)*( (Q.T @ C1) @ (C2 @ R) ) + (alpha/2)*gradLambda
    gradT = torch.diag(1/gQ) @ gradLambda @ torch.diag(1/gR) # (mass-reweighted form)
    gamma_T = gamma / torch.max(torch.abs(gradT))
    return gradT, gamma_T

def Wasserstein_Grad_LR(C_factors, Q, R, Lambda, device, \
                   dtype=torch.float64, full_grad=True):

    C1, C2 = C_factors[0], C_factors[1]
    
    gradQ = C1 @ ((C2 @ R) @ Lambda.T)
    
    if full_grad:
        # rank-one perturbation
        N1 = Q.shape[0]
        one_N1 = torch.ones((N1), device=device, dtype=dtype)
        gQ = Q.T @ one_N1
        w1 = torch.diag( (gradQ.T @ Q) @ torch.diag(1/gQ) )
        gradQ -= torch.outer(one_N1, w1)
    
    # linear term
    gradR = C2.T @ ((C1.T @ Q) @ Lambda)
    if full_grad:
        # rank-one perturbation
        N2 = R.shape[0]
        one_N2 = torch.ones((N2), device=device, dtype=dtype)
        gR = R.T @ one_N2
        w2 = torch.diag( torch.diag(1/gR) @ (R.T @ gradR) )
        gradR -= torch.outer(one_N2, w2)
    
    return gradQ, gradR


