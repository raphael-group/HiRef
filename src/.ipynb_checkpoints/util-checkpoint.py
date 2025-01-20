import torch
import util
from sklearn.cluster import KMeans
import random
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np

'''
A number of useful functions and sub-routines for the main FRLC routine FRLC_iteration
'''

def initialize_couplings(a, b, gQ, gR, gamma, \
                         full_rank=True, device='cpu', \
                         dtype=torch.float64, rank2_random=False, \
                        max_iter=50):
    '''
    ------Parameters------
    a: torch tensor
        Left outer marginal, should be positive and sum to 1.0
    b: torch tensor
        Right outer marginal, should be positive and sum to 1.0
    gQ: torch tensor
        Left inner marginal, should be positive and sum to 1.0
    gR: torch tensor
        Right inner marginal, should be positive and sum to 1.0
    gamma: float
        Step-size of the coordinate MD
    full_rank: bool
        If True, initialize a full-rank set of sub-couplings.
        Else if False, initialize with a rank-2 initialization.
    device: str
        'cpu' if running on CPU, else 'cuda' for GPU
    dtype: torch dtype
        Defaults to float64
    rank2_random: bool
        If False, use deterministic rank 2 initialization of Scetbon '21
        Else, use an initialization with randomly sampled vector on simplex.
    max_iter: int
        The maximum number of Sinkhorn iterations for initialized sub-couplings.
    '''
    N1, N2 = a.size(dim=0), b.size(dim=0)
    r, r2 = gQ.size(dim=0), gR.size(dim=0)
    one_N1 = torch.ones((N1), device=device, dtype=dtype)
    one_N2 = torch.ones((N2), device=device, dtype=dtype)
    
    if full_rank:
        '''
        A means of initializing full-rank sub-coupling matrices using randomly sampled matrices
        and Sinkhorn projection onto the polytope of feasible couplings.

        Only non-diagonal initialization for the LC-factorization and handles the case of unequal
        inner left and right ranks (non-square latent couplings).
        '''
        # 1. Q-generation
        # Generate a random (full-rank) matrix as our coupling initialization
        C_random = torch.rand((N1,r), device=device, dtype=dtype)
        '''
        # Generate a random Kernel
        xi_random = torch.exp( -C_random )
        # Generate a random coupling
        u, v = Sinkhorn(xi_random, a, gQ, N1, r, gamma, device=device, max_iter=max_iter, dtype=dtype)
        Q = torch.diag(u) @ xi_random @ torch.diag(v)
        '''
        Q = logSinkhorn(C_random, a, gQ, gamma, max_iter = max_iter, \
                         device=device, dtype=dtype, balanced=True, unbalanced=False)
        
        # 2. R-generation
        C_random = torch.rand((N2,r2), device=device, dtype=dtype)
        '''
        xi_random = torch.exp( -C_random )
        u, v = Sinkhorn(xi_random, b, gR, N2, r2, gamma, device=device, max_iter=max_iter, dtype=dtype)
        R = torch.diag(u) @ xi_random @ torch.diag(v)'''
        R = logSinkhorn(C_random, b, gR, gamma, max_iter = max_iter, \
                         device=device, dtype=dtype, balanced=True, unbalanced=False)
        
        # 3. T-generation
        gR, gQ = R.T @ one_N2, Q.T @ one_N1
        C_random = torch.rand((r,r2), device=device, dtype=dtype)
        '''
        xi_random = torch.exp( -C_random )
        u, v = Sinkhorn(xi_random, gQ, gR, r, r2, gamma, device=device, max_iter=max_iter, dtype=dtype)
        T = torch.diag(u) @ xi_random @ torch.diag(v)
        '''
        T = logSinkhorn(C_random, gQ, gR, gamma, max_iter = max_iter, \
                         device=device, dtype=dtype, balanced=True, unbalanced=False)
        
        # Use this to form the inner inverse coupling
        if r == r2:
            Lambda = torch.linalg.inv(T)
        else:
            Lambda = torch.diag(1/gQ) @ T @ torch.diag(1/gR)
            #also, could do: torch.diag(1/gQ) @ T @ torch.diag(1/gR)
    elif r == r2:
        '''
        Rank-2 initialization which requires equal inner ranks and gQ = gR = g.
        This is adapted from "Low-Rank Sinkhorn Factorization" at https://arxiv.org/pdf/2103.04737
        We advise setting full_rank = True and using the first initialization.
        '''
        g = gQ
        lambd = torch.min(torch.tensor([torch.min(a), torch.min(b), torch.min(g)])) / 2

        if rank2_random:
            # Take random sample from probability simplex
            a1 = random_simplex_sample(N1, device=device, dtype=dtype)
            b1 = random_simplex_sample(N2, device=device, dtype=dtype)
            g1 = random_simplex_sample(r, device=device, dtype=dtype)
        else:
            # or initialize exactly as in scetbon 21' ott-jax repo
            g1 = torch.arange(1, r + 1, device=device, dtype=dtype)
            g1 /= g1.sum()
            a1 = torch.arange(1, N1 + 1, device=device, dtype=dtype)
            a1 /= a1.sum()
            b1 = torch.arange(1, N2 + 1, device=device, dtype=dtype)
            b1 /= b1.sum()
        
        a2 = (a - lambd*a1)/(1 - lambd)
        b2 = (b - lambd*b1)/(1 - lambd)
        g2 = (g - lambd*g1)/(1 - lambd)
        
        # Generate Rank-2 Couplings
        Q = lambd*torch.outer(a1, g1).to(device) + (1 - lambd)*torch.outer(a2, g2).to(device)
        R = lambd*torch.outer(b1, g1).to(device) + (1 - lambd)*torch.outer(b2, g2).to(device)
        
        # This is already determined as g (but recomputed anyway)
        gR, gQ = R.T @ one_N2, Q.T @ one_N1
        
        # Last term adds very tiny off-diagonal component for the non-diagonal LC-factorization (o/w the matrix stays fully diagonal)
        T = (1-lambd)*torch.diag(g) + lambd*torch.outer(gR, gQ).to(device)
        Lambda = torch.linalg.inv(T)
    
    return Q, R, T, Lambda


def k_means_initialization(x0, x1, r1, r2=None, \
                           a=None, b=None, gQ=None, gR=None, \
                           eps = 1e-3, device = 'cpu', \
                           dtype=torch.float64):
    '''
    An initialization relying on a pair of k-means clusterings on the first and second dataset.
    ------Parameters------
    x0: torch.tensor
        First N1 x d dataset for d the data-dimension
    x1: torch.tensor
        Second N2 x d dataset for d the data-dimension
    r1: int
        Latent source rank
    r2: int
        Latent target rank
    a: torch tensor
        Left outer marginal, should be positive and sum to 1.0
    b: torch tensor
        Right outer marginal, should be positive and sum to 1.0
    gQ: torch tensor
        Left inner marginal, should be positive and sum to 1.0
    gR: torch tensor
        Right inner marginal, should be positive and sum to 1.0
    eps: float
        Epsilon used for Sinkhorn to generate the sub-couplings.
    device: str
        'cpu' if running on CPU, else 'cuda' for GPU
    dtype: torch dtype
        Defaults to float64
    '''
    n, m =  x0.size(dim=0), x1.size(dim=0)
    # Initialize outer marginals
    if a is None:
        one_n = torch.ones((n), device=device, dtype=dtype)
        a = one_n / n
    if b is None:
        one_m = torch.ones((m), device=device, dtype=dtype)
        b = one_m / m
    # Set ranks equal if second rank not given  
    if r2 is None:
        r2 = r1
    
    if gQ is None:
        one_r1 = torch.ones((r1), device=device, dtype=dtype)
        gQ = one_r1 / r1
    if gR is None:
        one_r2 = torch.ones((r2), device=device, dtype=dtype)
        gR = one_r2 / r2
    _x0, _x1 = x0.cpu().numpy(), x1.cpu().numpy()
    # Compute optimal clustering to initialize OT alignment
    y0 = KMeans(n_clusters=r1, n_init="auto").fit(_x0).cluster_centers_
    y1 = KMeans(n_clusters=r2, n_init="auto").fit(_x1).cluster_centers_
    # Move back to tensor
    x0,x1=x0.double(),x1.double()
    y0,y1 = torch.from_numpy(y0).to(device).double(),torch.from_numpy(y1).to(device).double()
    # Compute distance matrices
    CQ,CT,CR = torch.cdist(x0, y0), torch.cdist(y0, y1), torch.cdist(x1,y1)
    # Generate Kernel
    xiQ, xiR, xiT = torch.exp( -CQ / eps ), torch.exp( -CR / eps ), torch.exp( -CT / eps )
    
    # Generate couplings
    u, v = util.Sinkhorn(xiQ, a, gQ, n, r1, eps, device=device)
    Q = torch.diag(u) @ xiQ @ torch.diag(v)
    u, v = util.Sinkhorn(xiR, b, gR, m, r2, eps, device=device)
    R = torch.diag(u) @ xiR @ torch.diag(v)
    u, v = util.Sinkhorn(xiT, gQ, gR, r1, r2, eps, device=device)
    T = torch.diag(u) @ xiT @ torch.diag(v)
    
    return (Q,R,T)




def random_simplex_sample(N, device='cpu', dtype=torch.float64):
    # Samples a random N-dimensional vector from the simplex
    d = torch.exp(torch.randn(N, device=device, dtype=dtype))
    return d / torch.sum(d)




def semi_project_Left(xi1, a, g, N1, r, gamma_k, tau, max_iter = 50, \
                      delta = 1e-9, device='cpu', dtype=torch.float64):
    '''
    Semi-relaxed Sinkhorn with tight left marginal.
    '''
    u = torch.ones((N1), device=device, dtype=dtype)
    v = torch.ones((r), device=device, dtype=dtype)
    u_tild = u
    v_tild = v
    i = 0
    while i == 0 or (i < max_iter and 
                     gamma_k**-1 * torch.max(torch.tensor([torch.max(torch.log(u/u_tild)),torch.max(torch.log(v/v_tild))])) > delta ):
        u_tild = u
        v_tild = v
        u = (a / (xi1 @ v))**(tau/(tau + gamma_k**-1 ))
        v = (g / (xi1.T @ u))
        i+=1
    
    return u, v




def semi_project_Right(xi2, b, g, N2, r, gamma_k, tau, max_iter = 50, \
                       delta = 1e-9, device='cpu', dtype=torch.float64):
    '''
    Semi-relaxed Sinkhorn with tight right marginal.
    '''
    u = torch.ones((N2), device=device, dtype=dtype)
    v = torch.ones((r), device=device, dtype=dtype)
    u_tild = u
    v_tild = v
    i = 0
    while i == 0 or (i < max_iter and 
                     gamma_k**-1 * torch.max(torch.tensor([torch.max(torch.log(u/u_tild)),torch.max(torch.log(v/v_tild))])) > delta ):
        u_tild = u
        v_tild = v
        u = (b / (xi2 @ v))**(tau/(tau + gamma_k**-1 ))
        v = (g / (xi2.T @ u))
        i+=1
    
    return u, v




def semi_project_Balanced(xi1, a, g, N1, r, gamma_k, tau, max_iter = 50, \
                          delta = 1e-9, device='cpu', dtype=torch.float64):
    # Lax-inner marginal
    u = torch.ones((N1), device=device, dtype=dtype)
    v = torch.ones((r), device=device, dtype=dtype)
    u_tild = u
    v_tild = v
    i = 0
    while i == 0 or (i < max_iter and 
                     gamma_k**-1 * torch.max(torch.tensor([torch.max(torch.log(u/u_tild)),torch.max(torch.log(v/v_tild))])) > delta ):
        u_tild = u
        v_tild = v
        v = (g / (xi1.T @ u))**(tau/(tau + gamma_k**-1 ))
        u = (a / (xi1 @ v))
        i+=1
    
    return u, v


def project_Unbalanced(xi1, a, g, N1, r, gamma_k, tau, max_iter = 50, \
                       delta = 1e-9, device='cpu', dtype=torch.float64):
    '''
    Fully-relaxed Sinkhorn with relaxed left and right marginals.
    '''
    # Unbalanced
    u = torch.ones((N1), device=device, dtype=dtype)
    v = torch.ones((r), device=device, dtype=dtype)
    u_tild = u
    v_tild = v
    i = 0
    while i == 0 or (i < max_iter and 
                     gamma_k**-1 * torch.max(torch.tensor([torch.max(torch.log(u/u_tild)),torch.max(torch.log(v/v_tild))])) > delta ):
        u_tild = u
        v_tild = v
        v = (g / (xi1.T @ u))**(tau/(tau + gamma_k**-1 ))
        u = (a / (xi1 @ v))**(tau/(tau + gamma_k**-1 ))
        i+=1
    
    return u, v

def logSinkhorn(grad, a, b, gamma_k, max_iter = 50, \
             device='cpu', dtype=torch.float64, balanced=True, unbalanced=False, tau=None, tau2=None):
    
    log_a = torch.log(a)
    log_b = torch.log(b)

    n, m = a.size(0), b.size(0)
    
    f_k = torch.zeros((n), device=device)
    g_k = torch.zeros((m), device=device)

    epsilon = gamma_k**-1
    
    if not balanced:
        ubc = (tau/(tau + epsilon ))
        if tau2 is not None:
            ubc2 = (tau2/(tau2 + epsilon ))
    
    for i in range(max_iter):
        if balanced and not unbalanced:
            # Balanced
            f_k = f_k + epsilon*(log_a - torch.logsumexp(Cost(f_k, g_k, grad, epsilon, device=device), axis=1))
            g_k = g_k + epsilon*(log_b - torch.logsumexp(Cost(f_k, g_k, grad, epsilon, device=device), axis=0))
        elif not balanced and unbalanced:
            # Unbalanced
            f_k = ubc*(f_k + epsilon*(log_a - torch.logsumexp(Cost(f_k, g_k, grad, epsilon, device=device), axis=1)) )
            g_k = ubc2*(g_k + epsilon*(log_b - torch.logsumexp(Cost(f_k, g_k, grad, epsilon, device=device), axis=0)) )
        else:
            # Semi-relaxed
            f_k = (f_k + epsilon*(log_a - torch.logsumexp(Cost(f_k, g_k, grad, epsilon, device=device), axis=1)) )
            g_k = ubc*(g_k + epsilon*(log_b - torch.logsumexp(Cost(f_k, g_k, grad, epsilon, device=device), axis=0)) )

    P = torch.exp(Cost(f_k, g_k, grad, epsilon, device=device))
    
    return P

def Sinkhorn(xi, a, b, N1, r, gamma_k, max_iter = 50, \
             delta = 1e-9, device='cpu', dtype=torch.float64):
    '''
    A lightweight impl of Sinkhorn.
    ------Parameters------
    xi: torch tensor
        An n x m matrix of the exponentiated positive Sinkhorn kernel.
    a: torch tensor
        Left outer marginal, should be positive and sum to 1.0
    b: torch tensor
        Right outer marginal, should be positive and sum to 1.0
    N1: int
        Dimension 1
    r: int
        Dimension 2
    gamma_k: float
        Step-size used for scaling convergence criterion.
    max_iter: int
        Maximum number of iterations for Sinkhorn loop
    delta: float
        Used for determining convergence to marginals
    device: str
        'cpu' if running on CPU, else 'cuda' for GPU
    dtype: torch dtype
        Defaults to float64
    '''
    u = torch.ones((N1), device=device, dtype=dtype)
    v = torch.ones((r), device=device, dtype=dtype)
    u_tild = u
    v_tild = v
    i = 0
    
    while i == 0 or (i < max_iter and 
                     gamma_k**-1 * torch.max(torch.tensor([torch.max(torch.log(u/u_tild)),torch.max(torch.log(v/v_tild))])) > delta ):
        
        u_tild = u
        v_tild = v
        u = (a / (xi @ v))
        v = (b / (xi.T @ u))
        i+=1
        
    return u, v

def Cost(f, g, Grad, epsilon, device='cpu', dtype=torch.float64):
    '''
    A matrix which is using for the broadcasted log-domain log-sum-exp trick-based updates.
    ------Parameters------
    f: torch.tensor (N1)
        First dual variable of semi-unbalanced Sinkhorn
    g: torch.tensor (N2)
        Second dual variable of semi-unbalanced Sinkhorn
    Grad: torch.tensor (N1 x N2)
        A collection of terms in our gradient for the update
    epsilon: float
        Entropic regularization for Sinkhorn
    device: 'str'
        Device tensors placed on
    '''
    return -( Grad - torch.outer(f, torch.ones(Grad.size(dim=1), device=device)) - torch.outer(torch.ones(Grad.size(dim=0), device=device), g) ) / epsilon



def Delta(vark, varkm1, gamma_k):
    '''
    Convergence criterion for FRLC.
    ------Parameters------
    vark: tuple of 3-tensors
        Tuple of coordinate MD block variables (Q,R,T) at current iter
    varkm1:  tuple of 3-tensors
        Tuple of coordinate MD block variables (Q,R,T) at previous iter
    gamma_k: float
        Coordinate MD step-size
    '''
    Q, R, T = vark
    Q_prev, R_prev, T_prev = varkm1
    error = (gamma_k**-2)*(torch.norm(Q - Q_prev) + torch.norm(R - R_prev) + torch.norm(T - T_prev))
    return error


def low_rank_distance_factorization(X1, X2, r, eps, device='cpu', dtype=torch.float64):
    n = X1.shape[0]
    m = X2.shape[0]
    '''
    Indyk '19
    '''
    # low-rank distance matrix factorization of Bauscke, Indyk, Woodruff
    
    t = int(r/eps) # this is poly(1/eps, r) in general -- this t might not achieve the correct bound tightly
    i_star = random.randint(1, n)
    j_star = random.randint(1, m)
    
    # Define probabilities of sampling
    p = (torch.cdist(X1, X2[j_star][None,:])**2 \
            + torch.cdist(X1[i_star,:][None,:], X2[j_star,:][None,:])**2 \
                    + (torch.sum(torch.cdist(X1[i_star][None,:], X2))/m) )[:,0]**2
    
    p_dist = (p / p.sum())
    
    # Use random choice to sample rows
    indices_p = torch.from_numpy(np.random.choice(n, size=(t), p=p_dist.cpu().numpy())).to(device)
    X1_t = X1[indices_p, :]
    '''
    Frieze '04
    '''
    P_t = torch.sqrt(p[indices_p]*t)
    S = torch.cdist(X1_t, X2)/P_t[:, None] # t x m
    
    # Define probabilities of sampling by row norms
    q = torch.norm(S, dim=0)**2 / torch.norm(S)**2 # m x 1
    q_dist = (q / q.sum())
    # Use random choice to sample rows
    indices_q = torch.from_numpy(np.random.choice(m, size=(t), p=q_dist.cpu().numpy())).to(device)
    S_t = S[:, indices_q] # t x t
    Q_t = torch.sqrt(q[indices_q]*t)
    W = S_t[:, :] / Q_t[None, :]
    # Find U
    U, Sig, Vh = torch.linalg.svd(W) # t x t for all
    F = U[:, :r] # t x r
    # U.T for the final return
    U_t = (S.T @ F) / torch.norm(W.T @ F) # m x r
    '''
    Chen & Price '17
    '''
    # Find V for the final return
    indices = torch.from_numpy(np.random.choice(m, size=(t))).to(device)
    X2_t = X2[indices, :] # t x dim
    D_t = torch.cdist(X1, X2_t) / np.sqrt(t) # n x t
    Q = U_t.T @ U_t # r x r
    U, Sig, Vh = torch.linalg.svd(Q)
    U = U / Sig # r x r
    U_tSub = U_t[indices, :].T # t x r
    B = U.T @ U_tSub / np.sqrt(t) # (r x r) (r x t)
    A = torch.linalg.inv(B @ B.T)
    Z = ((A @ B) @ D_t.T) # (r x r) (r x t) (t x n)
    V = Z.T @ U
    return V.double(), U_t.T.double()

def hadamard_square_lr(A1, A2, device='cpu'):
    """
    Input
        A1: torch.tensor, low-rank subcoupling of shape (n, r)
        A2: torch.tensor, low-rank subcoupling of shape (n, r)
                ( such that A \approx A1 @ A2.T )
    
    Output
        A1_tilde: torch.tensor, low-rank subcoupling of shape (n, r**2)
        A2_tilde: torch.tensor, low-rank subcoupling of shape (n, r**2)
               ( such that A * A \approx A1_tilde @ A2_tilde.T )
    """
    
    A1 = A1.to(device)
    A2 = A2.to(device)
    n, r = A1.shape
    A1_tilde = torch.einsum("ij,ik->ijk", A1, A1).reshape(n, r * r)
    A2_tilde = torch.einsum("ij,ik->ijk", A2, A2).reshape(n, r * r)
    
    return A1_tilde, A2_tilde


def hadamard_lr(A1, A2, B1, B2, device='cpu'):
    """
    Input
        A1: torch.tensor, low-rank subcoupling of shape (n, r)
        A2: torch.tensor, low-rank subcoupling of shape (n, r)
                ( such that A \approx A1 @ A2.T )
        
        B1: torch.tensor, low-rank subcoupling of shape (n, r)
        B2: torch.tensor, low-rank subcoupling of shape (n, r)
                ( such that B \approx B1 @ B2.T )
    
    Output
        M1_tilde: torch.tensor, low-rank subcoupling of shape (n, r**2)
        M2_tilde: torch.tensor, low-rank subcoupling of shape (n, r**2)
               ( such that A * B \approx M1_tilde @ M2_tilde.T given low-rank factorizations for A & B)
    """
    A1 = A1.to(device)
    A2 = A2.to(device)
    B1 = B1.to(device)
    B2 = B2.to(device)
    n, r = A1.shape

    M1_tilde = torch.einsum("ij,ik->ijk", A1, B1).reshape(n, r * r)
    M2_tilde = torch.einsum("ij,ik->ijk", A2, B2).reshape(n, r * r)
    
    return M1_tilde, M2_tilde

def LC_proj(X0, X1, Q, R):
    
    gQ = torch.sum(Q,axis=0)
    Q_barycenters = torch.diag(1/gQ) @ Q.T @ X0
    gR = torch.sum(R,axis=0)
    R_barycenters = torch.diag(1/gR) @ R.T @ X1

    return Q_barycenters, R_barycenters