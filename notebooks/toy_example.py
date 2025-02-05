import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def generate_sample_data(N = 4096):
    # 1. Set up cluster centers
    center_t1_A = np.array([0, 0])
    # 2. Set up cluster covariance matrices
    cov_t1_A = np.array([[6, 0], [0, 1]])
    # Set number of points
    num_A = N
    
    # 4. Variance scaling
    var = .1 # Variance of Gaussians used to make each cluster
    X1 = np.random.multivariate_normal(center_t1_A, var*cov_t1_A, size=num_A)
    
    def dy_dt(vars, t):
        tau = 0.0
        x, y = vars
        if y > 0 and x > tau:
            dxdt = 2
            dydt = 2
        elif y > 0 and x < -tau:
            dxdt = -2
            dydt = 2
        elif y < 0 and x > tau:
            dxdt = 2
            dydt = -2
        elif y < 0 and x < tau:
            dxdt = -2
            dydt = -2
        
        return [dxdt, dydt]
    
    X2 = np.zeros(X1.shape)
    
    ts = np.array([0, 1, 2])
    for i in range(X1.shape[0]):
        IC = X1[i,:]
        # ODE-Int is overkill, but why not...
        solution = np.array( odeint(dy_dt, [IC[0], IC[1]], ts) )
        X2[i,:] = solution[1,:]
        
    plt.scatter(X1[:,0], X1[:,1], c='b', alpha=0.5)
    plt.scatter(X2[:,0], X2[:,1], c='r', alpha=0.5)
    plt.axis('equal')
    plt.show()
    return X1, X2