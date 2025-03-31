import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

from tqdm import tqdm
import jax
import jax.numpy as jnp
from ott.geometry.geometry import Geometry
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn



###
# Random
###

def set_seed(seed=42):
    """
    Set the random seed for NumPy, 
    PyTorch, and Python's 
    random module to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    # Set seed for Python's built-in random module
    random.seed(seed)

    # Set seed for NumPy
    np.random.seed(seed)

    # Set seed for PyTorch on CPU
    torch.manual_seed(seed)

    # If using CUDA, set seed for all CUDA devices
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Ensure deterministic behavior in PyTorch (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


###
# Plotting 
###

plt.rcParams.update({
    'figure.figsize': (12, 10),
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.bottom': False,
    'axes.spines.left': False,
    'xtick.bottom': False,
    'ytick.left': False
})

# plotting only spatial coordinates
def plot_cell_positions(filename, point_size=1, alpha=0.2, color='red', theta_deg=0.0):
    """
    Plot cell positions from a CSV file with optional centering.
    
    Parameters:
    -----------
    filename : str
        Path to the CSV file containing cell data
    point_size : float
        Size of the scatter points
    alpha : float
        Transparency of points
    color : str
        Color of the scatter points
    center : bool
        Whether to center the coordinates
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    df : pandas DataFrame
        The loaded data
    """
    # Load data
    df = pd.read_csv(filename)
    print(f"Number of cells: {len(df)}")
    
    # Extract coordinates
    x = np.array(df['center_x'])
    y = np.array(df['center_y'])

    x_c = x - np.mean(x)
    y_c = y - np.mean(y)

    theta0 = (np.pi/180)*-theta_deg

    L = np.array([[np.cos(theta0), -np.sin(theta0)],
                [np.sin(theta0), np.cos(theta0)]])
    
    target_L = np.matmul(L, np.array([x_c, y_c]))
    x_c_L = target_L[0]
    y_c_L = target_L[1]
    
    # Create plot
    fig, ax = plt.subplots()
    ax.scatter(x_c_L, y_c_L, s=point_size, alpha=alpha, c=color)
    #ax.legend(markerscale=10)
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    return fig, ax, df, x, y, x_c, y_c, x_c_L, y_c_L

# plotting two sets of spatial coordinates
def plot_dual_scatter(x1, y1, x2, y2, s1=1, s2=1, alpha1=0.2, alpha2=0.1, c1='red', c2='blue', show_legend=False):
    """
    Create a dual scatter plot with clean styling.
    
    Parameters:
    -----------
    x1, y1 : array-like
        Coordinates for the first scatter plot
    x2, y2 : array-like
        Coordinates for the second scatter plot
    s1, s2 : float
        Point sizes for first and second scatter plots
    alpha1, alpha2 : float
        Transparency values for first and second scatter plots
    c1, c2 : str
        Colors for first and second scatter plots
    show_legend : bool
        Whether to show the legend
        
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    fig, ax = plt.subplots()
    
    # Create scatter plots
    scatter1 = ax.scatter(x1, y1, s=s1, alpha=alpha1, c=c1, label='Dataset 1')
    scatter2 = ax.scatter(x2, y2, s=s2, alpha=alpha2, c=c2, label='Dataset 2')
    
    # Add legend if requested
    if show_legend:
        ax.legend(markerscale=10)
    
    # Clean styling
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_aspect('equal')
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    return fig, ax

# plotting gene expression comparison
def plot_ge_comparison(S1, S2, expr1, expr2, expr_transferred, 
                                   s=0.5, cmap='viridis', alpha=0.8, 
                                   titles=None, gene_name='Grm4', figsize=(18, 6)):
    """
    Plot gene expression comparison across two tissue slices with transfer prediction.
    
    Parameters:
    -----------
    S1 : numpy.ndarray
        2D coordinates for slice 1, shape (n_cells, 2)
    S2 : numpy.ndarray
        2D coordinates for slice 2, shape (n_cells, 2)
    expr1 : numpy.ndarray
        Expression values for slice 1
    expr2 : numpy.ndarray
        Expression values for slice 2
    expr_transferred : numpy.ndarray
        Transferred/predicted expression values for slice 2
    s : float
        Marker size for scatter plots
    cmap : str
        Colormap name
    alpha : float
        Marker transparency
    titles : list of str, optional
        Custom titles for the three plots
    gene_name : str
        Name of the gene being visualized
    figsize : tuple
        Figure size (width, height)
        
    Returns:
    --------
    fig : matplotlib Figure
    axes : list of matplotlib Axes
    """
    
    # Set default titles if not provided
    if titles is None:
        titles = [
            f"Slice 1 {gene_name} abundance",
            f"Slice 2 {gene_name} abundance",
            f"Slice 2 predicted {gene_name}"
        ]
    
    # Create common color normalization
    overall_min = min(expr1.min(), expr2.min(), expr_transferred.min())
    overall_max = max(expr1.max(), expr2.max(), expr_transferred.max())
    norm = mpl.colors.Normalize(vmin=overall_min, vmax=overall_max)
    
    # Create figure with extra space on the right for the colorbar
    fig = plt.figure(figsize=figsize)
    
    # Create a gridspec layout with room for the colorbar
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.1])
    
    # Create three axes for the plots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    axes = [ax1, ax2, ax3]
    
    # Create colorbar axis
    cbar_ax = fig.add_subplot(gs[0, 3])
    
    # Plot data arrays and expression values
    data_arrays = [(S1, expr1), (S2, expr2), (S2, expr_transferred)]
    
    for i, ((coords, expr), ax, title) in enumerate(zip(data_arrays, axes, titles)):
        sns.scatterplot(
            x=coords[:, 0],
            y=coords[:, 1],
            hue=expr,
            palette=cmap,
            hue_norm=norm,
            s=s,
            alpha=alpha,
            legend=False,
            ax=ax
        )
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.axis("off")
    
    # Add the colorbar using the dedicated axis
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        cax=cbar_ax,
        orientation='vertical'
    )
    cbar.set_label(f'{gene_name} expression')
    
    plt.tight_layout()
    return fig, axes

def transfer_and_visualize_gene(source_df, target_df, source_coords, target_coords, 
                               gene_name, P,
                               num_bins_x, num_bins_y, s=0.5, cmap='viridis', 
                               alpha=0.8, figsize=(14, 6), ret_results=False):
    """
    Transfer gene expression from source to target dataset, calculate spatial averages,
    evaluate similarity, and visualize results.
    
    Parameters:
    -----------
    source_df : pandas.DataFrame
        Source dataset containing gene expression data
    target_df : pandas.DataFrame
        Target dataset containing gene expression data
    source_coords : numpy.ndarray
        2D coordinates for source dataset, shape (n_cells, 2)
    target_coords : numpy.ndarray
        2D coordinates for target dataset, shape (n_cells, 2)
    gene_name : str
        Name of gene to transfer and analyze
    P : numpy.ndarray
        Mapping indices for transferring expression from source to target
        Should be a 2D array where each row is [source_index, target_index]
    num_bins_x : int
        Number of bins for spatial averaging in x dimension
    num_bins_y : int
        Number of bins for spatial averaging in y dimension
    s : float
        Point size for scatter plots
    cmap : str
        Colormap name
    alpha : float
        Transparency for scatter points
    figsize : tuple
        Figure size (width, height)
        
    Returns:
    --------
    dict : Dictionary containing:
        'cosine_similarity': float, similarity between true and transferred expression
        'figure': matplotlib Figure
        'axes': list of matplotlib Axes
        'source_smoothed': pandas.Series, smoothed source expression
        'target_smoothed': pandas.Series, smoothed target expression
        'transferred_smoothed': pandas.Series, smoothed transferred expression
    """
    source_indices = P[:, 0]
    target_indices = P[:, 1]
    
    # Extract gene counts from source and target
    source_counts = source_df[gene_name]
    target_counts = target_df[gene_name]
    
    # Initialize empty array for transferred counts
    transferred_counts = np.zeros(len(target_coords), dtype=source_counts.dtype)
    
    # Transfer expression from source to target using mapping indices
    transferred_counts[target_indices] = source_counts.iloc[source_indices]
    
    # Create a series for the transferred counts
    transferred_series = pd.Series(
        transferred_counts, 
        index=target_df.index, 
        name="transferred_expression"
    )
    
    # Add transferred expression to a copy of the target dataframe
    target_df_with_transfer = target_df.copy()
    target_df_with_transfer["transferred_expression"] = transferred_series
    
    # Calculate spatial averages
    source_smoothed = spatial_avg(
        source_df, 
        source_coords, 
        gene=gene_name, 
        num_bins_x=num_bins_x, 
        num_bins_y=num_bins_y
    )
    
    target_smoothed = spatial_avg(
        target_df, 
        target_coords, 
        gene=gene_name, 
        num_bins_x=num_bins_x, 
        num_bins_y=num_bins_y
    )
    
    transferred_smoothed = spatial_avg(
        target_df_with_transfer,
        target_coords,
        gene="transferred_expression",
        num_bins_x=num_bins_x,
        num_bins_y=num_bins_y
    )
    
    # Calculate cosine similarity between true and transferred expression
    vec1 = target_smoothed.values.reshape(1, -1)
    vec2 = transferred_smoothed.values.reshape(1, -1)
    cos_sim = cosine_similarity(vec1, vec2)
    cos_sim_value = cos_sim[0, 0]
    
    print(f"Cosine similarity for gene {gene_name}: {cos_sim_value:.4f}")
    
    # Create visualization
    titles = [
        f"Source {gene_name} expression",
        f"Target {gene_name} expression",
        f"Transferred {gene_name} expression"
    ]
    
    fig, axes = plot_ge_comparison(
        source_coords, target_coords,
        source_smoothed,
        target_smoothed,
        transferred_smoothed,
        s=s, cmap=cmap, alpha=alpha,
        titles=titles, gene_name=gene_name, figsize=figsize
    )
    
    # Optionally return results
    if not ret_results:
        return None
    else:
        # Return results as a dictionary
        results = {
            'cosine_similarity': cos_sim_value,
            'figure': fig,
            'axes': axes,
            'source_smoothed': source_smoothed,
            'target_smoothed': target_smoothed,
            'transferred_smoothed': transferred_smoothed
        }
        
        return results

###
# Spatial utils 
###

def spatial_avg(X_df, S, gene, num_bins_x=100, num_bins_y=100):

    x_min, x_max = S[:, 0].min(), S[:, 0].max()
    y_min, y_max = S[:, 1].min(), S[:, 1].max()

    df_coords = pd.DataFrame({
        'x': S[:, 0],
        'y': S[:, 1]
    }, index=X_df.index)  # ensure the same index

    df_coords['x_bin'] = pd.cut(df_coords['x'], bins=num_bins_x, labels=False)
    df_coords['y_bin'] = pd.cut(df_coords['y'], bins=num_bins_y, labels=False)

    # 2) Group by (x_bin, y_bin) and compute mean expression in each bin
    #    We'll first combine coords+counts in a single DataFrame
    df_all = pd.concat([df_coords, X_df], axis=1)

    # We'll group by the bin indices:
    grouped = df_all.groupby(['x_bin', 'y_bin'])

    # Then compute the mean expression across all genes:
    bin_means = grouped[X_df.columns].mean()  # DataFrame indexed by (x_bin, y_bin)

    # 3) Map each cell’s bin back to the mean expression
    #    We can merge or reindex by (x_bin, y_bin).
    df_coords2 = df_coords.copy()
    df_coords2['x_bin_y_bin'] = list(zip(df_coords2['x_bin'], df_coords2['y_bin']))

    # bin_means is indexed by (x_bin, y_bin); we align on that:
    bin_means.index = bin_means.index.set_names(['x_bin', 'y_bin'])
    bin_means = bin_means.reset_index()  # turn multi-index into columns for merging
    bin_means['x_bin_y_bin'] = list(zip(bin_means['x_bin'], bin_means['y_bin']))

    # Merge each cell with the average expression of its bin
    df_smoothed = pd.merge(
        df_coords2[['x_bin_y_bin']],
        bin_means.drop(['x_bin', 'y_bin'], axis=1),
        on='x_bin_y_bin',
        how='left'
    )
    df_smoothed.index = df_coords2.index  # restore original cell index

    # df_smoothed now has the "smoothed" expression for each cell in X_df.columns
    smoothed_counts = df_smoothed[X_df.columns]

    # 4) For a single gene, e.g. "Grm4", you can see its smoothed version:
    smoothed_gene = smoothed_counts[gene]
    return smoothed_gene


###
# computing cost
###

def compute_transport_cost(S1, S2, transport_plan):
    """
    Compute the transport cost with vectorized operations while using linear space.
    
    Parameters:
    -----------
    S1 : numpy.ndarray
        Source distribution points of shape (n1, d)
    S2 : numpy.ndarray
        Target distribution points of shape (n2, d)
    transport_plan : numpy.ndarray
        An array of shape (n, 2) containing index pairs
    
    Returns:
    --------
    float
        The total transport cost
    """
    # Extract indices from transport plan
    indices = transport_plan.astype(int)
    i_indices = indices[:, 0]
    j_indices = indices[:, 1]
    
    # Extract the relevant points from S1 and S2
    S1_selected = S1[i_indices]
    S2_selected = S2[j_indices]
    
    # Compute Euclidean distances
    distances = np.sqrt(np.sum((S1_selected - S2_selected)**2, axis=1))
    
    # Sum all distances
    total_cost = np.sum(distances)
    
    return total_cost


def compute_transport_cost_from_plan(S1, S2, plan):
    """
    Compute transport cost from a transport plan matrix without instantiating
    the full cost matrix.
    
    Parameters:
    -----------
    S1 : numpy.ndarray
        Source distribution points of shape (n, d)
    S2 : numpy.ndarray
        Target distribution points of shape (n, d)
    plan : numpy.ndarray
        Transport plan matrix of shape (n, n)

    Returns:
    --------
    float
        The total transport cost
    """
    # total_cost = 0.0
    
    '''
    # Find non-zero entries in the plan to avoid unnecessary calculations
    nonzero_indices = np.argwhere(plan > 0)
    
    # Compute cost only for non-zero entries in the plan
    for i, j in nonzero_indices:
        # Calculate distance
        dist = np.sqrt(np.sum((S1[i] - S2[j])**2)) ** p
        
        # Add weighted cost
        total_cost += dist * plan[i, j]
    '''
    C = cdist(S1, S2)

    cost = np.sum(C * plan)
        
    return cost


def compute_transport_cost_from_factored_plan_jax(S1, S2, Q, R, g):
    """
    JAX-accelerated computation of transport cost from factored plan.
    
    Parameters:
    -----------
    S1 : jax.numpy.ndarray
        Source distribution points of shape (n, d)
    S2 : jax.numpy.ndarray
        Target distribution points of shape (m, d)
    Q : jax.numpy.ndarray
        Matrix of shape (n, r)
    R : jax.numpy.ndarray
        Matrix of shape (m, r)
    g : jax.numpy.ndarray
        Vector of length r
    
    Returns:
    --------
    float
        The total transport cost
    """
    # Create a function to compute individual cost contributions
    @jax.jit
    def compute_cost_for_row(i, total):
        # Get the i-th row of Q and scale it by 1/g
        q_i_scaled = Q[i] * (1.0 / g)
        
        # Calculate costs for all j's at once to maximize parallelism
        dists = jnp.sqrt(jnp.sum((S1[i, None, :] - S2) ** 2, axis=1))
        plan_vals = jnp.dot(q_i_scaled, R.T)
        row_cost = jnp.sum(dists * plan_vals)
        
        return total + row_cost
    
    # Use JAX's scan operation to loop through rows efficiently
    total_cost = jax.lax.fori_loop(0, S1.shape[0], compute_cost_for_row, 0.0)
    
    return total_cost


def efficient_normalized_cdist(S1, S2, device='cuda', dtype=torch.float32, p=2, batch_size=1000):
    """
    Efficiently compute normalized pairwise distances between two sets of points
    without instantiating the full distance matrix at once.
    
    Args:
        S1: First set of points
        S2: Second set of points
        device: Computation device ('cuda' or 'cpu')
        dtype: Data type for computation
        p: The p-norm to use for distance computation
        batch_size: Size of batches to use for computation
        
    Returns:
        Normalized pairwise distance matrix
    """
    import torch
    
    S1_ = torch.tensor(S1, device=device, dtype=dtype)
    S2_ = torch.tensor(S2, device=device, dtype=dtype)
    
    n1 = S1_.shape[0]
    n2 = S2_.shape[0]
    
    # First pass: find max distance without storing full matrix
    max_dist = 0.0
    
    for i in range(0, n1, batch_size):
        end_i = min(i + batch_size, n1)
        S1_batch = S1_[i:end_i]
        
        for j in range(0, n2, batch_size):
            end_j = min(j + batch_size, n2)
            S2_batch = S2_[j:end_j]
            
            # Compute batch of distances
            C_batch = torch.cdist(S1_batch, S2_batch, p=p)
            
            # Update max distance
            batch_max = torch.max(C_batch)
            max_dist = max(max_dist, batch_max.item())
    
    # Second pass: create and fill normalized distance matrix
    C_normalized = torch.zeros((n1, n2), device=device, dtype=dtype)
    
    for i in range(0, n1, batch_size):
        end_i = min(i + batch_size, n1)
        S1_batch = S1_[i:end_i]
        
        for j in range(0, n2, batch_size):
            end_j = min(j + batch_size, n2)
            S2_batch = S2_[j:end_j]
            
            # Compute batch of distances and normalize
            C_batch = torch.cdist(S1_batch, S2_batch, p=p)
            C_normalized[i:end_i, j:end_j] = C_batch / max_dist
    
    return C_normalized
    
    # Now compute normalized distances in batches
    def get_normalized_distances(i, j, end_i, end_j):
        S1_batch = S1_[i:end_i]
        S2_batch = S2_[j:end_j]
        C_batch = torch.cdist(S1_batch, S2_batch, p=p)
        return C_batch / max_dist
    
    # Example of using the function to compute a specific batch
    # (You can integrate this into your workflow as needed)
    # normalized_batch = get_normalized_distances(0, 0, batch_size, batch_size)
    
    return get_normalized_distances, max_dist


def compute_transport_cost_from_factored_plan_torch(S1, S2, Q, R, g, chunk_size=1000, device=None):
    """
    Memory-efficient PyTorch implementation of transport cost calculation
    from factored plan representation.
    
    Parameters:
    -----------
    S1 : torch.Tensor
        Source distribution points of shape (n, d)
    S2 : torch.Tensor
        Target distribution points of shape (m, d)
    Q : torch.Tensor
        Matrix of shape (n, r)
    R : torch.Tensor
        Matrix of shape (m, r)
    g : torch.Tensor
        Vector of length r
    chunk_size : int, optional
        Size of chunks to process at once. Default is 1000.
    device : torch.device, optional
        Device to run computation on. Default is None (uses current device).
    
    Returns:
    --------
    float
        The total transport cost
    """
    # Make sure everything is on the same device
    if device is not None:
        S1 = S1.to(device)
        S2 = S2.to(device)
        Q = Q.to(device)
        R = R.to(device)
        g = g.to(device)
    
    n = S1.shape[0]
    m = S2.shape[0]
    
    # Precompute 1/g
    g_inv = 1.0 / g
    
    total_cost = torch.tensor(0.0, device=S1.device)
    
    # Process in chunks to minimize memory usage
    for i in range(0, n, chunk_size):
        # Clear CUDA cache if using GPU
        if device is not None and device.type == 'cuda':
            torch.cuda.empty_cache()
            
        i_end = min(i + chunk_size, n)
        S1_chunk = S1[i:i_end]
        Q_chunk = Q[i:i_end]
        
        # Scale Q by 1/g
        Q_scaled_chunk = Q_chunk * g_inv.unsqueeze(0)
        
        # Process S2 in chunks as well for very large problems
        for j in range(0, m, chunk_size):
            j_end = min(j + chunk_size, m)
            S2_chunk = S2[j:j_end]
            R_chunk = R[j:j_end]
            
            # Compute pairwise distances efficiently using broadcasting
            # Reshape for broadcasting: (chunk_size, 1, d) - (1, chunk_size, d)
            diff = S1_chunk.unsqueeze(1) - S2_chunk.unsqueeze(0)
            dist = torch.sqrt(torch.sum(diff**2, dim=2))
            
            # Compute plan values: (chunk_size, r) @ (r, chunk_size) -> (chunk_size, chunk_size)
            plan_vals = torch.mm(Q_scaled_chunk, R_chunk.t())
            
            # Multiply distances by plan values and sum
            chunk_cost = torch.sum(dist * plan_vals)
            total_cost += chunk_cost
            
            # Free memory
            del diff, dist, plan_vals
            if device is not None and device.type == 'cuda':
                torch.cuda.empty_cache()
    
    return total_cost.item()

###
# minibatch, with plan return
###


@jax.jit
def sinkhorn_batch(xs, ys, 
                   x0s, y0s,
                   x1s, y1s,
                   x2s, y2s,
                   x3s, y3s,
                   x4s, y4s,
                   p=1):
    # Pre-compile a jitted instance
    cost_mat = jnp.linalg.norm(xs[:, None, :] - ys[None, :, :], axis=-1) ** p

    cost_mat_0 = x0s[:, None] * y0s[None, :]
    cost_mat_1 = x1s[:, None] * y1s[None, :]
    cost_mat_2 = x2s[:, None] * y2s[None, :]
    cost_mat_3 = x3s[:, None] * y3s[None, :]
    cost_mat_4 = x4s[:, None] * y4s[None, :]

    geom = Geometry(cost_mat)
    ot_problem = linear_problem.LinearProblem(geom)
    solver = sinkhorn.Sinkhorn()
    ot_solution = solver(ot_problem)
    gamma = ot_solution.matrix

    batch_cost = jnp.sum(cost_mat * gamma)

    batch_cost_0 = jnp.sum(cost_mat_0 * gamma)
    batch_cost_1 = jnp.sum(cost_mat_1 * gamma)
    batch_cost_2 = jnp.sum(cost_mat_2 * gamma)
    batch_cost_3 = jnp.sum(cost_mat_3 * gamma)
    batch_cost_4 = jnp.sum(cost_mat_4 * gamma)

    return batch_cost, batch_cost_0, batch_cost_1, batch_cost_2, batch_cost_3, batch_cost_4


def minibatch_sinkhorn(X, Y, X0, Y0, X1, Y1, X2, Y2, X3, Y3, X4, Y4, batch_size, p=1):
    """
    Compute mini-batch OT using entropic regularization (Sinkhorn via ott-jax) without replacement.
    Implicit coupling corresponds to definition 6 of
    
    Parameters:
    -----------
    X: np.array, shape (n, d) - source samples.
    Y: np.array, shape (n, d) - target samples.
    batch_size: int - number of samples in each mini-batch.
    p: float - power for the cost (default is 1, but cost is computed as squared Euclidean).
    
    Returns:
    --------
    tuple: (transport cost over the mini-batches, full transport plan)
    """
    n = X.shape[0]
    assert X.shape[0] == Y.shape[0], "X and Y must have the same number of points."
    batch_size = min(n, batch_size)
    
    # Create a random permutation for batching without replacement.
    perm = np.random.permutation(n)
    batches = [(perm[i:i+batch_size], perm[i:i+batch_size])
              for i in range(0, n, batch_size)]
    
    total_cost = 0.0
    total_cost_0 = 0.0
    total_cost_1 = 0.0
    total_cost_2 = 0.0
    total_cost_3 = 0.0
    total_cost_4 = 0.0

    num_batches = len(batches)
    
    for idx_src, idx_tgt in tqdm(batches, desc="Mini-batch Sinkhorn"):
        # Convert the mini-batch data to jax.numpy arrays.
        xs = jnp.array(X[idx_src])
        ys = jnp.array(Y[idx_tgt])
        x0s = jnp.array(X0[idx_src])
        y0s = jnp.array(Y0[idx_tgt])
        x1s = jnp.array(X1[idx_src])
        y1s = jnp.array(Y1[idx_tgt])
        x2s = jnp.array(X2[idx_src])
        y2s = jnp.array(Y2[idx_tgt])
        x3s = jnp.array(X3[idx_src])
        y3s = jnp.array(Y3[idx_tgt])
        x4s = jnp.array(X4[idx_src])
        y4s = jnp.array(Y4[idx_tgt])
        
        # Use the precompiled Sinkhorn function.
        batch_cost, batch_cost_0, batch_cost_1, batch_cost_2, batch_cost_3, batch_cost_4 = sinkhorn_batch(xs, ys,
                                                                                                          x0s, y0s,
                                                                                                          x1s, y1s,
                                                                                                          x2s, y2s,
                                                                                                          x3s, y3s,
                                                                                                          x4s, y4s,
                                                                                                          p=p)
        total_cost += float(batch_cost)
        total_cost_0 += float(batch_cost_0)
        total_cost_1 += float(batch_cost_1)
        total_cost_2 += float(batch_cost_2)
        total_cost_3 += float(batch_cost_3)
        total_cost_4 += float(batch_cost_4)
        
    # Return the average cost across batches and the full transport plan.
    return total_cost / num_batches, total_cost_0, total_cost_1, total_cost_2, total_cost_3, total_cost_4