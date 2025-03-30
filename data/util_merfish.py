import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl



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