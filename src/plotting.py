import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_pts(x_source, x_target, bary_source, bary_target, P, title_str='', scale=20, file_handle=None):
    
    # Create a scatter plot
    plt.scatter(x_source[:,0], x_source[:,1], color='orange', label='Source', alpha=0.2)
    plt.scatter(x_target[:,0], x_target[:,1], c='green', label='Target', alpha=0.2)
    
    plt.scatter(bary_source[:,0], bary_source[:,1], color='blue', label='Latent source points')
    plt.scatter(bary_target[:,0], bary_target[:,1], c='red', label='Latent target points')
    
    # Draw lines between paired points
    for i in range(bary_source.shape[0]):
        for j in range(bary_target.shape[0]):
            plt.plot([bary_source[i,0], bary_target[j,0]],
                      [bary_source[i,1], bary_target[j,1]], \
                     'k-', c='b', alpha=0.5, linewidth=P[i,j]*scale)
        
    plt.title(title_str)
    plt.axis('off')
    plt.legend()
    if file_handle is None:
        plt.show()
    else:
        plt.savefig(file_handle)
        plt.show()
    return

def plot_pts_GT(x_source, x_target, P, title_str='', scale=50, file_handle=None):
    
    # Create a scatter plot
    plt.scatter(x_source[:,0], x_source[:,1], color='orange', label='Source', alpha=0.2)
    plt.scatter(x_target[:,0], x_target[:,1], c='green', label='Target', alpha=0.2)
    
    # Draw lines between paired points
    for i in range(x_source.shape[0]):
        print(i)
        for j in range(x_target.shape[0]):
            plt.plot([x_source[i,0], x_target[j,0]],
                      [x_source[i,1], x_target[j,1]], \
                     'k-', c='b', alpha=0.5, linewidth=P[i,j]*scale)
        
    plt.title(title_str)
    plt.axis('off')
    plt.legend()
    if file_handle is None:
        plt.show()
    else:
        plt.savefig(file_handle)
        plt.show()
    return

mouse_color_map = {}
mouse_color_map['CD_IC'] = sns.color_palette('tab20')[0]
mouse_color_map['CD_PC'] = sns.color_palette('tab20')[1]
mouse_color_map['CNT'] = sns.color_palette('tab20')[2]
mouse_color_map['DCT'] = sns.color_palette('tab20')[3]
mouse_color_map['Endo'] = sns.color_palette('tab20')[4]
mouse_color_map['Fib'] = sns.color_palette('tab20')[5]
mouse_color_map['Immune'] = sns.color_palette('tab20')[6]
mouse_color_map['JGA'] = sns.color_palette('tab20')[7]
mouse_color_map['LOH'] = sns.color_palette('tab20')[8]
mouse_color_map['PEC'] = sns.color_palette('tab20')[9]
mouse_color_map['PT(S1)'] = sns.color_palette('tab20')[10]
mouse_color_map['PT(S2)'] = sns.color_palette('tab20')[11]
mouse_color_map['PT(S3)'] = sns.color_palette('tab20')[12]
mouse_color_map['Podo'] = sns.color_palette('tab20')[13]
mouse_color_map['Uro'] = sns.color_palette('tab20')[14]
mouse_color_map['IM'] = sns.color_palette('tab20')[15]
mouse_color_map['NP'] = sns.color_palette('tab20')[16]
mouse_color_map['PT'] = sns.color_palette('tab20')[17]
mouse_color_map['UBP'] = sns.color_palette('tab20')[18]

cell_to_color_map = {}
cell_to_color_map['A'] = sns.color_palette('tab20')[0]
cell_to_color_map['B'] = sns.color_palette('tab20')[1]

def largest_indices(ary, n=None):
    """Returns the n largest indices from a numpy array."""
    if n is not None:
        flat = ary.flatten()
        indices = np.argpartition(flat, -n)[-n:]
        indices = indices[np.argsort(-flat[indices])]
        return np.unravel_index(indices, ary.shape)
    else:
        indices_slice1 = np.argmax(ary, axis=0)
        indices_slice2 = np.arange(ary.shape[1])
        return (indices_slice1, indices_slice2)

def plot2D_samples_mat(xs, xt, G, thr=1e-8, alpha=0.2, top=1000, weight_alpha=False, **kwargs):
    if ('color' not in kwargs) and ('c' not in kwargs):
        kwargs['color'] = 'k'
    mx = G.max()
    #     idx = np.where(G/mx>=thr)
    idx = largest_indices(G, top)
    print(len(idx[0]))
    for l in range(len(idx[0])):
        plt.plot([xs[idx[0][l], 0], xt[idx[1][l], 0]], [xs[idx[0][l], 1], xt[idx[1][l], 1]],
                 alpha=alpha * (1 - weight_alpha) + (weight_alpha * G[idx[0][l], idx[1][l]] / mx), c='k')
    return
def plot_slice_pairwise_alignment(slice_pair, pi, thr=1 - 1e-8, alpha=0.05, top=1000, weight_alpha=False):
    """
    Parameters:
    slice_pair_filename: path to the .h5ad file of the slice pair to align
    pi: alignment matrix of the slice pair
    Returns:
    Plots the top 1000 aligned pairs
    """
    slice1 = slice_pair[slice_pair.obs['timepoint'] == 1]
    slice2 = slice_pair[slice_pair.obs['timepoint'] == 2]
    coordinates1, coordinates2 = slice1.obsm['spatial'], slice2.obsm['spatial']
    offset = (coordinates1[:, 0].max() - coordinates2[:, 0].min()) * 1.1
    temp = np.zeros(coordinates2.shape)
    temp[:, 0] = offset
    plt.figure(figsize=(20, 10))
    plot2D_samples_mat(coordinates1, coordinates2 + temp, pi, thr=thr, c='k', alpha=alpha, top=top,
                       weight_alpha=weight_alpha)
    plt.scatter(coordinates1[:, 0], coordinates1[:, 1], linewidth=0, s=100, marker=".", color=list(slice1.obs['annotation'].astype('str').map(cell_to_color_map)))
    plt.scatter(coordinates2[:, 0] + offset, coordinates2[:, 1], linewidth=0, s=100, marker=".", color=list(slice2.obs['annotation'].astype('str').map(cell_to_color_map)))
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.show()
    return

def plot_slice_value(slice, value_vec, vmax, title=None, save_name=None):
    """
    Parameters: 
    slice: AnnData object of the slice
    value_vec: growth rate vector
    vmax: max value in value_vec to plot

    Returns:
    Plots the slice with each spot colored according to its value in value_vec
    """
    plt.figure()
    spatial = slice.obsm['spatial']
    sc = plt.scatter(spatial[:, 0], spatial[:, 1], c=value_vec, cmap='RdYlGn', s=50, vmax=vmax)
    cbar = plt.colorbar(sc)
    cbar.ax.tick_params(labelsize=20)
    plt.gca().invert_yaxis()
    plt.axis('off')

    fig = plt.gcf()
    fig_size = fig.get_size_inches()
    new_width = 20.0
    new_height = new_width * (fig_size[1] / fig_size[0])
    fig.set_size_inches(new_width, new_height)
    # fig.suptitle(title)
    if save_name:
        plt.savefig(save_name, dpi=300, transparent=True, bbox_inches="tight")
    plt.show()
    return


def plot_slice_value_binary(slice, value_vec):
    plt.figure()
    spatial = slice.obsm['spatial']
    color_mask = value_vec > 0
    sc = plt.scatter(spatial[:, 0], spatial[:, 1], c=color_mask, cmap='RdYlGn', s=50)
    plt.colorbar(sc, ticks=[0, 1], format='%.0f')
    plt.gca().invert_yaxis()

    fig = plt.gcf()
    fig_size = fig.get_size_inches()
    new_width = 20.0
    new_height = new_width * (fig_size[1] / fig_size[0])
    fig.set_size_inches(new_width, new_height)
    plt.show()
    return
