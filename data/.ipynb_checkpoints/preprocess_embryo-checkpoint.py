import scanpy as sc
import anndata as ad
import numpy as np
import json
import os

def intersect(lst1, lst2): 
    """
    param: lst1 - list
    param: lst2 - list
    
    return: list of common elements
    """

    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3

def preprocess_embryo(slice1_filename, slice2_filename, save_dir):
    """
    Preprocess spatial transcriptomics data from two embryonic slices.
    
    This function reads two `.h5ad` files, extracts and normalizes count data, 
    ensures gene set consistency, saves spatial coordinates and cell types, 
    performs PCA, and saves the processed features.
    
    Parameters:
    slice1_filename (str): Path to the first slice `.h5ad` file.
    slice2_filename (str): Path to the second slice `.h5ad` file.
    save_dir (str): Directory to save processed data.
    
    Returns:
    None: Saves processed data files for further analysis.
    """
    
    adata_t1 = sc.read_h5ad(slice1_filename)
    adata_t1.X = adata_t1.layers['count']
    adata_t1.obs['timepoint'] = [1] * adata_t1.shape[0]
    
    adata_t2 = sc.read_h5ad(slice2_filename)
    adata_t2.X = adata_t2.layers['count']
    adata_t2.obs['timepoint'] = [2] * adata_t2.shape[0]
    
    # Make sure t1 and t2 slices have the same set of genes
    common_genes = intersect(adata_t1.var.index, adata_t2.var.index)
    adata_t1 = adata_t1[:, common_genes]
    adata_t2 = adata_t2[:, common_genes]
    
    print('printing adata shapes')
    print(adata_t1.shape)
    print(adata_t2.shape)
    
    # Save the spot coordiantes
    slice1_coordinates = np.array(adata_t1.obsm['spatial'])
    slice2_coordinates = np.array(adata_t2.obsm['spatial'])
    np.save(save_dir + 'slice1_coordinates.npy', slice1_coordinates)
    np.save(save_dir + 'slice2_coordinates.npy', slice2_coordinates)
    
    # Save the cell types
    slice1_types = adata_t1.obs['annotation'].tolist()
    slice2_types = adata_t2.obs['annotation'].tolist()
    
    with open(save_dir+'slice1_types.json', 'w') as file:
        json.dump(slice1_types, file)
    with open(save_dir+'slice2_types.json', 'w') as file:
        json.dump(slice2_types, file)
    
    # Concatenate the datasets for a joint PCA
    joint_adata = ad.concat([adata_t1, adata_t2])
    joint_adata.obs['timepoint'] = joint_adata.obs['timepoint'].astype('category')
    
    print('joint adata constructed')
    
    # Normalize and log-transform the data
    sc.pp.normalize_total(joint_adata)
    sc.pp.log1p(joint_adata)
    print('log normalized')

    # Perform PCA
    sc.pp.pca(joint_adata, n_comps=30)
    print("pca done")
    
    slice1 = joint_adata[joint_adata.obs['timepoint'] == 1]
    slice2 = joint_adata[joint_adata.obs['timepoint'] == 2]

    # Extract and save PCA features from both datasets
    slice1_feature = np.array(slice1.obsm['X_pca'])
    slice2_feature = np.array(slice2.obsm['X_pca'])
    np.save(save_dir+'slice1_feature.npy', slice1_feature)
    np.save(save_dir+'slice2_feature.npy', slice2_feature)

    return


if __name__ == "__main__":
    
    embryo_directory = '/scratch/gpfs/ph3641/mouse_embryo/'
    embryo_stages = ['E9.5', 'E10.5', 'E11.5', 'E12.5', 'E13.5','E14.5', 'E15.5', 'E16.5']  # Example stages
    embryo_endhandle = '_E1S1.MOSTA.h5ad'
    save_directory = '/scratch/gpfs/ph3641/mouse_embryo_preprocessed/'
    
    for i in range(len(embryo_stages) - 1):
        
        print('Preprocessing embryo stages ' + embryo_stages[i] + ' and ' + embryo_stages[i+1])
        
        slice1_filename = embryo_directory + embryo_stages[i] + embryo_endhandle
        slice2_filename = embryo_directory + embryo_stages[i+1] + embryo_endhandle
        
        save_dir = save_directory + embryo_stages[i] + '_' + embryo_stages[i+1] + '/'
        
        if not os.path.exists(save_dir):
            print('Saving slice 1,2 pair to directory: ' + save_dir)
            os.makedirs(save_dir)
        
        preprocess_embryo(slice1_filename, slice2_filename, save_dir)
