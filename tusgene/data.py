"""
Data Loading and Preprocessing
==============================

Functions for loading gene expression data and brain atlas files.
"""

import numpy as np
import pandas as pd
import nibabel as nib
from . import config


def load_expression_data(genes=None):
    """
    Load gene expression data from CSV.

    The expression data contains gene expression values for 332 brain regions
    (300 cortical + 32 subcortical) from the Allen Human Brain Atlas.

    Args:
        genes: List of gene names to extract. If None, uses config.TUS_GENES.

    Returns:
        expression_df: Full pandas DataFrame
        expression_matrix: Expression matrix (n_regions x n_genes)
        expression_zscore: Z-scored expression matrix
    """
    if genes is None:
        genes = config.TUS_GENES

    print("Loading expression data...")
    expression_df = pd.read_csv(config.EXPRESSION_PATH)

    # Validate genes exist in data
    available_genes = [g for g in genes if g in expression_df.columns]
    missing_genes = [g for g in genes if g not in expression_df.columns]

    if missing_genes:
        print(f"  Warning: {len(missing_genes)} genes not found: {missing_genes}")

    # Extract expression matrix
    expression_matrix = expression_df[available_genes].to_numpy()

    # Z-score normalization (mean=0, std=1 per gene)
    # This ensures equal weighting of genes in clustering
    expression_zscore = (expression_matrix - expression_matrix.mean(axis=0)) / expression_matrix.std(axis=0)

    print(f"  Loaded {expression_matrix.shape[0]} brain regions x {expression_matrix.shape[1]} genes")
    print(f"  Cortical regions: {config.N_CORTICAL_REGIONS}")
    print(f"  Subcortical regions: {config.N_SUBCORTICAL_REGIONS}")

    return expression_df, expression_matrix, expression_zscore, available_genes


def load_atlas():
    """
    Load brain atlas NIfTI file.

    The atlas combines:
    - Yeo 2011 17-network cortical parcellation (300 parcels, labels 1-300)
    - Tian S2 subcortical parcellation (32 regions, labels 301-332)

    Returns:
        atlas_img: NIfTI image object
        atlas_data: 3D numpy array of atlas labels
    """
    print("Loading brain atlas...")
    atlas_img = nib.load(config.ATLAS_PATH)
    atlas_data = atlas_img.get_fdata()

    # Get atlas dimensions
    shape = atlas_data.shape
    n_labels = len(np.unique(atlas_data)) - 1  # Exclude 0 (background)

    print(f"  Atlas dimensions: {shape}")
    print(f"  Number of labeled regions: {n_labels}")

    return atlas_img, atlas_data


def get_all_available_genes():
    """
    Get list of all genes available in the expression dataset.

    Returns:
        List of gene names (column names excluding 'label')
    """
    expression_df = pd.read_csv(config.EXPRESSION_PATH)
    all_genes = [col for col in expression_df.columns if col != 'label']
    return all_genes


def get_region_info(labels, labels_parent=None):
    """
    Generate summary statistics for cluster assignments.

    Args:
        labels: Cluster assignment array (length = n_regions)
        labels_parent: Optional parent cluster labels (e.g., K_OPTIMAL labels
                       when analyzing K_GRANULAR clusters) for hierarchy mapping

    Returns:
        DataFrame with cluster composition information
    """
    k = len(np.unique(labels))
    n_regions = len(labels)

    summary = []
    for c in range(k):
        mask = (labels == c)
        n_total = mask.sum()
        n_cortical = (labels[:config.N_CORTICAL_REGIONS] == c).sum()
        n_subcortical = (labels[config.N_CORTICAL_REGIONS:] == c).sum()
        pct_cortical = n_cortical / n_total * 100 if n_total > 0 else 0

        row = {
            'cluster': c + 1,
            'n_regions': n_total,
            'n_cortical': n_cortical,
            'n_subcortical': n_subcortical,
            'pct_cortical': pct_cortical
        }

        # Add parent cluster info if parent labels provided
        if labels_parent is not None:
            regions_in_cluster = np.where(labels == c)[0]
            k_parent = len(np.unique(labels_parent))
            parent_counts = np.bincount(labels_parent[regions_in_cluster], minlength=k_parent)
            parent = np.argmax(parent_counts) + 1
            row['parent_cluster'] = parent

        summary.append(row)

    return pd.DataFrame(summary)