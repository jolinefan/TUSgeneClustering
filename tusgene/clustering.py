"""
Clustering Functions
====================

PCA dimensionality reduction and K-means clustering for brain region parcellation.

Key Methods:
    - PCA: Reduces 30-dimensional gene expression to 10 principal components
    - K-means: Clusters brain regions based on expression profiles
    - Silhouette analysis: Evaluates clustering quality across K values
    - Hierarchical clustering: Orders genes/clusters for visualization
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from . import config


def perform_pca(expression_matrix, n_components=None):
    """
    Perform PCA dimensionality reduction.

    Why PCA before clustering?
    - Reduces noise in high-dimensional gene expression data
    - 10 components capture ~90% of total variance
    - Improves K-means stability (curse of dimensionality)
    - Makes clustering more interpretable

    Args:
        expression_matrix: Expression matrix (n_regions x n_genes)
        n_components: Number of components. Default: config.N_PCA_COMPONENTS

    Returns:
        expression_pca: Transformed data (n_regions x n_components)
        pca: Fitted PCA object (for variance analysis)
    """
    if n_components is None:
        n_components = config.N_PCA_COMPONENTS

    pca = PCA(n_components=n_components, random_state=config.RANDOM_STATE)
    expression_pca = pca.fit_transform(expression_matrix)

    explained_var = pca.explained_variance_ratio_.sum()
    print(f"PCA: {n_components} components explain {explained_var:.1%} of variance")

    return expression_pca, pca


def cluster_kmeans(expression_pca, k, random_state=None):
    """
    Perform K-means clustering.

    Args:
        expression_pca: PCA-transformed expression data
        k: Number of clusters
        random_state: Random seed for reproducibility

    Returns:
        labels: Cluster assignments (0 to k-1)
        kmeans: Fitted KMeans object
    """
    if random_state is None:
        random_state = config.RANDOM_STATE

    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
    labels = kmeans.fit_predict(expression_pca)

    return labels, kmeans


def compute_silhouette_curve(expression_pca, k_range=None):
    """
    Compute silhouette scores across multiple K values.

    The silhouette score measures how similar each point is to its own cluster
    compared to other clusters. Range: [-1, 1], higher is better.

    Interpretation:
        - K=2 typically highest (~0.58) but trivially splits cortex/subcortex
        - K_OPTIMAL is the best for K>=3, revealing meaningful subdivisions
        - K_GRANULAR provides finer granularity (local maximum)

    Args:
        expression_pca: PCA-transformed expression data
        k_range: Range of K values to test. Default: config.K_RANGE

    Returns:
        k_values: Array of K values tested
        silhouettes: Corresponding silhouette scores
        optimal_k: K with highest silhouette (excluding K=2)
    """
    if k_range is None:
        k_range = config.K_RANGE

    k_values = []
    silhouettes = []

    print("Computing silhouette scores...")
    for k in k_range:
        labels, _ = cluster_kmeans(expression_pca, k)
        score = silhouette_score(expression_pca, labels)
        k_values.append(k)
        silhouettes.append(score)
        print(f"  K={k:2d}: silhouette = {score:.4f}")

    k_values = np.array(k_values)
    silhouettes = np.array(silhouettes)

    # Find optimal K (excluding K=2 which trivially splits cortex/subcortex)
    mask_k3_plus = k_values >= 3
    if mask_k3_plus.any():
        idx_best = np.argmax(silhouettes[mask_k3_plus])
        optimal_k = k_values[mask_k3_plus][idx_best]
        optimal_score = silhouettes[mask_k3_plus][idx_best]
        print(f"\nOptimal K (for K>=3): {optimal_k} (silhouette = {optimal_score:.4f})")
    else:
        optimal_k = k_values[np.argmax(silhouettes)]
        print(f"\nOptimal K: {optimal_k}")

    return k_values, silhouettes, optimal_k


def find_local_maxima(k_values, silhouettes):
    """
    Find local maxima in silhouette curve.

    Useful for identifying meaningful K values beyond the global optimum.

    Args:
        k_values: Array of K values
        silhouettes: Corresponding silhouette scores

    Returns:
        List of (k, score) tuples for local maxima
    """
    local_maxima = []
    for i in range(1, len(silhouettes) - 1):
        if silhouettes[i] > silhouettes[i-1] and silhouettes[i] > silhouettes[i+1]:
            local_maxima.append((k_values[i], silhouettes[i]))
    return local_maxima


def compute_cluster_means(expression_zscore, labels, genes=None):
    """
    Compute mean z-scored expression for each cluster.

    Creates the data matrix for heatmap visualization: each row is a cluster's
    average expression profile across all genes.

    Args:
        expression_zscore: Z-scored expression matrix
        labels: Cluster assignments
        genes: Gene list (for shape inference). Default: config.TUS_GENES

    Returns:
        cluster_means: Matrix of shape (n_clusters x n_genes)
    """
    if genes is None:
        genes = config.TUS_GENES

    k = len(np.unique(labels))
    n_genes = expression_zscore.shape[1]
    cluster_means = np.zeros((k, n_genes))

    for c in range(k):
        cluster_means[c] = expression_zscore[labels == c].mean(axis=0)

    return cluster_means


def hierarchical_cluster_genes(cluster_means, method='average'):
    """
    Hierarchically cluster genes based on their expression patterns.

    Used to order genes in heatmap so similar genes are adjacent.

    Args:
        cluster_means: Matrix of cluster expression profiles (k x n_genes)
        method: Linkage method ('average', 'ward', 'complete', etc.)

    Returns:
        linkage_matrix: Scipy linkage matrix for dendrogram
    """
    # Transpose so we cluster genes (columns become rows)
    gene_distances = pdist(cluster_means.T)
    linkage_matrix = linkage(gene_distances, method=method)
    return linkage_matrix


def hierarchical_cluster_clusters(cluster_means, method='average'):
    """
    Hierarchically cluster the clusters themselves.

    Used to show relationships between K_GRANULAR subclusters and their K_OPTIMAL parents.

    Args:
        cluster_means: Matrix of cluster expression profiles (k x n_genes)
        method: Linkage method

    Returns:
        linkage_matrix: Scipy linkage matrix for dendrogram
    """
    cluster_distances = pdist(cluster_means)
    linkage_matrix = linkage(cluster_distances, method=method)
    return linkage_matrix


def map_fine_to_coarse_clusters(labels_fine, labels_coarse):
    """
    Determine which coarse cluster each fine cluster primarily belongs to.

    This enables visualization of K_GRANULAR clusters colored by their K_OPTIMAL parent,
    showing hierarchical nesting of clustering solutions.

    Method: For each fine cluster, count how many regions belong to each
    coarse cluster, then assign to the majority.

    Args:
        labels_fine: Cluster assignments for fine solution (e.g., K_GRANULAR)
        labels_coarse: Cluster assignments for coarse solution (e.g., K_OPTIMAL)

    Returns:
        parent_map: Dict mapping fine cluster index -> coarse cluster index
    """
    k_fine = len(np.unique(labels_fine))
    k_coarse = len(np.unique(labels_coarse))

    parent_map = {}
    for c_fine in range(k_fine):
        # Find all regions in this fine cluster
        regions_in_fine = np.where(labels_fine == c_fine)[0]

        # Count membership in each coarse cluster
        coarse_counts = np.bincount(labels_coarse[regions_in_fine], minlength=k_coarse)

        # Assign to majority coarse cluster
        parent_map[c_fine] = int(np.argmax(coarse_counts))

    return parent_map


def get_cluster_results(expression_matrix, expression_zscore, k_values=None):
    """
    Run complete clustering pipeline for specified K values.

    This is the main entry point for clustering analysis.

    Args:
        expression_matrix: Raw expression matrix
        expression_zscore: Z-scored expression matrix
        k_values: List of K values to compute. Default: [K_OPTIMAL, K_GRANULAR]

    Returns:
        results: Dict with keys:
            - 'expression_pca': PCA-transformed data
            - 'pca': PCA object
            - k (int): Dict with 'labels', 'silhouette', 'cluster_means' for each K
    """
    if k_values is None:
        k_values = [config.K_OPTIMAL, config.K_GRANULAR]

    # PCA dimensionality reduction
    expression_pca, pca = perform_pca(expression_matrix)

    results = {
        'expression_pca': expression_pca,
        'pca': pca
    }

    # Cluster for each K
    for k in k_values:
        labels, kmeans = cluster_kmeans(expression_pca, k)
        silhouette = silhouette_score(expression_pca, labels)
        cluster_means = compute_cluster_means(expression_zscore, labels)

        # Also compute raw cluster means for magnitude visualization
        cluster_means_raw = np.zeros((k, expression_matrix.shape[1]))
        for c in range(k):
            cluster_means_raw[c] = expression_matrix[labels == c].mean(axis=0)

        results[k] = {
            'labels': labels,
            'silhouette': silhouette,
            'cluster_means': cluster_means,
            'cluster_means_raw': cluster_means_raw,
            'kmeans': kmeans
        }

        print(f"K={k}: silhouette = {silhouette:.4f}")

    return results


def compute_gene_significance(expression_matrix, labels, genes, alpha=0.05):
    """
    Compute statistical significance for each gene in each cluster.

    For each gene × cluster combination, tests whether expression in that
    cluster is significantly different from expression in other clusters
    using Wilcoxon rank-sum test with FDR correction.

    Args:
        expression_matrix: Raw expression matrix (n_regions x n_genes)
        labels: Cluster labels for each region (0-indexed)
        genes: List of gene names
        alpha: Significance threshold after FDR correction (default: 0.05)

    Returns:
        significance_matrix: Dict with keys:
            - 'p_values': Matrix (n_clusters x n_genes) of raw p-values
            - 'p_adjusted': Matrix (n_clusters x n_genes) of FDR-adjusted p-values
            - 'direction': Matrix (n_clusters x n_genes) of direction (+1 up, -1 down)
            - 'significant': Boolean matrix (n_clusters x n_genes)
            - 'stars': Matrix of star strings ('*', '**', '***', or '')
    """
    n_clusters = len(np.unique(labels))
    n_genes = len(genes)

    p_values = np.zeros((n_clusters, n_genes))
    directions = np.zeros((n_clusters, n_genes))

    for c in range(n_clusters):
        in_cluster = labels == c
        out_cluster = ~in_cluster

        for g in range(n_genes):
            expr_in = expression_matrix[in_cluster, g]
            expr_out = expression_matrix[out_cluster, g]

            # Wilcoxon rank-sum test (Mann-Whitney U)
            try:
                stat, p = mannwhitneyu(expr_in, expr_out, alternative='two-sided')
                p_values[c, g] = p
            except ValueError:
                # If all values are identical
                p_values[c, g] = 1.0

            # Direction: +1 if cluster mean > others, -1 if lower
            directions[c, g] = 1 if expr_in.mean() > expr_out.mean() else -1

    # FDR correction across all tests
    p_flat = p_values.flatten()
    reject, p_adj_flat, _, _ = multipletests(p_flat, alpha=alpha, method='fdr_bh')
    p_adjusted = p_adj_flat.reshape((n_clusters, n_genes))
    significant = p_adjusted < alpha

    # Create star matrix (* p<0.05, ** p<0.01, *** p<0.001)
    stars = np.empty((n_clusters, n_genes), dtype=object)
    for c in range(n_clusters):
        for g in range(n_genes):
            p = p_adjusted[c, g]
            if p < 0.001:
                stars[c, g] = '***'
            elif p < 0.01:
                stars[c, g] = '**'
            elif p < 0.05:
                stars[c, g] = '*'
            else:
                stars[c, g] = ''

    print(f"Significance testing: {np.sum(significant)} of {n_clusters * n_genes} gene-cluster pairs significant (FDR < {alpha})")

    return {
        'p_values': p_values,
        'p_adjusted': p_adjusted,
        'direction': directions,
        'significant': significant,
        'stars': stars,
        'genes': genes
    }
