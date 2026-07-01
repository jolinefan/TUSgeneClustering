#!/usr/bin/env python3
"""
Create Figure 1 Main: K_GRANULAR Clustering with Gene Expression Profiles
==========================================================================

Generates a combined figure with 5 rows showing K_GRANULAR cluster characterization:
  Row a: K_GRANULAR heatmap with hierarchical gene ordering
  Row b: 3D brain render (PyVista) - each cluster highlighted
  Row c: Glass brain projection (nilearn) - axial view
  Row d: Axial slice at cluster's peak location
  Row e: Bidirectional radar plot - all 30 genes (labeled a-z + Greek)

Output: output/figures/figure1_main.pdf

Usage:
    python scripts/create_mainFig_fast.py

Note:
    Requires expression data and atlas files.
    Run `python -m tusgene.main` first if you need cluster_gene_matrix files.
"""
import sys
import os

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tusgene import config, data, clustering, visualization


def main():
    """Generate Figure 1 Main."""
    print("=" * 60)
    print(f" Creating Figure 1 Main: K={config.K_GRANULAR} Clustering with Gene Profiles")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    expression_df, expression_matrix, expression_zscore, genes = data.load_expression_data()
    atlas_img, atlas_data = data.load_atlas()

    # Run clustering
    print("\nRunning clustering...")
    cluster_results = clustering.get_cluster_results(expression_matrix, expression_zscore)

    # Compute gene linkage from K_OPTIMAL cluster means (original method)
    cluster_means_opt = cluster_results[config.K_OPTIMAL]['cluster_means']
    gene_linkage = clustering.hierarchical_cluster_genes(cluster_means_opt)

    # Compute cluster linkage for K_GRANULAR
    cluster_means_gran = cluster_results[config.K_GRANULAR]['cluster_means']
    cluster_linkage_gran = clustering.hierarchical_cluster_clusters(cluster_means_gran)

    # Create cluster gene matrix
    cluster_gene_matrix_gran = pd.DataFrame(
        cluster_means_gran,
        index=[f'C{i+1}' for i in range(config.K_GRANULAR)],
        columns=genes
    )

    # Save cluster gene matrix for reference
    visualization.ensure_output_dirs()
    cluster_gene_matrix_gran.to_csv(os.path.join(config.TABLES_DIR, 'cluster_gene_matrix_granular.csv'))

    # Generate figure
    print("\nGenerating Figure 1 Main...")
    visualization.create_figure1_main(
        cluster_results=cluster_results,
        cluster_gene_matrix=cluster_gene_matrix_gran,
        gene_linkage=gene_linkage,
        cluster_linkage_9=cluster_linkage_gran,
        atlas_img=atlas_img,
        atlas_data=atlas_data,
        genes=genes
    )

    print("\n" + "=" * 60)
    print(" Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
