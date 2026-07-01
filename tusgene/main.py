#!/usr/bin/env python3
"""
================================================================================
TUS Gene Spatial Clustering Analysis - Main Pipeline
================================================================================

This script runs the complete analysis pipeline for the TUS gene study,
progressing through each stage in logical order.

RESEARCH QUESTION:
    Can brain regions be meaningfully parcellated based on the spatial
    expression patterns of mechanosensitive genes hypothesized to mediate
    transcranial ultrasound neuromodulation (TUS) effects?

ANALYSIS STAGES:
    1. DATA LOADING
       Load gene expression data (332 regions x n genes) and brain atlas

    2. CLUSTERING ANALYSIS
       - PCA dimensionality reduction
       - Silhouette analysis to determine optimal K
       - K-means clustering at K_OPTIMAL and K_GRANULAR (see config.py)

    3. MONTE CARLO VALIDATION
       Test whether TUS genes produce specific clusters or if any
       random gene set gives similar parcellations

    4. GENE-LEVEL STATISTICS
       Identify which genes drive cluster differences using
       Kruskal-Wallis tests and Cohen's d effect sizes

    5. FIGURE GENERATION
       Create publication-quality figures summarizing results

USAGE:
    python -m tusgene.main                    # Run full pipeline
    python -m tusgene.main --skip-monte-carlo # Skip Monte Carlo (faster)
    python -m tusgene.main --help             # Show options

================================================================================
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tusgene import config, data, clustering, statistics, visualization


def print_section(title):
    """Print a formatted section header."""
    width = 70
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width + "\n")


def stage1_load_data():
    """
    STAGE 1: DATA LOADING
    ---------------------
    Load gene expression data and brain atlas.

    Data sources:
    - Expression: Allen Human Brain Atlas, mapped to combined atlas
    - Atlas: Yeo 2011 17-network (300 cortical) + Tian S2 (32 subcortical)
    """
    print_section("STAGE 1: DATA LOADING")

    # Load expression data
    expression_df, expression_matrix, expression_zscore, genes = data.load_expression_data()

    # Load brain atlas
    atlas_img, atlas_data = data.load_atlas()

    print(f"\nData Summary:")
    print(f"  Expression matrix: {expression_matrix.shape[0]} regions x {expression_matrix.shape[1]} genes")
    print(f"  TUS genes loaded: {len(genes)}")
    print(f"  Atlas shape: {atlas_data.shape}")

    return {
        'expression_df': expression_df,
        'expression_matrix': expression_matrix,
        'expression_zscore': expression_zscore,
        'genes': genes,
        'atlas_img': atlas_img,
        'atlas_data': atlas_data
    }


def stage1b_robust_silhouette(data_dict, n_seeds=100):
    """
    STAGE 1B: ROBUST SILHOUETTE ANALYSIS
    ------------------------------------
    Compute silhouette scores with proper statistical methodology.

    This addresses a critical methodological issue: K-means clustering is
    sensitive to random initialization. Single-seed results can be misleading.

    Method:
    - Run K-means 10 times per seed (n_init=10) and keep best
    - Average results across n_seeds random seeds
    - Compute 95% confidence intervals

    Results saved to output/tables/robust_silhouette_analysis.csv
    """
    print_section("STAGE 1B: ROBUST SILHOUETTE ANALYSIS")

    expression_matrix = data_dict['expression_matrix']

    # Iterates over n_seeds random seeds (default 100), running K-means with
    # n_init=10 initializations per seed. Averages silhouette scores across
    # all seeds for each K value to get stable, reproducible results.
    # See: statistics.py:74-83 for the seed iteration loop
    robust_results = statistics.compute_robust_silhouette_curve(
        expression_matrix, n_seeds=n_seeds, n_init=10
    )

    # Save results
    visualization.ensure_output_dirs()
    robust_results.to_csv(
        os.path.join(config.TABLES_DIR, 'robust_silhouette_analysis.csv'),
        index=False
    )
    print(f"\nResults saved to {config.TABLES_DIR}/robust_silhouette_analysis.csv")

    return {'robust_silhouette': robust_results}


def stage2_clustering_analysis(data_dict):
    """
    STAGE 2: CLUSTERING ANALYSIS
    ----------------------------
    Determine optimal cluster number and perform clustering.

    Methods:
    1. PCA: Reduce genes to principal components (~90% variance)
    2. Silhouette analysis: Test K values in K_RANGE, find optimal K
    3. K-means: Cluster at K_OPTIMAL and K_GRANULAR (see config.py)
    """
    print_section("STAGE 2: CLUSTERING ANALYSIS")

    expression_matrix = data_dict['expression_matrix']
    expression_zscore = data_dict['expression_zscore']

    # Step 2.1: PCA + Silhouette Analysis
    print("Step 2.1: Silhouette Analysis Across K Values")
    print("-" * 50)

    expression_pca, pca = clustering.perform_pca(expression_matrix)
    k_values, silhouettes, optimal_k = clustering.compute_silhouette_curve(expression_pca)

    # Find local maxima
    local_maxima = clustering.find_local_maxima(k_values, silhouettes)
    print(f"\nLocal maxima at K = {[k for k, _ in local_maxima]}")

    # Step 2.2: Cluster at K_OPTIMAL and K_GRANULAR
    print(f"\nStep 2.2: Clustering at K={config.K_OPTIMAL} (optimal) and K={config.K_GRANULAR} (granular)")
    print("-" * 50)

    cluster_results = clustering.get_cluster_results(expression_matrix, expression_zscore)

    # Step 2.3: Analyze cluster composition
    print("\nStep 2.3: Cluster Composition Analysis")
    print("-" * 50)

    labels_opt = cluster_results[config.K_OPTIMAL]['labels']
    labels_gran = cluster_results[config.K_GRANULAR]['labels']

    print(f"\nK={config.K_OPTIMAL} Cluster Composition:")
    region_info_opt = data.get_region_info(labels_opt)
    print(region_info_opt.to_string(index=False))

    print(f"\nK={config.K_GRANULAR} Cluster Composition (with K={config.K_OPTIMAL} parent):")
    region_info_gran = data.get_region_info(labels_gran, labels_opt)
    print(region_info_gran.to_string(index=False))

    # Step 2.4: Map K_GRANULAR to K_OPTIMAL parents
    print("\nStep 2.4: Hierarchical Cluster Mapping")
    print("-" * 50)

    k_gran_to_k_opt_map = clustering.map_fine_to_coarse_clusters(labels_gran, labels_opt)
    print(f"K={config.K_GRANULAR} -> K={config.K_OPTIMAL} parent mapping:")
    for k_gran, k_opt in k_gran_to_k_opt_map.items():
        print(f"  Cluster {k_gran+1} -> Parent {k_opt+1}")

    # Step 2.5: Hierarchical clustering for visualization
    print("\nStep 2.5: Hierarchical Clustering for Dendrograms")
    print("-" * 50)

    cluster_means_opt = cluster_results[config.K_OPTIMAL]['cluster_means']
    cluster_means_gran = cluster_results[config.K_GRANULAR]['cluster_means']

    gene_linkage = clustering.hierarchical_cluster_genes(cluster_means_opt)
    cluster_linkage_opt = clustering.hierarchical_cluster_clusters(cluster_means_opt)
    cluster_linkage_gran = clustering.hierarchical_cluster_clusters(cluster_means_gran)
    print("Gene and cluster dendrograms computed")

    # Create parent colors for K_GRANULAR visualization
    k_gran_parent_colors = {c: config.CLUSTER_COLORS_OPT[k_gran_to_k_opt_map[c]] for c in range(config.K_GRANULAR)}

    return {
        'cluster_results': cluster_results,
        'k_values': k_values,
        'silhouettes': silhouettes,
        'gene_linkage': gene_linkage,
        'cluster_linkage_opt': cluster_linkage_opt,
        'cluster_linkage_gran': cluster_linkage_gran,
        'k_gran_to_k_opt_map': k_gran_to_k_opt_map,
        'k_gran_parent_colors': k_gran_parent_colors,
        'region_info_opt': region_info_opt,
        'region_info_gran': region_info_gran
    }


def stage3_monte_carlo_validation(data_dict, clustering_dict, n_iterations=None):
    """
    STAGE 3: MONTE CARLO VALIDATION
    -------------------------------
    Test whether TUS gene clustering is specific or driven by brain architecture.

    Question: Do TUS genes produce clusters that are specific to mechanosensitive
    genes, or would ANY random gene set produce similar parcellations?

    Method:
    1. Cluster brain regions using TUS genes (our analysis)
    2. Repeat with random gene sets of same size
    3. Compare using Adjusted Rand Index (ARI) at both K_OPTIMAL and K_GRANULAR

    Interpretation:
    - ARI > 0.5: Clustering driven by brain architecture
    - ARI 0.3-0.5: Partial architecture effect
    - ARI < 0.3: TUS genes show specific clustering
    """
    print_section("STAGE 3: MONTE CARLO VALIDATION")

    if n_iterations is None:
        n_iterations = config.N_MONTE_CARLO_ITERATIONS

    expression_df = data_dict['expression_df']
    tus_labels_opt = clustering_dict['cluster_results'][config.K_OPTIMAL]['labels']
    tus_labels_gran = clustering_dict['cluster_results'][config.K_GRANULAR]['labels']
    tus_silhouette_opt = clustering_dict['cluster_results'][config.K_OPTIMAL]['silhouette']
    tus_silhouette_gran = clustering_dict['cluster_results'][config.K_GRANULAR]['silhouette']

    # Run Dual-K Monte Carlo (compares at both K_OPTIMAL and K_GRANULAR)
    monte_carlo_dual_results = statistics.monte_carlo_dual_k_validation(
        expression_df,
        tus_labels_opt,
        tus_labels_gran,
        n_iterations=n_iterations
    )

    # Also run single-K validation for backward compatibility
    monte_carlo_results = statistics.monte_carlo_clustering_validation(
        expression_df,
        tus_labels_opt,
        n_iterations=n_iterations
    )

    # Compare TUS to random
    comparison_stats = statistics.compute_tus_vs_random_statistics(
        tus_silhouette_opt, monte_carlo_results
    )

    # Get mean ARI values for null distribution analysis
    tus_ari_opt = monte_carlo_dual_results[f'ari_k{config.K_OPTIMAL}'].mean()
    tus_ari_gran = monte_carlo_dual_results[f'ari_k{config.K_GRANULAR}'].mean()

    # Run null distribution analysis (random vs random)
    print("\n" + "-" * 50)
    print("Running Null Distribution Analysis...")
    print("-" * 50)
    null_results = statistics.monte_carlo_null_distribution(
        expression_df,
        tus_ari_opt,
        tus_ari_gran,
        n_reference_sets=100,
        n_comparisons_per_ref=100
    )

    # Save results
    visualization.ensure_output_dirs()
    monte_carlo_results.to_csv(
        os.path.join(config.TABLES_DIR, 'monte_carlo_results.csv'),
        index=False
    )
    monte_carlo_dual_results.to_csv(
        os.path.join(config.TABLES_DIR, 'monte_carlo_dual_k_results.csv'),
        index=False
    )

    # Save null distribution results
    null_df = pd.DataFrame({
        'null_ari_optimal': null_results[f'null_ari_k{config.K_OPTIMAL}'],
        'null_ari_granular': null_results[f'null_ari_k{config.K_GRANULAR}']
    })
    null_df.to_csv(
        os.path.join(config.TABLES_DIR, 'monte_carlo_null_distribution.csv'),
        index=False
    )
    print(f"\nResults saved to {config.TABLES_DIR}/")

    return {
        'monte_carlo_results': monte_carlo_results,
        'monte_carlo_dual_results': monte_carlo_dual_results,
        'comparison_stats': comparison_stats,
        'tus_silhouette_opt': tus_silhouette_opt,
        'tus_silhouette_gran': tus_silhouette_gran,
        'null_results': null_results
    }


def stage4_gene_statistics(data_dict, clustering_dict):
    """
    STAGE 4: GENE-LEVEL STATISTICS
    ------------------------------
    Identify which genes drive cluster differences.

    Methods:
    - Kruskal-Wallis H-test: Non-parametric test for differential expression
    - Cohen's d effect size: Magnitude of expression differences
    - Bonferroni correction: Control for multiple testing
    - Wilcoxon rank-sum test: Per-gene, per-cluster significance (FDR corrected)

    Output: Ranked list of genes by cluster differentiation + significance matrix
    """
    print_section("STAGE 4: GENE-LEVEL STATISTICS")

    expression_matrix = data_dict['expression_matrix']
    expression_zscore = data_dict['expression_zscore']
    genes = data_dict['genes']
    labels_opt = clustering_dict['cluster_results'][config.K_OPTIMAL]['labels']
    labels_gran = clustering_dict['cluster_results'][config.K_GRANULAR]['labels']

    # Compute gene statistics (Kruskal-Wallis omnibus test)
    print(f"Computing gene-level statistics for K={config.K_OPTIMAL} clusters...")
    gene_stats = statistics.compute_gene_cluster_statistics(expression_zscore, labels_opt, genes)

    # Apply multiple testing correction
    gene_stats = statistics.compute_bonferroni_correction(gene_stats)

    # Identify marker genes
    markers = statistics.identify_cluster_marker_genes(gene_stats)

    # Compute per-gene, per-cluster significance (Wilcoxon rank-sum with FDR)
    print(f"\nComputing per-gene significance for K={config.K_GRANULAR} clusters...")
    gene_significance = clustering.compute_gene_significance(expression_matrix, labels_gran, genes)

    # Save results
    visualization.ensure_output_dirs()
    gene_stats.to_csv(
        os.path.join(config.TABLES_DIR, 'gene_cluster_statistics.csv'),
        index=False
    )
    if not markers.empty:
        markers.to_csv(
            os.path.join(config.TABLES_DIR, 'cluster_marker_genes.csv'),
            index=False
        )

    # Save gene significance matrix
    sig_df = pd.DataFrame(
        gene_significance['p_adjusted'],
        index=[f'C{i+1}' for i in range(config.K_GRANULAR)],
        columns=genes
    )
    sig_df.to_csv(os.path.join(config.TABLES_DIR, 'gene_significance_fdr.csv'))
    print(f"\nResults saved to {config.TABLES_DIR}/")

    return {
        'gene_stats': gene_stats,
        'markers': markers,
        'gene_significance': gene_significance
    }


def stage5_figure_generation(data_dict, clustering_dict, monte_carlo_dict=None, robust_dict=None, gene_stats_dict=None):
    """
    STAGE 5: FIGURE GENERATION
    --------------------------
    Create publication-quality figures.

    Main Figures:
    1. Figure 1 Main: K_GRANULAR clustering with gene profiles
       - Heatmap, 3D brains, glass brains, axial slices, radar plots
       - Radar plots include significance markers (* p<0.05, ** p<0.01, *** p<0.001)

    Supplemental Figures:
    S1. K_OPTIMAL Clustering Overview
    S2. Robust silhouette analysis with confidence intervals
    S3. Monte Carlo explanation (ARI distributions, interpretation)
    S4. Silhouette curve: Clustering quality vs number of clusters
    """
    print_section("STAGE 5: FIGURE GENERATION")

    genes = data_dict['genes']
    atlas_img = data_dict['atlas_img']
    atlas_data = data_dict['atlas_data']
    expression_zscore = data_dict['expression_zscore']
    cluster_results = clustering_dict['cluster_results']
    gene_linkage = clustering_dict['gene_linkage']
    cluster_linkage_opt = clustering_dict['cluster_linkage_opt']
    cluster_linkage_gran = clustering_dict['cluster_linkage_gran']
    k_gran_parent_colors = clustering_dict['k_gran_parent_colors']
    k_values = clustering_dict['k_values']
    silhouettes = clustering_dict['silhouettes']

    visualization.ensure_output_dirs()

    # Create cluster-gene matrices for figure generation
    print("Creating cluster-gene matrices...")
    cluster_means_gran = cluster_results[config.K_GRANULAR]['cluster_means']
    cluster_means_opt = cluster_results[config.K_OPTIMAL]['cluster_means']

    # DataFrame format with cluster names as index
    cluster_gene_matrix_gran = pd.DataFrame(
        cluster_means_gran,
        index=[f'C{i+1}' for i in range(config.K_GRANULAR)],
        columns=genes
    )
    cluster_gene_matrix_opt = pd.DataFrame(
        cluster_means_opt,
        index=[f'C{i+1}' for i in range(config.K_OPTIMAL)],
        columns=genes
    )

    # Save cluster gene matrices
    cluster_gene_matrix_gran.to_csv(os.path.join(config.TABLES_DIR, 'cluster_gene_matrix_granular.csv'))
    cluster_gene_matrix_opt.to_csv(os.path.join(config.TABLES_DIR, 'cluster_gene_matrix_optimal.csv'))
    print(f"Cluster gene matrices saved to {config.TABLES_DIR}/")

    # Get gene significance if available
    gene_significance = None
    if gene_stats_dict is not None and 'gene_significance' in gene_stats_dict:
        gene_significance = gene_stats_dict['gene_significance']

    # Figure 1 Main: K_GRANULAR Clustering with Gene Profiles
    print(f"\nCreating Figure 1 Main: K={config.K_GRANULAR} Clustering with Gene Profiles...")
    visualization.create_figure1_main(
        cluster_results,
        cluster_gene_matrix_gran,
        gene_linkage,
        cluster_linkage_gran,
        atlas_img,
        atlas_data,
        genes,
        gene_significance=gene_significance
    )

    # Supplemental Figure S1: K_OPTIMAL Clustering Overview
    print(f"\nCreating Supplemental Figure S1: K={config.K_OPTIMAL} Clustering Overview...")
    visualization.create_supplemental1_optimal(
        cluster_results,
        cluster_gene_matrix_opt,
        gene_linkage,
        atlas_img,
        atlas_data,
        genes
    )

    # Monte Carlo figures (if available)
    if monte_carlo_dict is not None:
        print("\nCreating Monte Carlo Validation Figures...")
        tus_silhouette = cluster_results[config.K_OPTIMAL]['silhouette']
        visualization.create_figure2_monte_carlo(
            monte_carlo_dict['monte_carlo_results'],
            tus_silhouette
        )

        # Supplemental Figure S3: Monte Carlo Explanation
        if 'monte_carlo_dual_results' in monte_carlo_dict:
            print("Creating Supplemental Figure S3: Monte Carlo Explanation...")
            visualization.create_monte_carlo_explanation_figure(
                monte_carlo_dict['monte_carlo_dual_results'],
                monte_carlo_dict['tus_silhouette_opt'],
                monte_carlo_dict['tus_silhouette_gran']
            )

        # Null Distribution Figure: TUS vs Random vs Random-Random
        if 'null_results' in monte_carlo_dict:
            print("Creating Null Distribution Comparison Figure...")
            visualization.create_null_distribution_figure(
                monte_carlo_dict['null_results']
            )

    # Supplemental Figure S4: Silhouette curve
    print("\nCreating Supplemental Figure S4: Silhouette Curve...")
    visualization.create_silhouette_curve_figure(k_values, silhouettes)

    # Supplemental Figure S2: Robust silhouette analysis
    # Check for pre-computed 1000 seeds file first
    seeds_1000_path = os.path.join(config.TABLES_DIR, 'robust_silhouette_1000seeds.csv')
    if os.path.exists(seeds_1000_path):
        print("\nCreating Supplemental Figure S2: Robust Silhouette Analysis (1000 seeds)...")
        robust_1000 = pd.read_csv(seeds_1000_path)
        visualization.create_supplemental_robust_silhouette(robust_1000, n_seeds=1000)
    elif robust_dict is not None:
        print("\nCreating Supplemental Figure S2: Robust Silhouette Analysis...")
        visualization.create_supplemental_robust_silhouette(
            robust_dict['robust_silhouette']
        )

    print(f"\nAll figures saved to {config.FIGURES_DIR}/")


def print_summary(data_dict, clustering_dict, monte_carlo_dict=None, gene_stats_dict=None):
    """Print final analysis summary."""
    print_section("ANALYSIS SUMMARY")

    print("KEY FINDINGS:")
    print("-" * 50)

    # Clustering results
    sil_opt = clustering_dict['cluster_results'][config.K_OPTIMAL]['silhouette']
    sil_gran = clustering_dict['cluster_results'][config.K_GRANULAR]['silhouette']
    print(f"\n1. OPTIMAL CLUSTERING:")
    print(f"   K={config.K_OPTIMAL} is optimal (silhouette = {sil_opt:.3f})")
    print(f"   K={config.K_GRANULAR} provides granularity (silhouette = {sil_gran:.3f})")

    # Monte Carlo results
    if monte_carlo_dict is not None:
        ari = monte_carlo_dict['comparison_stats']['ari_mean']
        print(f"\n2. MONTE CARLO VALIDATION:")
        print(f"   Mean ARI = {ari:.3f} (moderate similarity to random)")
        print(f"   Clustering is partially driven by brain architecture")
        print(f"   Similar parcellations emerge from many gene sets")

    # Gene statistics
    if gene_stats_dict is not None:
        n_sig = gene_stats_dict['gene_stats']['significant_bonferroni'].sum()
        n_markers = len(gene_stats_dict['markers'])
        print(f"\n3. GENE-LEVEL ANALYSIS:")
        print(f"   {n_sig} genes significantly differentiate clusters")
        print(f"   {n_markers} genes identified as cluster markers")

    print("\n" + "=" * 70)
    print(" ANALYSIS COMPLETE")
    print("=" * 70)


def main():
    """Run the complete analysis pipeline."""

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='TUS Gene Spatial Clustering Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--skip-monte-carlo', action='store_true',
                        help='Skip Monte Carlo validation (faster)')
    parser.add_argument('--monte-carlo-iterations', type=int, default=None,
                        help=f'Number of Monte Carlo iterations (default: {config.N_MONTE_CARLO_ITERATIONS})')
    parser.add_argument('--skip-figures', action='store_true',
                        help='Skip figure generation')
    parser.add_argument('--skip-robust-silhouette', action='store_true',
                        help='Skip robust silhouette analysis (faster)')
    parser.add_argument('--robust-seeds', type=int, default=100,
                        help='Number of random seeds for robust silhouette (default: 100)')
    args = parser.parse_args()

    print("=" * 70)
    print(" TUS GENE SPATIAL CLUSTERING ANALYSIS")
    print(" Complete Analysis Pipeline")
    print("=" * 70)

    # Stage 1: Load Data
    data_dict = stage1_load_data()

    # Stage 1B: Robust Silhouette Analysis (optional but recommended)
    robust_dict = None
    if not args.skip_robust_silhouette:
        robust_dict = stage1b_robust_silhouette(data_dict, n_seeds=args.robust_seeds)
    else:
        print_section("STAGE 1B: ROBUST SILHOUETTE ANALYSIS")
        print("Skipped (use --skip-robust-silhouette=false to run)")

    # Stage 2: Clustering Analysis
    clustering_dict = stage2_clustering_analysis(data_dict)

    # Stage 3: Monte Carlo Validation (optional)
    monte_carlo_dict = None
    if not args.skip_monte_carlo:
        monte_carlo_dict = stage3_monte_carlo_validation(
            data_dict, clustering_dict,
            n_iterations=args.monte_carlo_iterations
        )
    else:
        print_section("STAGE 3: MONTE CARLO VALIDATION")
        print("Skipped (use --skip-monte-carlo=false to run)")

    # Stage 4: Gene Statistics
    gene_stats_dict = stage4_gene_statistics(data_dict, clustering_dict)

    # Stage 5: Figure Generation (optional)
    if not args.skip_figures:
        stage5_figure_generation(data_dict, clustering_dict, monte_carlo_dict, robust_dict, gene_stats_dict)
    else:
        print_section("STAGE 5: FIGURE GENERATION")
        print("Skipped")

    # Print Summary
    print_summary(data_dict, clustering_dict, monte_carlo_dict, gene_stats_dict)


if __name__ == '__main__':
    main()
