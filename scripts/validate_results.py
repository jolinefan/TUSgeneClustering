#!/usr/bin/env python3
"""
Validation Script for TUS Gene Analysis
=======================================

Performs numerical spot-checks to verify analysis correctness.
Run this after the main pipeline to catch potential bugs.

Usage:
    python scripts/validate_results.py

Exit codes:
    0 - All checks passed
    1 - One or more checks failed
"""
import sys
import os

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tusgene import config, data, clustering

# Track failures
FAILURES = []


def check(condition, name, details=""):
    """Record pass/fail for a check."""
    if condition:
        print(f"  [PASS] {name}")
        return True
    else:
        msg = f"{name}: {details}" if details else name
        FAILURES.append(msg)
        print(f"  [FAIL] {name}")
        if details:
            print(f"         {details}")
        return False


def validate_data_loading():
    """Verify data loads correctly with expected dimensions."""
    print("\n" + "=" * 60)
    print("1. DATA LOADING VALIDATION")
    print("=" * 60)

    expression_df, expression_matrix, expression_zscore, genes = data.load_expression_data()
    atlas_img, atlas_data = data.load_atlas()

    n_genes = len(config.TUS_GENES)

    # Check dimensions
    check(
        expression_matrix.shape == (332, n_genes),
        "Expression matrix shape",
        f"Expected (332, {n_genes}), got {expression_matrix.shape}"
    )

    check(
        len(genes) == n_genes,
        "Gene count",
        f"Expected {n_genes}, got {len(genes)}"
    )

    check(
        atlas_data.shape == (91, 109, 91),
        "Atlas dimensions",
        f"Expected (91, 109, 91), got {atlas_data.shape}"
    )

    # Check z-score properties
    col_means = np.mean(expression_zscore, axis=0)
    col_stds = np.std(expression_zscore, axis=0)

    check(
        np.allclose(col_means, 0, atol=1e-10),
        "Z-score means ~ 0",
        f"Max deviation: {np.max(np.abs(col_means)):.2e}"
    )

    check(
        np.allclose(col_stds, 1, atol=1e-10),
        "Z-score stds ~ 1",
        f"Max deviation: {np.max(np.abs(col_stds - 1)):.2e}"
    )

    # Check atlas parcel count
    unique_parcels = np.unique(atlas_data)
    n_parcels = len(unique_parcels[unique_parcels > 0])

    check(
        n_parcels == 332,
        "Atlas parcel count",
        f"Expected 332, got {n_parcels}"
    )

    return expression_matrix, expression_zscore, genes


def validate_pca(expression_matrix):
    """Verify PCA results match expected values."""
    print("\n" + "=" * 60)
    print("2. PCA VALIDATION")
    print("=" * 60)

    expression_pca, pca = clustering.perform_pca(expression_matrix)

    # Check output dimensions
    check(
        expression_pca.shape == (332, config.N_PCA_COMPONENTS),
        "PCA output shape",
        f"Expected (332, {config.N_PCA_COMPONENTS}), got {expression_pca.shape}"
    )

    # Check explained variance
    total_var = sum(pca.explained_variance_ratio_)

    check(
        0.89 < total_var < 0.92,
        "PCA explained variance ~ 90.6%",
        f"Got {total_var * 100:.1f}%"
    )

    # Verify with independent computation (no StandardScaler - matches pipeline)
    pca_verify = PCA(n_components=10, random_state=config.RANDOM_STATE)
    pca_verify.fit(expression_matrix)

    check(
        np.allclose(pca.explained_variance_ratio_, pca_verify.explained_variance_ratio_),
        "PCA variance matches independent computation"
    )

    return expression_pca


def validate_clustering(expression_pca):
    """Verify clustering results."""
    print("\n" + "=" * 60)
    print("3. CLUSTERING VALIDATION")
    print("=" * 60)

    k_opt = config.K_OPTIMAL
    k_gran = config.K_GRANULAR

    # Run clustering
    kmeans_opt = KMeans(n_clusters=k_opt, random_state=config.RANDOM_STATE, n_init=10)
    labels_opt = kmeans_opt.fit_predict(expression_pca)

    kmeans_gran = KMeans(n_clusters=k_gran, random_state=config.RANDOM_STATE, n_init=10)
    labels_gran = kmeans_gran.fit_predict(expression_pca)

    # Check label validity
    check(
        set(labels_opt) == set(range(k_opt)),
        f"K={k_opt} labels in {{0..{k_opt-1}}}",
        f"Got {set(labels_opt)}"
    )

    check(
        set(labels_gran) == set(range(k_gran)),
        f"K={k_gran} labels in {{0..{k_gran-1}}}",
        f"Got {set(labels_gran)}"
    )

    # Check all regions assigned
    check(
        len(labels_opt) == 332,
        f"K={k_opt}: All 332 regions assigned",
        f"Got {len(labels_opt)}"
    )

    check(
        len(labels_gran) == 332,
        f"K={k_gran}: All 332 regions assigned",
        f"Got {len(labels_gran)}"
    )

    # Verify silhouette scores are reasonable (not checking exact values since K may vary)
    sil_opt = silhouette_score(expression_pca, labels_opt)
    sil_gran = silhouette_score(expression_pca, labels_gran)

    check(
        0.15 < sil_opt < 0.35,
        f"K={k_opt} silhouette in reasonable range",
        f"Got {sil_opt:.4f}"
    )

    check(
        0.15 < sil_gran < 0.30,
        f"K={k_gran} silhouette in reasonable range",
        f"Got {sil_gran:.4f}"
    )

    # Check cluster sizes are reasonable (no empty clusters)
    for k, labels in [(k_opt, labels_opt), (k_gran, labels_gran)]:
        sizes = [np.sum(labels == i) for i in range(k)]
        min_size = min(sizes)
        check(
            min_size >= 5,
            f"K={k}: No tiny clusters",
            f"Smallest cluster has {min_size} regions"
        )

    return labels_opt, labels_gran


def validate_saved_outputs(expression_zscore, genes, labels_gran):
    """Verify saved CSV files match computed values."""
    print("\n" + "=" * 60)
    print("4. SAVED OUTPUT VALIDATION")
    print("=" * 60)

    k_gran = config.K_GRANULAR

    # Load saved cluster gene matrix (try multiple possible names)
    possible_paths = [
        os.path.join(config.TABLES_DIR, 'cluster_gene_matrix_granular.csv'),
        os.path.join(config.TABLES_DIR, f'cluster_gene_matrix_k{k_gran}.csv'),
    ]

    matrix_path = None
    for path in possible_paths:
        if os.path.exists(path):
            matrix_path = path
            break

    if matrix_path is None:
        print(f"  [SKIP] Cluster gene matrix not found - run main pipeline first")
        return

    saved_matrix = pd.read_csv(matrix_path, index_col=0)

    # Manually compute cluster means for spot check
    # Pick cluster 0 (C1) and first gene
    test_gene_idx = 0
    test_gene = genes[test_gene_idx]
    test_cluster = 0

    mask = labels_gran == test_cluster
    manual_mean = np.mean(expression_zscore[mask, test_gene_idx])
    saved_mean = saved_matrix.loc[f'C{test_cluster + 1}', test_gene]

    check(
        np.isclose(manual_mean, saved_mean, atol=0.01),
        f"Spot check: C1/{test_gene} mean matches",
        f"Manual: {manual_mean:.4f}, Saved: {saved_mean:.4f}"
    )

    # Spot check another value (middle cluster, middle gene)
    n_genes = len(genes)
    test_gene_idx = n_genes // 2
    test_gene = genes[test_gene_idx]
    test_cluster = min(k_gran // 2, k_gran - 1)

    mask = labels_gran == test_cluster
    manual_mean = np.mean(expression_zscore[mask, test_gene_idx])
    saved_mean = saved_matrix.loc[f'C{test_cluster + 1}', test_gene]

    check(
        np.isclose(manual_mean, saved_mean, atol=0.01),
        f"Spot check: C{test_cluster+1}/{test_gene} mean matches",
        f"Manual: {manual_mean:.4f}, Saved: {saved_mean:.4f}"
    )

    # Verify matrix dimensions
    check(
        saved_matrix.shape == (k_gran, n_genes),
        "Cluster gene matrix shape",
        f"Expected ({k_gran}, {n_genes}), got {saved_matrix.shape}"
    )


def validate_reproducibility(expression_matrix):
    """Verify results are deterministic with fixed seed."""
    print("\n" + "=" * 60)
    print("5. REPRODUCIBILITY VALIDATION")
    print("=" * 60)

    # Run clustering twice
    pca1, _ = clustering.perform_pca(expression_matrix)
    pca2, _ = clustering.perform_pca(expression_matrix)

    check(
        np.allclose(pca1, pca2),
        "PCA results identical across runs"
    )

    k_gran = config.K_GRANULAR
    kmeans1 = KMeans(n_clusters=k_gran, random_state=config.RANDOM_STATE, n_init=10)
    labels1 = kmeans1.fit_predict(pca1)

    kmeans2 = KMeans(n_clusters=k_gran, random_state=config.RANDOM_STATE, n_init=10)
    labels2 = kmeans2.fit_predict(pca2)

    check(
        np.array_equal(labels1, labels2),
        "K-means labels identical across runs"
    )


def validate_config_consistency():
    """Check config values are internally consistent."""
    print("\n" + "=" * 60)
    print("6. CONFIGURATION VALIDATION")
    print("=" * 60)

    n_genes = len(config.TUS_GENES)

    # Gene list consistency
    check(
        n_genes >= 30,
        f"TUS_GENES has at least 30 genes",
        f"Got {n_genes}"
    )

    # All genes in families
    genes_in_families = set()
    for family_genes in config.GENE_FAMILIES.values():
        genes_in_families.update(family_genes)

    check(
        genes_in_families == set(config.TUS_GENES),
        "GENE_FAMILIES covers all TUS_GENES"
    )

    # Color list lengths match K values
    check(
        len(config.CLUSTER_COLORS_OPT) == config.K_OPTIMAL,
        f"CLUSTER_COLORS_OPT has {config.K_OPTIMAL} colors",
        f"Got {len(config.CLUSTER_COLORS_OPT)}"
    )

    check(
        len(config.CLUSTER_COLORS_GRAN) == config.K_GRANULAR,
        f"CLUSTER_COLORS_GRAN has {config.K_GRANULAR} colors",
        f"Got {len(config.CLUSTER_COLORS_GRAN)}"
    )

    # K values make sense
    check(
        config.K_OPTIMAL < config.K_GRANULAR,
        "K_OPTIMAL < K_GRANULAR",
        f"{config.K_OPTIMAL} < {config.K_GRANULAR}"
    )


def main():
    """Run all validation checks."""
    print("=" * 60)
    print(" TUS GENE ANALYSIS VALIDATION")
    print("=" * 60)

    # Run validations
    expression_matrix, expression_zscore, genes = validate_data_loading()
    expression_pca = validate_pca(expression_matrix)
    labels_opt, labels_gran = validate_clustering(expression_pca)
    validate_saved_outputs(expression_zscore, genes, labels_gran)
    validate_reproducibility(expression_matrix)
    validate_config_consistency()

    # Summary
    print("\n" + "=" * 60)
    print(" VALIDATION SUMMARY")
    print("=" * 60)

    if FAILURES:
        print(f"\n  {len(FAILURES)} CHECK(S) FAILED:\n")
        for f in FAILURES:
            print(f"    - {f}")
        print()
        sys.exit(1)
    else:
        print("\n  ALL CHECKS PASSED\n")
        sys.exit(0)


if __name__ == '__main__':
    main()
