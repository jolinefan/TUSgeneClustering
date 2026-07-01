"""
Statistical Analysis Functions
==============================

Monte Carlo validation and gene-level statistics for TUS clustering analysis.

Key Analyses:
    1. Monte Carlo Validation: Tests whether TUS genes cluster differently than
       random gene sets, addressing whether observed clustering is TUS-specific
       or driven by general brain architecture.

    2. Gene-Level Statistics: Kruskal-Wallis tests and Cohen's d effect sizes
       to identify which genes drive cluster differences.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from tqdm import tqdm
from . import config


# =============================================================================
# ROBUST SILHOUETTE ANALYSIS
# =============================================================================

def compute_robust_silhouette_curve(expression_matrix, k_range=None, n_seeds=100, n_init=10):
    """
    Compute silhouette scores with proper statistical methodology.

    Standard K-means is sensitive to random initialization. This function
    addresses this by:
    1. Using n_init=10 (run K-means 10 times, keep best)
    2. Averaging across multiple random seeds
    3. Computing confidence intervals

    This provides publication-quality, reproducible results.

    Args:
        expression_matrix: Expression matrix (n_regions x n_genes)
        k_range: Range of K values to test. Default: config.K_RANGE
        n_seeds: Number of random seeds to average over (default: 100)
        n_init: Number of K-means initializations per run (default: 10)

    Returns:
        results_df: DataFrame with columns:
            - k: Number of clusters
            - mean_silhouette: Mean silhouette across seeds
            - std_silhouette: Standard deviation
            - ci95: 95% confidence interval half-width
            - is_local_max: Boolean indicating local maximum
    """
    if k_range is None:
        k_range = config.K_RANGE

    # PCA dimensionality reduction
    pca = PCA(n_components=config.N_PCA_COMPONENTS)
    expression_pca = pca.fit_transform(expression_matrix)

    print(f"Robust Silhouette Analysis:")
    print(f"  K range: {min(k_range)}-{max(k_range)}")
    print(f"  Random seeds: {n_seeds}")
    print(f"  K-means initializations per seed: {n_init}")

    results = []

    for k in k_range:
        scores = []
        for seed in range(n_seeds):
            kmeans = KMeans(n_clusters=k, random_state=seed, n_init=n_init, init='k-means++')
            labels = kmeans.fit_predict(expression_pca)
            scores.append(silhouette_score(expression_pca, labels))

        mean_sil = np.mean(scores)
        std_sil = np.std(scores)
        # 95% CI using standard error of mean: z * (σ / √n)
        # z=1.96 is the two-tailed critical value from standard normal distribution
        # (captures 95% of probability mass, 2.5% in each tail)
        ci95 = 1.96 * std_sil / np.sqrt(n_seeds)

        results.append({
            'k': k,
            'mean_silhouette': mean_sil,
            'std_silhouette': std_sil,
            'ci95': ci95
        })
        print(f"  K={k:2d}: {mean_sil:.4f} ± {ci95:.4f}")

    results_df = pd.DataFrame(results)

    # Identify local maxima
    sil_values = results_df['mean_silhouette'].values
    is_local_max = np.zeros(len(sil_values), dtype=bool)
    for i in range(1, len(sil_values) - 1):
        if sil_values[i] > sil_values[i-1] and sil_values[i] > sil_values[i+1]:
            is_local_max[i] = True
    results_df['is_local_max'] = is_local_max

    # Print local maxima
    local_max_df = results_df[results_df['is_local_max']]
    if not local_max_df.empty:
        print(f"\nLocal maxima at K = {local_max_df['k'].tolist()}")

    # Best K excluding K=2
    best_k3_plus = results_df[results_df['k'] >= 3].loc[
        results_df[results_df['k'] >= 3]['mean_silhouette'].idxmax()
    ]
    print(f"Best K (for K≥3): {int(best_k3_plus['k'])} (silhouette = {best_k3_plus['mean_silhouette']:.4f})")

    return results_df


# =============================================================================
# MONTE CARLO VALIDATION
# =============================================================================

def monte_carlo_clustering_validation(
    expression_df,
    tus_labels,
    n_iterations=None,
    n_genes=None,
    random_seed=None,
    show_progress=True
):
    """
    Monte Carlo validation of TUS gene clustering specificity.

    NOTE: This is the legacy single-K version. Use monte_carlo_dual_k_validation
    for comparing both K_OPTIMAL and K_GRANULAR simultaneously.
    """
    if n_iterations is None:
        n_iterations = config.N_MONTE_CARLO_ITERATIONS
    if n_genes is None:
        n_genes = len(config.TUS_GENES)
    if random_seed is None:
        random_seed = config.MONTE_CARLO_SEED

    all_genes = [col for col in expression_df.columns
                 if col != 'label' and col not in config.TUS_GENES]

    print(f"Monte Carlo Validation:")
    print(f"  Iterations: {n_iterations:,}")
    print(f"  Genes per set: {n_genes}")
    print(f"  Available non-TUS genes: {len(all_genes):,}")

    np.random.seed(random_seed)

    results = {
        'silhouette': [],
        'optimal_k': [],
        'ari': []
    }

    iterator = range(n_iterations)
    if show_progress:
        iterator = tqdm(iterator, desc="Monte Carlo")

    for _ in iterator:
        random_genes = np.random.choice(all_genes, size=n_genes, replace=False)
        expression_random = expression_df[random_genes].to_numpy()

        pca = PCA(n_components=config.N_PCA_COMPONENTS)
        expression_pca = pca.fit_transform(expression_random)

        best_score = -1
        best_k = config.K_OPTIMAL
        best_labels = None

        for k in config.K_RANGE:
            kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
            labels = kmeans.fit_predict(expression_pca)
            score = silhouette_score(expression_pca, labels)

            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels

        ari = adjusted_rand_score(tus_labels, best_labels)

        results['silhouette'].append(best_score)
        results['optimal_k'].append(best_k)
        results['ari'].append(ari)

    results_df = pd.DataFrame(results)

    print(f"\nResults Summary:")
    print(f"  Mean Silhouette: {results_df['silhouette'].mean():.4f} +/- {results_df['silhouette'].std():.4f}")
    print(f"  Mean Optimal K: {results_df['optimal_k'].mean():.1f}")
    print(f"  Mean ARI: {results_df['ari'].mean():.4f} +/- {results_df['ari'].std():.4f}")

    return results_df


def monte_carlo_dual_k_validation(
    expression_df,
    tus_labels_k_opt,
    tus_labels_k_granular,
    n_iterations=1000,
    n_genes=None,
    random_seed=None,
    show_progress=True
):
    """
    Monte Carlo validation comparing random gene sets to BOTH K_OPTIMAL and K_GRANULAR TUS clusters.

    SCIENTIFIC QUESTION:
    Do random gene sets produce the same brain parcellations as TUS genes?
    This tests whether TUS clustering reflects general brain architecture
    or is specific to mechanosensitive gene expression.

    METHOD:
    For each of N random gene sets:
    1. Perform PCA + K-means at FIXED K values (matching TUS configuration)
    2. Compute ARI between random clusters and TUS clusters at each K
    3. Track silhouette scores for clustering quality

    INTERPRETATION OF ARI:
    - ARI > 0.7: Strong similarity (brain architecture dominates)
    - ARI 0.4-0.7: Moderate similarity (partial architecture effect)
    - ARI < 0.4: Weak similarity (TUS genes are specific)
    - ARI ~ 0: No similarity beyond chance

    Args:
        expression_df: Full expression DataFrame
        tus_labels_k_opt: TUS cluster labels at K_OPTIMAL
        tus_labels_k_granular: TUS cluster labels at K_GRANULAR
        n_iterations: Number of Monte Carlo iterations (default: 1000)
        n_genes: Number of genes per random set (default: 30)
        random_seed: Random seed for reproducibility
        show_progress: Show progress bar

    Returns:
        results_df: DataFrame with columns:
            - silhouette_k{K_OPTIMAL}, silhouette_k{K_GRANULAR}: Clustering quality
            - ari_k{K_OPTIMAL}, ari_k{K_GRANULAR}: Similarity to TUS clusters
    """
    if n_genes is None:
        n_genes = len(config.TUS_GENES)
    if random_seed is None:
        random_seed = config.MONTE_CARLO_SEED

    k_opt = config.K_OPTIMAL
    k_gran = config.K_GRANULAR

    # Get all available genes (excluding TUS genes)
    all_genes = [col for col in expression_df.columns
                 if col != 'label' and col not in config.TUS_GENES]

    print(f"Monte Carlo Dual-K Validation:")
    print(f"  Iterations: {n_iterations:,}")
    print(f"  Genes per set: {n_genes}")
    print(f"  Available non-TUS genes: {len(all_genes):,}")
    print(f"  Comparing at K={k_opt} (optimal) and K={k_gran} (granular)")

    np.random.seed(random_seed)

    results = {
        f'silhouette_k{k_opt}': [],
        f'silhouette_k{k_gran}': [],
        f'ari_k{k_opt}': [],
        f'ari_k{k_gran}': []
    }

    iterator = range(n_iterations)
    if show_progress:
        iterator = tqdm(iterator, desc=f"Monte Carlo (K={k_opt} & K={k_gran})")

    for _ in iterator:
        # Sample random genes
        random_genes = np.random.choice(all_genes, size=n_genes, replace=False)
        expression_random = expression_df[random_genes].to_numpy()

        # PCA
        pca = PCA(n_components=config.N_PCA_COMPONENTS)
        expression_pca = pca.fit_transform(expression_random)

        # K-means at K_OPTIMAL
        kmeans_opt = KMeans(n_clusters=k_opt, random_state=0, n_init='auto')
        labels_opt = kmeans_opt.fit_predict(expression_pca)
        sil_opt = silhouette_score(expression_pca, labels_opt)
        ari_opt = adjusted_rand_score(tus_labels_k_opt, labels_opt)

        # K-means at K_GRANULAR
        kmeans_gran = KMeans(n_clusters=k_gran, random_state=0, n_init='auto')
        labels_gran = kmeans_gran.fit_predict(expression_pca)
        sil_gran = silhouette_score(expression_pca, labels_gran)
        ari_gran = adjusted_rand_score(tus_labels_k_granular, labels_gran)

        results[f'silhouette_k{k_opt}'].append(sil_opt)
        results[f'silhouette_k{k_gran}'].append(sil_gran)
        results[f'ari_k{k_opt}'].append(ari_opt)
        results[f'ari_k{k_gran}'].append(ari_gran)

    results_df = pd.DataFrame(results)

    # Print summary
    ari_opt_col = f'ari_k{k_opt}'
    ari_gran_col = f'ari_k{k_gran}'
    sil_opt_col = f'silhouette_k{k_opt}'
    sil_gran_col = f'silhouette_k{k_gran}'

    print(f"\nResults Summary:")
    print(f"  K={k_opt}: Mean ARI = {results_df[ari_opt_col].mean():.4f} +/- {results_df[ari_opt_col].std():.4f}")
    print(f"       Mean Silhouette = {results_df[sil_opt_col].mean():.4f}")
    print(f"  K={k_gran}: Mean ARI = {results_df[ari_gran_col].mean():.4f} +/- {results_df[ari_gran_col].std():.4f}")
    print(f"       Mean Silhouette = {results_df[sil_gran_col].mean():.4f}")

    # Interpretation
    ari_opt_mean = results_df[ari_opt_col].mean()
    ari_gran_mean = results_df[ari_gran_col].mean()

    print(f"\nInterpretation:")
    for k, ari in [(f'K={k_opt}', ari_opt_mean), (f'K={k_gran}', ari_gran_mean)]:
        if ari > 0.7:
            print(f"  {k}: HIGH similarity - clustering largely reflects brain architecture")
        elif ari > 0.4:
            print(f"  {k}: MODERATE similarity - partial brain architecture effect")
        else:
            print(f"  {k}: LOW similarity - TUS genes show specific clustering")

    return results_df


def compute_tus_vs_random_statistics(tus_silhouette, monte_carlo_results):
    """
    Compare TUS clustering quality to random gene sets.

    Args:
        tus_silhouette: Silhouette score from TUS gene clustering
        monte_carlo_results: DataFrame from monte_carlo_clustering_validation()

    Returns:
        stats_dict: Dictionary with comparison statistics
    """
    random_silhouettes = monte_carlo_results['silhouette'].values
    random_ari = monte_carlo_results['ari'].values

    # Silhouette percentile
    pct_lower = (random_silhouettes < tus_silhouette).mean() * 100

    # ARI interpretation
    pct_high_ari = (random_ari > 0.7).mean() * 100
    pct_moderate_ari = ((random_ari > 0.4) & (random_ari <= 0.7)).mean() * 100
    pct_low_ari = (random_ari <= 0.4).mean() * 100

    stats_dict = {
        'tus_silhouette': tus_silhouette,
        'random_silhouette_mean': random_silhouettes.mean(),
        'random_silhouette_std': random_silhouettes.std(),
        'silhouette_percentile': pct_lower,
        'ari_mean': random_ari.mean(),
        'ari_std': random_ari.std(),
        'pct_high_ari': pct_high_ari,
        'pct_moderate_ari': pct_moderate_ari,
        'pct_low_ari': pct_low_ari
    }

    print(f"\nTUS vs Random Comparison:")
    print(f"  TUS silhouette: {tus_silhouette:.4f}")
    print(f"  Random silhouette: {random_silhouettes.mean():.4f} +/- {random_silhouettes.std():.4f}")
    print(f"  TUS percentile: {pct_lower:.1f}% (lower = TUS clusters less compact)")
    print(f"\n  ARI Distribution:")
    print(f"    High similarity (>0.7): {pct_high_ari:.1f}%")
    print(f"    Moderate (0.4-0.7): {pct_moderate_ari:.1f}%")
    print(f"    Low similarity (<=0.4): {pct_low_ari:.1f}%")

    return stats_dict


def monte_carlo_null_distribution(
    expression_df,
    tus_ari_opt,
    tus_ari_gran,
    n_reference_sets=100,
    n_comparisons_per_ref=100,
    n_genes=None,
    random_seed=None,
    show_progress=True
):
    """
    Build null distribution of ARI values: random gene set vs. random gene set.

    This answers: "How similar are ANY two random gene sets to each other?"
    By comparing TUS-vs-random ARI to this null, we can determine if TUS
    clustering is more or less similar to random than expected by chance.

    SCIENTIFIC QUESTION:
    Is TUS behaving like a typical gene set, or does it show unusual
    similarity/dissimilarity to random gene clusterings?

    METHOD:
    1. Pick a random gene "reference" set, cluster brain regions
    2. Compare to N other random gene sets via ARI
    3. Repeat with M reference sets to build stable null distribution
    4. Compare TUS-vs-random ARI to this null

    INTERPRETATION:
    - If TUS ARI is within null distribution: TUS behaves like any gene set
    - If TUS ARI > null (higher similarity): TUS converges on same architecture
    - If TUS ARI < null (lower similarity): TUS has specificity!

    Args:
        expression_df: Full expression DataFrame with all genes
        tus_ari_opt: Mean ARI from TUS vs random at K_OPTIMAL (from prior analysis)
        tus_ari_gran: Mean ARI from TUS vs random at K_GRANULAR (from prior analysis)
        n_reference_sets: Number of reference gene sets to use (default: 100)
        n_comparisons_per_ref: Comparisons per reference set (default: 100)
        n_genes: Genes per set (default: matches TUS count)
        random_seed: Random seed for reproducibility
        show_progress: Show progress bar

    Returns:
        results_dict: Dictionary containing:
            - null_ari_opt: Array of null ARI values at K_OPTIMAL
            - null_ari_gran: Array of null ARI values at K_GRANULAR
            - tus_percentile_opt: Where TUS ARI falls in null distribution
            - tus_percentile_gran: Where TUS ARI falls in null distribution
            - interpretation: Text interpretation of results
    """
    if n_genes is None:
        n_genes = len(config.TUS_GENES)
    if random_seed is None:
        random_seed = config.MONTE_CARLO_SEED

    # Get all available genes (excluding TUS genes for fair comparison)
    all_genes = [col for col in expression_df.columns
                 if col != 'label' and col not in config.TUS_GENES]

    print(f"Null Distribution Analysis (Random vs Random):")
    print(f"  Reference sets: {n_reference_sets}")
    print(f"  Comparisons per reference: {n_comparisons_per_ref}")
    print(f"  Total comparisons: {n_reference_sets * n_comparisons_per_ref:,}")
    print(f"  Genes per set: {n_genes}")
    print(f"  Available genes: {len(all_genes):,}")

    np.random.seed(random_seed)

    null_ari_opt = []
    null_ari_gran = []

    k_opt = config.K_OPTIMAL
    k_gran = config.K_GRANULAR

    iterator = range(n_reference_sets)
    if show_progress:
        iterator = tqdm(iterator, desc="Building null distribution")

    for ref_idx in iterator:
        # Select reference gene set
        ref_genes = np.random.choice(all_genes, size=n_genes, replace=False)
        ref_expression = expression_df[ref_genes].to_numpy()

        # Cluster reference set
        pca_ref = PCA(n_components=config.N_PCA_COMPONENTS)
        ref_pca = pca_ref.fit_transform(ref_expression)

        kmeans_ref_opt = KMeans(n_clusters=k_opt, random_state=0, n_init='auto')
        ref_labels_opt = kmeans_ref_opt.fit_predict(ref_pca)

        kmeans_ref_gran = KMeans(n_clusters=k_gran, random_state=0, n_init='auto')
        ref_labels_gran = kmeans_ref_gran.fit_predict(ref_pca)

        # Compare to other random gene sets
        # Get remaining genes (exclude reference genes)
        remaining_genes = [g for g in all_genes if g not in ref_genes]

        for comp_idx in range(n_comparisons_per_ref):
            # Select comparison gene set from remaining genes
            # If not enough remaining, sample with replacement from all genes
            if len(remaining_genes) >= n_genes:
                comp_genes = np.random.choice(remaining_genes, size=n_genes, replace=False)
            else:
                comp_genes = np.random.choice(all_genes, size=n_genes, replace=False)

            comp_expression = expression_df[comp_genes].to_numpy()

            # Cluster comparison set
            pca_comp = PCA(n_components=config.N_PCA_COMPONENTS)
            comp_pca = pca_comp.fit_transform(comp_expression)

            kmeans_comp_opt = KMeans(n_clusters=k_opt, random_state=0, n_init='auto')
            comp_labels_opt = kmeans_comp_opt.fit_predict(comp_pca)

            kmeans_comp_gran = KMeans(n_clusters=k_gran, random_state=0, n_init='auto')
            comp_labels_gran = kmeans_comp_gran.fit_predict(comp_pca)

            # Compute ARI between reference and comparison
            ari_opt = adjusted_rand_score(ref_labels_opt, comp_labels_opt)
            ari_gran = adjusted_rand_score(ref_labels_gran, comp_labels_gran)

            null_ari_opt.append(ari_opt)
            null_ari_gran.append(ari_gran)

    null_ari_opt = np.array(null_ari_opt)
    null_ari_gran = np.array(null_ari_gran)

    # Calculate where TUS falls in null distribution
    tus_percentile_opt = (null_ari_opt < tus_ari_opt).mean() * 100
    tus_percentile_gran = (null_ari_gran < tus_ari_gran).mean() * 100

    # Print results
    print(f"\nNull Distribution Results:")
    print(f"  K={k_opt} (random vs random):")
    print(f"    Mean ARI: {null_ari_opt.mean():.4f} ± {null_ari_opt.std():.4f}")
    print(f"    TUS vs random ARI: {tus_ari_opt:.4f}")
    print(f"    TUS percentile: {tus_percentile_opt:.1f}%")
    print(f"  K={k_gran} (random vs random):")
    print(f"    Mean ARI: {null_ari_gran.mean():.4f} ± {null_ari_gran.std():.4f}")
    print(f"    TUS vs random ARI: {tus_ari_gran:.4f}")
    print(f"    TUS percentile: {tus_percentile_gran:.1f}%")

    # Interpretation
    print(f"\nInterpretation:")
    for k, tus_ari, null_ari, pct in [
        (k_opt, tus_ari_opt, null_ari_opt, tus_percentile_opt),
        (k_gran, tus_ari_gran, null_ari_gran, tus_percentile_gran)
    ]:
        if pct > 95:
            interp = "TUS MORE similar to random than random-to-random (architecture-driven)"
        elif pct < 5:
            interp = "TUS LESS similar to random than random-to-random (TUS-SPECIFIC!)"
        else:
            interp = "TUS similarity to random is TYPICAL (no special specificity)"
        print(f"  K={k}: {interp}")

    # Return with dynamic keys based on actual K values
    results_dict = {
        f'null_ari_k{k_opt}': null_ari_opt,
        f'null_ari_k{k_gran}': null_ari_gran,
        f'null_mean_k{k_opt}': null_ari_opt.mean(),
        f'null_std_k{k_opt}': null_ari_opt.std(),
        f'null_mean_k{k_gran}': null_ari_gran.mean(),
        f'null_std_k{k_gran}': null_ari_gran.std(),
        f'tus_ari_k{k_opt}': tus_ari_opt,
        f'tus_ari_k{k_gran}': tus_ari_gran,
        f'tus_percentile_k{k_opt}': tus_percentile_opt,
        f'tus_percentile_k{k_gran}': tus_percentile_gran
    }

    return results_dict


# =============================================================================
# GENE-LEVEL STATISTICS
# =============================================================================

def compute_gene_cluster_statistics(expression_zscore, labels, genes=None):
    """
    Compute statistics for each gene across clusters.

    Tests which genes show significant differential expression between clusters
    using Kruskal-Wallis H-test (non-parametric ANOVA) and Cohen's d effect sizes.

    Args:
        expression_zscore: Z-scored expression matrix (n_regions x n_genes)
        labels: Cluster assignments
        genes: Gene names. Default: config.TUS_GENES

    Returns:
        stats_df: DataFrame with columns:
            - gene: Gene name
            - category: Gene family
            - kruskal_H: Kruskal-Wallis H statistic
            - kruskal_p: p-value
            - cluster{i}_mean_z: Mean z-score for cluster i
            - cohens_d_C{i}_vs_C{j}: Effect size between clusters
    """
    if genes is None:
        genes = config.TUS_GENES

    k = len(np.unique(labels))
    results = []

    for gene_idx, gene in enumerate(genes):
        expression = expression_zscore[:, gene_idx]

        # Group expression by cluster
        groups = [expression[labels == c] for c in range(k)]

        # Kruskal-Wallis test (non-parametric one-way ANOVA)
        H, p = stats.kruskal(*groups)

        row = {
            'gene': gene,
            'category': config.GENE_CATEGORIES.get(gene, 'Unknown'),
            'kruskal_H': H,
            'kruskal_p': p
        }

        # Cluster means
        for c in range(k):
            row[f'cluster{c+1}_mean_z'] = groups[c].mean()

        # Cohen's d effect sizes between all pairs
        for i in range(k):
            for j in range(i+1, k):
                d = cohens_d(groups[i], groups[j])
                row[f'cohens_d_C{i+1}_vs_C{j+1}'] = d

        results.append(row)

    stats_df = pd.DataFrame(results)

    # Sort by significance
    stats_df = stats_df.sort_values('kruskal_p')

    # Print top genes
    print(f"\nTop 10 genes by cluster differentiation (Kruskal-Wallis):")
    for _, row in stats_df.head(10).iterrows():
        print(f"  {row['gene']:8s} ({row['category']:8s}): H={row['kruskal_H']:6.1f}, p={row['kruskal_p']:.2e}")

    return stats_df


def cohens_d(group1, group2):
    """
    Compute Cohen's d effect size between two groups.

    Cohen's d = (mean1 - mean2) / pooled_std

    Interpretation:
        |d| < 0.2: Small effect
        |d| 0.2-0.8: Medium effect
        |d| > 0.8: Large effect

    Args:
        group1, group2: Arrays of values

    Returns:
        d: Cohen's d effect size
    """
    n1, n2 = len(group1), len(group2)
    # Use ddof=1 for sample variance (required for Cohen's d formula)
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))

    if pooled_std == 0:
        return 0.0

    d = (group1.mean() - group2.mean()) / pooled_std
    return d


def identify_cluster_marker_genes(stats_df, threshold_p=0.05, threshold_d=0.8):
    """
    Identify genes that are markers for specific clusters.

    A marker gene shows:
    1. Significant differential expression (p < threshold)
    2. Large effect size (|d| > threshold) for at least one cluster pair

    Args:
        stats_df: Output from compute_gene_cluster_statistics()
        threshold_p: Significance threshold
        threshold_d: Minimum effect size threshold

    Returns:
        markers_df: DataFrame of marker genes with their characterization
    """
    # Filter significant genes
    sig_genes = stats_df[stats_df['kruskal_p'] < threshold_p].copy()

    # Find effect size columns
    d_cols = [c for c in stats_df.columns if c.startswith('cohens_d')]

    # Identify large effects
    markers = []
    for _, row in sig_genes.iterrows():
        large_effects = []
        for col in d_cols:
            if abs(row[col]) > threshold_d:
                clusters = col.replace('cohens_d_', '').split('_vs_')
                direction = 'higher in ' + clusters[0] if row[col] > 0 else 'higher in ' + clusters[1]
                large_effects.append({
                    'comparison': col.replace('cohens_d_', ''),
                    'd': row[col],
                    'direction': direction
                })

        if large_effects:
            markers.append({
                'gene': row['gene'],
                'category': row['category'],
                'p_value': row['kruskal_p'],
                'max_effect_size': max(abs(row[col]) for col in d_cols),
                'n_large_effects': len(large_effects),
                'effects': large_effects
            })

    markers_df = pd.DataFrame(markers)
    if not markers_df.empty:
        markers_df = markers_df.sort_values('max_effect_size', ascending=False)

    print(f"\nCluster Marker Genes (p<{threshold_p}, |d|>{threshold_d}):")
    print(f"  Found {len(markers_df)} marker genes")
    for _, row in markers_df.head(5).iterrows():
        print(f"  {row['gene']}: max |d| = {row['max_effect_size']:.2f}")

    return markers_df


def compute_bonferroni_correction(stats_df, alpha=0.05):
    """
    Apply Bonferroni correction for multiple testing.

    Args:
        stats_df: DataFrame with 'kruskal_p' column
        alpha: Family-wise error rate

    Returns:
        stats_df with additional columns:
            - bonferroni_p: Corrected p-value
            - significant_bonferroni: Boolean
    """
    n_tests = len(stats_df)
    stats_df = stats_df.copy()
    stats_df['bonferroni_p'] = stats_df['kruskal_p'] * n_tests
    stats_df['bonferroni_p'] = stats_df['bonferroni_p'].clip(upper=1.0)
    stats_df['significant_bonferroni'] = stats_df['bonferroni_p'] < alpha

    n_sig = stats_df['significant_bonferroni'].sum()
    print(f"\nBonferroni correction (alpha={alpha}, {n_tests} tests):")
    print(f"  {n_sig} genes remain significant")

    return stats_df
