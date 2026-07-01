"""
Unit Tests for Clustering Module
================================

Tests for PCA, K-means, and silhouette analysis functions.

Run with: pytest tests/test_clustering.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from tusgene import config, clustering


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def synthetic_data():
    """Create synthetic clustered data for testing."""
    X, y = make_blobs(n_samples=100, n_features=30, centers=4, random_state=42)
    return X, y


@pytest.fixture
def real_expression_data():
    """Load actual expression data."""
    from tusgene import data
    _, expression_matrix, _, genes = data.load_expression_data()
    return expression_matrix, genes


# =============================================================================
# PCA TESTS
# =============================================================================

class TestPerformPCA:
    """Tests for clustering.perform_pca()"""

    def test_output_shape(self, synthetic_data):
        """PCA output should have correct dimensions."""
        X, _ = synthetic_data
        X_pca, pca = clustering.perform_pca(X, n_components=10)

        assert X_pca.shape[0] == X.shape[0], "Number of samples should be preserved"
        assert X_pca.shape[1] == 10, "Should have requested number of components"

    def test_output_shape_fewer_components(self, synthetic_data):
        """PCA with fewer components than features."""
        X, _ = synthetic_data
        X_pca, pca = clustering.perform_pca(X, n_components=5)

        assert X_pca.shape[1] == 5

    def test_variance_explained_sum(self, real_expression_data):
        """Explained variance should sum to expected value."""
        X, _ = real_expression_data
        _, pca = clustering.perform_pca(X, n_components=10)

        total_var = sum(pca.explained_variance_ratio_)
        assert 0.85 < total_var < 0.95, f"Expected ~90% variance, got {total_var:.1%}"

    def test_deterministic_with_seed(self, synthetic_data):
        """PCA should be deterministic with fixed random state."""
        X, _ = synthetic_data

        X_pca1, _ = clustering.perform_pca(X)
        X_pca2, _ = clustering.perform_pca(X)

        np.testing.assert_array_almost_equal(X_pca1, X_pca2)

    def test_components_orthogonal(self, synthetic_data):
        """PCA components should be orthogonal."""
        X, _ = synthetic_data
        X_pca, _ = clustering.perform_pca(X, n_components=5)

        # Covariance matrix should be diagonal (orthogonal components)
        cov = np.cov(X_pca.T)
        off_diagonal = cov - np.diag(np.diag(cov))

        assert np.allclose(off_diagonal, 0, atol=1e-10)


# =============================================================================
# K-MEANS TESTS
# =============================================================================

class TestClusterKmeans:
    """Tests for clustering.cluster_kmeans()"""

    def test_correct_number_of_clusters(self, synthetic_data):
        """Should return exactly k unique labels."""
        X, _ = synthetic_data
        X_pca, _ = clustering.perform_pca(X, n_components=10)

        for k in [2, 4, 6, 9]:
            labels, _ = clustering.cluster_kmeans(X_pca, k)
            unique_labels = set(labels)
            assert unique_labels == set(range(k)), f"Expected labels 0-{k-1}, got {unique_labels}"

    def test_all_samples_assigned(self, synthetic_data):
        """All samples should be assigned to a cluster."""
        X, _ = synthetic_data
        X_pca, _ = clustering.perform_pca(X, n_components=10)
        labels, _ = clustering.cluster_kmeans(X_pca, k=4)

        assert len(labels) == X.shape[0]
        assert not any(np.isnan(labels))

    def test_deterministic_with_seed(self, synthetic_data):
        """K-means should be deterministic with fixed random state."""
        X, _ = synthetic_data
        X_pca, _ = clustering.perform_pca(X, n_components=10)

        labels1, _ = clustering.cluster_kmeans(X_pca, k=4, random_state=42)
        labels2, _ = clustering.cluster_kmeans(X_pca, k=4, random_state=42)

        np.testing.assert_array_equal(labels1, labels2)

    def test_different_seeds_may_differ(self, synthetic_data):
        """Different seeds can produce different results."""
        X, _ = synthetic_data
        X_pca, _ = clustering.perform_pca(X, n_components=10)

        labels1, _ = clustering.cluster_kmeans(X_pca, k=4, random_state=1)
        labels2, _ = clustering.cluster_kmeans(X_pca, k=4, random_state=999)

        # Labels might differ (though could be same by chance)
        # Just verify both are valid
        assert set(labels1) == set(range(4))
        assert set(labels2) == set(range(4))

    def test_labels_are_integers(self, synthetic_data):
        """Labels should be integers, not floats."""
        X, _ = synthetic_data
        X_pca, _ = clustering.perform_pca(X, n_components=10)
        labels, _ = clustering.cluster_kmeans(X_pca, k=4)

        assert labels.dtype in [np.int32, np.int64, int]


# =============================================================================
# SILHOUETTE TESTS
# =============================================================================

class TestSilhouetteAnalysis:
    """Tests for silhouette score computation."""

    def test_silhouette_in_valid_range(self, synthetic_data):
        """Silhouette scores must be in [-1, 1]."""
        X, _ = synthetic_data
        X_pca, _ = clustering.perform_pca(X, n_components=10)

        for k in [2, 4, 6]:
            labels, _ = clustering.cluster_kmeans(X_pca, k)
            from sklearn.metrics import silhouette_score
            score = silhouette_score(X_pca, labels)

            assert -1 <= score <= 1, f"Silhouette {score} out of range for k={k}"

    def test_well_separated_clusters_high_silhouette(self):
        """Well-separated clusters should have high silhouette."""
        # Create obviously separated clusters
        X, y = make_blobs(n_samples=100, centers=4, cluster_std=0.5, random_state=42)

        from sklearn.metrics import silhouette_score
        score = silhouette_score(X, y)

        assert score > 0.5, f"Well-separated clusters should have high silhouette, got {score}"


# =============================================================================
# HIERARCHICAL CLUSTERING TESTS
# =============================================================================

class TestHierarchicalClustering:
    """Tests for hierarchical clustering functions."""

    def test_gene_linkage_shape(self, real_expression_data):
        """Gene linkage should have correct shape."""
        X, genes = real_expression_data

        # Compute cluster means (simulate K=4)
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        cluster_means = np.array([X[labels == c].mean(axis=0) for c in range(4)])
        linkage = clustering.hierarchical_cluster_genes(cluster_means)

        n_genes = cluster_means.shape[1]
        assert linkage.shape == (n_genes - 1, 4)

    def test_cluster_linkage_shape(self, real_expression_data):
        """Cluster linkage should have correct shape."""
        X, _ = real_expression_data

        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=9, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        cluster_means = np.array([X[labels == c].mean(axis=0) for c in range(9)])
        linkage = clustering.hierarchical_cluster_clusters(cluster_means)

        n_clusters = 9
        assert linkage.shape == (n_clusters - 1, 4)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestGetClusterResults:
    """Tests for the main get_cluster_results function."""

    def test_returns_both_k_values(self, real_expression_data):
        """Should return results for both K_OPTIMAL and K_GRANULAR."""
        X, _ = real_expression_data
        from tusgene import data
        _, _, expression_zscore, _ = data.load_expression_data()

        results = clustering.get_cluster_results(X, expression_zscore)

        assert config.K_OPTIMAL in results
        assert config.K_GRANULAR in results

    def test_result_keys(self, real_expression_data):
        """Each result should have required keys."""
        X, _ = real_expression_data
        from tusgene import data
        _, _, expression_zscore, _ = data.load_expression_data()

        results = clustering.get_cluster_results(X, expression_zscore)

        for k in [config.K_OPTIMAL, config.K_GRANULAR]:
            assert 'labels' in results[k]
            assert 'silhouette' in results[k]
            assert 'cluster_means' in results[k]
            assert 'kmeans' in results[k]

    def test_labels_length(self, real_expression_data):
        """Labels should match number of brain regions."""
        X, _ = real_expression_data
        from tusgene import data
        _, _, expression_zscore, _ = data.load_expression_data()

        results = clustering.get_cluster_results(X, expression_zscore)

        assert len(results[config.K_OPTIMAL]['labels']) == 332
        assert len(results[config.K_GRANULAR]['labels']) == 332


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Edge case tests."""

    def test_single_cluster(self):
        """K=1 should assign all to cluster 0."""
        X = np.random.randn(50, 10)
        X_pca, _ = clustering.perform_pca(X, n_components=5)
        labels, _ = clustering.cluster_kmeans(X_pca, k=1)

        assert set(labels) == {0}
        assert len(labels) == 50

    def test_k_equals_n_samples(self):
        """K = n_samples should give each sample its own cluster."""
        X = np.random.randn(10, 5)
        X_pca, _ = clustering.perform_pca(X, n_components=3)
        labels, _ = clustering.cluster_kmeans(X_pca, k=10)

        assert len(set(labels)) == 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
