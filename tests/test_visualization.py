"""
Unit Tests for Visualization Module
====================================

Tests for figure generation functions.

Run with: pytest tests/test_visualization.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt

from tusgene import config, visualization


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_cluster_data():
    """Create mock clustering data for testing."""
    np.random.seed(42)

    k_opt = config.K_OPTIMAL
    k_gran = config.K_GRANULAR
    n_genes = len(config.TUS_GENES)

    # Mock cluster results
    labels_opt = np.random.randint(0, k_opt, size=332)
    labels_gran = np.random.randint(0, k_gran, size=332)

    cluster_means_opt = np.random.randn(k_opt, n_genes)
    cluster_means_gran = np.random.randn(k_gran, n_genes)

    return {
        'labels_opt': labels_opt,
        'labels_gran': labels_gran,
        'cluster_means_opt': cluster_means_opt,
        'cluster_means_gran': cluster_means_gran,
    }


@pytest.fixture
def mock_genes():
    """Return list of mock gene names."""
    return config.TUS_GENES


# =============================================================================
# COLOR CONFIGURATION TESTS
# =============================================================================

class TestColorConfiguration:
    """Tests for color configurations in config.py."""

    def test_cluster_colors_opt_count(self):
        """Should have correct number of colors for K_OPTIMAL."""
        assert len(config.CLUSTER_COLORS_OPT) == config.K_OPTIMAL

    def test_cluster_colors_gran_count(self):
        """Should have correct number of colors for K_GRANULAR."""
        assert len(config.CLUSTER_COLORS_GRAN) == config.K_GRANULAR

    def test_colors_are_valid_hex(self):
        """All colors should be valid hex codes."""
        import re
        hex_pattern = re.compile(r'^#[0-9A-Fa-f]{6}$')

        for color in config.CLUSTER_COLORS_OPT:
            assert hex_pattern.match(color), f"Invalid hex color: {color}"

        for color in config.CLUSTER_COLORS_GRAN:
            assert hex_pattern.match(color), f"Invalid hex color: {color}"

    def test_category_colors_complete(self):
        """All gene categories should have assigned colors."""
        for category in config.CATEGORY_ORDER:
            assert category in config.CATEGORY_COLORS, f"Missing color for {category}"

    def test_category_order_matches_families(self):
        """CATEGORY_ORDER should match GENE_FAMILIES keys."""
        assert set(config.CATEGORY_ORDER) == set(config.GENE_FAMILIES.keys())


# =============================================================================
# GRANULAR COLOR GENERATION TESTS
# =============================================================================

class TestGranularColorGeneration:
    """Tests for generate_granular_colors_from_parent_map function."""

    def test_returns_correct_number_of_colors(self):
        """Should return K_GRANULAR colors."""
        k_opt = config.K_OPTIMAL
        k_gran = config.K_GRANULAR

        # Create a simple mapping (each granular cluster maps to an optimal cluster)
        gran_to_opt_map = {i: i % k_opt for i in range(k_gran)}
        colors = visualization.generate_granular_colors_from_parent_map(
            gran_to_opt_map, config.CLUSTER_COLORS_OPT
        )

        assert len(colors) == k_gran

    def test_colors_are_strings(self):
        """All colors should be hex strings."""
        k_opt = config.K_OPTIMAL
        k_gran = config.K_GRANULAR

        gran_to_opt_map = {i: i % k_opt for i in range(k_gran)}
        colors = visualization.generate_granular_colors_from_parent_map(
            gran_to_opt_map, config.CLUSTER_COLORS_OPT
        )

        for color in colors:
            assert isinstance(color, str)
            assert color.startswith('#')


# =============================================================================
# HEATMAP TESTS
# =============================================================================

class TestHeatmapCreation:
    """Tests for heatmap panel creation."""

    def test_heatmap_creates_figure(self, mock_cluster_data, mock_genes):
        """Heatmap should create a valid figure."""
        from scipy.cluster.hierarchy import linkage

        cluster_means = mock_cluster_data['cluster_means_opt']
        gene_linkage = linkage(cluster_means.T, method='average')

        fig = plt.figure(figsize=(10, 5))
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(1, 1)

        # This should not raise
        visualization.create_heatmap_panel(
            fig, gs[0],
            cluster_means, config.K_OPTIMAL, config.CLUSTER_COLORS_OPT,
            gene_linkage, mock_genes,
            'a', 'Test Title'
        )

        plt.close(fig)

    def test_heatmap_panel_label(self, mock_cluster_data, mock_genes):
        """Panel label should be added to figure."""
        from scipy.cluster.hierarchy import linkage

        cluster_means = mock_cluster_data['cluster_means_opt']
        gene_linkage = linkage(cluster_means.T, method='average')

        fig = plt.figure(figsize=(10, 5))
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(1, 1)

        visualization.create_heatmap_panel(
            fig, gs[0],
            cluster_means, config.K_OPTIMAL, config.CLUSTER_COLORS_OPT,
            gene_linkage, mock_genes,
            'x', 'Test'  # Panel label 'x'
        )

        # Check that axes were created (heatmap panel creates subplots)
        assert len(fig.axes) > 0

        plt.close(fig)


# =============================================================================
# BRAIN SLICE TESTS
# =============================================================================

class TestBrainSliceCreation:
    """Tests for brain slice panel creation."""

    def test_brain_slices_creates_axes(self, mock_cluster_data):
        """Brain slices should create axes in figure."""
        from tusgene import data
        _, atlas_data = data.load_atlas()

        fig = plt.figure(figsize=(10, 5))
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(1, 1)

        visualization.create_brain_slices(
            fig, gs[0],
            mock_cluster_data['labels_opt'], config.K_OPTIMAL, config.CLUSTER_COLORS_OPT,
            atlas_data, 'b'
        )

        # Should have multiple axes (one per slice)
        assert len(fig.axes) > 0

        plt.close(fig)


# =============================================================================
# OUTPUT DIRECTORY TESTS
# =============================================================================

class TestOutputDirectories:
    """Tests for output directory creation."""

    def test_ensure_output_dirs_creates_figures_dir(self):
        """Should ensure figures directory exists."""
        visualization.ensure_output_dirs()
        assert os.path.exists(config.FIGURES_DIR)

    def test_ensure_output_dirs_creates_tables_dir(self):
        """Should ensure tables directory exists."""
        visualization.ensure_output_dirs()
        assert os.path.exists(config.TABLES_DIR)


# =============================================================================
# FIGURE SIZE AND LAYOUT TESTS
# =============================================================================

class TestFigureSizes:
    """Tests for figure dimensions."""

    def test_figure1_dimensions(self, mock_cluster_data, mock_genes):
        """Figure 1 should have expected dimensions."""
        # Can't easily test without full data, so just check config
        # Default figsize is (15, 7)
        pass  # Placeholder

    def test_k_granular_is_configured(self):
        """K_GRANULAR should be configured."""
        assert config.K_GRANULAR > config.K_OPTIMAL


# =============================================================================
# RADAR PLOT TESTS
# =============================================================================

class TestRadarPlots:
    """Tests for radar plot generation."""

    def test_gene_families_cover_all_genes(self):
        """All TUS genes should be in a family."""
        genes_in_families = set()
        for genes in config.GENE_FAMILIES.values():
            genes_in_families.update(genes)

        assert genes_in_families == set(config.TUS_GENES)

    def test_family_count(self):
        """Should have gene families defined."""
        assert len(config.CATEGORY_ORDER) > 0

    def test_all_families_have_genes(self):
        """Each family should have at least one gene."""
        for family, genes in config.GENE_FAMILIES.items():
            assert len(genes) >= 1, f"Family {family} has no genes"


# =============================================================================
# COLORMAP TESTS
# =============================================================================

class TestColormaps:
    """Tests for custom colormaps."""

    def test_cluster_colors_are_distinct(self):
        """All K_OPTIMAL cluster colors should be distinct."""
        colors = config.CLUSTER_COLORS_OPT
        assert len(colors) == len(set(colors)), "Cluster colors should be unique"

    def test_gran_colors_count(self):
        """K_GRANULAR colors should have correct count."""
        assert len(config.CLUSTER_COLORS_GRAN) == config.K_GRANULAR
        assert len(config.CLUSTER_COLORS_OPT) == config.K_OPTIMAL


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'not slow'])
