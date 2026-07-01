"""
TUS Gene Analysis Package
=========================

Spatial clustering analysis of transcranial ultrasound-sensitive (TUS) gene
expression patterns in the human brain.

Modules:
    config: Configuration constants (paths, genes, colors)
    data: Data loading and preprocessing
    clustering: PCA and K-means clustering functions
    statistics: Statistical analysis (Kruskal-Wallis, Monte Carlo)
    visualization: Figure generation functions

Usage:
    python -m tusgene.main  # Run complete analysis pipeline
"""

__version__ = "1.0.0"
__author__ = "TUS Gene Analysis Team"

from . import config
from . import data
from . import clustering
from . import statistics
from . import visualization
