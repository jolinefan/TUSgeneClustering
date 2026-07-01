"""
Configuration Constants for TUS Gene Analysis
==============================================

This module contains all configuration settings, file paths, gene lists,
and color schemes used throughout the analysis.
"""

import os
import matplotlib as mpl

# =============================================================================
# FILE PATHS
# =============================================================================

# Project root directory (parent of tusgene package)
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input data files
EXPRESSION_PATH = os.path.join(
    PROJECT_DIR,
    'expression_TianS2_lrmirror-bidirectional_missing-interpolate_ibfthreshold-p4.csv'
)
ATLAS_PATH = os.path.join(
    PROJECT_DIR,
    '300Parcels_Yeo2011_17Networks_and_TianS2_FSLMNI152_2mm.nii.gz'
)

# Output directories
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')
TABLES_DIR = os.path.join(OUTPUT_DIR, 'tables')


# =============================================================================
# ANALYSIS PARAMETERS
# =============================================================================

# PCA settings
# -----------------------------------------------------------------------------
# Rationale for N_PCA_COMPONENTS = 10:
#
# Dimensionality reduction via PCA is standard practice before clustering
# high-dimensional gene expression data. The number of components is typically
# chosen to capture a sufficient proportion of total variance (commonly 70-90%)
# while reducing noise and mitigating the "curse of dimensionality" that
# degrades K-means performance in high dimensions.
#
# For this dataset (332 brain regions × 35 genes), 10 components empirically
# captures 90.6% of total variance. This threshold balances:
#   1. Signal preservation - retaining dominant expression patterns
#   2. Noise reduction - discarding low-variance components
#   3. Clustering stability - avoiding sparse high-dimensional space
#
# Note: There is no universally accepted method for selecting the number of
# components. Common approaches include variance thresholds (70-90%),
# eigenvalue > 1 (Kaiser criterion), and scree plot inspection.
#
# References:
#   - Jolliffe IT, Cadima J. Principal component analysis: a review and recent
#     developments. Phil Trans R Soc A. 2016;374:20150202.
#     https://doi.org/10.1098/rsta.2015.0202
#   - Ma S, Dai Y. Principal component analysis based methods in bioinformatics
#     studies. Brief Bioinform. 2011;12(6):714-722.
#     https://doi.org/10.1093/bib/bbq090
# -----------------------------------------------------------------------------
N_PCA_COMPONENTS = 10  # Captures 90.6% variance (empirically determined)

# Clustering settings
# -----------------------------------------------------------------------------
# K values determined via robust silhouette analysis across multiple random seeds.
# See output/tables/robust_silhouette_analysis.csv for full results.
#
# K=2 trivially separates cortex from subcortex and is excluded from consideration.
# K_OPTIMAL: Highest mean silhouette score for K >= 3
# K_GRANULAR: Local maximum providing finer-grained clustering
#
# To regenerate: python -m tusgene.main (runs stage1b_robust_silhouette)
# -----------------------------------------------------------------------------

# =============================================================================
# MANUAL K SELECTION (HARDCODED)
# =============================================================================
# Override the data-driven K selection with a manually specified value.
# Set K_MANUAL to use a specific K for main analysis figures.
K_MANUAL = 7  # Manually selected K value for analysis

K_OPTIMAL = 5      # Highest mean silhouette for K >= 3
K_GRANULAR = K_MANUAL  # Using manually selected K=7 for granular analysis
K_RANGE = range(2, 16)  # Extended range for silhouette analysis (2-15)

# Monte Carlo settings
# -----------------------------------------------------------------------------
# N_MONTE_CARLO_ITERATIONS = 10,000:
#
# Monte Carlo permutation tests compare TUS gene clustering to random gene sets.
# The number of iterations determines:
#   1. P-value precision: minimum detectable p ≈ 1/N (here, p ≈ 0.0001)
#   2. Margin of error: ~7,500 iterations needed for <10% error at α=0.05
#   3. Computational cost: ~10-15 min for 10,000 iterations on modern hardware
#
# 10,000 is standard in genomics permutation tests, balancing precision and
# computational feasibility for genome-wide studies.
#
# Reference:
#   - Lin DY. An efficient Monte Carlo approach to assessing statistical
#     significance in genomic studies. Bioinformatics. 2005;21(6):781-7.
#     https://doi.org/10.1093/bioinformatics/bti053
# -----------------------------------------------------------------------------
N_MONTE_CARLO_ITERATIONS = 10000
MONTE_CARLO_SEED = 42

# Reproducibility
# -----------------------------------------------------------------------------
# RANDOM_STATE = 42:
#
# Fixed seed ensures identical results across runs for:
#   - K-means clustering initialization
#   - PCA (when using randomized SVD)
#   - Monte Carlo sampling
#
# The value 42 is arbitrary but conventional in scientific computing
# (a reference to "The Hitchhiker's Guide to the Galaxy"). Any fixed integer
# provides equivalent reproducibility; 42 is simply widely recognized.
# -----------------------------------------------------------------------------
RANDOM_STATE = 0


# =============================================================================
# TUS GENE LIST (n=46)
# =============================================================================
#
# Gene Selection Methodology
# --------------------------
# Genes were selected via a systematic, review-anchored pipeline:
#   1. Ion channel/receptor families identified from independent reviews
#      across 3 tiers (TUS-specific, mechanotransduction, force-from-lipids)
#   2. All human members enumerated via HGNC nomenclature (147 genes)
#   3. Filtered for Allen Human Brain Atlas availability (109 remain)
#   4. PubMed literature validation via systematic search (108 queries,
#      14 categories, 7,010+ articles screened)
#   5. Inclusion threshold: gene mentioned in >= 10 articles
#   6. Expert exclusions with documented rationale
#
# See scripts/systematic_gene_selection.py for full pipeline.
# See output/tables/systematic_gene_selection.csv for gene-level data.
# =============================================================================

TUS_GENES = [
    # =========================================================================
    # TIER 1: TUS-specific reviews (Song 2023, Prieto & Maduke 2024, Ranade 2015)
    # =========================================================================

    # Piezo mechanosensitive channels - primary ultrasound targets
    'PIEZO1', 'PIEZO2',

    # TRP channels - mechanosensitive subfamilies
    # TRPV: Vanilloid subfamily, respond to mechanical stretch
    'TRPV1', 'TRPV2', 'TRPV3',
    # PKD: Polycystin channels, primary cilia mechanosensors
    'PKD1', 'PKD2',
    # TRPC: Canonical subfamily, stretch-activated
    'TRPC1', 'TRPC3', 'TRPC4', 'TRPC5',
    # TRPM: Melastatin subfamily, mechanically modulated
    'TRPM2', 'TRPM3', 'TRPM4', 'TRPM7',

    # Two-pore potassium (K2P) channels - stretch-activated
    'KCNK1',   # TWIK-1
    'KCNK2',   # TREK-1: canonical stretch-activated K2P
    'KCNK3',   # TASK-1
    'KCNK5',   # TASK-2
    'KCNK9',   # TASK-3
    'KCNK10',  # TREK-2: stretch-activated

    # Acid-sensing ion channels (ASIC) - proton/mechanical gated
    'ASIC1', 'ASIC2', 'ASIC3', 'ASIC4',

    # =========================================================================
    # TIER 2: Broader mechanotransduction reviews
    # =========================================================================

    # Voltage-gated sodium (Nav) - ultrasound-sensitive (Legon & Strohman 2024)
    'SCN1A',   # Nav1.1
    'SCN2A',   # Nav1.2
    'SCN8A',   # Nav1.6

    # T-type calcium channels (Cav3) - mechanosensation role (Heppenstall & Lewin 2006)
    'CACNA1G', # Cav3.1
    'CACNA1H', # Cav3.2
    'CACNA1I', # Cav3.3

    # Inward rectifier potassium (Kir) - potassium channel family (Song 2023)
    'KCNJ10',  # Kir4.1: glial potassium regulation
    'KCNJ2',   # Kir2.1: mechanosensitive inward rectifier

    # Connexin/Pannexin - mechanosensitive gap junctions (Li 2024)
    'GJA1',    # Connexin-43: highly mechanosensitive gap junction
    'GJB6',    # Connexin-30: gap junction mechanical coupling
    'PANX1',   # Pannexin-1: mechanosensitive ATP release channel
    'PANX2',   # Pannexin-2: brain-expressed pannexin

    # Chloride/Anion channels - volume-regulated (Li 2024)
    'BEST1',   # Bestrophin-1: calcium-activated chloride channel
    'LRRC8A',  # SWELL1: volume-regulated anion channel

    # Integrin - ECM mechanosensors (Di 2023)
    'ITGB1',   # Integrin beta-1
    'ITGB2',   # Integrin beta-2
    'ITGB8',   # Integrin beta-8

    # =========================================================================
    # TIER 3: Force-from-lipids (Petersen 2016; Head 2014)
    # =========================================================================

    # Phospholipase D - membrane tension sensors (Petersen 2016)
    'PLD1',    # Phospholipase D1
    'PLD2',    # Phospholipase D2: direct mechanosensor upstream of TREK-1

    # Caveolin - lipid raft scaffolding proteins (Head, Patel & Insel 2014)
    'CAV1',    # Caveolin-1: caveolae mechanosensitive membrane domains
    'CAV2',    # Caveolin-2: co-expressed with CAV1 in caveolae
]

# Gene family categorization for figure legends
GENE_CATEGORIES = {
    # Piezo
    'PIEZO1': 'Piezo', 'PIEZO2': 'Piezo',
    # TRP (all subfamilies grouped together)
    'TRPV1': 'TRP', 'TRPV2': 'TRP', 'TRPV3': 'TRP',
    'PKD1': 'TRP', 'PKD2': 'TRP',
    'TRPC1': 'TRP', 'TRPC3': 'TRP', 'TRPC4': 'TRP', 'TRPC5': 'TRP',
    'TRPM2': 'TRP', 'TRPM3': 'TRP', 'TRPM4': 'TRP', 'TRPM7': 'TRP',
    # K2P
    'KCNK1': 'K2P', 'KCNK2': 'K2P', 'KCNK3': 'K2P', 'KCNK5': 'K2P',
    'KCNK9': 'K2P', 'KCNK10': 'K2P',
    # ASIC
    'ASIC1': 'ASIC', 'ASIC2': 'ASIC', 'ASIC3': 'ASIC', 'ASIC4': 'ASIC',
    # Nav
    'SCN1A': 'Nav', 'SCN2A': 'Nav', 'SCN8A': 'Nav',
    # Cav
    'CACNA1G': 'Cav', 'CACNA1H': 'Cav', 'CACNA1I': 'Cav',
    # Kir
    'KCNJ10': 'Kir', 'KCNJ2': 'Kir',
    # Connexin/Pannexin
    'GJA1': 'Connexin', 'GJB6': 'Connexin', 'PANX1': 'Connexin', 'PANX2': 'Connexin',
    # Chloride
    'BEST1': 'Chloride', 'LRRC8A': 'Chloride',
    # Integrin
    'ITGB1': 'Integrin', 'ITGB2': 'Integrin', 'ITGB8': 'Integrin',
    # Phospholipase
    'PLD1': 'Phospholipase', 'PLD2': 'Phospholipase',
    # Caveolin
    'CAV1': 'Caveolin', 'CAV2': 'Caveolin',
}

# Ordered list of categories for consistent legend ordering
CATEGORY_ORDER = ['Piezo', 'TRP', 'K2P', 'ASIC', 'Nav', 'Cav', 'Kir', 'Connexin', 'Chloride', 'Integrin', 'Phospholipase', 'Caveolin']

# Gene families (inverse of GENE_CATEGORIES - maps family to gene list)
GENE_FAMILIES = {
    'Piezo': ['PIEZO1', 'PIEZO2'],
    'TRP': ['TRPV1', 'TRPV2', 'TRPV3', 'PKD1', 'PKD2', 'TRPC1', 'TRPC3', 'TRPC4', 'TRPC5', 'TRPM2', 'TRPM3', 'TRPM4', 'TRPM7'],
    'K2P': ['KCNK1', 'KCNK2', 'KCNK3', 'KCNK5', 'KCNK9', 'KCNK10'],
    'ASIC': ['ASIC1', 'ASIC2', 'ASIC3', 'ASIC4'],
    'Nav': ['SCN1A', 'SCN2A', 'SCN8A'],
    'Cav': ['CACNA1G', 'CACNA1H', 'CACNA1I'],
    'Kir': ['KCNJ10', 'KCNJ2'],
    'Connexin': ['GJA1', 'GJB6', 'PANX1', 'PANX2'],
    'Chloride': ['BEST1', 'LRRC8A'],
    'Integrin': ['ITGB1', 'ITGB2', 'ITGB8'],
    'Phospholipase': ['PLD1', 'PLD2'],
    'Caveolin': ['CAV1', 'CAV2'],
}


# =============================================================================
# COLOR SCHEMES
# =============================================================================

# Gene category colors (Nature-style palette)
CATEGORY_COLORS = {
    'Piezo': '#E64B35',       # Red - primary mechanosensors
    'TRP': '#F39B7F',         # Salmon
    'K2P': '#4DBBD5',         # Cyan
    'Kir': '#8491B4',         # Blue-gray
    'Nav': '#91D1C2',         # Teal
    'Cav': '#00A087',         # Green
    'ASIC': '#B09C85',        # Tan
    'Integrin': '#7E6148',    # Brown
    'Connexin': '#DC9FB4',    # Pink - gap junctions
    'Phospholipase': '#9B59B6',  # Purple - lipid signaling
    'Chloride': '#999999',    # Gray - anion channels
    'Natriuretic': '#3498DB', # Blue - pressure sensing
    'Caveolin': '#2ECC71',    # Emerald green - lipid raft scaffolding
}

# =============================================================================
# CLUSTER COLOR PALETTES
# =============================================================================

# Cluster colors for different K values (neon palette for high contrast)
CLUSTER_COLORS_3 = ['#39ff14', '#ff073a', '#00f0ff']  # green, red, cyan
CLUSTER_COLORS_4 = ['#39ff14', '#ff073a', '#00f0ff', '#ff00ff']  # green, red, cyan, magenta
CLUSTER_COLORS_5 = ['#39ff14', '#ff073a', '#00f0ff', '#ff00ff', '#ffcc00']
CLUSTER_COLORS_7 = [
    '#27b20e', '#39ff14',  # green shades
    '#ff073a', '#ff6a88',  # red shades
    '#00f0ff',             # cyan
    '#ff00ff',             # magenta
    '#ffcc00'              # yellow
]

# Active color assignments (update these when changing K_OPTIMAL/K_GRANULAR)
CLUSTER_COLORS_OPT = CLUSTER_COLORS_5
CLUSTER_COLORS_GRAN = CLUSTER_COLORS_7


# =============================================================================
# FIGURE STYLE SETTINGS
# =============================================================================

# Line color (near-black for crisp printing)
LINE_COLOR = (30/255, 30/255, 30/255)

def setup_figure_style():
    """Configure matplotlib for publication-quality figures."""
    mpl.rcParams['pdf.fonttype'] = 42      # Editable text in PDFs
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['font.size'] = 10
    mpl.rcParams['axes.linewidth'] = 0.8
    mpl.rcParams['axes.edgecolor'] = LINE_COLOR
    mpl.rcParams['xtick.color'] = LINE_COLOR
    mpl.rcParams['ytick.color'] = LINE_COLOR
    mpl.rcParams['text.color'] = LINE_COLOR


# =============================================================================
# ATLAS INFORMATION
# =============================================================================

# The atlas combines:
# - Yeo 2011 17-network cortical parcellation (300 parcels)
# - Tian S2 subcortical parcellation (32 regions)
# Total: 332 brain regions

N_CORTICAL_REGIONS = 300
N_SUBCORTICAL_REGIONS = 32
N_TOTAL_REGIONS = N_CORTICAL_REGIONS + N_SUBCORTICAL_REGIONS