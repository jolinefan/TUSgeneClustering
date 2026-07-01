# TUS Gene Spatial Clustering Analysis

Spatial transcriptomic analysis of mechanosensitive genes hypothesized to mediate transcranial ultrasound stimulation (TUS) effects. This pipeline clusters brain regions based on expression profiles of 46 TUS-relevant genes to identify areas with potentially similar neuromodulation responsivity.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full analysis pipeline
python -m tusgene.main

# Skip Monte Carlo for faster iteration (figures only)
python -m tusgene.main --skip-monte-carlo --skip-robust-silhouette

# Validate results
python scripts/validate_results.py
```

## Analysis Pipeline

The pipeline progresses through five stages:

```
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 1: DATA LOADING                                              │
│  ├── Load gene expression (332 regions x 46 genes)                  │
│  └── Load brain atlas (Yeo 17-network + Tian S2 subcortical)        │
├─────────────────────────────────────────────────────────────────────┤
│  STAGE 2: CLUSTERING ANALYSIS                                       │
│  ├── PCA dimensionality reduction (46 -> 10 components, ~90% var)   │
│  ├── Robust silhouette analysis (multiple seeds x K=2-10)           │
│  ├── K-means clustering at K_OPTIMAL and K_GRANULAR                 │
│  └── Hierarchical clustering for dendrogram visualization           │
├─────────────────────────────────────────────────────────────────────┤
│  STAGE 3: MONTE CARLO VALIDATION                                    │
│  ├── Generate 10,000 random 46-gene sets                            │
│  └── Cluster each and compute ARI vs TUS clustering                 │
├─────────────────────────────────────────────────────────────────────┤
│  STAGE 4: GENE STATISTICS                                           │
│  ├── Kruskal-Wallis tests for differential expression               │
│  ├── Cohen's d effect sizes                                         │
│  ├── Bonferroni-corrected significance                              │
│  └── Per-gene per-cluster significance (Wilcoxon + FDR correction)  │
├─────────────────────────────────────────────────────────────────────┤
│  STAGE 5: FIGURE GENERATION                                         │
│  ├── Figure 1 Main: Combined heatmap, brain renders, radar plots    │
│  ├── Supplemental: K_OPTIMAL clustering overview                    │
│  ├── Supplemental: Robust silhouette curve                          │
│  └── Supplemental: Monte Carlo ARI distributions                    │
└─────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
tusgene/
├── tusgene/                  # Main package
│   ├── config.py             # Constants, gene lists, colors
│   ├── data.py               # Data loading functions
│   ├── clustering.py         # PCA, K-means, silhouette, significance
│   ├── statistics.py         # Monte Carlo, gene statistics
│   ├── visualization.py      # All figure generation
│   ├── main.py               # Pipeline orchestration
│   └── METHODS.txt           # Statistical methods documentation
├── pubMed/                   # PubMed literature search for gene selection
│   ├── systematic_gene_identification.py  # Queries NCBI across 14 categories
│   ├── literature_search.py  # PubMed API wrapper
│   ├── extract_genes.py      # Gene symbol extraction from abstracts
│   └── systematic_review_results/  # Search outputs (rankings, logs)
├── scripts/
│   ├── create_mainFig_fast.py         # Regenerate main figure (no simulations)
│   ├── create_gene_selection_flowchart.py  # Gene selection flowchart figure
│   ├── create_supplemental_gene_table.py   # Supplemental gene significance table
│   ├── systematic_gene_selection.py   # Final 46-gene list from family + PubMed pipeline
│   └── validate_results.py            # QC checks on pipeline output
├── output/
│   ├── figures/              # Generated figures (PDF/JPG)
│   └── tables/               # Generated CSV tables (gene stats, cluster matrices)
└── requirements.txt          # Python dependencies
```

## Data Requirements

Download the following files from Figshare ([doi:10.6084/m9.figshare.32862467](https://doi.org/10.6084/m9.figshare.32862467)) and place them in the project root:

- `expression_TianS2_lrmirror-bidirectional_missing-interpolate_ibfthreshold-p4.csv`
- `300Parcels_Yeo2011_17Networks_and_TianS2_FSLMNI152_2mm.nii.gz`

## TUS Gene List (n=46)

Mechanosensitive genes spanning 12 families, selected via the systematic, review-anchored pipeline below and available in the Allen Human Brain Atlas:

- **Piezo**: PIEZO1, PIEZO2
- **TRP**: TRPV1, TRPV2, TRPV3, PKD1, PKD2, TRPC1, TRPC3, TRPC4, TRPC5, TRPM2, TRPM3, TRPM4, TRPM7
- **K2P**: KCNK1, KCNK2, KCNK3, KCNK5, KCNK9, KCNK10
- **ASIC**: ASIC1, ASIC2, ASIC3, ASIC4
- **Nav**: SCN1A, SCN2A, SCN8A
- **Cav**: CACNA1G, CACNA1H, CACNA1I
- **Kir**: KCNJ10, KCNJ2
- **Connexin**: GJA1, GJB6, PANX1, PANX2
- **Chloride**: BEST1, LRRC8A
- **Integrin**: ITGB1, ITGB2, ITGB8
- **Phospholipase**: PLD1, PLD2
- **Caveolin**: CAV1, CAV2

## Gene Selection Methodology

Genes were selected through a systematic, review-anchored pipeline. The full
implementation is in `scripts/systematic_gene_selection.py`; per-gene results
are in `output/tables/systematic_gene_selection.csv` and a step-by-step record
in `output/tables/gene_selection_methodology.txt`.

### Pipeline

1. **Family definitions** — Mechanosensitive ion-channel/receptor families
   identified from independent reviews across three tiers:
   - *Tier 1 (TUS-specific):* Song et al. 2023, Prieto & Maduke 2024, Ranade et al. 2015 — Piezo, TRP, K2P, ASIC
   - *Tier 2 (broader mechanotransduction):* Nav, Cav (T-type), Kir, Connexin/Pannexin, Chloride/anion, Integrin
   - *Tier 3 (force-from-lipids):* Petersen 2016, Head et al. 2014 — Phospholipase D, Caveolin
2. **Member enumeration** — All human family members enumerated via HGNC nomenclature (147 genes).
3. **Atlas availability** — Filtered to genes present in the Allen Human Brain Atlas expression data (109 genes).
4. **Literature validation** — PubMed systematic search (108 predefined queries across 14 mechanotransduction categories; 7,000+ articles screened).
5. **Inclusion threshold** — Gene mentioned in ≥10 articles within the search results.
6. **Expert exclusions** — Genes removed with documented rationale 

This yields the final set of **46 genes**.
