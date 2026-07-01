"""Create supplemental table of gene significance per cluster (K=7).

Combines z-scores, FDR-corrected p-values, and direction for each gene
in each cluster. Output is a clean CSV suitable for supplemental material.
"""
import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from tusgene import config

# Load data
zscore_df = pd.read_csv('output/tables/cluster_gene_matrix_granular.csv', index_col=0)
pval_df = pd.read_csv('output/tables/gene_significance_fdr.csv', index_col=0)

# Build the table: one row per gene-cluster combination
rows = []
for gene in config.TUS_GENES:
    family = config.GENE_CATEGORIES.get(gene, 'Other')
    for cluster in range(1, config.K_GRANULAR + 1):
        cluster_name = f'C{cluster}'
        zscore = zscore_df.loc[cluster_name, gene] if gene in zscore_df.columns else np.nan
        pval = pval_df.loc[cluster_name, gene] if gene in pval_df.columns else np.nan

        # Significance stars
        if pd.isna(pval):
            stars = ''
        elif pval < 0.001:
            stars = '***'
        elif pval < 0.01:
            stars = '**'
        elif pval < 0.05:
            stars = '*'
        else:
            stars = 'ns'

        direction = 'enriched' if zscore > 0 else 'depleted' if zscore < 0 else 'neutral'

        rows.append({
            'Gene': gene,
            'Family': family,
            'Cluster': cluster_name,
            'Z-score': round(zscore, 3),
            'FDR p-value': f'{pval:.2e}' if not pd.isna(pval) else '',
            'Significance': stars,
            'Direction': direction,
        })

long_df = pd.DataFrame(rows)

# Save long format
long_df.to_csv('output/tables/supplemental_gene_significance_long.csv', index=False)
print(f'Long format: {len(long_df)} rows saved')

# Also create a wide-format summary: gene x cluster with z-scores and stars
wide_rows = []
for gene in config.TUS_GENES:
    family = config.GENE_CATEGORIES.get(gene, 'Other')
    row = {'Gene': gene, 'Family': family}
    for cluster in range(1, config.K_GRANULAR + 1):
        cn = f'C{cluster}'
        zscore = zscore_df.loc[cn, gene] if gene in zscore_df.columns else np.nan
        pval = pval_df.loc[cn, gene] if gene in pval_df.columns else np.nan

        if pd.isna(pval):
            stars = ''
        elif pval < 0.001:
            stars = '***'
        elif pval < 0.01:
            stars = '**'
        elif pval < 0.05:
            stars = '*'
        else:
            stars = ''

        row[f'{cn} z-score'] = round(zscore, 3)
        row[f'{cn} p(FDR)'] = f'{pval:.2e}' if not pd.isna(pval) else ''
        row[f'{cn} sig'] = stars
    wide_rows.append(row)

wide_df = pd.DataFrame(wide_rows)
wide_df.to_csv('output/tables/supplemental_gene_significance_wide.csv', index=False)
print(f'Wide format: {len(wide_df)} genes saved')

# Print summary of most significant gene-cluster associations
print('\nTop 20 most significant gene-cluster associations:')
top = long_df[long_df['Significance'] == '***'].copy()
top['pval_float'] = top['FDR p-value'].apply(lambda x: float(x) if x else 1.0)
top = top.sort_values('pval_float').head(20)
for _, r in top.iterrows():
    print(f"  {r['Gene']:10s} ({r['Family']:12s}) in {r['Cluster']}: "
          f"z={r['Z-score']:+.3f} ({r['Direction']}), p={r['FDR p-value']}")
