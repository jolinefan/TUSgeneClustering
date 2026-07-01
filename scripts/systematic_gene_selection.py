#!/usr/bin/env python3
"""
Systematic Gene Selection for TUS Analysis
===========================================

This script implements an objective, review-based gene selection methodology.

Pipeline:
    Step 1: Define mechanosensitive protein families from independent reviews
    Step 2: Enumerate all human gene members per family (HGNC nomenclature)
    Step 3: Filter for genes present in Allen Human Brain Atlas
    Step 4: Validate with PubMed literature frequency
    Step 5: Apply inclusion threshold (>= N articles)
    Step 6: Expert exclusions with documented rationale

References for family definitions:
    Tier 1 (TUS-specific reviews):
        - Song et al. (2023) Front Neurosci 17:1232308. PMID: 37583416
        - Prieto & Maduke (2024) Curr Opin Behav Sci 56:101355. PMID: 38505510
        - Ranade et al. (2015) Neuron 87(6):1162-1179. PMID: 26402601
    Tier 2 (Broader mechanotransduction reviews):
        - Legon & Strohman (2024) Nat Rev Methods Primers 4. DOI: 10.1038/s43586-024-00368-6
        - Heppenstall & Lewin (2006) Cell Calcium 40(2):165-74. PMID: 16777219
        - Li et al. (2024) CNS Neurosci Ther 30(6):e14809. PMID: 38923822
        - Di et al. (2023) Signal Transduct Target Ther 8:282. PMID: 37518181
    Tier 3 (Lipid-mediated mechanotransduction):
        - Petersen et al. (2016) Nat Commun 7:13873. PMID: 27976674
        - Head, Patel & Insel (2014) BBA Biomembranes 1838(2):532-45. PMID: 23899502

Usage:
    python scripts/systematic_gene_selection.py
"""

import os
import sys
import pandas as pd
import numpy as np

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tusgene import config

# =============================================================================
# STEP 1: DEFINE FAMILIES FROM INDEPENDENT REVIEWS
# =============================================================================
# Each family is traceable to specific peer-reviewed publications.
# Tier indicates the source type:
#   Tier 1: Named in TUS-specific reviews as ultrasound neuromodulation mediators
#   Tier 2: Classified as mechanosensitive in broader mechanotransduction reviews
#   Tier 3: Lipid-mediated mechanotransduction pathway

FAMILIES = {
    # -------------------------------------------------------------------------
    # TIER 1: TUS-specific reviews (Song 2023, Prieto & Maduke 2024, Ranade 2015)
    # -------------------------------------------------------------------------
    'Piezo': {
        'tier': 1,
        'sources': [
            'Song et al. 2023 Front Neurosci PMID:37583416',
            'Prieto & Maduke 2024 Curr Opin Behav Sci PMID:38505510',
            'Ranade et al. 2015 Neuron PMID:26402601',
        ],
        'rationale': 'Primary mechanosensitive ion channels; bona fide mechanosensors',
        # HGNC gene family: Piezo
        'hgnc_members': ['PIEZO1', 'PIEZO2'],
    },
    'TRP': {
        'tier': 1,
        'sources': [
            'Song et al. 2023 Front Neurosci PMID:37583416',
            'Prieto & Maduke 2024 Curr Opin Behav Sci PMID:38505510',
            'Ranade et al. 2015 Neuron PMID:26402601',
        ],
        'rationale': 'Calcium-permeable channels involved in sensory mechanotransduction',
        # HGNC: TRP superfamily (TRPV, TRPC, TRPM, TRPA, TRPP/PKD subfamilies)
        'hgnc_members': [
            # TRPV (vanilloid)
            'TRPV1', 'TRPV2', 'TRPV3', 'TRPV4', 'TRPV5', 'TRPV6',
            # TRPC (canonical)
            'TRPC1', 'TRPC3', 'TRPC4', 'TRPC5', 'TRPC6', 'TRPC7',
            # TRPM (melastatin)
            'TRPM1', 'TRPM2', 'TRPM3', 'TRPM4', 'TRPM5', 'TRPM6', 'TRPM7', 'TRPM8',
            # TRPA (ankyrin)
            'TRPA1',
            # TRPP/Polycystin (PKD)
            'PKD1', 'PKD2', 'PKD1L1', 'PKD1L2', 'PKD1L3', 'PKD2L1', 'PKD2L2',
            # TRPML (mucolipin)
            'MCOLN1', 'MCOLN2', 'MCOLN3',
        ],
    },
    'K2P': {
        'tier': 1,
        'sources': [
            'Song et al. 2023 Front Neurosci PMID:37583416',
            'Prieto & Maduke 2024 Curr Opin Behav Sci PMID:38505510',
            'Ranade et al. 2015 Neuron PMID:26402601',
        ],
        'rationale': 'Background leak K+ channels; TREK/TRAAK are bona fide mechanosensors',
        # HGNC: KCNK family (two-pore domain potassium channels)
        'hgnc_members': [
            'KCNK1',   # TWIK-1
            'KCNK2',   # TREK-1
            'KCNK3',   # TASK-1
            'KCNK4',   # TRAAK
            'KCNK5',   # TASK-2
            'KCNK6',   # TWIK-2
            'KCNK7',   # KCNK7
            'KCNK9',   # TASK-3
            'KCNK10',  # TREK-2
            'KCNK12',  # THIK-2
            'KCNK13',  # THIK-1
            'KCNK15',  # TASK-5
            'KCNK16',  # TALK-1
            'KCNK17',  # TALK-2
            'KCNK18',  # TRESK
        ],
    },
    'ASIC': {
        'tier': 1,
        'sources': [
            'Ranade et al. 2015 Neuron PMID:26402601',
            'Song et al. 2023 Front Neurosci PMID:37583416',
        ],
        'rationale': 'DEG/ENaC family; proton-gated with mechanosensitive properties',
        # HGNC: ASIC family
        'hgnc_members': ['ASIC1', 'ASIC2', 'ASIC3', 'ASIC4', 'ASIC5'],
    },

    # -------------------------------------------------------------------------
    # TIER 2: Broader mechanotransduction reviews
    # -------------------------------------------------------------------------
    'Nav': {
        'tier': 2,
        'sources': [
            'Legon & Strohman 2024 Nat Rev Methods Primers DOI:10.1038/s43586-024-00368-6',
        ],
        'rationale': 'Voltage-gated Na+ channels listed as ultrasound-sensitive ion channels that undergo conformational changes altering conductance and gating',
        # HGNC: SCN*A genes (alpha subunits, neuronal)
        'hgnc_members': [
            'SCN1A', 'SCN2A', 'SCN3A', 'SCN4A', 'SCN5A',
            'SCN7A', 'SCN8A', 'SCN9A', 'SCN10A', 'SCN11A',
        ],
    },
    'Cav': {
        'tier': 2,
        'sources': [
            'Heppenstall & Lewin 2006 Cell Calcium PMID:16777219',
        ],
        'rationale': 'T-type (Cav3) calcium channels play a role in mechanosensation',
        # HGNC: T-type calcium channel alpha subunits only
        # Non-T-type (CACNA1A, CACNA1C etc.) excluded: not T-type, not mechanosensitive
        'hgnc_members': ['CACNA1G', 'CACNA1H', 'CACNA1I'],
    },
    'Kir': {
        'tier': 2,
        'sources': [
            'Song et al. 2023 Front Neurosci PMID:37583416',
        ],
        'rationale': 'Inward rectifier K+ channels; classified alongside K2P and KCa as one of three main potassium channel families in TUS neuromodulation',
        # HGNC: KCNJ family
        'hgnc_members': [
            'KCNJ1', 'KCNJ2', 'KCNJ3', 'KCNJ4', 'KCNJ5', 'KCNJ6',
            'KCNJ8', 'KCNJ9', 'KCNJ10', 'KCNJ11', 'KCNJ12', 'KCNJ13',
            'KCNJ14', 'KCNJ15', 'KCNJ16',
        ],
    },
    'Connexin': {
        'tier': 2,
        'sources': [
            'Li et al. 2024 CNS Neurosci Ther PMID:38923822',
        ],
        'rationale': 'Connexins/pannexins are mechanosensitive gap junction channels',
        # HGNC: GJA, GJB, PANX families (brain-relevant subset)
        'hgnc_members': [
            'GJA1',   # Connexin-43 (most abundant brain connexin)
            'GJA3', 'GJA4', 'GJA5', 'GJA8', 'GJA9', 'GJA10',
            'GJB1',   # Connexin-32
            'GJB2',   # Connexin-26
            'GJB3', 'GJB4', 'GJB5',
            'GJB6',   # Connexin-30
            'GJB7',
            'GJC1', 'GJC2', 'GJC3',
            'GJD2', 'GJD3', 'GJD4',
            'PANX1', 'PANX2', 'PANX3',
        ],
    },
    'Chloride': {
        'tier': 2,
        'sources': [
            'Li et al. 2024 CNS Neurosci Ther PMID:38923822',
        ],
        'rationale': 'Volume-regulated anion channels respond to cell swelling/mechanical stress',
        # HGNC: LRRC8 (VRAC) and Bestrophin families
        'hgnc_members': [
            'LRRC8A', 'LRRC8B', 'LRRC8C', 'LRRC8D', 'LRRC8E',
            'BEST1', 'BEST2', 'BEST3', 'BEST4',
        ],
    },
    'Integrin': {
        'tier': 2,
        'sources': [
            'Di et al. 2023 Signal Transduct Target Ther PMID:37518181',
        ],
        'rationale': 'ECM-cytoskeleton mechanotransducers at focal adhesions',
        # HGNC: ITGA and ITGB families
        'hgnc_members': [
            'ITGA1', 'ITGA2', 'ITGA3', 'ITGA4', 'ITGA5', 'ITGA6',
            'ITGA7', 'ITGA8', 'ITGA9', 'ITGA10', 'ITGA11',
            'ITGAV', 'ITGAD', 'ITGAE', 'ITGAL', 'ITGAM', 'ITGAX',
            'ITGB1', 'ITGB2', 'ITGB3', 'ITGB4', 'ITGB5', 'ITGB6',
            'ITGB7', 'ITGB8',
        ],
    },

    # -------------------------------------------------------------------------
    # TIER 3: Lipid-mediated mechanotransduction (force-from-lipids)
    # -------------------------------------------------------------------------
    'Phospholipase': {
        'tier': 3,
        'sources': [
            'Petersen et al. 2016 Nat Commun PMID:27976674',
        ],
        'rationale': 'PLD2 is a membrane tension mechanosensor upstream of TREK-1 in neurons',
        # HGNC: PLD family
        'hgnc_members': ['PLD1', 'PLD2', 'PLD3', 'PLD4', 'PLD5', 'PLD6'],
    },
    'Caveolin': {
        'tier': 3,
        'sources': [
            'Head, Patel & Insel 2014 BBA Biomembranes PMID:23899502',
        ],
        'rationale': 'Caveolae are mechanosensitive lipid raft invaginations that buffer membrane tension and regulate Piezo1/ion channel activity',
        # HGNC: Caveolin family
        'hgnc_members': ['CAV1', 'CAV2', 'CAV3'],
    },
}

# =============================================================================
# STEP 2-3: CHECK ATLAS AVAILABILITY
# =============================================================================

def check_atlas_availability(expression_path):
    """Check which HGNC family members are in the Allen Brain Atlas."""
    # Read just the header to get available genes
    df = pd.read_csv(expression_path, nrows=0)
    atlas_genes = set(df.columns) - {'label'}

    results = {}
    total_hgnc = 0
    total_available = 0

    print("=" * 70)
    print("STEP 2-3: HGNC Family Enumeration & Atlas Availability")
    print("=" * 70)

    for family, info in FAMILIES.items():
        members = info['hgnc_members']
        available = [g for g in members if g in atlas_genes]
        missing = [g for g in members if g not in atlas_genes]

        total_hgnc += len(members)
        total_available += len(available)

        results[family] = {
            'tier': info['tier'],
            'hgnc_total': len(members),
            'atlas_available': len(available),
            'available_genes': available,
            'missing_genes': missing,
            'sources': info['sources'],
            'rationale': info['rationale'],
        }

        print(f"\n{family} (Tier {info['tier']}):")
        print(f"  HGNC members: {len(members)}")
        print(f"  Atlas available: {len(available)} ({', '.join(available)})")
        if missing:
            print(f"  Missing from atlas: {len(missing)} ({', '.join(missing)})")

    print(f"\n{'=' * 70}")
    print(f"TOTAL: {total_available}/{total_hgnc} HGNC members available in atlas")
    print(f"{'=' * 70}")

    return results, atlas_genes


# =============================================================================
# STEP 4: PUBMED LITERATURE VALIDATION
# =============================================================================

def validate_with_existing_pubmed(results, existing_ranking_path=None):
    """
    Cross-reference atlas-available genes with existing PubMed search results.

    Uses the gene_ranking.csv from the previous systematic review as a proxy
    for literature support. Genes found in the previous review already have
    validated article counts.
    """
    print("\n" + "=" * 70)
    print("STEP 4: PubMed Literature Validation")
    print("=" * 70)

    # Load existing ranking if available
    existing_genes = {}
    if existing_ranking_path and os.path.exists(existing_ranking_path):
        ranking_df = pd.read_csv(existing_ranking_path)
        for _, row in ranking_df.iterrows():
            existing_genes[row['gene'].upper()] = int(row['article_count'])
        print(f"  Loaded {len(existing_genes)} genes from existing PubMed search")

    validated = []
    not_in_pubmed = []

    for family, info in results.items():
        for gene in info['available_genes']:
            article_count = existing_genes.get(gene.upper(), 0)
            entry = {
                'gene': gene,
                'family': family,
                'tier': info['tier'],
                'article_count': article_count,
                'in_atlas': True,
            }
            if article_count > 0:
                validated.append(entry)
            else:
                not_in_pubmed.append(entry)

    validated_df = pd.DataFrame(validated).sort_values(
        ['tier', 'family', 'article_count'], ascending=[True, True, False]
    )

    not_found_df = pd.DataFrame(not_in_pubmed).sort_values(['family', 'gene'])

    print(f"\n  Genes with PubMed support: {len(validated)}")
    print(f"  Genes without PubMed hits: {len(not_in_pubmed)}")

    return validated_df, not_found_df


# =============================================================================
# STEP 5: APPLY INCLUSION THRESHOLD
# =============================================================================

def apply_threshold(validated_df, min_articles=5):
    """Apply article count threshold for final inclusion."""
    print(f"\n{'=' * 70}")
    print(f"STEP 5: Apply Inclusion Threshold (>= {min_articles} articles)")
    print(f"{'=' * 70}")

    above = validated_df[validated_df['article_count'] >= min_articles].copy()
    below = validated_df[validated_df['article_count'] < min_articles].copy()

    print(f"\n  Above threshold: {len(above)} genes")
    print(f"  Below threshold: {len(below)} genes")

    if not below.empty:
        print(f"\n  Genes below threshold (excluded):")
        for _, row in below.iterrows():
            print(f"    {row['gene']} ({row['family']}): {row['article_count']} articles")

    return above, below


# =============================================================================
# STEP 5B: MANUAL INCLUSIONS (bypass threshold with documented rationale)
# =============================================================================

MANUAL_INCLUSIONS = {
    'CAV1': 'Caveolin-1: caveolae mechanosensitive membrane scaffolding (Head et al. 2014 PMID:23899502)',
    'CAV2': 'Caveolin-2: co-expressed with CAV1 in caveolae (Head et al. 2014 PMID:23899502)',
}


def apply_manual_inclusions(above_df, below_df, not_found_df):
    """Add manually included genes that fall below threshold but have strong rationale."""
    print(f"\n{'=' * 70}")
    print("STEP 5B: Manual Inclusions (threshold bypass with documented rationale)")
    print(f"{'=' * 70}")

    already_included = set(above_df['gene'])
    to_add = []

    for gene, reason in MANUAL_INCLUSIONS.items():
        if gene in already_included:
            continue
        # Check below-threshold list first
        match = below_df[below_df['gene'] == gene]
        if not match.empty:
            to_add.append(match.iloc[0].to_dict())
            print(f"    {gene} ({match.iloc[0]['family']}): {match.iloc[0]['article_count']} articles - {reason}")
        else:
            # Check not-found list (0 articles)
            match = not_found_df[not_found_df['gene'] == gene]
            if not match.empty:
                to_add.append(match.iloc[0].to_dict())
                print(f"    {gene} ({match.iloc[0]['family']}): 0 articles - {reason}")

    if to_add:
        print(f"\n  Included {len(to_add)} genes below threshold:")
        above_df = pd.concat([above_df, pd.DataFrame(to_add)], ignore_index=True)

    print(f"\n  Gene count after inclusions: {len(above_df)}")
    return above_df


# =============================================================================
# STEP 6: EXPERT EXCLUSIONS
# =============================================================================

# Documented exclusions with rationale
EXPERT_EXCLUSIONS = {
    'TRPC2': 'Pseudogene in humans (non-functional)',
    'PLD3': 'Misnamed gene; functions as exonuclease, not phospholipase (Munck et al. 2020)',
    'SCN9A': 'Pain-specific Nav channel; not primary mechanosensor in brain',
    'CACNA1A': 'P/Q-type Cav; not T-type, not mechanosensitive',
    'CACNA1B': 'N-type Cav; not T-type, not mechanosensitive',
    'CACNA1C': 'L-type Cav; not T-type, not mechanosensitive',
    'CACNA1D': 'L-type Cav; not T-type, not mechanosensitive',
    'CACNA1E': 'R-type Cav; not T-type, not mechanosensitive',
    'CACNA1F': 'L-type Cav; not T-type, not mechanosensitive',
    'CACNA1S': 'L-type Cav; not T-type, not mechanosensitive',
}


def apply_expert_exclusions(df):
    """Apply documented expert exclusions."""
    print(f"\n{'=' * 70}")
    print("STEP 6: Expert Exclusions (documented rationale)")
    print(f"{'=' * 70}")

    excluded = df[df['gene'].isin(EXPERT_EXCLUSIONS)].copy()
    kept = df[~df['gene'].isin(EXPERT_EXCLUSIONS)].copy()

    if not excluded.empty:
        print(f"\n  Excluded {len(excluded)} genes:")
        for _, row in excluded.iterrows():
            reason = EXPERT_EXCLUSIONS.get(row['gene'], 'Unknown')
            print(f"    {row['gene']} ({row['family']}): {reason}")

    print(f"\n  Final gene count: {len(kept)}")
    return kept, excluded


# =============================================================================
# COMPARISON WITH ORIGINAL LIST
# =============================================================================

def compare_with_original(final_df):
    """Compare new gene list with original 35 TUS genes."""
    print(f"\n{'=' * 70}")
    print("COMPARISON: New vs Original Gene List")
    print(f"{'=' * 70}")

    original = set(config.TUS_GENES)
    new = set(final_df['gene'].tolist())

    in_both = original & new
    only_original = original - new
    only_new = new - original

    print(f"\n  Original list: {len(original)} genes")
    print(f"  New list: {len(new)} genes")
    print(f"  In both: {len(in_both)}")

    if only_original:
        print(f"\n  In ORIGINAL but NOT in new ({len(only_original)}):")
        for g in sorted(only_original):
            cat = config.GENE_CATEGORIES.get(g, '?')
            print(f"    {g} ({cat})")

    if only_new:
        print(f"\n  In NEW but NOT in original ({len(only_new)}):")
        for _, row in final_df[final_df['gene'].isin(only_new)].sort_values('family').iterrows():
            print(f"    {row['gene']} ({row['family']}, {row['article_count']} articles)")

    return in_both, only_original, only_new


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Fix Windows encoding
    import sys as _sys
    if _sys.platform == 'win32':
        _sys.stdout.reconfigure(encoding='utf-8')

    print("=" * 70)
    print("SYSTEMATIC GENE SELECTION FOR TUS ANALYSIS")
    print("Review-based -> HGNC families -> Atlas filter -> PubMed validation")
    print("=" * 70)

    # Step 1 is implicit in FAMILIES dict above

    # Steps 2-3: Check atlas availability
    results, atlas_genes = check_atlas_availability(config.EXPRESSION_PATH)

    # Step 4: PubMed validation
    ranking_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'pubMed', 'systematic_review_results', 'gene_ranking.csv'
    )
    validated_df, not_found_df = validate_with_existing_pubmed(results, ranking_path)

    # Step 5: Apply threshold
    above_threshold, below_threshold = apply_threshold(validated_df, min_articles=10)

    # Step 6: Expert exclusions
    final_df, excluded_df = apply_expert_exclusions(above_threshold)

    # Summary table
    print(f"\n{'=' * 70}")
    print("FINAL GENE LIST")
    print(f"{'=' * 70}")
    print(f"\n{'Gene':<12} {'Family':<14} {'Tier':<6} {'Articles'}")
    print("-" * 45)
    for _, row in final_df.sort_values(['tier', 'family', 'gene']).iterrows():
        print(f"{row['gene']:<12} {row['family']:<14} {row['tier']:<6} {row['article_count']}")

    # Comparison
    in_both, only_original, only_new = compare_with_original(final_df)

    # Save results
    output_dir = os.path.join(config.OUTPUT_DIR, 'tables')
    os.makedirs(output_dir, exist_ok=True)

    final_df.to_csv(os.path.join(output_dir, 'systematic_gene_selection.csv'), index=False)
    print(f"\nResults saved to {output_dir}/systematic_gene_selection.csv")

    # Save detailed methodology report
    report_path = os.path.join(output_dir, 'gene_selection_methodology.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("SYSTEMATIC GENE SELECTION METHODOLOGY\n")
        f.write("=" * 70 + "\n\n")

        f.write("STEP 1: Family definitions from independent reviews\n")
        f.write("-" * 50 + "\n")
        for family, info in FAMILIES.items():
            f.write(f"\n{family} (Tier {info['tier']}):\n")
            f.write(f"  Rationale: {info['rationale']}\n")
            f.write(f"  Sources:\n")
            for src in info['sources']:
                f.write(f"    - {src}\n")
            f.write(f"  HGNC members: {len(info['hgnc_members'])}\n")

        f.write(f"\n\nSTEP 2-3: Atlas availability\n")
        f.write("-" * 50 + "\n")
        for family, info in results.items():
            f.write(f"  {family}: {info['atlas_available']}/{info['hgnc_total']} in atlas\n")

        f.write(f"\n\nSTEP 5: Threshold >= 10 articles\n")
        f.write("-" * 50 + "\n")
        f.write(f"  Above: {len(above_threshold)}, Below: {len(below_threshold)}\n")

        f.write(f"\n\nSTEP 6: Expert exclusions\n")
        f.write("-" * 50 + "\n")
        for gene, reason in EXPERT_EXCLUSIONS.items():
            f.write(f"  {gene}: {reason}\n")

        f.write(f"\n\nFINAL: {len(final_df)} genes\n")

    print(f"Methodology report saved to {report_path}")

    # Return the final gene list for downstream use
    return final_df['gene'].tolist()


if __name__ == '__main__':
    final_genes = main()
