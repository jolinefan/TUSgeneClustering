#!/usr/bin/env python3
"""
Create supplemental flowchart figure for gene selection methodology.

Panel a: Gene selection pipeline overview (left-to-right flow)
Panel b: PubMed literature validation methodology
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
import numpy as np
from tusgene import config

config.setup_figure_style()

# Colors
TIER1 = '#D4EDDA'   # green - TUS reviews
TIER2 = '#D1ECF1'   # blue - mechanotransduction
TIER3 = '#FFF3CD'   # yellow - force-from-lipids
FLOW = '#E8F4FD'    # light blue - pipeline steps
RESULT = '#D5F5E3'  # light green - results
DARK = '#2C3E50'    # dark text/borders
GRAY = '#6C757D'
RED = '#E74C3C'
GREEN = '#27AE60'


def draw_box(ax, x, y, w, h, text, color=FLOW, edgecolor=DARK,
             fontsize=8, fontweight='normal', text_color=DARK, alpha=1.0):
    """Draw a rounded box with centered text."""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.04",
                         facecolor=color, edgecolor=edgecolor,
                         linewidth=1.0, alpha=alpha, zorder=2)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            fontweight=fontweight, color=text_color, zorder=3,
            linespacing=1.3)
    return box


def draw_arrow(ax, x1, y1, x2, y2, color=DARK):
    """Draw an arrow between two points."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=1.3, connectionstyle='arc3,rad=0'),
                zorder=1)


def create_panel_a(ax):
    """Panel a: Gene selection pipeline (left-to-right flow)."""
    ax.set_xlim(-0.5, 16.5)
    ax.set_ylim(-2.0, 5.5)
    ax.axis('off')

    # Panel label
    ax.text(-0.3, 5.3, 'a', ha='left', va='top', fontsize=16,
            fontweight='bold', color='#1A1A1A')

    # --- Row 1: Literature sources (three tier boxes) ---
    tier_y = 4.2
    tier_w = 3.8
    tier_h = 1.0

    draw_box(ax, 2.0, tier_y, tier_w, tier_h,
             'Tier 1: TUS reviews\nSong 2023, Prieto & Maduke 2024,\nRanade 2015',
             color=TIER1, fontsize=7, fontweight='bold')

    draw_box(ax, 6.2, tier_y, tier_w, tier_h,
             'Tier 2: Mechanotransduction\nLegon & Strohman 2024,\nHeppenstall & Lewin 2006, Li 2024',
             color=TIER2, fontsize=7, fontweight='bold')

    draw_box(ax, 10.4, tier_y, tier_w, tier_h,
             'Tier 3: Force-from-lipids\nPetersen 2016, Head 2014',
             color=TIER3, fontsize=7, fontweight='bold')

    # Arrows from tiers down to merge point
    merge_y = 3.0
    draw_arrow(ax, 2.0, tier_y - tier_h/2, 6.2, merge_y + 0.35)
    draw_arrow(ax, 6.2, tier_y - tier_h/2, 6.2, merge_y + 0.35)
    draw_arrow(ax, 10.4, tier_y - tier_h/2, 6.2, merge_y + 0.35)

    # Merge box: 12 families identified
    draw_box(ax, 6.2, merge_y, 4.0, 0.6,
             '12 gene families identified',
             color=FLOW, fontsize=8, fontweight='bold')

    # --- Row 2: Main pipeline (left to right) ---
    pipe_y = 1.5
    box_w = 2.4
    box_h = 1.1
    positions = [1.5, 4.5, 7.5, 10.5, 13.5]

    labels = [
        'HGNC/IUPHAR\nenumeration\nof all members',
        'Allen Human\nBrain Atlas\navailability',
        'PubMed literature\nvalidation\n(see panel b)',
        'Inclusion\nthreshold\n(\u226510 articles)',
        'Expert review\n& exclusions',
    ]

    counts_above = [
        '147 genes',
        '38 excluded',
        '18 genes:\n0 articles',
        '43 below\nthreshold',
        'PLD3, SCN9A\nexcluded',
    ]

    counts_below = [
        None,
        '109 remain',
        '91 with\nsupport',
        '48 remain',
        None,
    ]

    colors_above = [GRAY, RED, RED, RED, RED]
    colors_below = [None, GREEN, GREEN, GREEN, None]

    # Arrow from merge down then left to first pipeline box
    draw_arrow(ax, 6.2, merge_y - 0.3, 1.5, pipe_y + box_h/2 + 0.1)

    for i, (xp, lab) in enumerate(zip(positions, labels)):
        draw_box(ax, xp, pipe_y, box_w, box_h, lab,
                 color=FLOW, fontsize=7.5, fontweight='bold')

        # Annotation above
        if counts_above[i]:
            ax.text(xp, pipe_y + box_h/2 + 0.15, counts_above[i],
                    ha='center', va='bottom', fontsize=6.5,
                    color=colors_above[i], fontstyle='italic',
                    linespacing=1.1, zorder=5)

        # Annotation below
        if counts_below[i]:
            ax.text(xp, pipe_y - box_h/2 - 0.1, counts_below[i],
                    ha='center', va='top', fontsize=6.5,
                    color=colors_below[i], fontstyle='italic',
                    linespacing=1.1)

        # Arrow to next box
        if i < len(positions) - 1:
            draw_arrow(ax, xp + box_w/2 + 0.08, pipe_y,
                       positions[i+1] - box_w/2 - 0.08, pipe_y)

    # Arrow to final result
    draw_arrow(ax, positions[-1] + box_w/2 + 0.08, pipe_y,
               positions[-1] + box_w/2 + 0.7, pipe_y)

    # Final result box
    final_x = 16.0
    draw_box(ax, final_x, pipe_y, 1.2, box_h,
             '46\ngenes',
             color=DARK, fontsize=10, fontweight='bold',
             text_color='white')

    # --- Bottom: Gene family summary (compact single row) ---
    table_y = -0.7
    families = [
        ('Piezo', 2, TIER1), ('TRP', 13, TIER1), ('K2P', 6, TIER1),
        ('ASIC', 4, TIER1), ('Nav', 3, TIER2), ('Cav', 3, TIER2),
        ('Kir', 2, TIER2), ('Connexin', 4, TIER2), ('Chloride', 2, TIER2),
        ('Integrin', 3, TIER2), ('PLD', 2, TIER3), ('Caveolin', 2, TIER3),
    ]

    n_fam = len(families)
    total_w = 14.5
    cell_w = total_w / n_fam
    start_x = 1.0

    for i, (name, count, color) in enumerate(families):
        cx = start_x + i * cell_w + cell_w / 2
        rect = FancyBboxPatch((cx - cell_w/2 + 0.03, table_y - 0.3),
                              cell_w - 0.06, 0.6,
                              boxstyle="round,pad=0.03",
                              facecolor=color, edgecolor='#BDC3C7',
                              linewidth=0.5, alpha=0.8, zorder=1)
        ax.add_patch(rect)
        ax.text(cx, table_y + 0.05, name, fontsize=6.5, ha='center',
                va='center', fontweight='bold', color=DARK)
        ax.text(cx, table_y - 0.15, f'n={count}', fontsize=6, ha='center',
                va='center', color=GRAY)

    # Tier legend
    leg_y = -1.5
    for tier, label, color in [
        (1, 'Tier 1: TUS reviews', TIER1),
        (2, 'Tier 2: Mechanotransduction', TIER2),
        (3, 'Tier 3: Force-from-lipids', TIER3),
    ]:
        xp = 2.5 + (tier - 1) * 4.5
        rect = FancyBboxPatch((xp - 1.5, leg_y - 0.15), 3.0, 0.3,
                              boxstyle="round,pad=0.02",
                              facecolor=color, edgecolor='#BDC3C7',
                              linewidth=0.5, zorder=1)
        ax.add_patch(rect)
        ax.text(xp, leg_y, label, fontsize=6.5, ha='center',
                va='center', color=DARK)


def create_panel_b(ax):
    """Panel b: PubMed literature validation methodology."""
    ax.set_xlim(-0.5, 16.5)
    ax.set_ylim(-1.0, 4.5)
    ax.axis('off')

    # Panel label
    ax.text(-0.3, 4.3, 'b', ha='left', va='top', fontsize=16,
            fontweight='bold', color='#1A1A1A')

    ax.text(8.0, 4.1, 'PubMed Literature Validation Detail',
            ha='center', va='center', fontsize=12, fontweight='bold',
            color='#1A1A1A')

    # --- Horizontal pipeline flow ---
    flow_y = 2.8
    box_w = 2.4
    box_h = 0.9
    positions = [1.5, 4.5, 7.5, 10.5, 13.5]

    labels = [
        '14 Search\nCategories',
        '108 Boolean\nQueries',
        'NCBI PubMed\nE-utilities API',
        '10,893 Unique\nArticles',
        'Gene Symbol\nExtraction',
    ]

    sublabels = [
        'all mechanotrans-\nduction pathways',
        'MeSH + free text\nwith neural context',
        'ESearch + EFetch\nmax 150/query',
        'deduplicated\nacross categories',
        'from titles &\nabstracts',
    ]

    for i, (xp, lab) in enumerate(zip(positions, labels)):
        c = RESULT if i == 3 else FLOW
        draw_box(ax, xp, flow_y, box_w, box_h, lab,
                 color=c, fontsize=7.5, fontweight='bold')
        ax.text(xp, flow_y - box_h/2 - 0.15, sublabels[i],
                ha='center', va='top', fontsize=6, color=GRAY,
                linespacing=1.2)

    # Arrows between boxes
    for i in range(len(positions) - 1):
        x1 = positions[i] + box_w/2 + 0.08
        x2 = positions[i+1] - box_w/2 - 0.08
        draw_arrow(ax, x1, flow_y, x2, flow_y)

    # Arrow to output
    draw_arrow(ax, positions[-1] + box_w/2 + 0.08, flow_y,
               positions[-1] + box_w/2 + 0.7, flow_y)

    # Output box
    draw_box(ax, 16.0, flow_y, 1.2, box_h,
             'Article\ncount\nper gene',
             color=RESULT, fontsize=7, fontweight='bold')

    # --- Example query ---
    ex_y = 0.8
    ax.text(8.0, ex_y + 0.5, 'Example Query',
            ha='center', va='center', fontsize=9, fontweight='bold',
            color=DARK)

    ex_box = FancyBboxPatch((2.0, ex_y - 0.25), 12.0, 0.5,
                            boxstyle="round,pad=0.04",
                            facecolor='#F4F6F7', edgecolor='#95A5A6',
                            linewidth=0.8, zorder=2)
    ax.add_patch(ex_box)
    ax.text(8.0, ex_y,
            '"mechanosensitive channel" AND (brain OR neuron OR nervous system)',
            ha='center', va='center', fontsize=7.5, color=DARK,
            family='monospace', zorder=3)

    ax.text(8.0, ex_y - 0.5,
            'All queries include neural context terms to restrict results '
            'to neural mechanotransduction literature',
            ha='center', va='center', fontsize=6.5, color=GRAY)


def create_flowchart():
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 1, height_ratios=[7.5, 5], hspace=0.15)

    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])

    create_panel_a(ax_a)
    create_panel_b(ax_b)

    # Save
    output_dir = config.FIGURES_DIR
    os.makedirs(output_dir, exist_ok=True)

    for ext in ['pdf', 'jpg']:
        path = os.path.join(output_dir, f'supplemental_gene_selection.{ext}')
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"Flowchart saved to {output_dir}/supplemental_gene_selection.pdf/.jpg")


if __name__ == '__main__':
    create_flowchart()
