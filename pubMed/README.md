# PubMed Systematic Literature Review

Systematic literature review methodology for identifying mechanosensitive genes expressed in the nervous system.

## Overview

This module queries PubMed across 14 mechanotransduction pathway categories using 92 predefined search terms, extracts gene symbols from article titles and abstracts, and ranks genes by literature frequency.

## Usage

```bash
python systematic_gene_identification.py --email your.email@example.com
```

### Options

- `--email` (required): Email address for NCBI API
- `--api-key`: NCBI API key for faster searches (optional)
- `--max-articles`: Maximum articles per query (default: 200)
- `--output-dir`: Output directory (default: systematic_review_results/)

## Output Files

Results are saved to `systematic_review_results/`:

| File | Description |
|------|-------------|
| `gene_ranking.csv` | All genes ranked by article frequency |
| `search_log.csv` | Record of all search queries and results |
| `methodology_report.txt` | Methods text suitable for publication |
| `top_genes.txt` | Quick reference list of top 50 genes |

## Files

| File | Description |
|------|-------------|
| `systematic_gene_identification.py` | Main systematic review script |
| `literature_search.py` | PubMed API wrapper (NCBI E-utilities) |
| `extract_genes.py` | Gene symbol extraction from text |
| `METHODS_OVERVIEW.txt` | Detailed methodology documentation |

## Requirements

```bash
pip install -r requirements.txt
```

## Search Categories

1. Mechanosensitive ion channels
2. Mechanotransduction signaling
3. Two-pore potassium channels (K2P)
4. Inward rectifier potassium channels
5. Voltage-gated calcium channels
6. Voltage-gated sodium channels
7. Acid-sensing ion channels (ASIC)
8. TRP channels
9. Integrin mechanotransduction
10. Gap junction mechanosensitivity
11. Lipid-mediated mechanotransduction
12. Chloride and anion channels
13. Natriuretic peptide signaling
14. Ultrasound neuromodulation mechanisms

See `METHODS_OVERVIEW.txt` for detailed rationale and references for each category.
