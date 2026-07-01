"""
Systematic Literature Review for Mechanosensitive Gene Identification
=====================================================================

This script implements a systematic, reproducible methodology for identifying
mechanosensitive genes expressed in the nervous system through comprehensive
PubMed literature review.

Methodology:
-----------
1. Query PubMed using predefined search terms covering known mechanotransduction pathways
2. Extract gene symbols from article titles and abstracts
3. Rank genes by literature frequency (number of articles mentioning each gene)
4. Categorize genes by mechanosensitive family
5. Output ranked gene list for manual expert review

The search strategy covers established mechanotransduction pathway categories:
- Mechanosensitive ion channels (Piezo, TRP, K2P, ASIC)
- Voltage-gated channels with mechanical modulation (Cav, Nav)
- Integrin-mediated mechanotransduction
- Gap junction mechanosensitivity (connexins, pannexins)
- Lipid-mediated mechanotransduction (phospholipases)
- Other mechanosensitive signaling (natriuretic peptides, chloride channels)

Usage:
    python systematic_gene_identification.py --email your.email@example.com

Output:
    - systematic_review_results/gene_ranking.csv: Ranked genes with article counts
    - systematic_review_results/search_log.csv: Search queries and results
    - systematic_review_results/methodology_report.txt: Methods description for paper
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from typing import Optional

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pubMed.literature_search import PubMedSearcher
from pubMed.extract_genes import extract_genes_from_text, KNOWN_GENES


# =============================================================================
# SYSTEMATIC SEARCH QUERIES
# =============================================================================
# Search queries organized by mechanotransduction pathway category.
# Each category targets a distinct mechanism of cellular mechanosensitivity.
#
# Query design principles:
# 1. Cover established mechanotransduction pathway categories
# 2. Include neural context (brain, neuron, glia, nervous system)
# 3. Use standard nomenclature from ion channel and mechanobiology literature

SYSTEMATIC_QUERIES = {
    # =========================================================================
    # PRIMARY MECHANOSENSITIVE MECHANISMS
    # =========================================================================
    "Mechanosensitive ion channels": [
        'mechanosensitive channel AND (brain OR neuron OR nervous system)',
        'mechanosensitive ion channel AND neural',
        '"stretch-activated channel" AND (brain OR neuron)',
        '"stretch-activated" AND "ion channel" AND neural',
        'mechanogated channel AND (brain OR neuron)',
        'mechanosensitive channel AND (glia OR astrocyte OR neuromodulation)',
    ],

    "Mechanotransduction signaling": [
        'mechanotransduction AND (channel OR receptor) AND (brain OR neuron)',
        'mechanosensor AND (channel OR ion)',
        '"mechanical force" AND "ion channel" AND neuron',
        'mechanosensitive AND signaling AND (brain OR neural)',
        '"force transduction" AND channel AND neuron',
        'mechanotransduction AND (channel OR receptor) AND (glia OR neuromodulation)',
    ],

    # =========================================================================
    # POTASSIUM CHANNELS
    # =========================================================================
    "Two-pore potassium channels (K2P)": [
        '"two-pore domain" potassium AND (brain OR neuron)',
        '"two-pore" potassium channel AND nervous system',
        '"leak potassium channel" AND (brain OR neuron)',
        '"background potassium channel" AND (brain OR neural)',
        '"tandem pore" potassium AND brain',
        # TREK/TRAAK/TASK subfamily terms
        '(TREK OR TRAAK) AND channel AND (brain OR neuron)',
        'TASK channel AND (brain OR neuron OR pH)',
        '"acid-sensitive" potassium AND brain',
        # K2P and mechanosensitivity
        '"two-pore" AND mechanosensitive',
        'K2P channel AND (brain OR mechanosensitive)',
        '(TREK OR TRAAK OR TASK) AND channel AND (glia OR astrocyte)',
    ],

    "Inward rectifier potassium channels": [
        '"inward rectifier" potassium AND brain',
        '"inwardly rectifying" AND channel AND (glia OR astrocyte)',
        '"inward rectifier" AND (neuron OR neural)',
        'Kir channel AND (brain OR glia)',
        '"potassium spatial buffering" AND brain',
        # Kir4.1 specific context
        '"glial potassium" AND channel',
        'astrocyte AND "potassium channel" AND brain',
    ],

    # =========================================================================
    # VOLTAGE-GATED CHANNELS
    # =========================================================================
    "Voltage-gated calcium channels": [
        '"T-type calcium" AND mechanosensitive',
        '"T-type calcium" AND (brain OR neuron)',
        '"voltage-gated calcium" AND mechanical AND brain',
        '"low-threshold calcium" AND (neuron OR thalamus)',
        '"low-voltage activated" calcium AND brain',
        # Cav3 family context
        'Cav3 AND (brain OR neuron OR thalamus)',
        '"T-type" AND "calcium channel" AND neural',
        '"T-type calcium" AND (glia OR astrocyte)',
    ],

    "Voltage-gated sodium channels": [
        '"voltage-gated sodium" AND mechanosensitive',
        '"voltage-gated sodium" AND mechanical AND neuron',
        '"voltage-gated sodium" AND (brain OR cortex) AND channel',
        'Nav channel AND (brain OR neuron)',
        '"sodium channel" AND "action potential" AND mechanosensitive',
        # Nav1.2 context
        '"neuronal sodium channel" AND brain',
        '"voltage-gated sodium" AND (glia OR neuromodulation)',
    ],

    # =========================================================================
    # ACID-SENSING AND TRP CHANNELS
    # =========================================================================
    "Acid-sensing ion channels": [
        '"acid-sensing ion channel" AND (brain OR neuron)',
        'ASIC AND mechanosensitive AND neural',
        'ASIC AND (brain OR neuron OR pain)',
        '"proton-gated" AND channel AND (brain OR neuron)',
        '"acid-sensing" AND mechanotransduction',
        'ASIC AND (glia OR astrocyte) AND channel',
    ],

    "TRP channels": [
        '"transient receptor potential" AND mechanosensitive AND brain',
        'TRP channel AND mechanical AND neuron',
        '"transient receptor potential" AND (brain OR neuron) AND channel',
        # TRPV subfamily
        'TRPV AND mechanosensitive',
        'TRPV AND (brain OR neuron OR sensory)',
        # Polycystin channels (PKD1/PKD2)
        'polycystin AND (mechanosensitive OR channel OR cilia)',
        'PKD AND channel AND (brain OR neuron OR cilia)',
        '"primary cilium" AND mechanosensitive AND channel',
        'TRP channel AND (glia OR astrocyte OR neuromodulation)',
    ],

    # =========================================================================
    # CELL ADHESION AND ECM MECHANOTRANSDUCTION
    # =========================================================================
    "Integrin mechanotransduction": [
        'integrin AND mechanotransduction AND (brain OR neuron)',
        'integrin AND "cell adhesion" AND brain',
        '"focal adhesion" AND mechanosensitive AND neural',
        # Broader integrin neural context
        'integrin AND (neuron OR glia OR brain) AND signaling',
        '"integrin alpha" AND (brain OR neuron)',
        '"integrin beta" AND (brain OR neuron)',
        # ECM mechanotransduction
        '"extracellular matrix" AND mechanotransduction AND (brain OR neural)',
        'integrin AND "mechanical force" AND (neuron OR brain)',
        'integrin AND (glia OR astrocyte) AND mechanotransduction',
    ],

    # =========================================================================
    # GAP JUNCTIONS AND CONNEXINS
    # =========================================================================
    "Gap junction mechanosensitivity": [
        '(connexin OR "gap junction") AND mechanosensitive',
        'connexin AND mechanical AND brain',
        '"gap junction" AND (brain OR glia OR astrocyte)',
        'connexin AND (astrocyte OR glia) AND brain',
        # Hemichannel context
        'hemichannel AND (brain OR glia)',
        'pannexin AND (brain OR neuron)',
        # Connexin30/43 context
        '"connexin 30" OR "connexin 43" AND brain',
    ],

    # =========================================================================
    # MEMBRANE AND LIPID-MEDIATED MECHANOTRANSDUCTION
    # =========================================================================
    "Lipid-mediated mechanotransduction": [
        'phospholipase AND mechanosensor',
        'phospholipase D AND (brain OR neuron)',
        '"membrane tension" AND signaling AND (brain OR neuron)',
        '"lipid raft" AND mechanotransduction',
        '"membrane curvature" AND signaling AND (brain OR neuron)',
        # PLD specific context
        '"phospholipase D" AND membrane AND (brain OR neural)',
        '"lipid signaling" AND mechanosensitive',
        # Caveolin / caveolae mechanosensitivity
        'caveolin AND (brain OR neuron OR glia)',
        'caveolae AND (mechanosensitive OR mechanotransduction)',
        '"lipid raft" AND (brain OR neuron OR glia) AND channel',
        'caveolin AND "ion channel"',
        '"membrane tension" AND signaling AND (glia OR neuromodulation)',
    ],

    # =========================================================================
    # CHLORIDE AND ANION CHANNELS
    # =========================================================================
    "Chloride and anion channels": [
        '"chloride channel" AND mechanosensitive AND brain',
        '(bestrophin OR "volume-regulated") AND channel AND brain',
        '"calcium-activated chloride" AND (brain OR neuron)',
        '"volume-regulated anion" AND (brain OR glia)',
        'VRAC AND (brain OR astrocyte)',
        # Bestrophin context
        'bestrophin AND (brain OR glia OR retina)',
        '"anion channel" AND mechanosensitive',
        '"volume-regulated" AND channel AND (glia OR astrocyte OR neuromodulation)',
    ],

    # =========================================================================
    # NATRIURETIC PEPTIDE SIGNALING
    # =========================================================================
    "Natriuretic peptide signaling": [
        '"natriuretic peptide receptor" AND brain',
        'ANP receptor AND (neuron OR neural)',
        '"natriuretic peptide" AND (brain OR neuron) AND receptor',
        'NPR AND (brain OR neuron) AND receptor',
        # Pressure/volume sensing context
        '"guanylyl cyclase" receptor AND brain',
        '"natriuretic peptide" AND (glia OR neuromodulation)',
    ],

    # =========================================================================
    # ULTRASOUND NEUROMODULATION CONTEXT
    # =========================================================================
    "Ultrasound neuromodulation mechanisms": [
        'ultrasound AND mechanosensitive AND (channel OR neuron)',
        '"focused ultrasound" AND "ion channel"',
        'ultrasound AND neuromodulation AND (channel OR mechanism)',
        '"transcranial ultrasound" AND (channel OR mechanosensitive)',
        'sonication AND (neuron OR channel) AND mechanism',
        'ultrasound AND (glia OR astrocyte) AND (channel OR mechanism)',
    ],
}

# Gene families for categorization
GENE_FAMILY_PATTERNS = {
    'Piezo': r'^PIEZO\d$',
    'TRP': r'^(TRPV|TRPC|TRPM|TRPA|TRPP|PKD)\d',
    'K2P': r'^KCNK\d+$',
    'Kir': r'^KCNJ\d+$',
    'Kv': r'^(KCNA|KCNB|KCNC|KCND|KCNQ)\d',
    'Nav': r'^SCN\d+A$',
    'Cav': r'^CACNA\d[A-Z]$',
    'ASIC': r'^ASIC\d',
    'Integrin': r'^ITG[AB]\d',
    'Connexin': r'^(GJ[AB]\d|CX\d|PANX)',
    'Chloride': r'^(BEST\d|CLCN|ANO|LRRC8)',
    'Phospholipase': r'^PL[ACD]\d',
    'Caveolin': r'^CAV[IN]*\d',
    'Other': r'.*',  # Catch-all
}


# =============================================================================
# GENE EXTRACTION AND RANKING
# =============================================================================

def categorize_gene(gene: str) -> str:
    """Assign a gene to its mechanosensitive family category."""
    gene_upper = gene.upper()
    for family, pattern in GENE_FAMILY_PATTERNS.items():
        if re.match(pattern, gene_upper):
            return family
    return 'Other'


def is_likely_gene(symbol: str) -> bool:
    """
    Filter to identify likely gene symbols vs. acronyms/abbreviations.

    Inclusion criteria:
    - Known gene symbols from curated list
    - Matches common gene naming patterns (letters + numbers)
    - Not a common non-gene abbreviation
    """
    # Known genes always included
    if symbol.upper() in KNOWN_GENES:
        return True

    # Common non-gene abbreviations to exclude
    EXCLUDE = {
        'MRI', 'FMRI', 'EEG', 'MEG', 'PET', 'CT', 'US', 'TUS', 'FUS', 'LIFU', 'HIFU',
        'CNS', 'PNS', 'BBB', 'CSF', 'ECG', 'EMG', 'LFP', 'ERP',
        'DNA', 'RNA', 'ATP', 'ADP', 'GTP', 'GDP', 'AMP', 'CAMP',
        'FDA', 'NIH', 'WHO', 'IRB', 'USA', 'UK',
        'PCR', 'ELISA', 'WB', 'IHC', 'ICC', 'FISH',
        'SEM', 'SD', 'SE', 'CI', 'OR', 'HR', 'RR', 'NS', 'NA',
        'ANOVA', 'ROC', 'AUC', 'ROI', 'VOI',
        'WT', 'KO', 'KI', 'HET', 'TG', 'AAV', 'GFP', 'RFP', 'YFP',
        'LTP', 'LTD', 'EPSP', 'IPSP', 'EPSC', 'IPSC',
        'GABA', 'NMDA', 'AMPA', 'BOLD',
        'PRF', 'PRR', 'DC', 'AC', 'Hz', 'KHz', 'MHz',
        'MIN', 'MAX', 'AVG', 'SUM', 'DIV', 'MUL',
        'MSC', 'ESC', 'IPSC', 'NSC',  # Stem cell abbreviations
        'DEG', 'GO', 'KEGG',  # Bioinformatics terms
    }

    if symbol.upper() in EXCLUDE:
        return False

    # Must be 2-10 characters
    if len(symbol) < 2 or len(symbol) > 10:
        return False

    # Should match gene-like pattern: letters optionally followed by numbers
    if not re.match(r'^[A-Z]{2,6}\d{0,3}[A-Z]?$', symbol.upper()):
        return False

    return True


class SystematicReview:
    """Conducts systematic literature review for mechanosensitive genes."""

    def __init__(self, email: str, api_key: Optional[str] = None,
                 output_dir: str = None, max_articles_per_query: int = 500):
        self.searcher = PubMedSearcher(email=email, api_key=api_key)
        self.max_articles = max_articles_per_query

        # Results storage
        self.gene_articles = defaultdict(set)  # gene -> set of PMIDs
        self.gene_mentions = Counter()  # gene -> total mention count
        self.gene_categories = defaultdict(set)  # gene -> set of query categories
        self.search_log = []  # Record of all searches
        self.all_pmids = set()

        # Output directory
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(__file__), 'systematic_review_results')
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.start_time = None

    def run(self):
        """Execute the systematic review."""
        self.start_time = datetime.now()

        print("=" * 70)
        print("Systematic Literature Review for TUS-Relevant Genes")
        print("=" * 70)
        print(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Output directory: {self.output_dir}")
        print()

        # Run all systematic queries
        total_queries = sum(len(queries) for queries in SYSTEMATIC_QUERIES.values())
        query_num = 0

        for category, queries in SYSTEMATIC_QUERIES.items():
            print(f"\n{category}")
            print("-" * 50)

            for query in queries:
                query_num += 1
                print(f"  [{query_num}/{total_queries}] {query[:60]}...")

                try:
                    self._process_query(query, category)
                except Exception as e:
                    print(f"    Error: {e}")
                    self.search_log.append({
                        'category': category,
                        'query': query,
                        'n_results': 0,
                        'n_articles_processed': 0,
                        'error': str(e),
                    })

        # Generate outputs
        print("\n" + "=" * 70)
        print("Generating results...")
        self._save_results()
        self._generate_report()

        elapsed = datetime.now() - self.start_time
        print(f"\nCompleted in {elapsed}")
        print(f"Results saved to: {self.output_dir}")

    def _process_query(self, query: str, category: str):
        """Process a single search query."""
        # Search PubMed
        pmids = self.searcher.search(query, max_results=self.max_articles)

        n_results = len(pmids)
        n_new = len(set(pmids) - self.all_pmids)

        if not pmids:
            self.search_log.append({
                'category': category,
                'query': query,
                'n_results': 0,
                'n_articles_processed': 0,
                'n_new_articles': 0,
                'genes_found': '',
            })
            print(f"    No results")
            return

        # Fetch article details
        articles = self.searcher.fetch_details(pmids)

        # Extract genes
        genes_this_query = set()
        for article in articles:
            pmid = article.get('pmid', '')
            text = f"{article.get('title', '')} {article.get('abstract', '')}"

            # Extract genes using existing extractor
            extracted = extract_genes_from_text(text)

            for found_term, canonical in extracted:
                gene = canonical.upper()

                # Apply filtering
                if not is_likely_gene(gene):
                    continue

                self.gene_articles[gene].add(pmid)
                self.gene_mentions[gene] += 1
                self.gene_categories[gene].add(category)
                genes_this_query.add(gene)

        self.all_pmids.update(pmids)

        # Log search
        self.search_log.append({
            'category': category,
            'query': query,
            'n_results': n_results,
            'n_articles_processed': len(articles),
            'n_new_articles': n_new,
            'genes_found': ','.join(sorted(genes_this_query)[:20]),
        })

        print(f"    {len(articles)} articles, {len(genes_this_query)} genes")

    def _save_results(self):
        """Save all results to files."""

        # 1. Gene ranking CSV
        ranking_path = os.path.join(self.output_dir, 'gene_ranking.csv')

        # Sort by article count (primary) and mention count (secondary)
        ranked_genes = sorted(
            self.gene_articles.keys(),
            key=lambda g: (len(self.gene_articles[g]), self.gene_mentions[g]),
            reverse=True
        )

        with open(ranking_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'rank', 'gene', 'family', 'article_count', 'mention_count',
                'categories', 'category_count'
            ])

            for rank, gene in enumerate(ranked_genes, 1):
                categories = sorted(self.gene_categories[gene])
                writer.writerow([
                    rank,
                    gene,
                    categorize_gene(gene),
                    len(self.gene_articles[gene]),
                    self.gene_mentions[gene],
                    '; '.join(categories),
                    len(categories),
                ])

        print(f"  - gene_ranking.csv ({len(ranked_genes)} genes)")

        # 2. Search log CSV
        log_path = os.path.join(self.output_dir, 'search_log.csv')
        with open(log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'category', 'query', 'n_results', 'n_articles_processed',
                'n_new_articles', 'genes_found'
            ])
            writer.writeheader()
            for row in self.search_log:
                # Only include expected fields
                clean_row = {k: row.get(k, '') for k in writer.fieldnames}
                writer.writerow(clean_row)

        print(f"  - search_log.csv ({len(self.search_log)} queries)")

        # 3. Top genes for quick reference
        top_path = os.path.join(self.output_dir, 'top_genes.txt')
        with open(top_path, 'w', encoding='utf-8') as f:
            f.write("Top Mechanosensitive Genes by Literature Frequency\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"{'Rank':<6}{'Gene':<12}{'Family':<12}{'Articles':<10}{'Categories'}\n")
            f.write("-" * 60 + "\n")

            for rank, gene in enumerate(ranked_genes[:50], 1):
                family = categorize_gene(gene)
                n_art = len(self.gene_articles[gene])
                n_cat = len(self.gene_categories[gene])
                f.write(f"{rank:<6}{gene:<12}{family:<12}{n_art:<10}{n_cat}\n")

        print(f"  - top_genes.txt (top 50)")

    def _generate_report(self):
        """Generate methodology report for paper methods section."""
        report_path = os.path.join(self.output_dir, 'methodology_report.txt')

        elapsed = datetime.now() - self.start_time
        n_queries = len(self.search_log)
        n_categories = len(SYSTEMATIC_QUERIES)
        n_articles = len(self.all_pmids)
        n_genes = len(self.gene_articles)

        # Count genes by family
        family_counts = Counter(categorize_gene(g) for g in self.gene_articles.keys())

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("SYSTEMATIC LITERATURE REVIEW METHODOLOGY\n")
            f.write("Identification of Neural Mechanosensitive Genes\n")
            f.write("=" * 70 + "\n\n")

            f.write("METHODS SECTION TEXT (for paper):\n")
            f.write("-" * 70 + "\n\n")

            methods_text = f"""
Mechanosensitive gene identification was performed through systematic literature
review of PubMed. We searched {n_categories} mechanotransduction pathway categories
using {n_queries} predefined queries targeting: mechanosensitive ion channels,
two-pore potassium (K2P) channels, inward rectifier potassium channels, voltage-gated
calcium and sodium channels, acid-sensing ion channels (ASICs), TRP channels,
integrin-mediated mechanotransduction, gap junctions, lipid-mediated signaling,
and natriuretic peptide receptors.

Searches were conducted on {self.start_time.strftime('%Y-%m-%d')} using the NCBI
E-utilities API. Gene symbols were extracted from article titles and abstracts
using pattern matching against known gene nomenclature. A total of {n_articles:,}
unique articles were screened, identifying {n_genes} candidate genes.

Genes were ranked by literature frequency (number of articles mentioning each gene)
and categorized by mechanosensitive family. The final gene list was manually reviewed
by domain experts to exclude false positives (e.g., genes mentioned in methodological
context rather than as mechanosensors) and to verify mechanosensitive function based
on primary literature.

Inclusion criteria for the final gene list:
1. Mentioned in ≥3 articles within mechanosensitive literature
2. Evidence of mechanosensitive or stretch-activated function
3. Expression in neural tissue (brain, neurons, or glia)
4. Not primarily a methodological marker or housekeeping gene
"""
            f.write(methods_text.strip() + "\n\n")

            f.write("-" * 70 + "\n")
            f.write("SEARCH CATEGORIES AND QUERIES:\n")
            f.write("-" * 70 + "\n\n")

            for category, queries in SYSTEMATIC_QUERIES.items():
                f.write(f"{category}:\n")
                for q in queries:
                    f.write(f"  - {q}\n")
                f.write("\n")

            f.write("-" * 70 + "\n")
            f.write("SUMMARY STATISTICS:\n")
            f.write("-" * 70 + "\n\n")

            f.write(f"Date of search: {self.start_time.strftime('%Y-%m-%d')}\n")
            f.write(f"Total search queries: {n_queries}\n")
            f.write(f"Total unique articles screened: {n_articles:,}\n")
            f.write(f"Total candidate genes identified: {n_genes}\n")
            f.write(f"Processing time: {elapsed}\n\n")

            f.write("Genes identified by family:\n")
            for family, count in sorted(family_counts.items(), key=lambda x: -x[1]):
                f.write(f"  {family}: {count}\n")

            f.write("\n" + "=" * 70 + "\n")

        print(f"  - methodology_report.txt")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Systematic literature review for neural mechanosensitive genes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script performs a systematic, reproducible literature review to identify
genes involved in mechanotransduction in the nervous system.

Output files:
  - gene_ranking.csv: All identified genes ranked by literature frequency
  - search_log.csv: Record of all search queries and results
  - top_genes.txt: Quick reference list of top 50 genes
  - methodology_report.txt: Methods text for paper + search details
        """
    )

    parser.add_argument('--email', required=True,
                       help='Email address (required by NCBI)')
    parser.add_argument('--api-key', dest='api_key', default=None,
                       help='NCBI API key for faster searches (optional)')
    parser.add_argument('--max-articles', type=int, default=500, dest='max_articles',
                       help='Max articles to fetch per query (default: 500)')
    parser.add_argument('--output-dir', dest='output_dir', default=None,
                       help='Output directory (default: pubMed/systematic_review_results)')

    args = parser.parse_args()

    review = SystematicReview(
        email=args.email,
        api_key=args.api_key,
        output_dir=args.output_dir,
        max_articles_per_query=args.max_articles,
    )

    review.run()


if __name__ == '__main__':
    main()
