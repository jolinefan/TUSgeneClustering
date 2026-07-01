"""
Extract gene names from text using pattern matching against known gene nomenclature.
"""
from __future__ import annotations

import re

# Known gene families and patterns relevant to ultrasound neuromodulation
KNOWN_GENES = {
    # Mechanosensitive Ion Channels
    "PIEZO1", "PIEZO2",

    # TRP Channels
    "TRPV1", "TRPV2", "TRPV3", "TRPV4", "TRPA1", "TRPC1", "TRPC2", "TRPC3",
    "TRPC4", "TRPC5", "TRPC6", "TRPP1", "TRPP2", "TRPM1", "TRPM2", "TRPM3",
    "TRPM4", "TRPM5", "TRPM6", "TRPM7", "TRPM8", "TMC1", "TMC2",
    "PKD1", "PKD2",

    # Two-Pore (K2P) Channels
    "KCNK1", "KCNK2", "KCNK3", "KCNK4", "KCNK5", "KCNK6", "KCNK7", "KCNK9",
    "KCNK10", "KCNK12", "KCNK13", "KCNK15", "KCNK16", "KCNK17", "KCNK18",
    "TREK1", "TREK2", "TRAAK", "TASK1", "TASK2", "TASK3", "TRESK", "TWIK1",
    "KCNJ10",

    # Voltage-Gated Sodium Channels
    "SCN1A", "SCN2A", "SCN3A", "SCN4A", "SCN5A", "SCN8A", "SCN9A", "SCN10A", "SCN11A",
    "NAV1.1", "NAV1.2", "NAV1.3", "NAV1.4", "NAV1.5", "NAV1.6", "NAV1.7", "NAV1.8", "NAV1.9",

    # Voltage-Gated Calcium Channels
    "CACNA1A", "CACNA1B", "CACNA1C", "CACNA1D", "CACNA1E", "CACNA1F",
    "CACNA1G", "CACNA1H", "CACNA1I", "CACNA1S",
    "CAV1.1", "CAV1.2", "CAV1.3", "CAV1.4", "CAV2.1", "CAV2.2", "CAV2.3",
    "CAV3.1", "CAV3.2", "CAV3.3",

    # Voltage-Gated Potassium Channels
    "KCNA1", "KCNA2", "KCNA3", "KCNA4", "KCNA5", "KCNA6", "KCNA7",
    "KCNB1", "KCNB2", "KCNC1", "KCNC2", "KCNC3", "KCNC4",
    "KCND1", "KCND2", "KCND3", "KCNQ1", "KCNQ2", "KCNQ3", "KCNQ4", "KCNQ5",

    # Inwardly Rectifying K Channels
    "KCNJ1", "KCNJ2", "KCNJ3", "KCNJ4", "KCNJ5", "KCNJ6", "KCNJ8",
    "KCNJ9", "KCNJ11", "KCNJ12", "KCNJ13", "KCNJ14", "KCNJ15", "KCNJ16",
    "KIR2.1", "KIR3.1", "KIR4.1", "KIR6.1", "KIR6.2",

    # Acid-Sensing Ion Channels
    "ASIC1", "ASIC1A", "ASIC1B", "ASIC2", "ASIC2A", "ASIC2B", "ASIC3", "ASIC4",

    # Lipid-Mediated Signaling / Phospholipases
    "PLD1", "PLD2", "PLD3", "PLD4", "PLD5", "PLD6",
    "PLA2", "PLA2G1B", "PLA2G2A", "PLA2G4A", "PLA2G6",
    "PLCB1", "PLCB2", "PLCB3", "PLCB4", "PLCG1", "PLCG2", "PLCD1", "PLCD3", "PLCD4",
    "PLC", "PLCE1", "PLCH1", "PLCH2",
    "PIP2", "PIP3", "PI3K", "PIK3CA", "PIK3CB", "PIK3CD",
    "GM1", "S1P", "S1PR1", "S1PR2", "S1PR3",
    "SPHK1", "SPHK2",  # Sphingosine kinases
    "LPCAT1", "LPCAT2",  # Lysophosphatidylcholine acyltransferases
    "DAGLA", "DAGLB",  # Diacylglycerol lipases

    # Integrins
    "ITGA1", "ITGA2", "ITGA3", "ITGA4", "ITGA5", "ITGA6", "ITGA7", "ITGA8", "ITGA9", "ITGA10",
    "ITGB1", "ITGB2", "ITGB3", "ITGB4", "ITGB5", "ITGB6", "ITGB7", "ITGB8",

    # Other mechanosensors/transducers
    "BEST1", "GJB6", "NPR3", "MEC4", "MEC10",
    "STOML3", "OSCA1", "TMEM63",
    "LRRC8A", "LRRC8B", "LRRC8C", "LRRC8D", "LRRC8E",  # VRAC subunits
    "ITGAV",  # Integrin alpha-V

    # Caveolins
    "CAV1", "CAV2", "CAV3",
    "CAVIN1", "CAVIN2", "CAVIN3", "CAVIN4",

    # Connexins/Pannexins
    "CX26", "CX30", "CX32", "CX43", "PANX1", "PANX2", "PANX3",
    "GJA1", "GJB1", "GJB2",

    # Other relevant genes
    "BDNF", "CREB", "CFOS", "FOS", "ARC", "NGFR", "NTRK1", "NTRK2",
    "GFAP", "SYN1", "PSD95", "DLG4", "CAMK2A", "CAMK2B",
    "GRIN1", "GRIN2A", "GRIN2B", "GRIA1", "GRIA2",  # Glutamate receptors
    "GABRA1", "GABRB2", "GAD1", "GAD2",  # GABAergic
    "SLC17A7", "SLC32A1",  # Vesicular transporters
}

# Common gene name aliases
GENE_ALIASES = {
    "TREK-1": "KCNK2",
    "TREK-2": "KCNK10",
    "TRAAK": "KCNK4",
    "TASK-1": "KCNK3",
    "TASK-2": "KCNK5",
    "TASK-3": "KCNK9",
    "TRESK": "KCNK18",
    "TWIK-1": "KCNK1",
    "Kir4.1": "KCNJ10",
    "Kir2.1": "KCNJ2",
    "Nav1.2": "SCN2A",
    "Nav1.5": "SCN5A",
    "Nav1.9": "SCN11A",
    "Cav1.2": "CACNA1C",
    "Cav3.1": "CACNA1G",
    "Cav3.2": "CACNA1H",
    "Cav3.3": "CACNA1I",
    "Connexin 30": "GJB6",
    "Connexin 43": "GJA1",
    "Bestrophin-1": "BEST1",
    "Piezo1": "PIEZO1",
    "Piezo2": "PIEZO2",
    "Caveolin-1": "CAV1",
    "Caveolin-2": "CAV2",
    "Caveolin-3": "CAV3",
    "SWELL1": "LRRC8A",
    "Pannexin-1": "PANX1",
    "Pannexin-2": "PANX2",
}


def extract_genes_from_text(text: str) -> list[tuple[str, str]]:
    """
    Extract gene names from text.
    Returns list of (found_term, canonical_gene_name) tuples.
    """
    if not text:
        return []

    found_genes = []
    text_upper = text.upper()

    # Search for known genes (case-insensitive)
    for gene in KNOWN_GENES:
        # Use word boundary matching
        pattern = r'\b' + re.escape(gene) + r'\b'
        if re.search(pattern, text_upper):
            found_genes.append((gene, gene))

    # Search for aliases
    for alias, canonical in GENE_ALIASES.items():
        pattern = r'\b' + re.escape(alias.upper()) + r'\b'
        if re.search(pattern, text_upper):
            found_genes.append((alias, canonical))
        # Also try with different formatting
        pattern_nodash = r'\b' + re.escape(alias.upper().replace("-", "")) + r'\b'
        if re.search(pattern_nodash, text_upper):
            found_genes.append((alias, canonical))

    # Generic patterns for gene-like symbols not in our list
    # Pattern: 2-6 uppercase letters followed by optional numbers (e.g., ABC1, ABCD12)
    generic_pattern = r'\b([A-Z]{2,6}[0-9]{0,2}[A-Z]?)\b'
    for match in re.finditer(generic_pattern, text):
        candidate = match.group(1)
        # Filter out common non-gene words
        exclude = {'THE', 'AND', 'FOR', 'WITH', 'FROM', 'THAT', 'THIS', 'HAVE', 'WERE',
                   'BEEN', 'ALSO', 'INTO', 'THAN', 'ONLY', 'MOST', 'SUCH', 'WHEN',
                   'EACH', 'BOTH', 'AFTER', 'BEFORE', 'BETWEEN', 'THROUGH', 'DURING',
                   'USA', 'MRI', 'TUS', 'FUS', 'LIFU', 'HIFU', 'CNS', 'PNS', 'EEG',
                   'MEG', 'FMRI', 'PET', 'FDA', 'NIH', 'WHO', 'DNA', 'RNA', 'ATP',
                   'ADP', 'GTP', 'GDP', 'BOLD', 'LTP', 'LTD', 'EPSP', 'IPSP'}
        if candidate not in exclude and len(candidate) >= 3:
            # Only include if it looks like a gene (has letters + optional numbers)
            if re.match(r'^[A-Z]+[0-9]*[A-Z]?$', candidate):
                found_genes.append((candidate, candidate))

    return found_genes


