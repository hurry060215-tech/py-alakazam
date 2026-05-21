"""Constants and reference tables for pyalakazam.

Faithful Python copies of the data objects shipped in the R alakazam
package (``R/Data.R`` and ``R/sysdata.rda``).
"""
from __future__ import annotations

# --------------------------------------------------------------------------
# IUPAC ambiguous character codes (alakazam R/Data.R)
# --------------------------------------------------------------------------
IUPAC_DNA: dict[str, list[str]] = {
    "A": ["A"], "C": ["C"], "G": ["G"], "T": ["T"],
    "M": ["A", "C"], "R": ["A", "G"], "W": ["A", "T"], "S": ["C", "G"],
    "Y": ["C", "T"], "K": ["G", "T"],
    "V": ["A", "C", "G"], "H": ["A", "C", "T"], "D": ["A", "G", "T"],
    "B": ["C", "G", "T"], "N": ["A", "C", "G", "T"],
}

IUPAC_AA: dict[str, list[str]] = {
    "A": ["A"], "B": ["N", "R"], "C": ["C"], "D": ["D"], "E": ["E"],
    "F": ["F"], "G": ["G"], "H": ["H"], "I": ["I"], "J": ["I", "L"],
    "K": ["K"], "L": ["L"], "M": ["M"], "N": ["N"], "P": ["P"],
    "Q": ["Q"], "R": ["R"], "S": ["S"], "T": ["T"], "V": ["V"],
    "W": ["W"],
    "X": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
          "M", "N", "P", "Q", "R", "S", "T", "V", "W", "X", "Y", "Z", "*"],
    "Y": ["Y"], "Z": ["E", "Q"], "*": ["*"],
}

DNA_IUPAC: dict[str, str] = {
    "A": "A", "C": "C", "G": "G", "T": "T",
    "AC": "M", "AG": "R", "AT": "W", "CG": "S", "CT": "Y", "GT": "K",
    "ACG": "V", "ACT": "H", "AGT": "D", "CGT": "B", "ACGT": "N",
}

# --------------------------------------------------------------------------
# Default color palettes (alakazam R/Data.R)
# --------------------------------------------------------------------------
# Nucleotide colors, named by base.
DNA_COLORS: dict[str, str] = {
    "A": "#64F73F", "C": "#FFB340", "G": "#EB413C", "T": "#3C88EE",
}

# Immunoglobulin isotype/chain colors, named by C-region call.
IG_COLORS: dict[str, str] = {
    "IGHA": "#377EB8", "IGHD": "#FF7F00", "IGHE": "#E41A1C",
    "IGHG": "#4DAF4A", "IGHM": "#984EA3", "IGHK": "#E5C494",
    "IGHL": "#FFD92F",
}

# T-cell receptor chain colors, named by locus.
TR_COLORS: dict[str, str] = {
    "TRA": "#CBD5E8", "TRB": "#F4CAE4", "TRD": "#FDCDAC", "TRG": "#E6F5C9",
}

# --------------------------------------------------------------------------
# Amino acid abbreviations (single letter -> three letter)
# --------------------------------------------------------------------------
ABBREV_AA: dict[str, str] = {
    "A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys",
    "Q": "Gln", "E": "Glu", "G": "Gly", "H": "His", "I": "Ile",
    "L": "Leu", "K": "Lys", "M": "Met", "F": "Phe", "P": "Pro",
    "S": "Ser", "T": "Thr", "W": "Trp", "Y": "Tyr", "V": "Val",
}

# --------------------------------------------------------------------------
# IMGT V-region boundaries (1-based, inclusive)
# --------------------------------------------------------------------------
IMGT_REGIONS: dict[str, tuple[int, int]] = {
    "fwr1": (1, 78),
    "cdr1": (79, 114),
    "fwr2": (115, 165),
    "cdr2": (166, 195),
    "fwr3": (196, 312),
}

# --------------------------------------------------------------------------
# Amino-acid physicochemical scales (alakazam R/sysdata.rda)
# --------------------------------------------------------------------------
# Kyte & Doolittle, 1982 hydropathy index
HYDROPATHY_KYTJ82: dict[str, float] = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5,
    "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9,
    "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9,
    "Y": -1.3, "V": 4.2,
}

# Zimmerman et al, 1968 bulkiness scale
BULKINESS_ZIMJ68: dict[str, float] = {
    "A": 11.50, "R": 14.28, "N": 12.82, "D": 11.68, "C": 13.46,
    "Q": 14.45, "E": 13.57, "G": 3.40, "H": 13.69, "I": 21.40,
    "L": 21.40, "K": 15.71, "M": 16.25, "F": 19.80, "P": 17.43,
    "S": 9.47, "T": 15.77, "W": 21.67, "Y": 18.03, "V": 21.57,
}

# Grantham, 1974 polarity scale
POLARITY_GRAR74: dict[str, float] = {
    "A": 8.1, "R": 10.5, "N": 11.6, "D": 13.0, "C": 5.5, "Q": 10.5,
    "E": 12.3, "G": 9.0, "H": 10.4, "I": 5.2, "L": 4.9, "K": 11.3,
    "M": 5.7, "F": 5.2, "P": 8.0, "S": 9.2, "T": 8.6, "W": 5.4,
    "Y": 6.2, "V": 5.9,
}

# EMBOSS pK values for charged amino acids
PK_EMBOSS: dict[str, float] = {
    "C": 8.5, "D": 3.9, "E": 4.1, "H": 6.5, "K": 10.8, "R": 12.5,
    "Y": 10.1,
}

# --------------------------------------------------------------------------
# Standard (and ambiguous) genetic code, used by translateDNA.
# Built to match seqinr::translate(ambiguous=TRUE): a codon translates to a
# residue when *all* nucleotide expansions resolve to the same residue.
# --------------------------------------------------------------------------
_STANDARD_CODON_TABLE: dict[str, str] = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


def _build_ambiguous_codon_table() -> dict[str, str]:
    """Expand the standard codon table to all IUPAC ambiguous codons.

    Matches seqinr's ``translate(ambiguous=TRUE)``: an ambiguous codon
    translates to a residue iff every concrete codon it expands to gives
    the same residue, otherwise it translates to ``X``.
    """
    table = dict(_STANDARD_CODON_TABLE)
    bases = list(IUPAC_DNA.keys())
    for b1 in bases:
        for b2 in bases:
            for b3 in bases:
                codon = b1 + b2 + b3
                if codon in table:
                    continue
                residues = set()
                for n1 in IUPAC_DNA[b1]:
                    for n2 in IUPAC_DNA[b2]:
                        for n3 in IUPAC_DNA[b3]:
                            residues.add(_STANDARD_CODON_TABLE[n1 + n2 + n3])
                table[codon] = residues.pop() if len(residues) == 1 else "X"
    return table


CODON_TABLE: dict[str, str] = _build_ambiguous_codon_table()
