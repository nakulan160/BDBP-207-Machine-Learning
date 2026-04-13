from Bio import pairwise2
from Bio.Align import substitution_matrices

# Sequences
text = "HEAGAWGHEE"
pattern = "PAWHEAE"

# Load BLOSUM62
blosum62 = substitution_matrices.load("BLOSUM62")

# Gap penalties
gap_open = -10
gap_extend = -1


# Function to run alignment
def run_alignment(seq1, seq2, matrix, mode):
    if mode == "global":
        alignments = pairwise2.align.globalds(seq1, seq2, matrix, gap_open, gap_extend)
    elif mode == "local":
        alignments = pairwise2.align.localds(seq1, seq2, matrix, gap_open, gap_extend)
    else:
        raise ValueError("Mode must be 'global' or 'local'")

    # Take best alignment
    return alignments[0]


# Function to print alignment
def print_alignment(title, alignment):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(pairwise2.format_alignment(*alignment))


# Function to print substitution scores for aligned residues
def print_substitution_scores(alignment, matrix, title):
    seqA, seqB, score, start, end = alignment
    print("\n" + "=" * 60)
    print(f"Substitution Scores — {title}")
    print("=" * 60)
    for a, b in zip(seqA, seqB):
        if a != "-" and b != "-":
            print(f"{a} - {b} : {matrix[a, b]}")
        else:
            print(f"{a} - {b} : GAP")


# Global alignment
global_b62 = run_alignment(text, pattern, blosum62, "global")
# Local alignment
local_b62 = run_alignment(text, pattern, blosum62, "local")

# Print alignments
print_alignment("GLOBAL ALIGNMENT — BLOSUM62", global_b62)
print_alignment("LOCAL ALIGNMENT — BLOSUM62", local_b62)

# Print substitution scores
print_substitution_scores(global_b62, blosum62, "GLOBAL BLOSUM62")
print_substitution_scores(local_b62, blosum62, "LOCAL BLOSUM62")
