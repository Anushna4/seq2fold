import numpy as np

amino_acids = "ACDEFGHIKLMNPQRSTVWY"

def extract_features(sequence):
    sequence = sequence.upper()
    length = len(sequence)

    if length == 0:
        return np.zeros(420)

    # 🔹 Amino Acid Composition (20)
    aac = [sequence.count(aa) / length for aa in amino_acids]

    # 🔹 Dipeptide Composition (400)
    dipeptides = [a+b for a in amino_acids for b in amino_acids]
    total = length - 1

    dpc = []
    for dp in dipeptides:
        dpc.append(sequence.count(dp) / total if total > 0 else 0)

    return np.array(aac + dpc)