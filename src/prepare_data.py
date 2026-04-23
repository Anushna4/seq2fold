from Bio import SeqIO
import pandas as pd

data = []

file_path = "../data/astral_40.fa"

for record in SeqIO.parse(file_path, "fasta"):
    sequence = str(record.seq)
    header = record.description

    try:
        scop_id = header.split()[1]
        parts = scop_id.split('.')

        protein_class = parts[0]
        fold = parts[1]
        superfamily = parts[2]

        data.append({
            "sequence": sequence,
            "class": protein_class,
            "fold": fold,
            "superfamily": superfamily
        })

    except:
        continue

df = pd.DataFrame(data)

# Save files
df.to_csv("../data/full_data.csv", index=False)
df[["sequence", "fold"]].to_csv("../data/fold_data.csv", index=False)
df[["sequence", "superfamily"]].to_csv("../data/homology_data.csv", index=False)

print("✅ Data prepared successfully!")
print(df.shape)