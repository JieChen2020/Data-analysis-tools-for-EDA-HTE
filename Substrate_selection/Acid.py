import csv
from rdkit import Chem
import pandas as pd
import re

pattern1 = Chem.MolFromSmarts('[NH2]')
pattern = Chem.MolFromSmarts('C(=O)[OH]')


def extract_acid_structures(smi_file_path):
    acid = []
    with (open(smi_file_path, 'r') as smi_file):
        for line in smi_file:
            line = line.strip()
            if line:
                smiles, mol_id = line.split()
                if len(smiles) < 15 or len(smiles) > 85:
                    continue

                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                mol = Chem.AddHs(mol)

                if mol.HasSubstructMatch(pattern1):
                    continue
                if mol.HasSubstructMatch(pattern):
                    print(smiles)
                    acid.append((smiles, mol_id))
        return acid


def save_to_csv(acid, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['SMILES', 'ID', 'Type']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for smiles, mol_id in acid:
            writer.writerow({'SMILES': smiles, 'ID': mol_id, 'Type': 'acid'})


smi_file_path = 'mcule_purchasable_in_stock_250309.smi'
acid = extract_acid_structures(smi_file_path)

output_file = 'extract_acid.csv'
save_to_csv(acid, output_file)

print(f"{output_file}")


file_path = 'extract_acid.csv'
df = pd.read_csv(file_path)

sampled_df = df.sample(n=10000, random_state=1)

sampled_df.to_csv('sampled_file.csv', index=False)

print(sampled_df.head())



