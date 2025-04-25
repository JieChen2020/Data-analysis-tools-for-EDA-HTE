import csv
from rdkit import Chem
import pandas as pd
import re

N_pattern = r'N'
pattern1 = Chem.MolFromSmarts('[NH2]')
pattern2 = Chem.MolFromSmarts('C1CCCC[NH]1')
pattern3 = Chem.MolFromSmarts('C1C=CC=C([NH]C)C=1')
pattern4 = Chem.MolFromSmarts('C1(CCCCC1)[NH]C')
pattern5 = Chem.MolFromSmarts('[NH]1CCCC1')
pattern6 = Chem.MolFromSmarts('C1CN(C)CC[NH]1')
pattern7 = Chem.MolFromSmarts('C1C=CC=C(C[NH]C)C=1')
pattern = Chem.MolFromSmarts('C(=O)[OH]')


def extract_amine_structures(smi_file_path):
    amine = []
    with (open(smi_file_path, 'r') as smi_file):
        for line in smi_file:
            line = line.strip()
            if line:
                smiles, mol_id = line.split()
                if len(smiles) < 15 or len(smiles) > 85 or len(re.findall(N_pattern, smiles)) > 3:
                    continue

                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                mol = Chem.AddHs(mol)

                if mol.HasSubstructMatch(pattern):
                    continue
                if mol.HasSubstructMatch(pattern1) or mol.HasSubstructMatch(pattern2):
                    print(smiles)
                    amine.append((smiles, mol_id))
        return amine


def save_to_csv(amine, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['SMILES', 'ID', 'Type']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for smiles, mol_id in amine:
            writer.writerow({'SMILES': smiles, 'ID': mol_id, 'Type': 'Amine'})


smi_file_path = 'mcule_purchasable_in_stock_250309.smi'
amine = extract_amine_structures(smi_file_path)

output_file = 'data10.csv'
save_to_csv(amine, output_file)

print(f"{output_file}")


file_path = 'data10.csv'
df = pd.read_csv(file_path)

sampled_df = df.sample(n=10000, random_state=1)

sampled_df.to_csv('sampled_file.csv', index=False)

print(sampled_df.head())
