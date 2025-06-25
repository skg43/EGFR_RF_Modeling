import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from tqdm import tqdm

# --- File paths ---
input_file = f"../data/egfr_04_bioactivity_data_3class_pIC50.csv"
output_file = f"../data/egfr_rdkit_features.csv"

# --- Load input CSV ---
df = pd.read_csv(input_file)

# --- Use the correct SMILES column (from your file) ---
smiles_col = "smiles"

img=Chem.MolFromSmiles(df['smiles'].iloc[0])
img1=Draw.MolToImage(img)
img1.show()

if smiles_col not in df.columns:
    raise ValueError(f"Column '{smiles_col}' not found.")

# --- Set Morgan fingerprint parameters ---
FP_RADIUS = 2
FP_BITS = 1024

# --- Initialize progress bar ---
tqdm.pandas()

# --- Define feature extraction function ---
def extract_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [np.nan] * (3 + FP_BITS)
    
    # Scalar descriptors
    mol_wt = Descriptors.MolWt(mol)
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    
    # Morgan fingerprint (bit vector)
    fp = GetMorganFingerprintAsBitVect(mol, radius=FP_RADIUS, nBits=FP_BITS)
    fp_bits = list(fp)
    
    return [mol_wt, rot_bonds, tpsa] + fp_bits

# --- Apply descriptor generation ---
print("Generating features with RDKit...")
feature_data = df[smiles_col].progress_apply(extract_features)

# --- Build feature DataFrame ---
scalar_names = ["MolWt", "NumRotatableBonds", "TPSA"]
fp_names = [f"FP_{i}" for i in range(FP_BITS)]
all_feature_names = scalar_names + fp_names

features_df = pd.DataFrame(feature_data.tolist(), columns=all_feature_names)

# --- Combine original data with descriptors ---
output_df = pd.concat([df, features_df], axis=1)

# --- Save output ---
output_df.to_csv(output_file, index=False)
print(f"Feature CSV saved: {output_file}")

