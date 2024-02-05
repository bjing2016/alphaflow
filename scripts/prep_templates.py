import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pdb', type=str, required=True)
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--outdir', type=str, default='./templates_dir')
args = parser.parse_args()

import os
from alphaflow.utils import protein
from openfold.data.data_pipeline import make_protein_features
import pandas as pd 
import numpy as np

os.makedirs(args.outdir, exist_ok=True)

with open(args.pdb) as f:
    prot = protein.from_pdb_string(f.read())
    
pdb_feats = make_protein_features(prot, args.name)
pdb_feats['all_atom_positions'] = pdb_feats['all_atom_positions'][None]
print({key: pdb_feats[key].shape for key in pdb_feats})
np.savez(f"{args.outdir}/{args.name}.npz", **pdb_feats)