import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, default='splits/atlas.csv')
parser.add_argument('--atlas_dir', type=str, required=True)
parser.add_argument('--outdir', type=str, default='./data_atlas')
parser.add_argument('--num_workers', type=int, default=1)
args = parser.parse_args()

import mdtraj, os, tempfile, tqdm
from betafold.utils import protein
from openfold.data.data_pipeline import make_protein_features
import pandas as pd 
from multiprocessing import Pool
import numpy as np

os.makedirs(args.outdir, exist_ok=True)

df = pd.read_csv(args.split, index_col='name')

def main():
    jobs = []
    for name in df.index:
        #if os.path.exists(f'{args.outdir}/{name}.npz'): continue
        jobs.append(name)

    if args.num_workers > 1:
        p = Pool(args.num_workers)
        p.__enter__()
        __map__ = p.imap
    else:
        __map__ = map
    for _ in tqdm.tqdm(__map__(do_job, jobs), total=len(jobs)):
        pass
    if args.num_workers > 1:
        p.__exit__(None, None, None)

def do_job(name):
    traj = mdtraj.load(f'{args.atlas_dir}/{name}/{name}_prod_R1_fit.xtc', top=f'{args.atlas_dir}/{name}/{name}.pdb') \
        + mdtraj.load(f'{args.atlas_dir}/{name}/{name}_prod_R2_fit.xtc', top=f'{args.atlas_dir}/{name}/{name}.pdb') \
        + mdtraj.load(f'{args.atlas_dir}/{name}/{name}_prod_R3_fit.xtc', top=f'{args.atlas_dir}/{name}/{name}.pdb')
    ref = mdtraj.load(f'{args.atlas_dir}/{name}/{name}.pdb')
    traj = ref + traj
    f, temp_path = tempfile.mkstemp(); os.close(f)
    positions_stacked = []
    for i in tqdm.trange(0, len(traj), 100):
        traj[i].save_pdb(temp_path)
    
        with open(temp_path) as f:
            prot = protein.from_pdb_string(f.read())
            pdb_feats = make_protein_features(prot, name)
            positions_stacked.append(pdb_feats['all_atom_positions'])
            
    
    pdb_feats['all_atom_positions'] = np.stack(positions_stacked)
    print({key: pdb_feats[key].shape for key in pdb_feats})
    np.savez(f"{args.outdir}/{name}.npz", **pdb_feats)
    os.unlink(temp_path)

main()