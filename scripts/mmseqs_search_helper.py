import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, required=True) 
parser.add_argument('--db_dir', type=str, default='./dbbase') 
parser.add_argument('--outdir', type=str, default='./alignment_dir') 
args = parser.parse_args()

import pandas as pd
import subprocess, os

df = pd.read_csv(args.split)
os.makedirs(args.outdir, exist_ok=True)
with open('/tmp/tmp.fasta', 'w') as f:
    for _, row in df.iterrows():
        f.write(f'>{row["name"]}\n{row.seqres}\n')

cmd = f'python -m scripts.mmseqs_search /tmp/tmp.fasta {args.db_dir} {args.outdir}'
os.system(cmd)

for name in os.listdir(args.outdir):
    if '.a3m' not in name:
        continue
    with open(f'{args.outdir}/{name}') as f:
        pdb_id = next(f).strip()[1:]
    cmd = f'mkdir -p {args.outdir}/{pdb_id}/a3m'
    print(cmd)
    os.system(cmd)
    cmd = f'mv {args.outdir}/{name} {args.outdir}/{pdb_id}/a3m/{pdb_id}.a3m'
    os.system(cmd)
    print(cmd)