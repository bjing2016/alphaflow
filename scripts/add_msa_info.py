import pandas as pd
import os, tqdm
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--openfold_dir', type=str, required=True)
parser.add_argument('--incsv', type=str, default='pdb_chains.csv')
parser.add_argument('--outcsv', type=str, default='pdb_chains_msa.csv')
args = parser.parse_args()

msas = os.listdir(f'{args.openfold_dir}/pdb')

msa_queries = {}
for pdb_id in tqdm.tqdm(msas):
    a3m_paths = [
        f'{args.openfold_dir}/pdb/{pdb_id}/a3m/bfd_uniclust_hits.a3m',
        f'{args.openfold_dir}/pdb/{pdb_id}/a3m/mgnify_hits.a3m',
        f'{args.openfold_dir}/pdb/{pdb_id}/a3m/uniref90_hits.a3m',
    ]
    for a3m_path in a3m_paths:
        if os.path.exists(a3m_path):
            break
    with open(a3m_path) as f:
        _ = next(f)
        msa_queries[pdb_id] = next(f).strip()

msa_queries = pd.Series(msa_queries)

df = pd.read_csv(args.incsv)
msa_id = [None]*len(df)

freqs = defaultdict(int)
done, skipped = 0, 0
for seqres, sub_df in tqdm.tqdm(df.groupby('seqres')):
    freqs[len(sub_df)] += 1    
    found = list(filter(lambda n: n in msa_queries.index, sub_df.name))
    if not found:
        skipped += 1
        continue
    done += 1
    if len(found) == 1:
        if seqres == msa_queries.loc[found[0]]:
            for idx in sub_df.index: msa_id[idx] = found[0]
        else:
            print('Mismatch', found[0])
    if len(found) != 1:
        print('Found multiple', found)
        for pdb_id in found:
            if seqres == msa_queries.loc[pdb_id]:
                for idx in sub_df.index: msa_id[idx] = found[0]
                print('Match', pdb_id)
            else:
                print('Mismatch', pdb_id)

df['msa_id'] = msa_id
df[~df.msa_id.isnull()].set_index('name').to_csv(args.outcsv)

    