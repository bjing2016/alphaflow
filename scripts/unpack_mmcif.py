import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mmcif_dir', type=str, required=True)
parser.add_argument('--outdir', type=str, default='./data')
parser.add_argument('--outcsv', type=str, default='./pdb_mmcif.csv')
parser.add_argument('--num_workers', type=int, default=15)
args = parser.parse_args()

import warnings, tqdm, os, io, logging
import pandas as pd
import numpy as np
from multiprocessing import Pool
from betafold.data.data_pipeline import DataPipeline
from openfold.data import mmcif_parsing

pipeline = DataPipeline(template_featurizer=None)

def main():
    dirs = os.listdir(args.data)
    files = [os.listdir(f"{args.data}/{dir}") for dir in dirs]
    files = sum(files, [])
    if args.num_workers > 1:
        p = Pool(args.num_workers)
        p.__enter__()
        __map__ = p.imap
    else:
        __map__ = map
    infos = list(tqdm.tqdm(__map__(unpack_mmcif, files), total=len(files)))
    if args.num_workers > 1:
        p.__exit__(None, None, None)
    info = []
    for inf in infos:
        info.extend(inf)
    df = pd.DataFrame(info).set_index('name')
    df.to_csv(args.outcsv)    
    
def unpack_mmcif(name):
    path = f"{args.mmcif_dir}/{name[1:3]}/{name}"

    with open(path, 'r') as f:
        mmcif_string = f.read()

    
    mmcif = mmcif_parsing.parse(
        file_id=name[:-4], mmcif_string=mmcif_string
    )
    if mmcif.mmcif_object is None:
        logging.info(f"Could not parse {name}. Skipping...")
        return []
    else:
        mmcif = mmcif.mmcif_object

    out = []
    for chain, seq in mmcif.chain_to_seqres.items():
        out.append({
            "name": f"{name[:-4]}_{chain}",
            "release_date":  mmcif.header["release_date"],
            "seqres": seq,
            "resolution": mmcif.header["resolution"],
        })
        
        data = pipeline.process_mmcif(mmcif=mmcif, chain_id=chain)
        out_dir = f"{args.outdir}/{name[1:3]}"
        os.makedirs(out_dir, exist_ok=True)
        out_path = f"{out_dir}/{name[:-4]}_{chain}.npz"
        np.savez(out_path, **data)
        
    
    return out
    
if __name__ == "__main__":
    main()