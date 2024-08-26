import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, default='splits/transporters_only.csv')
parser.add_argument('--templates_dir', type=str, default=None)
parser.add_argument('--msa_dir', type=str, default='./alignment_dir')
parser.add_argument('--mode', choices=['alphafold', 'esmfold'], default='alphafold')
parser.add_argument('--samples', type=int, default=10)
parser.add_argument('--steps', type=int, default=10)
parser.add_argument('--outpdb', type=str, default='./outpdb/default')
parser.add_argument('--weights', type=str, default=None)
parser.add_argument('--ckpt', type=str, default=None)
parser.add_argument('--original_weights', action='store_true')
parser.add_argument('--pdb_id', nargs='*', default=[])
parser.add_argument('--subsample', type=int, default=None)
parser.add_argument('--resample', action='store_true')
parser.add_argument('--tmax', type=float, default=1.0)
parser.add_argument('--no_diffusion', action='store_true', default=False)
parser.add_argument('--self_cond', action='store_true', default=False)
parser.add_argument('--noisy_first', action='store_true', default=False)
parser.add_argument('--runtime_json', type=str, default=None)
parser.add_argument('--no_overwrite', action='store_true', default=False)
args = parser.parse_args()

import torch, tqdm, os, wandb, json, time, pickle, gzip, re, io, string
import pandas as pd
from biopandas.pdb import PandasPdb
import pytorch_lightning as pl
import numpy as np
from collections import defaultdict
from alphaflow.data.data_modules import collate_fn
from alphaflow.model.wrapper import AlphaFoldWrapper, ESMFoldWrapper
from alphaflow.utils.tensor_utils import tensor_tree_map
import alphaflow.utils.protein as protein
from alphaflow.data.inference import AlphaFoldCSVDataset, CSVDataset
from collections import defaultdict
from openfold.utils.import_weights import import_jax_weights_
from alphaflow.config import model_config

from alphaflow.utils.logging import get_logger
logger = get_logger(__name__)
torch.set_float32_matmul_precision("high")

config = model_config(
    'initial_training',
    train=True, 
    low_prec=True
) 
schedule = np.linspace(args.tmax, 0, args.steps+1)
if args.tmax != 1.0:
    schedule = np.array([1.0] + list(schedule))
loss_cfg = config.loss
data_cfg = config.data
data_cfg.common.use_templates = False
data_cfg.common.max_recycling_iters = 0

if args.subsample: # https://elifesciences.org/articles/75751#s3
    data_cfg.predict.max_msa_clusters = args.subsample // 2
    data_cfg.predict.max_extra_msa = args.subsample

@torch.no_grad()
def main():

    valset = {
        'alphafold': AlphaFoldCSVDataset,
        'esmfold': CSVDataset,
    }[args.mode](
        data_cfg,
        args.input_csv,
        msa_dir=args.msa_dir,
        templates_dir=args.templates_dir,
    )
    # valset[0]
    logger.info("Loading the model")
    model_class = {'alphafold': AlphaFoldWrapper, 'esmfold': ESMFoldWrapper}[args.mode]

    if args.weights:
        ckpt = torch.load(args.weights, map_location='cpu')
        model = model_class(**ckpt['hyper_parameters'], training=False)
        model.model.load_state_dict(ckpt['params'], strict=False)
        model = model.cuda()
        
    
    elif args.original_weights:
        model = model_class(config, None, training=False)
        if args.mode == 'esmfold':
            path = "esmfold_3B_v1.pt"
            model_data = torch.load(path, map_location='cpu')
            model_state = model_data["model"]
            model.model.load_state_dict(model_state, strict=False)
            model = model.to(torch.float).cuda()
            
        elif args.mode == 'alphafold':
            import_jax_weights_(model.model, 'params_model_1.npz', version='model_3')
            model = model.cuda()
        
    else:
        model = model_class.load_from_checkpoint(args.ckpt, map_location='cpu')
        model.load_ema_weights()
        model = model.cuda()
    model.eval()
    
    logger.info("Model has been loaded")
    
    results = defaultdict(list)
    os.makedirs(args.outpdb, exist_ok=True)
    runtime = defaultdict(list)
    for i, item in enumerate(valset):
        if args.pdb_id and item['name'] not in args.pdb_id:
            continue
        if args.no_overwrite and os.path.exists(f'{args.outpdb}/{item["name"]}.pdb'):
            continue
        result = []
        output = []
        for j in tqdm.trange(args.samples):
            if args.subsample or args.resample:
                item = valset[i] # resample MSA
            
            batch = collate_fn([item])
            batch = tensor_tree_map(lambda x: x.cuda(), batch)  
            start = time.time()
            prots, outputs = model.inference(batch, as_protein=False, noisy_first=args.noisy_first,
                        no_diffusion=args.no_diffusion, schedule=schedule, self_cond=args.self_cond)
            runtime[item['name']].append(time.time() - start)
            result.append(prots[-1])

            dat = tensor_tree_map(lambda x: x.cpu().numpy(), outputs[-1])
            cols = ['name', 'seqres', 'ptm', 
                    'predicted_aligned_error', 'max_predicted_aligned_error']
            dat = {col: dat[col] for col in cols}
            output.append(dat)
      
        # with open(f'{args.outpdb}/{item["name"]}.pdb', 'w') as f:
        #     f.write(protein.prots_to_pdb(result))      

        pdbs = PandasPdb().read_pdb_from_list(protein.prots_to_pdb(result).splitlines(True))
        a = pdbs.df["ATOM"].residue_number.unique()
        seq = np.split(a, np.where(np.diff(a) != 1)[0]+1)
        
        chain_a = pdbs.df["ATOM"][pdbs.df["ATOM"]['residue_number']<=seq[0][-1]].copy()
        chain_a['chain_id'] = 'A'
        
        chains = []
        for i in range(len(seq)-1):
            chain_id = list(string.ascii_uppercase)[i+1]
            chain = pdbs.df["ATOM"][pdbs.df["ATOM"]['residue_number']>seq[i][-1]].copy()
            chain['chain_id'] = chain_id
            chain['residue_number'] = chain['residue_number'] - seq[i][-1] -1
            chains.append(chain)
            
        chains = pd.concat(chains)
        pdbs.df["ATOM"] = pd.concat([chain_a,chains]).sort_index()
        pdbs.to_pdb(path=f'{args.outpdb}/{item["name"]}.pdb', gz=False, append_newline=True)

            
        output = pd.DataFrame(output)
        output['name'] = output['name'].apply(lambda x: x[0])
        output['ptm'] = output['ptm'].apply(lambda x: x[0])
        # output['max_predicted_aligned_error'] = output.max_predicted_aligned_error.apply(lambda x: x[0])       
        output['linker'] = output.seqres.apply(lambda x: [m.start() for m in re.finditer(':', x[0])])
        output['seqres'] = output.seqres.apply(lambda x: x[0].split(':'))
        output['predicted_aligned_error'] = output.predicted_aligned_error.apply(lambda x: x[0])
        output['pae'] = output[['predicted_aligned_error','linker']].values.tolist()
        output['pae'] = output['pae'].apply(lambda x: [np.delete(i, x[1]) for i in x[0]])
        output['pae'] = output[['pae','linker']].values.tolist()
        output['pae'] = output['pae'].apply(lambda x: [i for j, i in enumerate(x[0]) if j not in x[1]])
        output['predicted_aligned_error'] = output['pae'].apply(lambda x: np.array(x))
        output.drop(['pae','linker'], axis=1, inplace=True)
        output.to_pickle(f'{args.outpdb}/{item["name"]}.pkl.gz', protocol=4)
        

    if args.runtime_json:
        with open(args.runtime_json, 'w') as f:
            f.write(json.dumps(dict(runtime)))
if __name__ == "__main__":
    main()
