import torch
import pandas as pd
import numpy as np
from openfold.np import residue_constants
from .data_pipeline import DataPipeline
from .feature_pipeline import FeaturePipeline
from openfold.data.data_transforms import make_atom14_masks
import alphaflow.utils.protein as protein

def seq_to_tensor(seq):
    unk_idx = residue_constants.restype_order_with_x["X"]
    encoded = torch.tensor(
        [residue_constants.restype_order_with_x.get(aa, unk_idx) for aa in seq]
    )
    return encoded

class AlphaFoldCSVDataset:
    def __init__(self, config, path, data_dir=None, mmcif_dir=None, msa_dir=None, templates=False):
        super().__init__()
        self.pdb_chains = pd.read_csv(path, index_col='name')#.sort_index()
        self.data_dir = data_dir
        self.msa_dir = msa_dir
        self.mmcif_dir = mmcif_dir
        self.data_pipeline = DataPipeline(template_featurizer=None)
        self.feature_pipeline = FeaturePipeline(config) 
        self.templates = templates
        
    def __len__(self):
        return len(self.pdb_chains)
        
    def __getitem__(self, idx):

        item = self.pdb_chains.iloc[idx]
        
        mmcif_feats = self.data_pipeline.process_str(item.seqres, item.name)
        if self.templates:
            path = f"{self.data_dir}/{item.name}.npz"
            mmcif_feats = dict(np.load(path, allow_pickle=True))
            extra_all_atom_positions = mmcif_feats['all_atom_positions'][0]
            
        
        try: msa_id = item.msa_id
        except: msa_id = item.name
        msa_features = self.data_pipeline._process_msa_feats(f'{self.msa_dir}/{msa_id}', item.seqres, alignment_index=None)
        data = {**mmcif_feats, **msa_features}

        feats = self.feature_pipeline.process_features(data, mode='predict') 
        if self.templates:
            feats['extra_all_atom_positions'] = torch.from_numpy(extra_all_atom_positions)
        feats['pseudo_beta_mask'] = torch.ones(len(item.seqres))
        feats['name'] = item.name
        feats['seqres'] = item.seqres
        make_atom14_masks(feats)

        pdb_id, chain = item.name.split('_')
        if self.mmcif_dir is not None:
            with open(f"{self.mmcif_dir}/{pdb_id[1:3]}/{pdb_id}.cif") as f:
                feats['ref_prot'] = protein.from_mmcif_string(f.read(), chain, name=item.name)
                
        return feats

class CSVDataset:
    def __init__(self, config, path, mmcif_dir=None, data_dir=None, msa_dir=None, templates=False):
        super().__init__()
        self.df = pd.read_csv(path, index_col='name')#.sort_index()
        self.data_dir = data_dir
        self.templates = templates
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        batch = {
            'name': row.name,
            'seqres': row.seqres,
            'aatype': seq_to_tensor(row.seqres),
            'residue_index': torch.arange(len(row.seqres)),
            'pseudo_beta_mask': torch.ones(len(row.seqres)),
            'seq_mask': torch.ones(len(row.seqres))
        }
        make_atom14_masks(batch)

        if self.templates:
            path = f"{self.data_dir}/{row.name}.npz"
            mmcif_feats = dict(np.load(path, allow_pickle=True))
            extra_all_atom_positions = mmcif_feats['all_atom_positions'][0]
            batch['extra_all_atom_positions'] = torch.from_numpy(extra_all_atom_positions)
            
        return batch