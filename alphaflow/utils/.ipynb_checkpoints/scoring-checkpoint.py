import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
import os, math

def score_complex(path_coords, path_CB_inds, path_plddt):
    '''
    Score all interfaces in the current complex

    Modified from the score_complex() function in MoLPC repo: 
    https://gitlab.com/patrickbryant1/molpc/-/blob/main/src/complex_assembly/score_entire_complex.py#L106-154
    '''

    chains = [*path_coords.keys()]
    chain_inds = np.arange(len(chains))
    complex_score = 0
    #Get interfaces per chain
    for i in chain_inds:
        chain_i = chains[i]
        chain_coords = np.array(path_coords[chain_i])
        chain_CB_inds = path_CB_inds[chain_i]
        l1 = len(chain_CB_inds)
        chain_CB_coords = chain_coords[chain_CB_inds]
        chain_plddt = path_plddt[chain_i]
 
        for int_i in np.setdiff1d(chain_inds, i):
            int_chain = chains[int_i]
            int_chain_CB_coords = np.array(path_coords[int_chain])[path_CB_inds[int_chain]]
            int_chain_plddt = path_plddt[int_chain]
            #Calc 2-norm
            mat = np.append(chain_CB_coords,int_chain_CB_coords,axis=0)
            a_min_b = mat[:,np.newaxis,:] -mat[np.newaxis,:,:]
            dists = np.sqrt(np.sum(a_min_b.T ** 2, axis=0)).T
            contact_dists = dists[:l1,l1:]
            contacts = np.argwhere(contact_dists<=8)
            #The first axis contains the contacts from chain 1
            #The second the contacts from chain 2
            if contacts.shape[0]>0:
                av_if_plDDT = np.concatenate((chain_plddt[contacts[:,0]], int_chain_plddt[contacts[:,1]])).mean()
                complex_score += np.log10(contacts.shape[0]+1)*av_if_plDDT

    return complex_score, len(chains)

def calculate_mpDockQ(complex_score):
    """
    A function that returns a complex's mpDockQ score after 
    calculating complex_score
    """
    L = 0.827
    x_0 = 261.398
    k = 0.036
    b = 0.221
    return L/(1+math.exp(-1*k*(complex_score-x_0))) + b


def mpdockq_scores(pdb):
    name = os.path.split(pdb)[-1].split('.')[0]
    pdbs = PandasPdb().read_pdb(pdb)
    start_idx = pdbs.df['OTHERS'][pdbs.df['OTHERS'].record_name=='MODEL'].reset_index()[['line_idx']]
    end_idx = pdbs.df['OTHERS'][pdbs.df['OTHERS'].record_name=='ENDMDL'].reset_index()[['line_idx']]
    idx = pd.concat([start_idx,end_idx],axis=1).line_idx.values.tolist()

    scores = []
    for k,v in enumerate(idx):
        chain_ids = pdbs.df['ATOM'].chain_id.unique()
    
        chains = [pdbs.df['ATOM'][(pdbs.df['ATOM'].chain_id==i) & 
                   (pdbs.df['ATOM'].line_idx>v[0]) & 
                   (pdbs.df['ATOM'].line_idx<v[1])].reset_index(drop=True) 
                  for i in chain_ids]
        
        chain_coords = {chain.chain_id.unique()[0] :chain[['x_coord', 'y_coord', 'z_coord']].values.tolist() for chain in chains}
        
        plddt_per_chain = {chain.chain_id.unique()[0]: chain[chain.atom_name=='CA'].b_factor.to_numpy() for chain in chains}
        
        chain_CB_idx = {chain.chain_id.unique()[0]: chain[(chain.atom_name=='CB')
        | ((chain.atom_name=='CA') & (chain.residue_name=='GLY'))].index.tolist() for chain in chains}

        complex_score, num_chains = score_complex(chain_coords, chain_CB_idx, plddt_per_chain)
        
        if num_chains>1:
            scores.append([name, k, calculate_mpDockQ(complex_score)])
        else:
            print('Not a complex. Please make sure your PDB file is a biological assembly')
            
    scores = pd.DataFrame(scores)
    scores.columns = ['name','model','mpDockQ']
    
    return scores