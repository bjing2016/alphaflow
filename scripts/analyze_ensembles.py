import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--atlas_dir', type=str, required=True)
parser.add_argument('--pdbdir', type=str, required=True) 
parser.add_argument('--pdb_id', nargs='*', default=[])
parser.add_argument('--bb_only', action='store_true')
parser.add_argument('--ca_only', action='store_true')
parser.add_argument('--num_workers', type=int, default=1)

args = parser.parse_args()
from sklearn.decomposition import PCA
import mdtraj, pickle, tqdm, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from scipy.optimize import linear_sum_assignment

def get_pca(xyz):
    traj_reshaped = xyz.reshape(xyz.shape[0], -1)
    pca = PCA(n_components=min(traj_reshaped.shape))
    coords = pca.fit_transform(traj_reshaped)
    return pca, coords

def get_rmsds(traj1, traj2, broadcast=False):
    n_atoms = traj1.shape[1]
    traj1 = traj1.reshape(traj1.shape[0], n_atoms * 3)
    traj2 = traj2.reshape(traj2.shape[0], n_atoms * 3)
    if broadcast:
        traj1, traj2 = traj1[:,None], traj2[None]
    distmat = np.square(traj1 - traj2).sum(-1)**0.5 / n_atoms**0.5 * 10
    return distmat

def condense_sidechain_sasas(sasas, top):
    assert top.n_residues > 1

    if top.n_atoms != sasas.shape[1]:
        raise exception.DataInvalid(
            f"The number of atoms in top ({top.n_atoms}) didn't match the "
            f"number of SASAs provided ({sasas.shape[1]}). Make sure you "
            f"computed atom-level SASAs (mode='atom') and that you've passed "
            "the correct topology file and array of SASAs"
        )

    sc_mask = np.array([a.name not in ['CA', 'C', 'N', 'O', 'OXT'] for a in top.atoms])
    res_id = np.array([a.residue.index for a in top.atoms])
    
    rsd_sasas = np.zeros((sasas.shape[0], top.n_residues), dtype='float32')

    for i in range(top.n_residues):
        rsd_sasas[:, i] = sasas[:, sc_mask & (res_id == i)].sum(1)
    return rsd_sasas

def sasa_mi(sasa): 
    N, L = sasa.shape
    joint_probs = np.zeros((L, L, 2, 2))

    joint_probs[:,:,1,1] = (sasa[:,:,None] & sasa[:,None,:]).mean(0)
    joint_probs[:,:,1,0] = (sasa[:,:,None] & ~sasa[:,None,:]).mean(0)
    joint_probs[:,:,0,1] = (~sasa[:,:,None] & sasa[:,None,:]).mean(0)
    joint_probs[:,:,0,0] = (~sasa[:,:,None] & ~sasa[:,None,:]).mean(0)

    marginal_probs = np.stack([1-sasa.mean(0), sasa.mean(0)], -1)
    indep_probs = marginal_probs[None,:,None,:] * marginal_probs[:,None,:,None]
    mi = np.nansum(joint_probs * np.log(joint_probs / indep_probs), (-1, -2))
    mi[np.arange(L), np.arange(L)] = 0
    return mi

       
def get_mean_covar(xyz):
    mean = xyz.mean(0)
    xyz = xyz - mean
    covar = (xyz[...,None] * xyz[...,None,:]).mean(0)
    return mean, covar


def sqrtm(M):
    D, P = np.linalg.eig(M)
    out = (P * np.sqrt(D[:,None])) @ np.linalg.inv(P)
    return out


def get_wasserstein(distmat, p=2):
    assert distmat.shape[0] == distmat.shape[1]
    distmat = distmat ** p
    row_ind, col_ind = linear_sum_assignment(distmat)
    return distmat[row_ind, col_ind].mean() ** (1/p)

def align_tops(top1, top2):
    names1 = [repr(a) for a in top1.atoms]
    names2 = [repr(a) for a in top2.atoms]

    intersection = [nam for nam in names1 if nam in names2]
    
    mask1 = [names1.index(nam) for nam in intersection]
    mask2 = [names2.index(nam) for nam in intersection]
    return mask1, mask2
    
def main(name):
    print('Analyzing', name)
    out = {}
    ref_aa = mdtraj.load(f'{args.atlas_dir}/{name}/{name}.pdb')
    topfile = f'{args.atlas_dir}/{name}/{name}.pdb'
    print('Loading reference trajectory')
    traj_aa = mdtraj.load(f'{args.atlas_dir}/{name}/{name}_prod_R1_fit.xtc', top=topfile) \
        + mdtraj.load(f'{args.atlas_dir}/{name}/{name}_prod_R2_fit.xtc', top=topfile) \
        + mdtraj.load(f'{args.atlas_dir}/{name}/{name}_prod_R3_fit.xtc', top=topfile)
    print(f'Loaded {traj_aa.n_frames} reference frames')
    
    print('Loading AF2 conformers')
    aftraj_aa = mdtraj.load(f'{args.pdbdir}/{name}.pdb')
        
    print(f'Loaded {aftraj_aa.n_frames} AF2 conformers')   
    print(f'Reference has {traj_aa.n_atoms} atoms')
    print(f'Crystal has {ref_aa.n_atoms} atoms')
    print(f'AF has {aftraj_aa.n_atoms} atoms')

    print('Removing hydrogens')

    traj_aa.atom_slice([a.index for a in traj_aa.top.atoms if a.element.symbol != 'H'], True)
    ref_aa.atom_slice([a.index for a in ref_aa.top.atoms if a.element.symbol != 'H'], True)
    aftraj_aa.atom_slice([a.index for a in aftraj_aa.top.atoms if a.element.symbol != 'H'], True)

    print(f'Reference has {traj_aa.n_atoms} atoms')
    print(f'Crystal has {ref_aa.n_atoms} atoms')
    print(f'AF has {aftraj_aa.n_atoms} atoms')
    
    if args.bb_only:
        print('Removing sidechains')
        aftraj_aa.atom_slice([a.index for a in aftraj_aa.top.atoms if a.name in ['CA', 'C', 'N', 'O', 'OXT']], True)
        print(f'AF has {aftraj_aa.n_atoms} atoms')

    elif args.ca_only:
        print('Removing sidechains')
        aftraj_aa.atom_slice([a.index for a in aftraj_aa.top.atoms if a.name == 'CA'], True)
        print(f'AF has {aftraj_aa.n_atoms} atoms')

    
    refmask, afmask = align_tops(traj_aa.top, aftraj_aa.top)
    traj_aa.atom_slice(refmask, True)
    ref_aa.atom_slice(refmask, True)
    aftraj_aa.atom_slice(afmask, True)

    print(f'Aligned on {aftraj_aa.n_atoms} atoms')

    np.random.seed(137)
    RAND1 = np.random.randint(0, traj_aa.n_frames, aftraj_aa.n_frames)
    RAND2 = np.random.randint(0, traj_aa.n_frames, aftraj_aa.n_frames)
    RAND1K = np.random.randint(0, traj_aa.n_frames, 1000)
        
    traj_aa.superpose(ref_aa)
    aftraj_aa.superpose(ref_aa)

    out['ca_mask'] = ca_mask = [a.index for a in traj_aa.top.atoms if a.name == 'CA']
    traj = traj_aa.atom_slice(ca_mask, False)
    ref = ref_aa.atom_slice(ca_mask, False)
    aftraj = aftraj_aa.atom_slice(ca_mask, False)
    print(f'Sliced {aftraj.n_atoms} C-alphas')
    
    traj.superpose(ref)
    aftraj.superpose(ref)

    
    n_atoms = aftraj.n_atoms

    print(f'Doing PCA')

    ref_pca, ref_coords = get_pca(traj.xyz)
    af_coords_ref_pca = ref_pca.transform(aftraj.xyz.reshape(aftraj.n_frames, -1))
    seed_coords_ref_pca = ref_pca.transform(ref.xyz.reshape(1, -1))
    
    af_pca, af_coords = get_pca(aftraj.xyz)
    ref_coords_af_pca = af_pca.transform(traj.xyz.reshape(traj.n_frames, -1))
    seed_coords_af_pca = af_pca.transform(ref.xyz.reshape(1, -1))
    
    joint_pca, _ = get_pca(np.concatenate([traj[RAND1].xyz, aftraj.xyz]))
    af_coords_joint_pca = joint_pca.transform(aftraj.xyz.reshape(aftraj.n_frames, -1))
    ref_coords_joint_pca = joint_pca.transform(traj.xyz.reshape(traj.n_frames, -1))
    seed_coords_joint_pca = joint_pca.transform(ref.xyz.reshape(1, -1))
    
    out['ref_variance'] = ref_pca.explained_variance_ / n_atoms * 100
    out['af_variance'] = af_pca.explained_variance_ / n_atoms * 100
    out['joint_variance'] = joint_pca.explained_variance_ / n_atoms * 100

    out['af_rmsf'] = mdtraj.rmsf(aftraj_aa, ref_aa) * 10
    out['ref_rmsf'] = mdtraj.rmsf(traj_aa, ref_aa) * 10
    
    print(f'Computing atomic EMD')
    ref_mean, ref_covar = get_mean_covar(traj_aa[RAND1K].xyz)
    af_mean, af_covar = get_mean_covar(aftraj_aa.xyz)
    out['emd_mean'] = (np.square(ref_mean - af_mean).sum(-1) ** 0.5) * 10
    try:
        out['emd_var'] = (np.trace(ref_covar + af_covar - 2*sqrtm(ref_covar @ af_covar), axis1=1,axis2=2) ** 0.5) * 10
    except:
        out['emd_var'] = np.trace(ref_covar) ** 0.5 * 10


    print(f'Analyzing SASA')
    sasa_thresh = 0.02
    af_sasa = mdtraj.shrake_rupley(aftraj_aa, probe_radius=0.28)
    af_sasa = condense_sidechain_sasas(af_sasa, aftraj_aa.top)
    ref_sasa = mdtraj.shrake_rupley(traj_aa[RAND1K], probe_radius=0.28)
    ref_sasa = condense_sidechain_sasas(ref_sasa, traj_aa.top)
    crystal_sasa = mdtraj.shrake_rupley(ref_aa, probe_radius=0.28)
    out['crystal_sasa'] = condense_sidechain_sasas(crystal_sasa, ref_aa.top)
    
    out['ref_sa_prob'] = (ref_sasa > sasa_thresh).mean(0)
    out['af_sa_prob'] = (af_sasa > sasa_thresh).mean(0)
    out['ref_mi_mat'] = sasa_mi(ref_sasa > sasa_thresh)
    out['af_mi_mat'] = sasa_mi(af_sasa > sasa_thresh)
    
    ref_distmat = np.linalg.norm(traj[RAND1].xyz[:,None,:] - traj[RAND1].xyz[:,:,None], axis=-1)
    af_distmat = np.linalg.norm(aftraj.xyz[:,None,:] - aftraj.xyz[:,:,None], axis=-1)

    out['ref_contact_prob'] = (ref_distmat < 0.8).mean(0)
    out['af_contact_prob'] = (af_distmat < 0.8).mean(0)
    out['crystal_distmat'] = np.linalg.norm(ref.xyz[0,None,:] - ref.xyz[0,:,None], axis=-1)
    
    out['ref_mean_pairwise_rmsd'] = get_rmsds(traj[RAND1].xyz, traj[RAND2].xyz, broadcast=True).mean()
    out['af_mean_pairwise_rmsd'] = get_rmsds(aftraj.xyz, aftraj.xyz, broadcast=True).mean()

    out['ref_rms_pairwise_rmsd'] = np.square(get_rmsds(traj[RAND1].xyz, traj[RAND2].xyz, broadcast=True)).mean() ** 0.5
    out['af_rms_pairwise_rmsd'] = np.square(get_rmsds(aftraj.xyz, aftraj.xyz, broadcast=True)).mean() ** 0.5

    out['ref_self_mean_pairwise_rmsd'] = get_rmsds(traj[RAND1].xyz, traj[RAND1].xyz, broadcast=True).mean()
    out['ref_self_rms_pairwise_rmsd'] = np.square(get_rmsds(traj[RAND1].xyz, traj[RAND1].xyz, broadcast=True)).mean() ** 0.5
    
    out['cosine_sim'] = (ref_pca.components_[0] * af_pca.components_[0]).sum() 
    

    def get_emd(ref_coords1, ref_coords2, af_coords, seed_coords, K=None):
        if len(ref_coords1.shape) == 3:
            ref_coords1 = ref_coords1.reshape(ref_coords1.shape[0], -1)
            ref_coords2 = ref_coords2.reshape(ref_coords2.shape[0], -1)
            af_coords = af_coords.reshape(af_coords.shape[0], -1)
            seed_coords = seed_coords.reshape(seed_coords.shape[0], -1)
        if K is not None:
            ref_coords1 = ref_coords1[:,:K]
            ref_coords2 = ref_coords2[:,:K]
            af_coords = af_coords[:,:K]
            seed_coords = seed_coords[:,:K]
        emd = {}
        emd['ref|ref mean'] = (np.square(ref_coords1 - ref_coords1.mean(0)).sum(-1)).mean()**0.5 / n_atoms ** 0.5 * 10
        
        distmat = np.square(ref_coords1[:,None] - ref_coords2[None]).sum(-1) 
        distmat = distmat ** 0.5 / n_atoms ** 0.5 * 10
        emd['ref|ref2'] = get_wasserstein(distmat)
        emd['ref mean|ref2 mean'] = np.square(ref_coords1.mean(0) - ref_coords2.mean(0)).sum() ** 0.5 / n_atoms ** 0.5 * 10
        
        distmat = np.square(ref_coords1[:,None] - af_coords[None]).sum(-1) 
        distmat = distmat ** 0.5 / n_atoms ** 0.5 * 10
        emd['ref|af'] = get_wasserstein(distmat)
        emd['ref mean|af mean'] = np.square(ref_coords1.mean(0) - af_coords.mean(0)).sum() ** 0.5 / n_atoms ** 0.5 * 10

        emd['ref|seed'] = (np.square(ref_coords1 - seed_coords).sum(-1)).mean()**0.5 / n_atoms ** 0.5 * 10
        emd['ref mean|seed'] = (np.square(ref_coords1.mean(0) - seed_coords).sum(-1)).mean()**0.5 / n_atoms ** 0.5 * 10

        emd['af|seed'] = (np.square(af_coords - seed_coords).sum(-1)).mean()**0.5 / n_atoms ** 0.5 * 10
        emd['af|af mean'] = (np.square(af_coords - af_coords.mean(0)).sum(-1)).mean()**0.5 / n_atoms ** 0.5 * 10
        emd['af mean|seed'] = (np.square(af_coords.mean(0) - seed_coords).sum(-1)).mean()**0.5 / n_atoms ** 0.5 * 10
        return emd
    
    K=2
    out[f'EMD,ref'] = get_emd(ref_coords[RAND1], ref_coords[RAND2], af_coords_ref_pca, seed_coords_ref_pca, K=K)
    out[f'EMD,af2'] = get_emd(ref_coords_af_pca[RAND1], ref_coords_af_pca[RAND2], af_coords, seed_coords_af_pca, K=K)
    out[f'EMD,joint'] = get_emd(ref_coords_joint_pca[RAND1], ref_coords_joint_pca[RAND2], af_coords_joint_pca, seed_coords_joint_pca, K=K)
    return name, out 


if args.pdb_id:
    pdb_id = args.pdb_id
else:
    pdb_id = [nam.split('.')[0] for nam in os.listdir(args.pdbdir) if '.pdb' in nam]

if args.num_workers > 1:
    p = Pool(args.num_workers)
    p.__enter__()
    __map__ = p.imap
else:
    __map__ = map
out = dict(tqdm.tqdm(__map__(main, pdb_id), total=len(pdb_id)))
if args.num_workers > 1:
    p.__exit__(None, None, None)

with open(f"{args.pdbdir}/out.pkl", 'wb') as f:
    f.write(pickle.dumps(out))