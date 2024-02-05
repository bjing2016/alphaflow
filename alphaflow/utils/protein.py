from typing import Sequence, Optional
from openfold.np.protein import to_pdb
from openfold.np.protein import from_pdb_string as _from_pdb_string
from openfold.data import mmcif_parsing
from openfold.np import residue_constants
from alphaflow.utils.tensor_utils import tensor_tree_map
import subprocess, tempfile, os, dataclasses
import numpy as np
from Bio import pairwise2

@dataclasses.dataclass(repr=False)
class Protein:
    """Protein structure representation."""

    # Cartesian coordinates of atoms in angstroms. The atom types correspond to
    # residue_constants.atom_types, i.e. the first three are N, CA, CB.
    atom_positions: np.ndarray  # [num_res, num_atom_type, 3]

    aatype: np.ndarray  # [num_res]
    seqres: str
    name: str
    
    # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
    # is present and 0.0 if not. This should be used for loss masking.
    atom_mask: np.ndarray  # [num_res, num_atom_type]

    # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
    residue_index: np.ndarray  # [num_res]

    # B-factors, or temperature factors, of each residue (in sq. angstroms units),
    # representing the displacement of the residue from its ground truth mean
    # value.
    b_factors: np.ndarray  # [num_res, num_atom_type]

    # Chain indices for multi-chain predictions
    chain_index: Optional[np.ndarray] = None

    # Optional remark about the protein. Included as a comment in output PDB
    # files
    remark: Optional[str] = None

    # Templates used to generate this protein (prediction-only)
    parents: Optional[Sequence[str]] = None

    # Chain corresponding to each parent
    parents_chain_index: Optional[Sequence[int]] = None
    
    def __repr__(self):
        ca_pos = residue_constants.atom_order["CA"]
        present = int(self.atom_mask[..., ca_pos].sum())
        total = self.atom_mask.shape[0]
        return f"Protein(name={self.name} seqres={self.seqres} residues={present}/{total} b_mean={self.b_factors[...,ca_pos].mean()})"

    def present(self):
        ca_pos = residue_constants.atom_order["CA"]
        return int(self.atom_mask[..., ca_pos].sum())

    def total(self):
        return self.atom_mask.shape[0]

def output_to_protein(output):
    """Returns the pbd (file) string from the model given the model output."""
    output = tensor_tree_map(lambda x: x.cpu().numpy(), output)
    final_atom_positions = output['final_atom_positions']
    final_atom_mask = output["atom37_atom_exists"]
    pdbs = []
    for i in range(output["aatype"].shape[0]):
        unk_idx = residue_constants.restype_order_with_x["X"]
        seqres = ''.join(
            [residue_constants.restypes[idx] if idx != unk_idx else "X" for idx in output["aatype"][i]]
        )
        pred = Protein(
            name=output['name'][i],
            aatype=output["aatype"][i],
            seqres=seqres,
            atom_positions=final_atom_positions[i],
            atom_mask=final_atom_mask[i],
            residue_index=output["residue_index"][i] + 1,
            b_factors=np.repeat(output["plddt"][i][...,None], residue_constants.atom_type_num, axis=-1),
            chain_index=output["chain_index"][i] if "chain_index" in output else None,
        )
        pdbs.append(pred)
    return pdbs

def from_dict(prot):
    name = prot['domain_name'].item().decode(encoding='utf-8')
    seq = prot['sequence'].item().decode(encoding='utf-8')
    return Protein(
        name=name,
        aatype=np.nonzero(prot["aatype"])[1],
        atom_positions=prot["all_atom_positions"],
        seqres=seq,
        atom_mask=prot["all_atom_mask"],
        residue_index=prot['residue_index'] + 1,
        b_factors=np.zeros((len(seq), 37))
    )

def from_pdb_string(pdb_string, name=''):
    prot = _from_pdb_string(pdb_string)
    
    unk_idx = residue_constants.restype_order_with_x["X"]
    seqres = ''.join(
        [residue_constants.restypes[idx] if idx != unk_idx else "X" for idx in prot.aatype]
    )
    prot = Protein(
        **prot.__dict__, 
        seqres=seqres,
        name=name,
    )
    return prot

def from_mmcif_string(mmcif_string, chain, name='', is_author_chain=False):
    mmcif_object = mmcif_parsing.parse(file_id = '', mmcif_string=mmcif_string)
    
    if(mmcif_object.mmcif_object is None):
        raise list(mmcif_object.errors.values())[0]
        
    mmcif_object = mmcif_object.mmcif_object

    atom_coords, atom_mask = mmcif_parsing.get_atom_coords(mmcif_object, chain)
    L = atom_coords.shape[0]
    seq = mmcif_object.chain_to_seqres[chain]
    
    unk_idx = residue_constants.restype_order_with_x["X"]
    aatype = np.array(
        [residue_constants.restype_order_with_x.get(aa, unk_idx) for aa in seq]
    )
    prot = Protein(
        aatype=aatype,
        name=name,
        seqres=seq,
        atom_positions=atom_coords,
        atom_mask=atom_mask,
        residue_index=np.arange(L) + 1,
        b_factors=np.zeros((L, 37)), # maybe replace 37 later
    )
    return prot


def global_metrics(ref_prot, pred_prot, lddt=False, symmetric=False):
    if lddt or symmetric:
        ref_prot, pred_prot = align_residue_numbering(ref_prot, pred_prot, mask=symmetric)

    f, ref_path = tempfile.mkstemp(); os.close(f)
    f, pred_path = tempfile.mkstemp(); os.close(f)
    with open(ref_path, 'w') as f:
        f.write(to_pdb(ref_prot))
    with open(pred_path, 'w') as f:
        f.write(to_pdb(pred_prot))
    
    out = tmscore(ref_path, pred_path)
    if lddt:
        out['lddt'] = my_lddt_func(ref_path, pred_path)
    
    os.unlink(ref_path)
    os.unlink(pred_path)
    return out

def prots_to_pdb(prots):
    ss = ''
    for i, prot in enumerate(prots):
        ss += f'MODEL {i}\n'
        prot = to_pdb(prot)
        ss += '\n'.join(prot.split('\n')[1:-2])
        ss += '\nENDMDL\n'
    return ss
    
def align_residue_numbering(prot1, prot2, mask=False):
    prot1 = Protein(**prot1.__dict__)
    prot2 = Protein(**prot2.__dict__)
    
    alignment = pairwise2.align.globalxx(prot1.seqres, prot2.seqres)[0]
    prot1.residue_index = np.array([i for i, c in enumerate(alignment.seqA) if c != '-'])
    prot2.residue_index = np.array([i for i, c in enumerate(alignment.seqB) if c != '-'])

    if mask:
        ca_pos = residue_constants.atom_order["CA"]
        mask1 = np.zeros(len(alignment.seqA))
        mask1[prot1.residue_index[prot1.atom_mask[..., ca_pos] == 1]] = 1
        mask2 = np.zeros(len(alignment.seqA))
        mask2[prot2.residue_index[prot2.atom_mask[..., ca_pos] == 1]] = 1
    
        mask = (mask1 == 1) & (mask2 == 1)
        
        prot1.atom_mask = prot1.atom_mask * mask[prot1.residue_index].reshape(-1, 1)
        prot2.atom_mask = prot2.atom_mask * mask[prot2.residue_index].reshape(-1, 1)
    
    return prot1, prot2
# ca_pos = residue_constants.atom_order["CA"]
# ref_ca = ref_prot.atom_positions[..., ca_pos, :]
# pred_ca = pred_prot.atom_positions[...,ca_pos, :]
# mask = ref_prot.atom_mask[..., ca_pos].astype(bool)
# trans_ca, rms = superimposition._superimpose_np(ref_ca[mask], pred_ca[mask])

def tmscore(ref_path, pred_path):
    
        
    out = subprocess.check_output(['TMscore', '-seq', pred_path, ref_path], 
                    stderr=open('/dev/null', 'w'))
    
    start = out.find(b'RMSD')
    end = out.find(b'rotation')
    out = out[start:end]
    
    rmsd, _, tm, _, gdt_ts, gdt_ha, _, _ = out.split(b'\n')
    
    result = {
        'rmsd': float(rmsd.split(b'=')[-1]),
        'tm': float(tm.split(b'=')[1].split()[0]),
        'gdt_ts': float(gdt_ts.split(b'=')[1].split()[0]),
        'gdt_ha': float(gdt_ha.split(b'=')[1].split()[0]),
    }
    return result

def drmsd(prot1, prot2, align=False, eps=1e-10):
    ca_pos = residue_constants.atom_order["CA"]
    if align:
        prot1, prot2 = align_residue_numbering(prot1, prot2)    
    N = max(prot1.residue_index.max(), prot2.residue_index.max()) 
    mask1, mask2 = np.zeros(N), np.zeros(N)    
    mask1[prot1.residue_index - 1] = prot1.atom_mask[:,ca_pos]
    mask2[prot2.residue_index - 1] = prot2.atom_mask[:,ca_pos]
    pos1, pos2 = np.zeros((N,3)), np.zeros((N,3))

    pos1[prot1.residue_index - 1] = prot1.atom_positions[:,ca_pos]
    pos2[prot2.residue_index - 1] = prot2.atom_positions[:,ca_pos]
    
    dmat1 = np.sqrt(eps + np.sum((pos1[..., None, :] - pos1[..., None, :, :]) ** 2, axis=-1))
    dmat2 = np.sqrt(eps + np.sum((pos2[..., None, :] - pos2[..., None, :, :]) ** 2, axis=-1))

    dists_to_score = mask1 * mask1[:,None] * mask2 * mask2[:,None] * (1.0 - np.eye(N))
    score = np.square(dmat1 - dmat2)

    return np.sqrt((score * dists_to_score).sum() / dists_to_score.sum())

def lddt_ca(prot1, prot2, cutoff=15.0, eps=1e-10, align=False, per_residue=False, symmetric=False):

    ca_pos = residue_constants.atom_order["CA"]
    
    if align:
        prot1, prot2 = align_residue_numbering(prot1, prot2)    
    N = max(prot1.residue_index.max(), prot2.residue_index.max()) 
    mask1, mask2 = np.zeros(N), np.zeros(N)    
    mask1[prot1.residue_index - 1] = prot1.atom_mask[:,ca_pos]
    mask2[prot2.residue_index - 1] = prot2.atom_mask[:,ca_pos]
    pos1, pos2 = np.zeros((N,3)), np.zeros((N,3))
    
    pos1[prot1.residue_index - 1] = prot1.atom_positions[:,ca_pos]
    pos2[prot2.residue_index - 1] = prot2.atom_positions[:,ca_pos]
    
    # mask1, mask2 = prot1.atom_mask[:,ca_pos], prot2.atom_mask[:,ca_pos]
    # pos1, pos2 = prot1.atom_positions[:,ca_pos], prot2.atom_positions[:,ca_pos]
    
    dmat1 = np.sqrt(eps + np.sum((pos1[..., None, :] - pos1[..., None, :, :]) ** 2, axis=-1))
    dmat2 = np.sqrt(eps + np.sum((pos2[..., None, :] - pos2[..., None, :, :]) ** 2, axis=-1))

    if symmetric:
        dists_to_score = (dmat1 < cutoff) | (dmat2 < cutoff)
    else:
        dists_to_score = (dmat1 < cutoff)
    dists_to_score = dists_to_score * mask1 * mask1[:,None] * mask2 * mask2[:,None] * (1.0 - np.eye(N))
    dist_l1 = np.abs(dmat1 - dmat2)
    score = (dist_l1[...,None] < np.array([0.5, 1.0, 2.0, 4.0])).mean(-1)

    if per_residue:
        score = (dists_to_score * score).sum(-1) / dists_to_score.sum(-1)
        return score[prot1.residue_index - 1]
    else:
        score = (dists_to_score * score).sum() / dists_to_score.sum()

    return score
def my_lddt_func(ref_path, pred_path):
    
    out = subprocess.check_output(['lddt', '-c', '-x', pred_path, ref_path],  
            stderr=open('/dev/null', 'w'))

    result = None
    for line in out.split(b'\n'):
        if b'Global LDDT score' in line:
            result = float(line.split(b':')[-1].strip())

    return result
