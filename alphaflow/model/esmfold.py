# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import typing as T
from functools import partial

import torch
from torch import nn
from torch.nn import LayerNorm

import esm
from esm import Alphabet

from alphaflow.utils.misc import (
    categorical_lddt,
    batch_encode_sequences,
    collate_dense_tensors,
    output_to_pdb,
)

from .trunk import FoldingTrunk
from .layers import GaussianFourierProjection
from .input_stack import InputPairStack

from openfold.data.data_transforms import make_atom14_masks
from openfold.np import residue_constants
from openfold.model.heads import PerResidueLDDTCaPredictor
from openfold.model.primitives import Linear
from openfold.utils.feats import atom14_to_atom37, pseudo_beta_fn


load_fn = esm.pretrained.load_model_and_alphabet
esm_registry = {
    "esm2_8M": partial(load_fn, "esm2_t6_8M_UR50D_500K"),
    "esm2_8M_270K": esm.pretrained.esm2_t6_8M_UR50D,
    "esm2_35M": partial(load_fn, "esm2_t12_35M_UR50D_500K"),
    "esm2_35M_270K": esm.pretrained.esm2_t12_35M_UR50D,
    "esm2_150M": partial(load_fn, "esm2_t30_150M_UR50D_500K"),
    "esm2_150M_270K": partial(load_fn, "esm2_t30_150M_UR50D_270K"),
    "esm2_650M": esm.pretrained.esm2_t33_650M_UR50D,
    "esm2_650M_270K": partial(load_fn, "esm2_t33_650M_270K_UR50D"),
    "esm2_3B": esm.pretrained.esm2_t36_3B_UR50D,
    "esm2_3B_270K": partial(load_fn, "esm2_t36_3B_UR50D_500K"),
    "esm2_15B": esm.pretrained.esm2_t48_15B_UR50D,
}


class ESMFold(nn.Module):
    def __init__(self, cfg, extra_input=False):
        super().__init__()

        self.cfg = cfg
        cfg = self.cfg

        self.distogram_bins = 64
        self.esm, self.esm_dict = esm_registry.get(cfg.esm_type)()

        self.esm.requires_grad_(False)
        self.esm.half()

        self.esm_feats = self.esm.embed_dim
        self.esm_attns = self.esm.num_layers * self.esm.attention_heads
        self.register_buffer("af2_to_esm", ESMFold._af2_to_esm(self.esm_dict).float()) # hack to get EMA working
        self.esm_s_combine = nn.Parameter(torch.zeros(self.esm.num_layers + 1))

        c_s = cfg.trunk.sequence_state_dim
        c_z = cfg.trunk.pairwise_state_dim

        self.esm_s_mlp = nn.Sequential(
            LayerNorm(self.esm_feats),
            nn.Linear(self.esm_feats, c_s),
            nn.ReLU(),
            nn.Linear(c_s, c_s),
        )
        ######################
        self.input_pair_embedding = Linear(
            cfg.input_pair_embedder.no_bins, 
            cfg.trunk.pairwise_state_dim,
            init="final",
        )
        self.input_time_projection = GaussianFourierProjection(
            embedding_size=cfg.input_pair_embedder.time_emb_dim
        )
        self.input_time_embedding = Linear(
            cfg.input_pair_embedder.time_emb_dim, 
            cfg.trunk.pairwise_state_dim,
            init="final",
        )
        torch.nn.init.zeros_(self.input_pair_embedding.weight)
        torch.nn.init.zeros_(self.input_pair_embedding.bias)
        self.input_pair_stack = InputPairStack(**cfg.input_pair_stack)

        self.extra_input = extra_input
        if extra_input:
            self.extra_input_pair_embedding = Linear(
                cfg.input_pair_embedder.no_bins, 
                cfg.evoformer_stack.c_z,
                init="final",
            )   
            self.extra_input_pair_stack = InputPairStack(**cfg.input_pair_stack)
        
        #######################

        # 0 is padding, N is unknown residues, N + 1 is mask.
        self.n_tokens_embed = residue_constants.restype_num + 3
        self.pad_idx = 0
        self.unk_idx = self.n_tokens_embed - 2
        self.mask_idx = self.n_tokens_embed - 1
        self.embedding = nn.Embedding(self.n_tokens_embed, c_s, padding_idx=0)

        self.trunk = FoldingTrunk(cfg.trunk)

        self.distogram_head = nn.Linear(c_z, self.distogram_bins)
        # self.ptm_head = nn.Linear(c_z, self.distogram_bins)
        # self.lm_head = nn.Linear(c_s, self.n_tokens_embed)
        self.lddt_bins = 50
        
        self.lddt_head = PerResidueLDDTCaPredictor(
            no_bins=self.lddt_bins,
            c_in=cfg.trunk.structure_module.c_s,
            c_hidden=cfg.lddt_head_hid_dim
        )

    @staticmethod
    def _af2_to_esm(d: Alphabet):
        # Remember that t is shifted from residue_constants by 1 (0 is padding).
        esm_reorder = [d.padding_idx] + [
            d.get_idx(v) for v in residue_constants.restypes_with_x
        ]
        return torch.tensor(esm_reorder)

    def _af2_idx_to_esm_idx(self, aa, mask):
        aa = (aa + 1).masked_fill(mask != 1, 0)
        return self.af2_to_esm.long()[aa]

    def _compute_language_model_representations(
        self, esmaa: torch.Tensor
    ) -> torch.Tensor:
        """Adds bos/eos tokens for the language model, since the structure module doesn't use these."""
        batch_size = esmaa.size(0)

        bosi, eosi = self.esm_dict.cls_idx, self.esm_dict.eos_idx
        bos = esmaa.new_full((batch_size, 1), bosi)
        eos = esmaa.new_full((batch_size, 1), self.esm_dict.padding_idx)
        esmaa = torch.cat([bos, esmaa, eos], dim=1)
        # Use the first padding index as eos during inference.
        esmaa[range(batch_size), (esmaa != 1).sum(1)] = eosi

        res = self.esm(
            esmaa,
            repr_layers=range(self.esm.num_layers + 1),
            need_head_weights=self.cfg.use_esm_attn_map,
        )
        esm_s = torch.stack(
            [v for _, v in sorted(res["representations"].items())], dim=2
        )
        esm_s = esm_s[:, 1:-1]  # B, L, nLayers, C
        esm_z = (
            res["attentions"].permute(0, 4, 3, 1, 2).flatten(3, 4)[:, 1:-1, 1:-1, :]
            if self.cfg.use_esm_attn_map
            else None
        )
        return esm_s, esm_z

    def _mask_inputs_to_esm(self, esmaa, pattern):
        new_esmaa = esmaa.clone()
        new_esmaa[pattern == 1] = self.esm_dict.mask_idx
        return new_esmaa

    def _get_input_pair_embeddings(self, dists, mask):

        mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        
        lower = torch.linspace(
            self.cfg.input_pair_embedder.min_bin,
            self.cfg.input_pair_embedder.max_bin,
            self.cfg.input_pair_embedder.no_bins, 
        device=dists.device)
        dists = dists.unsqueeze(-1)
        inf = self.cfg.input_pair_embedder.inf
        upper = torch.cat([lower[1:], lower.new_tensor([inf])], dim=-1)
        dgram = ((dists > lower) * (dists < upper)).type(dists.dtype)

        inp_z = self.input_pair_embedding(dgram * mask.unsqueeze(-1))
        inp_z = self.input_pair_stack(inp_z, mask, chunk_size=None)
        return inp_z

    def _get_extra_input_pair_embeddings(self, dists, mask):

        mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        
        lower = torch.linspace(
            self.cfg.input_pair_embedder.min_bin,
            self.cfg.input_pair_embedder.max_bin,
            self.cfg.input_pair_embedder.no_bins, 
        device=dists.device)
        dists = dists.unsqueeze(-1)
        inf = self.cfg.input_pair_embedder.inf
        upper = torch.cat([lower[1:], lower.new_tensor([inf])], dim=-1)
        dgram = ((dists > lower) * (dists < upper)).type(dists.dtype)

        inp_z = self.extra_input_pair_embedding(dgram * mask.unsqueeze(-1))
        inp_z = self.extra_input_pair_stack(inp_z, mask, chunk_size=None)
        return inp_z

                                   
    def forward(
        self,
        batch,
        prev_outputs=None,
    ):
        """Runs a forward pass given input tokens. Use `model.infer` to
        run inference from a sequence.

        Args:
            aa (torch.Tensor): Tensor containing indices corresponding to amino acids. Indices match
                openfold.np.residue_constants.restype_order_with_x.
            mask (torch.Tensor): Binary tensor with 1 meaning position is unmasked and 0 meaning position is masked.
            residx (torch.Tensor): Residue indices of amino acids. Will assume contiguous if not provided.
            masking_pattern (torch.Tensor): Optional masking to pass to the input. Binary tensor of the same size
                as `aa`. Positions with 1 will be masked. ESMFold sometimes produces different samples when
                different masks are provided.
            num_recycles (int): How many recycle iterations to perform. If None, defaults to training max
                recycles, which is 3.
        """
        aa = batch['aatype']
        mask = batch['seq_mask']
        residx = batch['residue_index']
       
        # === ESM ===
        
        esmaa = self._af2_idx_to_esm_idx(aa, mask)
        esm_s, _ = self._compute_language_model_representations(esmaa)

        # Convert esm_s to the precision used by the trunk and
        # the structure module. These tensors may be a lower precision if, for example,
        # we're running the language model in fp16 precision.
        esm_s = esm_s.to(self.esm_s_combine.dtype)
        esm_s = esm_s.detach()

        # === preprocessing ===
        esm_s = (self.esm_s_combine.softmax(0).unsqueeze(0) @ esm_s).squeeze(2)
        s_s_0 = self.esm_s_mlp(esm_s)
        s_s_0 += self.embedding(aa)
        #######################
        if 'noised_pseudo_beta_dists' in batch:
            inp_z = self._get_input_pair_embeddings(
                batch['noised_pseudo_beta_dists'], 
                batch['pseudo_beta_mask']
            )
            inp_z = inp_z + self.input_time_embedding(self.input_time_projection(batch['t']))[:,None,None]
        else: # have to run the module, else DDP wont work
            B, L = batch['aatype'].shape
            inp_z = self._get_input_pair_embeddings(
                s_s_0.new_zeros(B, L, L), 
                batch['pseudo_beta_mask'] * 0.0
            )
            inp_z = inp_z + self.input_time_embedding(self.input_time_projection(inp_z.new_zeros(B)))[:,None,None]
        ##########################
        #############################
        if self.extra_input:
            if 'extra_all_atom_positions' in batch:
                extra_pseudo_beta = pseudo_beta_fn(batch['aatype'], batch['extra_all_atom_positions'], None)
                extra_pseudo_beta_dists = torch.sum((extra_pseudo_beta.unsqueeze(-2) - extra_pseudo_beta.unsqueeze(-3)) ** 2, dim=-1)**0.5
                extra_inp_z = self._get_extra_input_pair_embeddings(
                    extra_pseudo_beta_dists, 
                    batch['pseudo_beta_mask'],
                )
                
            else: # otherwise DDP complains
                B, L = batch['aatype'].shape
                extra_inp_z = self._get_extra_input_pair_embeddings(
                    inp_z.new_zeros(B, L, L), 
                    inp_z.new_zeros(B, L),
                ) * 0.0
    
            inp_z = inp_z + extra_inp_z
        ########################


        
        s_z_0 = inp_z 
        if prev_outputs is not None:
            s_s_0 = s_s_0 + self.trunk.recycle_s_norm(prev_outputs['s_s'])
            s_z_0 = s_z_0 + self.trunk.recycle_z_norm(prev_outputs['s_z'])
            s_z_0 = s_z_0 + self.trunk.recycle_disto(FoldingTrunk.distogram(
                prev_outputs['sm']["positions"][-1][:, :, :3],
                3.375,
                21.375,
                self.trunk.recycle_bins,
            ))

        else:
            s_s_0 = s_s_0 + self.trunk.recycle_s_norm(torch.zeros_like(s_s_0)) * 0.0
            s_z_0 = s_z_0 + self.trunk.recycle_z_norm(torch.zeros_like(s_z_0)) * 0.0
            s_z_0 = s_z_0 + self.trunk.recycle_disto(s_z_0.new_zeros(s_z_0.shape[:-2], dtype=torch.long)) * 0.0


            
        structure: dict = self.trunk(
            s_s_0, s_z_0, aa, residx, mask, no_recycles=0 # num_recycles
        )
        disto_logits = self.distogram_head(structure["s_z"])
        disto_logits = (disto_logits + disto_logits.transpose(1, 2)) / 2
        structure["distogram_logits"] = disto_logits

        '''
        lm_logits = self.lm_head(structure["s_s"])
        structure["lm_logits"] = lm_logits
        '''
        
        structure["aatype"] = aa
        make_atom14_masks(structure)

        for k in [
            "atom14_atom_exists",
            "atom37_atom_exists",
        ]:
            structure[k] *= mask.unsqueeze(-1)
        structure["residue_index"] = residx
        lddt_head = self.lddt_head(structure['sm']["single"])
        structure["lddt_logits"] = lddt_head
        plddt = categorical_lddt(lddt_head, bins=self.lddt_bins)
        structure["plddt"] = 100 * plddt
        # we predict plDDT between 0 and 1, scale to be between 0 and 100.

        '''
        ptm_logits = self.ptm_head(structure["s_z"])
        seqlen = mask.type(torch.int64).sum(1)
        structure["tm_logits"] = ptm_logits
        structure["ptm"] = torch.stack(
            [
                compute_tm(
                    batch_ptm_logits[None, :sl, :sl],
                    max_bins=31,
                    no_bins=self.distogram_bins,
                )
                for batch_ptm_logits, sl in zip(ptm_logits, seqlen)
            ]
        )
        structure.update(
            compute_predicted_aligned_error(
                ptm_logits, max_bin=31, no_bins=self.distogram_bins
            )
        )
        '''

        structure["final_atom_positions"] = atom14_to_atom37(structure["sm"]["positions"][-1], batch)
        structure["final_affine_tensor"] = structure["sm"]["frames"][-1]
        if "name" in batch: structure["name"] = batch["name"]
        return structure

    @torch.no_grad()
    def infer(
        self,
        sequences: T.Union[str, T.List[str]],
        residx=None,
        masking_pattern: T.Optional[torch.Tensor] = None,
        num_recycles: T.Optional[int] = None,
        residue_index_offset: T.Optional[int] = 512,
        chain_linker: T.Optional[str] = "G" * 25,
    ):
        """Runs a forward pass given input sequences.

        Args:
            sequences (Union[str, List[str]]): A list of sequences to make predictions for. Multimers can also be passed in,
                each chain should be separated by a ':' token (e.g. "<chain1>:<chain2>:<chain3>").
            residx (torch.Tensor): Residue indices of amino acids. Will assume contiguous if not provided.
            masking_pattern (torch.Tensor): Optional masking to pass to the input. Binary tensor of the same size
                as `aa`. Positions with 1 will be masked. ESMFold sometimes produces different samples when
                different masks are provided.
            num_recycles (int): How many recycle iterations to perform. If None, defaults to training max
                recycles (cfg.trunk.max_recycles), which is 4.
            residue_index_offset (int): Residue index separation between chains if predicting a multimer. Has no effect on
                single chain predictions. Default: 512.
            chain_linker (str): Linker to use between chains if predicting a multimer. Has no effect on single chain
                predictions. Default: length-25 poly-G ("G" * 25).
        """
        if isinstance(sequences, str):
            sequences = [sequences]

        aatype, mask, _residx, linker_mask, chain_index = batch_encode_sequences(
            sequences, residue_index_offset, chain_linker
        )

        if residx is None:
            residx = _residx
        elif not isinstance(residx, torch.Tensor):
            residx = collate_dense_tensors(residx)

        aatype, mask, residx, linker_mask = map(
            lambda x: x.to(self.device), (aatype, mask, residx, linker_mask)
        )
        
        output = self.forward(
            aatype,
            mask=mask,
            residx=residx,
            masking_pattern=masking_pattern,
            num_recycles=num_recycles,
        )

        output["atom37_atom_exists"] = output[
            "atom37_atom_exists"
        ] * linker_mask.unsqueeze(2)

        output["mean_plddt"] = (output["plddt"] * output["atom37_atom_exists"]).sum(dim=(1, 2)) / output["atom37_atom_exists"].sum(dim=(1, 2))
        output["chain_index"] = chain_index

        return output

    def output_to_pdb(self, output: T.Dict) -> T.List[str]:
        """Returns the pbd (file) string from the model given the model output."""
        return output_to_pdb(output)

    def infer_pdbs(self, seqs: T.List[str], *args, **kwargs) -> T.List[str]:
        """Returns list of pdb (files) strings from the model given a list of input sequences."""
        output = self.infer(seqs, *args, **kwargs)
        return self.output_to_pdb(output)

    def infer_pdb(self, sequence: str, *args, **kwargs) -> str:
        """Returns the pdb (file) string from the model given an input sequence."""
        return self.infer_pdbs([sequence], *args, **kwargs)[0]

    def set_chunk_size(self, chunk_size: T.Optional[int]):
        # This parameter means the axial attention will be computed
        # in a chunked manner. This should make the memory used more or less O(L) instead of O(L^2).
        # It's equivalent to running a for loop over chunks of the dimension we're iterative over,
        # where the chunk_size is the size of the chunks, so 128 would mean to parse 128-lengthed chunks.
        # Setting the value to None will return to default behavior, disable chunking.
        self.trunk.set_chunk_size(chunk_size)

    @property
    def device(self):
        return self.esm_s_combine.device
