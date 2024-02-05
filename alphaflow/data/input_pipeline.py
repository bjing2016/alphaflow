# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import itertools

from openfold.data import data_transforms
NUM_RES = "num residues placeholder"
NUM_MSA_SEQ = "msa placeholder"
NUM_EXTRA_SEQ = "extra msa placeholder"
NUM_TEMPLATES = "num templates placeholder"

@data_transforms.curry1
def random_crop_to_size(
    protein,
    crop_size,
    max_templates,
    shape_schema,
    subsample_templates=False,
    seed=None,
):
    """Crop randomly to `crop_size`, or keep as is if shorter than that."""
    # We want each ensemble to be cropped the same way

    g = torch.Generator(device=protein["seq_length"].device)
    if seed is not None:
        g.manual_seed(seed)

    seq_length = protein["seq_length"]

    if "template_mask" in protein:
        num_templates = protein["template_mask"].shape[-1]
    else:
        num_templates = 0

    # No need to subsample templates if there aren't any
    subsample_templates = subsample_templates and num_templates

    num_res_crop_size = min(int(seq_length), crop_size)

    def _randint(lower, upper):
        return int(torch.randint(
                lower,
                upper + 1,
                (1,),
                device=protein["seq_length"].device,
                generator=g,
        )[0])

    if subsample_templates:
        templates_crop_start = _randint(0, num_templates)
        templates_select_indices = torch.randperm(
            num_templates, device=protein["seq_length"].device, generator=g
        )
    else:
        templates_crop_start = 0

    num_templates_crop_size = min(
        num_templates - templates_crop_start, max_templates
    )

    n = seq_length - num_res_crop_size
    if "use_clamped_fape" in protein and protein["use_clamped_fape"] == 1.:
        right_anchor = n
    else:
        x = _randint(0, n)
        right_anchor = n - x
    num_res_crop_start = _randint(0, right_anchor)

    for k, v in protein.items():
        if k not in shape_schema or (
            "template" not in k and NUM_RES not in shape_schema[k]
        ):
            continue

        # randomly permute the templates before cropping them.
        if k.startswith("template") and subsample_templates:
            v = v[templates_select_indices]

        slices = []
        for i, (dim_size, dim) in enumerate(zip(shape_schema[k], v.shape)):
            is_num_res = dim_size == NUM_RES
            if i == 0 and k.startswith("template"):
                crop_size = num_templates_crop_size
                crop_start = templates_crop_start
            else:
                crop_start = num_res_crop_start if is_num_res else 0
                crop_size = num_res_crop_size if is_num_res else dim
            slices.append(slice(crop_start, crop_start + crop_size))
        protein[k] = v[slices] #### MODIFIED 
    
    protein["seq_length"] = protein["seq_length"].new_tensor(num_res_crop_size)
    
    return protein



@data_transforms.curry1
def make_fixed_size(
    protein,
    shape_schema,
    num_res=0,
    num_templates=0,
):

    """Guess at the MSA and sequence dimension to make fixed size."""
    pad_size_map = {
        NUM_RES: num_res,
        NUM_TEMPLATES: num_templates,
    }

    for k, v in protein.items():
        # Don't transfer this to the accelerator.
        if k == "extra_cluster_assignment":
            continue
        shape = list(v.shape)
        schema = shape_schema[k]
        msg = "Rank mismatch between shape and shape schema for"
        assert len(shape) == len(schema), f"{msg} {k}: {shape} vs {schema}"
        pad_size = [
            pad_size_map.get(s2, None) or s1 for (s1, s2) in zip(shape, schema)
        ]

        padding = [(0, p - v.shape[i]) for i, p in enumerate(pad_size)]
        padding.reverse()
        padding = list(itertools.chain(*padding))
        if padding:
            protein[k] = torch.nn.functional.pad(v, padding)
            protein[k] = torch.reshape(protein[k], pad_size)
    
    return protein


def nonensembled_transform_fns(common_cfg, mode_cfg):
    """Input pipeline data transformers that are not ensembled."""
    transforms = [
        data_transforms.cast_to_64bit_ints,
        data_transforms.correct_msa_restypes,
        data_transforms.squeeze_features,
        data_transforms.randomly_replace_msa_with_unknown(0.0),
        data_transforms.make_seq_mask,
        data_transforms.make_msa_mask,
        data_transforms.make_hhblits_profile,
    ]
    if common_cfg.use_templates:
        transforms.extend(
            [
                data_transforms.fix_templates_aatype,
                data_transforms.make_template_mask,
                data_transforms.make_pseudo_beta("template_"),
            ]
        )
        if common_cfg.use_template_torsion_angles:
            transforms.extend(
                [
                    data_transforms.atom37_to_torsion_angles("template_"),
                ]
            )

    transforms.extend(
        [
            data_transforms.make_atom14_masks,
        ]
    )

    if mode_cfg.supervised:
        transforms.extend(
            [
                data_transforms.make_atom14_positions,
                data_transforms.atom37_to_frames,
                data_transforms.atom37_to_torsion_angles(""),
                data_transforms.make_pseudo_beta(""),
                data_transforms.get_backbone_frames,
                data_transforms.get_chi_angles,
            ]
        )

    ###############

    if "max_distillation_msa_clusters" in mode_cfg:
        transforms.append(
            data_transforms.sample_msa_distillation(
                mode_cfg.max_distillation_msa_clusters
            )
        )

    if common_cfg.reduce_msa_clusters_by_max_templates:
        pad_msa_clusters = mode_cfg.max_msa_clusters - mode_cfg.max_templates
    else:
        pad_msa_clusters = mode_cfg.max_msa_clusters

    max_msa_clusters = pad_msa_clusters
    max_extra_msa = mode_cfg.max_extra_msa

    msa_seed = None
    if(not common_cfg.resample_msa_in_recycling):
        msa_seed = ensemble_seed
    
    transforms.append(
        data_transforms.sample_msa(
            max_msa_clusters, 
            keep_extra=True,
            seed=msa_seed,
        )
    )

    if "masked_msa" in common_cfg:
        # Masked MSA should come *before* MSA clustering so that
        # the clustering and full MSA profile do not leak information about
        # the masked locations and secret corrupted locations.
        transforms.append(
            data_transforms.make_masked_msa(
                common_cfg.masked_msa, mode_cfg.masked_msa_replace_fraction
            )
        )

    if common_cfg.msa_cluster_features:
        transforms.append(data_transforms.nearest_neighbor_clusters())
        transforms.append(data_transforms.summarize_clusters())

    # Crop after creating the cluster profiles.
    if max_extra_msa:
        transforms.append(data_transforms.crop_extra_msa(max_extra_msa))
    else:
        transforms.append(data_transforms.delete_extra_msa)

    transforms.append(data_transforms.make_msa_feat())

    ##################
    crop_feats = dict(common_cfg.feat)
    # return transforms

    if mode_cfg.fixed_size:
        transforms.append(data_transforms.select_feat(list(crop_feats)))
        
        transforms.append(
            data_transforms.random_crop_to_size(
                mode_cfg.crop_size,
                mode_cfg.max_templates,
                crop_feats,
                mode_cfg.subsample_templates,
            )
            # random_crop_to_size(
            #     mode_cfg.crop_size,
            #     mode_cfg.max_templates,
            #     crop_feats,
            #     mode_cfg.subsample_templates,
            # )
        )
        transforms.append(
            data_transforms.make_fixed_size(
                crop_feats,
                pad_msa_clusters,
                mode_cfg.max_extra_msa,
                mode_cfg.crop_size,
                mode_cfg.max_templates,
            )
            # make_fixed_size(
            #     crop_feats,
            #     mode_cfg.crop_size,
            #     mode_cfg.max_templates,
            # )
        )
    '''
    else:
        transforms.append(
            data_transforms.crop_templates(mode_cfg.max_templates)
        )

    '''
    return transforms


def process_tensors_from_config(tensors, common_cfg, mode_cfg):
    """Based on the config, apply filters and transformations to the data."""

    no_templates = True
    if("template_aatype" in tensors):
        no_templates = tensors["template_aatype"].shape[0] == 0

    
    nonensembled = nonensembled_transform_fns(
        common_cfg,
        mode_cfg,
    )

    tensors = compose(nonensembled)(tensors)

    return tensors



@data_transforms.curry1
def compose(x, fs):
    for f in fs:
        x = f(x)
    return x


def map_fn(fun, x):
    ensembles = [fun(elem) for elem in x]
    features = ensembles[0].keys()
    ensembled_dict = {}
    for feat in features:
        ensembled_dict[feat] = torch.stack(
            [dict_i[feat] for dict_i in ensembles], dim=-1
        )
    return ensembled_dict
