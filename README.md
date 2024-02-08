# AlphaFlow

AlphaFlow is a modified version of AlphaFold, fine-tuned with a flow matching objective, capable of generative modeling of protein conformational ensembles. In particular, AlphaFlow models:
* Experimental ensembles, i.e, potential conformational states as they would be deposited in the PDB
* Molecular dynamics ensembles at physiological temperatures

We also provide a similarly fine-tuned version of ESMFold called ESMFlow. Technical details and thorough benchmarking results can be found in our paper, [AlphaFold Meets Flow Matching for Generating Protein Ensembles](https://arxiv.org/abs/2402.04845), by Bowen Jing, Bonnie Berger, Tommi Jaakkola. This repository contains all code, instructions and model weights necessary to run the method. If you have any questions, feel free to open an issue or reach out at bjing@mit.edu.


![imgs/ensembles.gif](imgs/ensembles.gif)

## Installation
In an environment with Python 3.9 (for example, `conda create -n [NAME] python=3.9`), run:
```
pip install numpy==1.21.2 pandas==1.5.3
pip install torch==1.12.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install biopython==1.79 dm-tree==0.1.6 modelcif==0.7 ml-collections==0.1.0 scipy==1.7.1 absl-py einops
pip install pytorch_lightning==2.0.4 fair-esm
pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@5484c38'
```
We ran installation on a machine with CUDA 11.6 and have tested with A100 and A6000 GPUs.

## Model weights

We provide several versions of AlphaFlow (and similarly named versions of ESMFlow).

* **AlphaFlow-PDB**&mdash;trained on PDB structures to model experimental ensembles from X-ray crystallography or cryo-EM under different conditions
* **AlphaFlow-MD**&mdash;trained on all-atom, explicit solvent MD trajectories at 300K
* **AlphaFlow-MD+Templates**&mdash;trained to additionally take a PDB structure as input, and models the corresponding MD ensemble at 300K

For all models, the **distilled** version runs significantly faster at the cost of some loss of accuracy (benchmarked in the paper).

### AlphaFlow models
| Model|Version|Weights|
|:---|:--|:--|
| AlphaFlow-PDB | base | https://alphaflow.s3.amazonaws.com/params/alphaflow_pdb_base_202402.pt |
| AlphaFlow-PDB | distilled | https://alphaflow.s3.amazonaws.com/params/alphaflow_pdb_distilled_202402.pt |
| AlphaFlow-MD | base | https://alphaflow.s3.amazonaws.com/params/alphaflow_md_base_202402.pt |
| AlphaFlow-MD | distilled | https://alphaflow.s3.amazonaws.com/params/alphaflow_md_distilled_202402.pt |
| AlphaFlow-MD+Templates | base | https://alphaflow.s3.amazonaws.com/params/alphaflow_md_templates_base_202402.pt |
| AlphaFlow-MD+Templates | distilled | https://alphaflow.s3.amazonaws.com/params/alphaflow_md_templates_distilled_202402.pt |

### ESMFlow models
| Model|Version|Weights|
|:---|:--|:--|
| ESMFlow-PDB | base | https://alphaflow.s3.amazonaws.com/params/esmflow_pdb_base_202402.pt |
| ESMFlow-PDB | distilled | https://alphaflow.s3.amazonaws.com/params/esmflow_pdb_distilled_202402.pt |
| ESMFlow-MD | base | https://alphaflow.s3.amazonaws.com/params/esmflow_md_base_202402.pt |
| ESMFlow-MD | distilled | https://alphaflow.s3.amazonaws.com/params/esmflow_md_distilled_202402.pt |
| ESMFlow-MD+Templates | base | https://alphaflow.s3.amazonaws.com/params/esmflow_md_templates_base_202402.pt |
| ESMFlow-MD+Templates | distilled | https://alphaflow.s3.amazonaws.com/params/esmflow_md_templates_distilled_202402.pt |

Training checkpoints (from which fine-tuning can be resumed) are available upon request; please reach out if you'd like to collaborate!

## Running inference

### Preparing input files

1. Prepare a input CSV with an `name` and `sequence` entry for each row. See `splits/` for examples.
2. If running an **AlphaFlow** model, prepare an MSA directory and place the alignments in `.a3m` format at the following paths: `{alignment_dir}/{name}/a3m/{name}.a3m`
3. If running an **MD+Templates** model, the template PDB file needs to be converted to zipped numpy arrays. Prepare a templates directory and run `pythom -m scripts.prep_templates --pdb [PATH] --name [NAME] --outdir [DIR]` for each PDB file of interest. The PDB file should include only a single chain with no residue gaps. The specified name must match the name in the input CSV. 

### Running the model

The basic command for running inference with **AlphaFlow** is:
```
python predict.py --mode alphafold --input_csv [PATH] --msa_dir [DIR] --weights [PATH] --samples [N] --outpdb [DIR]
```
If running the **PDB model**, we recommend appending `--self_cond --resample` for improved performance.

The basic command for running inference with **ESMFlow** is
```
python predict.py --mode esmfold --input_csv [PATH] --weights [PATH] --samples [N] --outpdb [DIR]
```
Additional command line arguments for either model:
* Use the `--pdb_id` argument to select (one or more) rows in the CSV. If no argument is specified, inference is run on all rows.
* If running the **MD  model with templates**, append `--templates --templates_dir [DIR]`.
* If running any **distilled** model, append the arguments `--noisy_first --no_diffusion`.
* To truncate the inference process for increased precision and reduced diversity, append (for example) `--tmax 0.2 --steps 2`. The default inference settings correspond to `--tmax 1.0 --steps 10`. See Appendix B.1 in the paper for more details.
  

## Coming soon

* Colab notebook
* Training workflow
* Analysis scripts to reproduce paper results
* Coarse-grained MD data

## License
MIT. Other licenses may apply to third-party source code noted in file headers.

## Citation
```
@misc{jing2024alphafold,
      title={AlphaFold Meets Flow Matching for Generating Protein Ensembles}, 
      author={Bowen Jing and Bonnie Berger and Tommi Jaakkola},
      year={2024},
      eprint={2402.04845},
      archivePrefix={arXiv},
      primaryClass={q-bio.BM}
}
```
