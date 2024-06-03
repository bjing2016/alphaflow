# Original Copyright 2021 DeepMind Technologies Limited
# Modification Copyright 2022 # Copyright 2021 AlQuraishi Laboratory
# Modifications Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# This container build on the OpenFold container, and installs AlphaFlow.
# At the end, you may wish to download the weights to run ESMFlow, so they are cached in the image.
# It is a large image, about 20GB without weights, 25GB with weights.
#
# OpenFold is quite difficult to get working, as it installs custom torch kernels, so it is used as the base.
# Adapted from https://github.com/aws-solutions-library-samples/aws-batch-arch-for-protein-folding/blob/main/infrastructure/docker/openfold/Dockerfile
#
# To run most recent image (after building), with GPUS, and mounting a directory `outputs`
# docker run --gpus all -v "$(pwd)/outputs:/outputs" -it "$(docker image ls -q | head -n1)" bash
#
# Note that you may need to install nvidia-container-toolkit to run.
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.15.0/install-guide.html
#
# Test command to output into mounted directory:
# python predict.py --mode esmfold --input_csv splits/atlas_test.csv --pdb 6o2v_A --weights params/esmflow_md_base_202402.pt --samples 5 --outpdb /outputs

FROM nvcr.io/nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu18.04

RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
  wget \
  libxml2 \
  cuda-minimal-build-11-3 \
  libcusparse-dev-11-3 \
  libcublas-dev-11-3 \
  libcusolver-dev-11-3 \
  git \
  awscli \
  && rm -rf /var/lib/apt/lists/* \
  && apt-get autoremove -y \
  && apt-get clean

RUN wget -q -P /tmp -O /tmp/miniconda.sh \
  "https://repo.anaconda.com/miniconda/Miniconda3-py39_23.5.2-0-Linux-$(uname -m).sh" \
  && bash /tmp/miniconda.sh -b -p /opt/conda \
  && rm /tmp/miniconda.sh

ENV PATH /opt/conda/bin:$PATH

RUN git clone https://github.com/aqlaboratory/openfold.git /opt/openfold \
  && cd /opt/openfold \
  && git checkout 1d878a1203e6d662a209a95f71b90083d5fc079c

# installing into the base environment since the docker container wont do anything other than run openfold and alphaflow
# RUN conda install -qy conda==4.13.0 \
RUN conda env update -n base --file /opt/openfold/environment.yml \
  && conda clean --all --force-pkgs-dirs --yes

RUN wget -q -P /opt/openfold/openfold/resources \
  https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt

RUN patch -p0 -d /opt/conda/lib/python3.9/site-packages/ < /opt/openfold/lib/openmm.patch

# Install OpenFold
RUN cd /opt/openfold \
  && pip3 install --upgrade pip --no-cache-dir \
  && python3 setup.py install

# Install alphaflow
RUN git clone https://github.com/bjing2016/alphaflow.git /opt/alphaflow

# Install alphaflow packages ~ as defined in README
# torch CUDA version should match your machine
RUN python -m pip install numpy==1.21.2 pandas==1.5.3 && \
    python -m pip install torch==1.13.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html && \
    python -m pip install biopython==1.79 dm-tree==0.1.6 modelcif==0.7 ml-collections==0.1.0 scipy==1.7.3 absl-py einops && \
    python -m pip install pytorch_lightning==2.0.4 fair-esm mdtraj wandb

WORKDIR /opt/alphaflow

# Optionally, download weights as part of the image, so the cached image contains them and we don't re-download each time
# ESMFlow and ESM2
# RUN mkdir params && \
#     aws s3 cp s3://alphaflow/params/esmflow_md_base_202402.pt params/esmflow_pdb_md_202402.pt && \
#    mkdir -p /root/.cache/torch/hub/checkpoints && \
#     wget -q -O /root/.cache/torch/hub/checkpoints/esm2_t36_3B_UR50D.pt https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt && \
#     wget -q -O /root/.cache/torch/hub/checkpoints/esm2_t36_3B_UR50D-contact-regression.pt https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t36_3B_UR50D-contact-regression.pt
