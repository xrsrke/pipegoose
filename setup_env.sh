#!/usr/bin/sh

# TODO: This script only work on bash. Make it work on zsh etc.

wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh

eval "$(/root/miniconda3/bin/conda shell.bash hook)"

conda create -n env-pipegoose python=3.9.2 --y

conda activate env-pipegoose

pip install -e .
pip install ninja

conda install cuda -c nvidia