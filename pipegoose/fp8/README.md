- Install conda 
```
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
zsh Miniforge3-Linux-x86_64.sh
```
- Create conda env
```
conda create -n env-pipegoose python=3.9.2 --y
conda activate env-pipegoose
pip install -e .
pip install git+https://github.com/NVIDIA/TransformerEngine.git@v0.11
conda install cuda -c nvidia
```
