<div align="center">

<div id="user-content-toc" style="margin-bottom: 50px">
  <ul align="center" style="list-style: none;">
    <summary>
      <h1>Drift Q-Learning</h1>
      <br>
      <h2><a href="https://arxiv.org/abs/2606.00350">Paper</a> &emsp; <a href="https://driftql.github.io/">Project page</a></h2>
    </summary>
  </ul>
</div>

<img src="assets/driftql.gif" width="90%" alt="Drift Q-Learning Animation">

</div>

## Overview

Drift Q-learning (DriftQL) is a simple and performant data-driven RL algorithm that leverages an expressive policy to model complex action distributions in data.

> **Note:** DriftQL's codebase is based on [FQL's implementation](https://github.com/seohongpark/fql/), with Diffusion-QL and Implicit Diffusion-QL (IDQL) added, both based on official author implementations.


## Installation

All packages are based on FQL's codebase, and the installation process is the same as FQL's. For convenience, we provide the installation instructions here again.

The current project requires `Python 3.10+` and is based on JAX. The main dependencies are `jax >= 0.6.2`, `ogbench == 1.1.0`, and `gymnasium == 0.29.1`. To install the full dependencies, simply run:
```bash
pip install -r requirements.txt
```
To use D4RL environments, you need to additionally set up MuJoCo 2.1.0. `mujoco-py` expects the library at `~/.mujoco/mujoco210`:
```bash
mkdir -p ~/.mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O /tmp/mujoco210.tar.gz
tar -xzf /tmp/mujoco210.tar.gz -C ~/.mujoco
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia' >> ~/.bashrc
source ~/.bashrc
```

Alternatively, you can set up an isolated environment with [Mamba](https://mamba.readthedocs.io/) (or Conda):
```bash
mamba create -n driftql python=3.10 -y
mamba activate driftql
pip install -r requirements.txt
```

To ensure that Jax is installed correctly with GPU support, run the following:
```bash
python jax_check.py 
```

By default, `pip install jax` only installs the CPU wheel. If `jax_check.py` reports a `CpuDevice` instead of `CudaDevice`, install the CUDA-bundled wheels (requires an NVIDIA driver supporting CUDA 12):
```bash
pip install -U "jax[cuda12]"
```

If you see the following output, then Jax is successfully installed and can access the GPU:
```
JAX version: 0.6.2
Devices: [CudaDevice(id=0)]

1. Creating 10000x10000 matrices...
✓ Arrays loaded to GPU.

2. Compiling (JIT)...
✓ Compiled.

3. Stressing GPU (check nvidia-smi now!)...
✓ Completed 50 heavy operations in 0.95 seconds.
```


## Usage

The main implementation of DriftQL is in [agents/driftql.py](agents/driftql.py), and our implementations of baselines can also be found in the same directory.
Here are some example commands:

```bash
# DriftQL on OGBench antsoccer-arena
python main.py \
    --env_name=antsoccer-arena-navigate-singletask-v0 \
    --agent.discount=0.995 \
    --agent.alpha=10 \
    --agent.drift_temp=0.5 \
    --agent.q_agg=mean

# DriftQL on D4RL halfcheetah-medium-expert
python main.py \
    --env_name=halfcheetah-medium-expert-v2 \
    --agent.alpha=300
```


## Citation

If you find this work useful, please cite:

```bibtex
@misc{houssaini2026driftqlearning,
      title={Drift Q-Learning},
      author={Anas Houssaini and Mohamad H. Danesh and Amin Abyaneh and Scott Fujimoto and Hsiu-Chin Lin and David Meger},
      year={2026},
      eprint={2606.00350},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.00350},
}
```
