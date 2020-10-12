# ActionReplay

The goal is to build a Generative Replay(GR) model that can recognize human actions in a continual learning fashion.

## Setup
```bash
git clone https://github.com/cookiekop/ActionReplay/
mkdir datasets models
```
### Prerequisites
- [Python](https://www.python.org/)
- [CUDA](https://developer.nvidia.com/cuda-toolkit)
- [PyTorch](https://pytorch.org/)
- [Matplotlib](https://matplotlib.org/)
- [Jupyter](https://jupyter.org/)

## Train
```bash
python pretrain_vae.py
```

## Evaluate
```bash
jupyter notebook vae_experiment.ipynb
```
