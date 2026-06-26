# 5UTRnet

Code for paper: 5UTRnet: A Deep Learning Model for Designing High-Efficiency 5’ UTRs in mRNA Therapeutics.
The workflow is as follows:

1. Environment Setup

```bash
conda create -n 5utrnet python=3.8
conda activate 5utrnet
pip install -r requirements.txt
```

2. Data Preparing

Download the UTR dataset and place it under the `data/` directory.

3. Model Training

```bash
CUDA_VISIBLE_DEVICES=0 python multi_task_main.py
```