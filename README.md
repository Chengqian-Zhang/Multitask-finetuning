# Multi-Task Fine-Tuning Enables Robust Out-of-Distribution Generalization in Atomistic Models

To reproduce the results of the paper "Multi-Task Fine-Tuning Enables Robust Out-of-Distribution Generalization in Atomistic Models" (https://arxiv.org/abs/2601.08486), one can follow the guidance below.

# Prerequisites
Before running the examples, you must install the deepmd-kit software. It is recommended to use the official installation guide:
https://docs.deepmodeling.com/projects/deepmd/en/stable/install/easy-install.html.

A simple way is:
```
git clone https://github.com/deepmodeling/deepmd-kit.git
cd deepmd-kit
git checkout devel
pip install -e .
```

# Repository Structure
The reproduction repository is organized as follows:

`examples/` contains the input configuration files (typically input.json) and the training commands specifically for the QM9 band gap prediction task covering training from scratch (scratch), fine-tuning (ft), linear probing (lp) and multi-task fine-tuning (mft).

`data/` provides the training data for both the downstream property prediction and the auxiliary force-field tasks. All data is stored in the `deepmd/npy` format.

`scripts/` contains the analysis and plotting scripts required to regenerate the tables and figures presented in the paper.

# Reproduction Steps
Step 1: Data Preparation

Ensure the data in the `data/` folder is correctly linked.
Multi-task fine-tuning (MFT) requires both downstream property prediction dataset and auxiliary force-field dataset.


Step 2: Training

Navigate to the `examples/` folder and execute the training command. Please refer to the specific `run.sh` within the examples/ directory for the exact command and hyperparameter settings used for training from scratch (scratch), fine-tuning (ft), linear probing (lp) and multi-task fine-tuning (mft).

Step 3: Evaluation

After training, one the evaluate the model performance on out-of-distribution (OOD) test sets and in-distribution (ID) test sets. If one employs multi-task fine-tuning (MFT), please follow:
```
dp test -m model.ckpt.pt -n 1000000 -s ./data/qm9_gap/id_test/ --head property
dp test -m model.ckpt.pt -n 1000000 -s ./data/qm9_gap/ood_test/ --head property
```
otherwise, please follow:
```
dp test -m model.ckpt.pt -n 1000000 -s ./data/qm9_gap/id_test/
dp test -m model.ckpt.pt -n 1000000 -s ./data/qm9_gap/ood_test/
```