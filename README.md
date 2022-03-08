# Extracting Effective Subnetworks with Gumebel-Softmax
_by Robin Dupont, Mohammed Amine Alaoui, Hichem Sahbi, Alice Lebois_

📜 arxiv : https://arxiv.org/abs/2202.12986

This repository contains the code for the implementation our Aribtrarily Shifted Log Parameterization.

- [Extracting Effective Subnetworks with Gumebel-Softmax](#extracting-effective-subnetworks-with-gumebel-softmax)
    - [⬇️ Setup](#️-setup)
  - [⚙️ Training models](#️-training-models)
    - [⏩ Quickstart](#-quickstart)
    - [🔎 Advanced options](#-advanced-options)
    - [📜 Paper configurations examples](#-paper-configurations-examples)

### ⬇️ Setup

First clone the repository :
``` bash
git clone https://github.com/N0ciple/ASLP.git
```

After cloning the repository install the dependencies with:
```bash
pip install -r requirements.txt
```
(don't forget to `cd` into the repo directory first!)


## ⚙️ Training models
### ⏩ Quickstart
```bash
python main.py
```
Simple as that !
This will train a `Conv4` model with our method (ASLP), **without** weight rescale, **without** signed constant and **with** data augmentation.

### 🔎 Advanced options

| option                   | default     | comment                                                                                                                                         |
| ------------------------ | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `--lr`                   | 50          | Set the learning for the masks optimizer                                                                                                        |
| `--momentum`             | 0.9         | Set the momentum fot the masks optimizer                                                                                                        |
| `--batch-size`           | 256         | Set the batch size (make shure it fits in you GPU memory, 256 should be ok for most GPUs)                                                       |
| `--strategy`             | ASLP        | Select the used strategy. Can be `ASLP` (our paper), `supermask` [1] or `edge-popup` [2]                                                        |
| `--weight-rescale`       | *N/A  (flag)* | Activate the Weight Rescale (depending on the chosen strategy)                                                                                  |
| `--signed-constant`      | *N/A  (flag)* | Activate the signed constant weight distribution                                                                                                |
| `--network`              | Conv4       | Set the used network architecture. Can be `Conv2`, `Conv4` or `Conv6`                                                                           |
| `--name`                 | Experiment  | Name of the experiment (for the tensorboard logger)                                                                                             |
| `--data-path`            | .           | Path where the data will be downloaded                                                                                                          |
| `--prune-and-test`       | *N/A (flag)*  | If this flag is present, the network  with the best validation accuracy will be pruned (according to the method) and tested on the test dataset |
| `--no-data-augmentation` | *N/A (flag)*  | If this flag is present, data augmentation will be disabled.                                                                                    |

### 📜 Paper configurations examples

**ASLP**
`Conv6` network, with weight rescale (WR), signed constant (SR) and data augmentation (DA)
```bash
python main.py \
    --strategy ASLP \
    --network Conv6 \
    --weight-rescale \
    --signed-constant \
    --name Conv6+DA+SC+WR \
    --prune-and-test
```

`Conv2` network without data augmentation (no-DA)
```bash
python main.py \
    --strategy ASLP \
    --network Conv2 \
    --no-data-augment \
    --name Conv2+no-DA \
    --prune-and-test
```

**References**
- [1] H. Zhou, J. Lan, R. Liu, and J. Yosinski, “Deconstructing lottery tickets: Zeros, signs, and the supermask,” in NeurIPS, 2019.
- [2] V. Ramanujan, M. Wortsman, A. Kembhavi, A. Farhadi, and M. Rastegari, “What’s hidden in a randomly weighted neural network?,” in CVPR. 2020.
