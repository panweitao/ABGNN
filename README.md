# ABGNN: A Unioned Graph Neural Network for Hardware Trojan Node Detection

Official research code for the paper:

> **A unioned graph neural network based hardware Trojan node detection**
> Weitao Pan, Meng Dong, Cong Wen, Hongjin Liu, Shaolin Zhang, Bo Shi, Zhixiong Di, Zhiliang Qiu, Yiming Gao, Ling Zheng
> *IEICE Electronics Express*, Vol. 20, No. 13, 20230204 (2023)
> DOI: [10.1587/elex.20.20230204](https://doi.org/10.1587/elex.20.20230204)

---

## Overview

Hardware Trojans (HT) are malicious modifications inserted into integrated circuits.
This work detects **Trojan nodes at the gate level**, directly on the netlist, **without any
golden reference**, so the method can be integrated into a standard IC design flow.

The core idea is a **unioned GNN**: for a directed gate-level netlist, two GraphSAGE
encoders are applied to the **input side** and the **output side** of the graph, and their
node embeddings are **unioned (fused)** to form a representative embedding for each node,
which is then classified as Trojan or normal.

Average results across designs reported in the paper:

| Metric | Score |
|---|---|
| Recall | 93.4% |
| F-measure | 91.4% |
| Precision | 90.7% |

## Method at a Glance

1. **Graph construction** — a gate-level netlist is converted into a directed graph; each
   node (gate) gets a 25-dimensional feature vector (19-dim one-hot cell type + 6 continuous
   structural features).
2. **Two directed views** — from the same edge list, two adjacency structures are built:
   the *input side* (`*_adj_lists`) and the *output side* (`*_adj_lists_2`).
3. **Unioned GNN** — one GraphSAGE encoder per view; the two node embeddings are fused
   (element-wise add; a concatenation variant is available in `src/utils.py`).
4. **Classification** — a class-weighted classifier predicts the Trojan/normal label per
   node, addressing the strong label imbalance.

## Repository Structure

```
.
├── main.py                     # Training / evaluation entry point
├── src/
│   ├── DataCenter.py           # Data loading; builds the two directed adjacency sets
│   ├── utils.py                # Unioned GraphSAGE: apply_model_2 / evaluate_2
│   └── experiments.conf        # Model & path configuration
├── models/
│   ├── Graphsage.py            # GraphSAGE implementation (SageLayer, MEAN/MAX agg.)
│   ├── BasicModel.py           # Base model (save / load)
│   └── model_best_debug_ep*.torch   # 15 reference checkpoints (see below)
├── ht/                         # Pre-processed graph data (17 designs)
│   └── *.content  *.cites  *.nodes
├── result/                     # Output directory (results are written here)
├── MATERIALS.md                # Detailed materials manifest
└── NOTICE.md                   # Data provenance and usage terms
```

## Requirements

- Python 3.x
- PyTorch
- `numpy`, `scikit-learn`, `pyhocon`
- A CUDA-capable GPU (recommended; the code sets `device = "cuda"`)

```bash
pip install torch numpy scikit-learn pyhocon
```

## Data Format

### `ht/<design>.content`
One node per line, 27 whitespace-separated columns:

```
<node_id>  <f1> ... <f25>  <label>
```

- `f1..f19` — 19-dim one-hot encoding of the cell / gate type
- `f20..f25` — 6 continuous structural features (centrality-based)
- `label` — `1` = Trojan node, `0` = normal node

The line number (0-based) is the node index used by the adjacency lists.

### `ht/<design>.cites`
One directed edge per line, given as node indices:

```
<u>  <v>     # an edge u -> v
```

### `ht/<design>.nodes`
A list of Trojan node **names** (available for `s38584-T100/T200/T300`).

### Designs included (17)

| Design | Nodes | Design | Nodes |
|---|---|---|---|
| RS232-T1000 | 215 | s35932-T200 | 5438 |
| RS232-T1100 | 216 | s35932-T300 | 5462 |
| RS232-T1200 | 216 | s38417-T100 | 5341 |
| RS232-T1300 | 213 | s38417-T200 | 5344 |
| RS232-T1400 | 215 | s38417-T300 | 5373 |
| RS232-T1500 | 216 | s38584-T100 | 6480 |
| RS232-T1600 | 214 | s38584-T200 | 6556 |
| s15850-T100 | 2182 | s38584-T300 | 7204 |
| s35932-T100 | 5441 | | |

## Usage

Run from the repository root (the config uses relative paths `./ht/`, `./result/`):

```bash
# default configuration: 2 layers, hidden 64, 25-dim features, 2 classes
python main.py --dataSet ht
```

Common options:

```bash
python main.py --dataSet ht --epochs 63 --b_sz 64 --seed 636 \
               --agg_func MAX --learn_method sup
```

Results (TPR, TNR, F1, and a `classification_report`) are appended to `result/<design>.txt`.

### Notes

- **GPU:** `main.py` sets `device = torch.device("cuda")` unconditionally. To run on CPU,
  change that line and remove the `.cuda()` calls.
- **Output directory:** create `result/` before running (an empty one is included).
- **Research prototype:** this is the code as used in our experiments; paths are relative and
  some helper scripts from the authors' working tree are not included.
- **Evaluation protocol:** the loop in `main.py` holds out one design at a time and trains on
  the remaining designs.

## Trained Checkpoints

`models/model_best_debug_ep*_*.torch` are intermediate checkpoints from development runs
(same format as `torch.save([graphSage, graphSage_2, classification])`). They are provided as
a reference point and are **not** guaranteed to reproduce the exact numbers in the paper.

## Data Provenance

The graph data in `ht/` is derived from the **Trust-Hub** hardware-Trojan benchmarks
(<https://www.trust-hub.org/>). Only the **pre-processed graph representations** are
distributed here; the original RTL/netlists and standard-cell libraries are **not** included
and should be obtained directly from Trust-Hub under their terms. See [`NOTICE.md`](NOTICE.md).

## Citation

If you find this code useful, please cite:

```bibtex
@article{pan2023unioned,
  title   = {A unioned graph neural network based hardware Trojan node detection},
  author  = {Pan, Weitao and Dong, Meng and Wen, Cong and Liu, Hongjin and Zhang, Shaolin
             and Shi, Bo and Di, Zhixiong and Qiu, Zhiliang and Gao, Yiming and Zheng, Ling},
  journal = {IEICE Electronics Express},
  volume  = {20},
  number  = {13},
  pages   = {20230204},
  year    = {2023},
  doi     = {10.1587/elex.20.20230204}
}
```

## License

The source code is released for academic, non-commercial research use.
A formal license file is pending; the data follows the terms in [`NOTICE.md`](NOTICE.md).
