# Materials Manifest

**Project:** A unioned graph neural network based hardware Trojan node detection
**Reference:** IEICE Electronics Express, Vol. 20, No. 13, 20230204 (2023)

---

## 1. Source code

| File | Role |
|---|---|
| `main.py` | Training / evaluation entry (`--dataSet ht`) |
| `src/DataCenter.py` | Loads `.content` / `.cites`; builds `*_adj_lists` and `*_adj_lists_2` (the two directions) |
| `src/utils.py` | `apply_model_2` / `evaluate_2`: two GraphSAGE encoders fused, then classifier; class-weighted loss |
| `src/experiments.conf` | Config: `num_layers=2`, `hidden_emb_size=64`, `feature_size=25`, `num_labels=2` |
| `models/Graphsage.py` | GraphSAGE (SageLayer; MEAN / MAX aggregation) |
| `models/BasicModel.py` | Base model with `save` / `load` |

## 2. Trained checkpoints

| Item | Count | Size | Contents |
|---|---|---|---|
| `models/model_best_debug_ep*_*.torch` | 15 | ~105 KB each | `[graphSage, graphSage_2, classification]` |

## 3. Datasets (`ht/`)

| Item | Count |
|---|---|
| `.content` (node features + labels) | 17 |
| `.cites` (directed edges) | 17 |
| `.nodes` (Trojan node names) | 3 (`s38584-T100/T200/T300`) |

Formats:

- `.content`: `node_id  f1 ... f25  label` — 27 columns; 25-dim feature + binary label.
- `.cites`: `u  v` — directed edge as node indices.
- `.nodes`: one Trojan node name per line.

Detailed per-design node counts are listed in `README.md` (§4).

## 4. Configuration summary

| Parameter | Value |
|---|---|
| `num_layers` | 2 |
| `hidden_emb_size` | 64 |
| `feature_size` | 25 |
| `num_labels` | 2 |
| `agg_func` | MAX |
| Fusion | element-wise add (concat variant commented in `src/utils.py`) |

## 5. Environment

Python + PyTorch, `pyhocon`, `numpy`, `scikit-learn`; a CUDA GPU is expected by the code as
shipped.

## 6. Not included / obtainable elsewhere

- **Cora / PubMed** datasets (environment sanity checks only) — not required for Trojan detection.
- **Raw RTL / gate-level netlists and the standard-cell library** — see `NOTICE.md`.
