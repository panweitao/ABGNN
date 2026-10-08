# NOTICE — Data Provenance and Usage Terms

## 1. Graph data (`ht/`)

The graph representations in `ht/` are derived from the **Trust-Hub** hardware-Trojan
benchmarks (https://www.trust-hub.org/). Specifically, the designs are built on the
RS232 and the s-series (ISCAS89 / ITC99) circuits with Trojans inserted in the Trust-Hub
style.

- This package distributes only the **pre-processed graph representations**
  (`.content` node features and `.cites` edge lists) that were generated for the paper.
- The **original RTL / gate-level netlists** and any **standard-cell library** files are
  **not** redistributed here. Please obtain them directly from Trust-Hub and the relevant
  library provider, and comply with their respective license terms.

## 2. Code

The source code is provided for **academic, non-commercial research and reproduction**
purposes. If you use it, please cite the paper:

> W. Pan, M. Dong, C. Wen, H. Liu, S. Zhang, B. Shi, Z. Di, Z. Qiu, Y. Gao, L. Zheng,
> "A unioned graph neural network based hardware Trojan node detection,"
> IEICE Electronics Express, Vol. 20, No. 13, 20230204, 2023.

## 3. No warranty

The code is a research prototype and is provided "as is", without warranty of any kind.
It is not guaranteed to reproduce the exact figures reported in the paper.
