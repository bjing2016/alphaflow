### Comparison of 48-layer and 12-layer AlphaFlow-MD+Templates
| |48l (base)|48l (distilled)|12l (base)|12l (distilled)|
|:-|:-:|:-:|:-:|:-:|
| Pariwise RMSD | 2.18 | 1.73 | 1.94 | 1.40
| Pairwise RMSD $r$ | 0.94 | 0.92 | 0.81 | 0.76
| All-atom RMSF | 1.31 | 1.00 | 1.01 | 0.76
| Global RMSF $r$ | 0.91 | 0.89 | 0.78 | 0.74
| Per-target RMSF $r$ | 0.90 | 0.88 | 0.89 | 0.86
| Root mean $\mathcal{W}_2$-dist | 1.95 | 2.18 | 2.26 | 2.43
| MD PCA $\mathcal{W}_2$-dist | 1.25 | 1.41 | 1.40 | 1.56
| Joint PCA $\mathcal{W}_2$-dist | 1.58 | 1.68 | 1.78 | 1.90
| % PC-sim > 0.5 | 44 | 43 | 46 | 39
| Weak contacts $J$ | 0.62 | 0.51 | 0.60 | 0.56
| Transient contacts $J$ | 0.47 | 0.42 | 0.36 | 0.24
| Exposed residue $J$ | 0.50 | 0.47 | 0.47 | 0.44
| Exposed MI matrix $\rho$ | 0.25 | 0.18 | 0.21 | 0.13
| **Runtime (s)** | 38 | 3.8 | 15.2 | 1.56