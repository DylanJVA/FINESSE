# FINESSE

**F**idelity-**IN**tegrated **E**quivalence-aware **S**wap **S**election and **E**xecution

FINESSE is MIRAGE with one change: SABRE's heuristic H is extended to use fidelity-weighted
distances. Because MIRAGE already uses H for both SWAP selection and mirror acceptance, this
single change makes both decisions fidelity-aware simultaneously. FinesseLayout adds a
fidelity-aware initial layout pass on top.

## Heuristic H

There is one H. The distance metric D switches based on whether a fidelity matrix is provided:

```
D[i,j] = D_hop[i,j]   shortest path in hops          (no fidelity_matrix)
          D_fid[i,j]   Dijkstra with −log F weights   (fidelity_matrix provided)

L[i,j] = −log F[i,j]   edge log-infidelity (same units as D_fid)
```

The routing lookahead over front layer F and extended set E is:

```
H_dist(layout) = Σ_{g∈F} D[π(g)]  +  W · avg_{g∈E} D[π(g)]
```

where π(g) = (physical qubit 0, physical qubit 1) of gate g under the current layout.

**SWAP selection:** each candidate edge (p0, p1) is scored after simulating the SWAP:

```
H(p0,p1) = max(δ[p0], δ[p1]) · H_dist(layout after SWAP(p0,p1))
```

When no fidelity matrix is provided, `H_dist` uses hop-count distance. When a
fidelity matrix is provided, `H_dist` uses `D_fid`, the Dijkstra shortest-path
distance on `-log F` edge weights. The best SWAP minimizes `H`.

**Mirror acceptance** (aggression > 0): when gate g on edge (gp0, gp1) becomes routable,
compare executing as U versus the mirror form U' = SWAP·U:

```
cost(U)  = k_U  · L[gp0,gp1]  +  H_dist(current layout,  stuck gates)
cost(U') = k_U' · L[gp0,gp1]  +  H_dist(permuted layout, stuck gates)
```

Accept U' if cost(U') ≤ cost(U) [aggression 2] or cost(U') < cost(U) [aggression 1].
k_U, k_U' are the native-gate decomposition costs of U and SWAP·U respectively.

H_dist here uses the same D as SWAP selection (D_hop or D_fid), summed over currently
stuck 2Q gates and their extended set. The decay δ from SWAP selection does not appear.

## Getting started

```bash
pip install -e ".[notebooks]"   # installs the package + registers finesse-download etc.
finesse-download                # downloads circuits into circuits/ — needed for benchmarks
```

The `pip install -e .` is a one-time setup. After that, notebooks and scripts have access to the `finesse` package and the `finesse-download` / `finesse-redqueen` commands. The correctness suite (`run_correctness_suite`) works without downloading — its circuits are bundled. Everything else needs `finesse-download` run first.

For MQT Bench circuits (generated on the fly, no download needed):
```bash
pip install -e ".[mqt]"
```

## Notebooks

`sabre.ipynb`, `lightsabre.ipynb`, and `mirage.ipynb` document and verify each layer of the stack — good starting point if you want to see how things fit together.

## Benchmarking

```bash
finesse-redqueen --seeds 20 --out rq.json
```

Runs our SABRE vs Qiskit on the full red-queen suite. Saves results to `rq.json` and a readable `rq.txt` after every circuit, so you can Ctrl+C and resume.

## Ablation configurations

| # | config key | Configuration | `mode` | `aggression` | `fidelity_matrix` |
|---|---|---|---|---|---|
| — | `qiskit_sabre` | Qiskit SABRE (reference) | — | — | — |
| 1 | `sabre` | SABRE | `'sabre'` | 0 | — |
| 2 | `lightsabre` | LightSABRE | `'lightsabre'` | 0 | — |
| 3 | `mirage_sabre` | MIRAGE (SABRE) | `'sabre'` | 2 | — |
| 4 | `mirage` | MIRAGE (LightSABRE) | `'lightsabre'` | 2 | — |
| 5 | `sabre_fid` | SABRE + fidelity | `'sabre'` | 0 | F |
| 6 | `lightsabre_fid` | LightSABRE + fidelity | `'lightsabre'` | 0 | F |
| 7 | `mirage_fid_sabre` | MIRAGE + fidelity (SABRE) | `'sabre'` | 2 | F |
| 8 | `mirage_fid` | MIRAGE + fidelity (LightSABRE) | `'lightsabre'` | 2 | F |
| 9 | `finesse` | **FINESSE** | `'lightsabre'` | 2 | F |

Row 9 additionally uses `FinesseLayout` (fidelity-aware layout pass). All other rows use trivial layout.

Key comparisons:
- 1→2, 3→4: isolates LightSABRE vs SABRE base
- 1→3, 2→4: isolates mirror absorption
- 1→5, 2→6: isolates fidelity in H (no mirrors)
- 3→7, 4→8: isolates fidelity in H (with mirrors) — because MIRAGE uses H, fidelity propagates to mirror acceptance automatically
- 8→9: tests whether FinesseLayout adds value beyond fidelity-aware routing alone

## References

- [SABRE](https://arxiv.org/abs/1809.02573)
- [LightSABRE](https://arxiv.org/abs/2409.08368)
- [MIRAGE](https://arxiv.org/abs/2308.03874)
