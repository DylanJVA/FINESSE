# FINESSE

Fidelity-aware quantum circuit transpilation for SNAIL-based superconducting architectures. Extends MIRAGE/LightSABRE with fidelity-weighted routing heuristics and mirror gate acceptance.

## Setup

```bash
git clone git@github.com:DylanJVA/MirageFidelities.git
cd MirageFidelities
python -m venv .venv && source .venv/bin/activate
pip install -e ".[mqt]"
pip install qiskit-ibm-runtime
mkdir -p Results logs
```

## Running benchmarks

Single seed (for testing):
```bash
python FrequencyAllocationRuns.py --paper --seed 0
```

Full parallel run on a server:
```bash
nohup ./run_parallel.sh paper 20 24 > logs/master.log 2>&1 &
```

Toronto (IBM heavy-hex) benchmark:
```bash
nohup ./run_parallel.sh toronto 20 24 > logs/master.log 2>&1 &
```

Results land in `Results/paper.csv` and `Results/toronto.csv`. Figures are generated in `PaperPlots.ipynb`.
