#!/bin/bash
# Usage:
#   ./run_parallel.sh paper           [seeds] [jobs]
#   ./run_parallel.sh stress          [seeds] [jobs]
#   ./run_parallel.sh ibm             [seeds] [jobs]
#   ./run_parallel.sh grid            [seeds] [jobs]
#   ./run_parallel.sh grid  --ibm     [seeds] [jobs]
#   ./run_parallel.sh dense --ibm     [seeds] [jobs]
#   ./run_parallel.sh transpile       [seeds] [jobs]
#   ./run_parallel.sh transpile --ibm [seeds] [jobs]
#
# Examples:
#   ./run_parallel.sh paper              # 20 seeds, all cores
#   ./run_parallel.sh paper 20 8         # 20 seeds, 8 at a time
#   ./run_parallel.sh grid 32            # alpha/beta grid, 32 seeds, all cores
#   ./run_parallel.sh transpile 20 8     # finesse_transpile vs SABRE, 20 seeds, 8 jobs

MODE=${1:-paper}
IBM_FLAG=""
if [[ "$MODE" = "grid" || "$MODE" = "dense" || "$MODE" = "transpile" ]] && [ "$2" = "--ibm" ]; then
    IBM_FLAG="--ibm"
    SEEDS=${3:-32}
    NJOBS=${4:-$(nproc)}
else
    SEEDS=${2:-20}
    NJOBS=${3:-$(nproc)}
fi

mkdir -p Results logs

# Clear per-seed files before launching so append-mode writes don't accumulate across runs
TAG_EARLY=$([ -n "$IBM_FLAG" ] && echo "ibm" || echo "snail")
rm -f Results/${MODE}_${TAG_EARLY}_s*.csv

echo "=== $MODE: seeds 0-$((SEEDS-1)), $NJOBS parallel slots, $(nproc) cores available ==="

pids=()

for seed in $(seq 0 $((SEEDS - 1))); do
    # Throttle: wait for a slot to open if we're at capacity
    while [ ${#pids[@]} -ge $NJOBS ]; do
        # Poll for any finished job
        new_pids=()
        for pid in "${pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                new_pids+=("$pid")
            fi
        done
        pids=("${new_pids[@]}")
        [ ${#pids[@]} -ge $NJOBS ] && sleep 2
    done

    echo "  launching seed $seed ($(date +%H:%M:%S))"
    python3 FrequencyAllocationRuns.py --$MODE $IBM_FLAG --seed $seed \
        > logs/${MODE}_s${seed}.log 2>&1 &
    pids+=($!)
done

# Wait for all remaining
echo "All seeds launched, waiting for completion..."
for pid in "${pids[@]}"; do
    wait "$pid"
done

echo "=== All seeds done ($(date +%H:%M:%S)) ==="

if [ "$MODE" = "paper" ]; then
    echo "Merging Results/paper_s*.csv → Results/paper.csv"
    python3 FrequencyAllocationRuns.py --merge
elif [ "$MODE" = "stress" ]; then
    echo "Merging Results/stress_s*.csv → Results/stress.csv"
    python3 - <<'EOF'
import glob, pandas as pd, sys
files = sorted(glob.glob("Results/stress_s*.csv"))
if not files:
    print("No per-seed files found."); sys.exit(1)
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df = df.sort_values(["device","circuit","router","beta","seed"]).reset_index(drop=True)
df.to_csv("Results/stress.csv", index=False)
print(f"Merged {len(files)} files → Results/stress.csv ({len(df)} rows)")
EOF
elif [ "$MODE" = "grid" ]; then
    TAG=$([ -n "$IBM_FLAG" ] && echo "ibm" || echo "snail")
    echo "Merging Results/grid_${TAG}_s*.csv → Results/grid_${TAG}.csv"
    python3 - <<EOF
import glob, pandas as pd, sys
tag = "$TAG"
files = sorted(glob.glob(f"Results/grid_{tag}_s*.csv"))
if not files:
    print("No per-seed files found."); sys.exit(1)
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df = df.sort_values(["device","circuit","router","beta","seed"]).reset_index(drop=True)
df.to_csv(f"Results/grid_{tag}.csv", index=False)
print(f"Merged {len(files)} files -> Results/grid_{tag}.csv ({len(df)} rows)")
EOF
elif [ "$MODE" = "ibm" ]; then
    echo "Merging Results/ibm_s*.csv → Results/ibm.csv"
    python3 - <<'EOF'
import glob, pandas as pd, sys
files = sorted(glob.glob("Results/ibm_s*.csv"))
if not files:
    print("No per-seed files found."); sys.exit(1)
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df = df.sort_values(["device","circuit","router","beta","seed"]).reset_index(drop=True)
df.to_csv("Results/ibm.csv", index=False)
print(f"Merged {len(files)} files → Results/ibm.csv ({len(df)} rows)")
print(df.groupby(["device","router","beta"])[["swaps","depth","lf_cost"]].mean().round(2).to_string())
EOF
elif [ "$MODE" = "dense" ]; then
    TAG=$([ -n "$IBM_FLAG" ] && echo "ibm" || echo "snail")
    echo "Merging Results/dense_${TAG}_s*.csv → Results/dense_${TAG}.csv"
    python3 - <<EOF
import glob, pandas as pd, sys
tag = "$TAG"
files = sorted(glob.glob(f"Results/dense_{tag}_s*.csv"))
if not files:
    print("No per-seed files found."); sys.exit(1)
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df = df.sort_values(["device","circuit","router","beta","seed"]).reset_index(drop=True)
df.to_csv(f"Results/dense_{tag}.csv", index=False)
print(f"Merged {len(files)} files -> Results/dense_{tag}.csv ({len(df)} rows)")
EOF
elif [ "$MODE" = "transpile" ]; then
    TAG=$([ -n "$IBM_FLAG" ] && echo "ibm" || echo "snail")
    echo "Merging Results/transpile_${TAG}_s*.csv → Results/transpile_${TAG}.csv"
    python3 - <<EOF
import glob, pandas as pd, sys
tag = "$TAG"
files = sorted(glob.glob(f"Results/transpile_{tag}_s*.csv"))
if not files:
    print("No per-seed files found."); sys.exit(1)
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df = df.sort_values(["device","circuit","router","beta","seed"]).reset_index(drop=True)
df.to_csv(f"Results/transpile_{tag}.csv", index=False)
print(f"Merged {len(files)} files -> Results/transpile_{tag}.csv ({len(df)} rows)")
EOF
    echo "Running FINESSE_AUTO (21 trials, single call)..."
    python3 FrequencyAllocationRuns.py --transpile $IBM_FLAG --finesse-only
fi

echo "Done. Check logs/ for per-seed output."
