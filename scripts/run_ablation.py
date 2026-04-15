"""
Run configurable transpilation benchmarks and save the results to parquet.

Examples:
    finesse-bench
    finesse-bench --configs qiskit_sabre,sabre,lightsabre,mirage
    finesse-bench --configs qiskit_sabre_fid,sabre_fid,mirage_fid --trials 5
    finesse-bench --circuits rd84_142,ham7_104 --configs mirage,mirage_fid
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from finesse import BENCH_CIRCUITS, QUICK_CIRCUITS
from finesse.ablation import make_synthetic_fidelity
from finesse.benchmarks import BenchmarkConfig, make_tokyo, run_benchmark


CONFIG_PRESETS = {
    "qiskit_sabre":     ("qiskit_sabre", False, 0),
    "qiskit_sabre_fid": ("qiskit_sabre", True, 0),
    "sabre":            ("sabre", False, 0),
    "sabre_fid":        ("sabre", True, 0),
    "lightsabre":       ("lightsabre", False, 0),
    "lightsabre_fid":   ("lightsabre", True, 0),
    "mirage":           ("mirage", False, 0),
    "mirage_fid":       ("mirage", True, 0),
    "finesse":          ("mirage", True, 1),
}


def _parse_config(name: str, trials: int, aggression: int) -> BenchmarkConfig:
    try:
        router, use_fidelity, bidir_passes = CONFIG_PRESETS[name]
    except KeyError as exc:
        valid = ", ".join(CONFIG_PRESETS)
        raise ValueError(f"Unknown config {name!r}. Valid configs: {valid}") from exc
    cfg_aggression = aggression if router in {"mirage", "mirage_sabre"} else None
    return BenchmarkConfig(
        name=name,
        router=router,
        use_fidelity=use_fidelity,
        trials=trials,
        bidir_passes=bidir_passes,
        aggression=cfg_aggression,
    )


def _parse_circuits(value: str):
    if value == "quick":
        return QUICK_CIRCUITS
    if value == "bench":
        return BENCH_CIRCUITS
    out = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        label = item[:-5] if item.endswith(".qasm") else item
        qasm_name = item if item.endswith(".qasm") else f"{item}.qasm"
        out.append((label, qasm_name))
    return out


def main():
    parser = argparse.ArgumentParser(description="Configurable FINESSE benchmark runner")
    parser.add_argument(
        "--configs",
        default="qiskit_sabre,sabre,lightsabre,mirage,mirage_fid",
        help="comma-separated config presets",
    )
    parser.add_argument(
        "--circuits",
        default="quick",
        help="quick, bench, or a comma-separated list like rd84_142,ham7_104",
    )
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument(
        "--aggression",
        type=int,
        default=2,
        choices=[0, 1, 2, 3],
        help="mirror acceptance aggression for MIRAGE-style configs",
    )
    parser.add_argument("--out", default="benchmark.parquet")
    args = parser.parse_args()

    config_names = [item.strip() for item in args.configs.split(",") if item.strip()]
    configs = [_parse_config(name, args.trials, args.aggression) for name in config_names]
    circuits = _parse_circuits(args.circuits)

    cm = make_tokyo()
    F = make_synthetic_fidelity(cm, seed=42)

    print("=" * 60)
    print(
        f"Benchmark: {len(configs)} configs × {len(circuits)} circuits × "
        f"{args.seeds} seeds × {args.trials} trials"
    )
    print("=" * 60)
    df = run_benchmark(
        configs=configs,
        circuits=circuits,
        coupling_map=cm,
        n_seeds=args.seeds,
        fidelity_matrix=F,
        verbose=True,
    )

    rows = df.to_dict("records") if hasattr(df, "to_dict") else df

    meta = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "configs": ",".join(config_names),
        "circuits": args.circuits,
        "n_seeds": str(args.seeds),
        "n_trials": str(args.trials),
        "aggression": str(args.aggression),
    }
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError:
        out_path = args.out if args.out.endswith(".json") else f"{args.out}.json"
        with open(out_path, "w") as f:
            json.dump({"meta": meta, "rows": rows}, f, indent=2)
        print(f"\nSaved {len(rows):,} rows -> {out_path}")
    else:
        if hasattr(df, "to_dict"):
            table = pa.Table.from_pandas(df)
        else:
            table = pa.Table.from_pylist(rows)
        encoded_meta = {k: v.encode() if isinstance(v, str) else v for k, v in meta.items()}
        table = table.replace_schema_metadata({**(table.schema.metadata or {}), **encoded_meta})
        pq.write_table(table, args.out)
        print(f"\nSaved {len(rows):,} rows -> {args.out}")

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["config"]].append(row)

    print(f'\n{"Config":<18} {"Avg SWAPs":>10} {"Avg depth":>10} {"Avg -logF":>12}')
    print("-" * 56)
    for cfg in config_names:
        cfg_rows = grouped[cfg]
        avg_swaps = float(np.mean([row["swap_count"] for row in cfg_rows]))
        avg_depth = float(np.mean([row["gate_depth"] for row in cfg_rows]))
        lf_vals = np.array([row["lf_cost"] for row in cfg_rows], dtype=float)
        avg_lf = float(np.nanmean(lf_vals))
        lf = f"{avg_lf:.2f}" if np.isfinite(avg_lf) else "n/a"
        print(f"{cfg:<18} {avg_swaps:>10.1f} {avg_depth:>10.1f} {lf:>12}")


if __name__ == "__main__":
    main()
