"""
compare_experiments.py — compare results across all experiments.

Usage:
    python compare_experiments.py               # all experiments, sorted by Recall@3
    python compare_experiments.py --sort mrr    # sort by MRR
    python compare_experiments.py --sort recall3
    python compare_experiments.py --sort name   # alphabetical

Each experiment must have a config.json in its subdirectory under experiments/.
"""

import argparse
import json
import sys
import io
from pathlib import Path

# Ensure UTF-8 output on Windows (needed for table formatting with special characters)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

EXPERIMENTS_DIR = Path(__file__).parent / "experiments"

SORT_OPTIONS = ["recall1", "recall3", "recall5", "recall10", "mrr", "name"]


def load_experiments() -> list[dict]:
    """Load all experiment configs from experiments/*/config.json."""
    exps = []
    for config_file in sorted(EXPERIMENTS_DIR.glob("*/config.json")):
        try:
            cfg = json.loads(config_file.read_text(encoding="utf-8"))
            cfg["_dir"] = config_file.parent.name
            exps.append(cfg)
        except Exception as e:
            print(f"[Warning] Could not load {config_file}: {e}")
    return exps


def main():
    parser = argparse.ArgumentParser(description="Compare RAG experiment results")
    parser.add_argument("--sort", default="recall3", choices=SORT_OPTIONS,
                        help="metric to sort by (default: recall3)")
    args = parser.parse_args()

    exps = load_experiments()
    if not exps:
        print(f"No experiments found in {EXPERIMENTS_DIR}")
        return

    # Sort by chosen metric (descending), except name (ascending)
    sort_key = {
        "recall1":  lambda e: (e.get("metrics") or {}).get("Recall@1",  -1),
        "recall3":  lambda e: (e.get("metrics") or {}).get("Recall@3",  -1),
        "recall5":  lambda e: (e.get("metrics") or {}).get("Recall@5",  -1),
        "recall10": lambda e: (e.get("metrics") or {}).get("Recall@10", -1),
        "mrr":      lambda e: (e.get("metrics") or {}).get("MRR",       -1),
        "name":     lambda e: e.get("name", ""),
    }
    exps.sort(key=sort_key[args.sort], reverse=(args.sort != "name"))

    # Print results table
    print("\n" + "=" * 90)
    print(f"{'Experiment':<28} {'Retriever':<16} {'Questions':>9} "
          f"{'R@1':>7} {'R@3':>7} {'R@5':>7} {'R@10':>7} {'MRR':>7}  {'Date'}")
    print("=" * 90)

    for e in exps:
        m   = e.get("metrics") or {}
        n   = m.get("n_questions", "?")
        r1  = f"{m['Recall@1']*100:.1f}%"  if "Recall@1"  in m else "-"
        r3  = f"{m['Recall@3']*100:.1f}%"  if "Recall@3"  in m else "-"
        r5  = f"{m['Recall@5']*100:.1f}%"  if "Recall@5"  in m else "-"
        r10 = f"{m['Recall@10']*100:.1f}%" if "Recall@10" in m else "-"
        mrr = f"{m['MRR']:.4f}"            if "MRR"       in m else "-"
        ts  = e.get("timestamp", "")[:10]   # date portion only

        print(f"{e.get('name','?'):<28} {e.get('retriever','?'):<16} {str(n):>9} "
              f"{r1:>7} {r3:>7} {r5:>7} {r10:>7} {mrr:>7}  {ts}")

    print("=" * 90)
    print(f"Total experiments: {len(exps)}")

    # Highlight the best experiment by Recall@3
    best = max(exps, key=lambda e: (e.get("metrics") or {}).get("Recall@3", -1))
    if best.get("metrics"):
        r3_best = best["metrics"].get("Recall@3", 0)
        print(f"\nBest (Recall@3): {best['name']}  ->  {r3_best*100:.1f}%")


if __name__ == "__main__":
    main()
