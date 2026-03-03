import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def shorten(name: str) -> str:
    """Make x-axis labels human-readable."""
    mapping = {
        "distractor_drift": "Distractor",
        "multi_turn_drift": "Multi-Turn",
        "single_prompt_style": "Single Prompt",
        "single_prompts": "Single Prompt",
        "multi_turn_conversations": "Multi-Turn",
        "distractor_prompts": "Distractor",
    }
    return mapping.get(name, name)


def pick_group_col(df: pd.DataFrame) -> str | None:
    """Try to guess what column to group by."""
    candidates = [
        "category",
        "test",
        "bucket",
        "group",
        "prompt_type",
        "dataset",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def require_col(df: pd.DataFrame, col: str, name: str):
    if col not in df.columns:
        raise ValueError(
            f"{name} is missing required column '{col}'. "
            f"Found columns: {df.columns.tolist()}"
        )


def load_json_as_df(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        # in case the JSON is wrapped like {"results": [...]}
        if "results" in data and isinstance(data["results"], list):
            data = data["results"]
        else:
            data = [data]

    if not isinstance(data, list):
        raise ValueError(f"{path} does not contain a JSON list of records.")

    return pd.DataFrame(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="Baseline scored JSON")
    ap.add_argument("--pirate", required=True, help="Pirate scored JSON")
    ap.add_argument("--out_prefix", default="bert", help="Prefix for saved plot files")
    ap.add_argument("--group_col", default=None, help="Column name to group by (optional)")
    ap.add_argument("--score_col", default="prob_pirate", help="Score column name")
    args = ap.parse_args()

    df_base = load_json_as_df(args.baseline)
    df_pir = load_json_as_df(args.pirate)

    require_col(df_base, args.score_col, "baseline JSON")
    require_col(df_pir, args.score_col, "pirate JSON")

    # Clean score column
    df_base[args.score_col] = pd.to_numeric(df_base[args.score_col], errors="coerce")
    df_pir[args.score_col] = pd.to_numeric(df_pir[args.score_col], errors="coerce")
    df_base = df_base.dropna(subset=[args.score_col]).copy()
    df_pir = df_pir.dropna(subset=[args.score_col]).copy()

    # Decide grouping column
    group_col = args.group_col
    if group_col is None:
        group_col = pick_group_col(df_base)
        if group_col is None:
            group_col = pick_group_col(df_pir)

    if group_col is not None and (group_col not in df_base.columns or group_col not in df_pir.columns):
        print(f"[Warning] group_col='{group_col}' not present in BOTH JSON files. Disabling grouped plot.")
        group_col = None

    # =========================
    # Plot 1: Histogram overlay
    # =========================
    plt.figure(figsize=(8, 5))
    plt.hist(df_base[args.score_col].values, bins=40, alpha=0.6, label="Baseline")
    plt.hist(df_pir[args.score_col].values, bins=40, alpha=0.6, label="Pirate")
    plt.xlabel("BERT Pirate Score")
    plt.ylabel("Count")
    plt.title("BERT Pirate Score Distribution")
    plt.legend()
    plt.tight_layout()
    out1 = f"{args.out_prefix}_hist.png"
    plt.savefig(out1, dpi=200)
    plt.close()
    print("Saved:", out1)

    # ======================
    # Plot 2: Boxplot compare
    # ======================
    plt.figure(figsize=(6, 5))
    plt.boxplot(
        [df_base[args.score_col].values, df_pir[args.score_col].values],
        tick_labels=["Baseline", "Pirate"],
        showmeans=True
    )
    plt.ylabel("BERT Pirate Score")
    plt.title("BERT Pirate Score: Baseline vs Pirate")
    plt.tight_layout()
    out2 = f"{args.out_prefix}_box.png"
    plt.savefig(out2, dpi=200)
    plt.close()
    print("Saved:", out2)

    # ==========================================
    # Plot 3: Mean BERT score by prompt group
    # ==========================================
    if group_col is not None:
        g_base = df_base.groupby(group_col)[args.score_col].mean()
        g_pir = df_pir.groupby(group_col)[args.score_col].mean()

        # Align groups that exist in both
        idx = sorted(set(g_base.index).intersection(set(g_pir.index)))
        g_base = g_base.loc[idx]
        g_pir = g_pir.loc[idx]

        labels = [shorten(str(x)) for x in idx]
        x = np.arange(len(labels))
        width = 0.35

        plt.figure(figsize=(8, 5))
        plt.bar(x - width / 2, g_base.values, width, label="Baseline")
        plt.bar(x + width / 2, g_pir.values, width, label="Pirate")

        plt.xticks(x, labels, rotation=0)
        plt.ylabel("Mean BERT Pirate Score")
        plt.title("Mean BERT Pirate Score by Prompt Type")
        plt.legend()
        plt.tight_layout()

        out3 = f"{args.out_prefix}_bar_clean.png"
        plt.savefig(out3, dpi=200)
        plt.close()
        print("Saved:", out3)
    else:
        print("[Info] No grouping column detected. Skipped grouped bar chart.")

    # ==========================================
    # Optional: print summary stats
    # ==========================================
    base_mean = float(df_base[args.score_col].mean())
    pir_mean = float(df_pir[args.score_col].mean())
    print("\n===== SUMMARY =====")
    print(f"Baseline mean: {base_mean:.4f}")
    print(f"Pirate mean  : {pir_mean:.4f}")
    print(f"Delta        : {pir_mean - base_mean:.4f}")


if __name__ == "__main__":
    main()