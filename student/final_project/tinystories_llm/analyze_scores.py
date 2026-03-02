import argparse
import pandas as pd

def summarize(df, name="model"):
    print(f"\n===== SUMMARY: {name} =====")

    if "pirate_style_score" not in df.columns:
        raise ValueError("CSV missing pirate_style_score column")

    print("\nOverall Average Style Score:")
    print(df["pirate_style_score"].mean())

    if "contains_pirate_word" in df.columns:
        print("\nPirate Word Rate:")
        print(df["contains_pirate_word"].mean())

    if "unique_word_ratio" in df.columns:
        print("\nAverage Unique Word Ratio:")
        print(df["unique_word_ratio"].mean())

    if "bucket" in df.columns:
        print("\nPer Bucket Style Score:")
        print(df.groupby("bucket")["pirate_style_score"].mean())

    if "max_new_tokens" in df.columns:
        print("\nPer Length Style Score:")
        print(df.groupby("max_new_tokens")["pirate_style_score"].mean())

    if "temperature" in df.columns:
        print("\nPer Temperature Style Score:")
        print(df.groupby("temperature")["pirate_style_score"].mean())


def compare(df1, df2, name1="baseline", name2="patched"):
    print(f"\n===== DELTA ({name2} - {name1}) =====")

    merge_cols = ["bucket", "max_new_tokens", "temperature"]
    available = [c for c in merge_cols if c in df1.columns and c in df2.columns]

    if not available:
        print("Not enough shared grouping columns to compute delta.")
        return

    g1 = df1.groupby(available)["pirate_style_score"].mean().reset_index()
    g2 = df2.groupby(available)["pirate_style_score"].mean().reset_index()

    merged = g1.merge(g2, on=available, suffixes=("_base", "_patched"))
    merged["delta"] = merged["pirate_style_score_patched"] - merged["pirate_style_score_base"]

    print(merged.sort_values("delta", ascending=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--patched", required=False)
    args = parser.parse_args()

    df_base = pd.read_csv(args.baseline)
    summarize(df_base, "baseline")

    if args.patched:
        df_patch = pd.read_csv(args.patched)
        summarize(df_patch, "patched")
        compare(df_base, df_patch)