import pandas as pd
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--baseline", required=True)
ap.add_argument("--patched", required=True)
ap.add_argument("--out_csv", default="cosine_comparison.csv")
args = ap.parse_args()

df_base = pd.read_csv(args.baseline)
df_patch = pd.read_csv(args.patched)

# Merge on common keys
merge_cols = ["test", "max_new_tokens", "temperature", "top_p"]

merged = df_base.merge(
    df_patch,
    on=merge_cols,
    suffixes=("_base", "_patched")
)

merged["cosine_delta"] = (
    merged["cosine_style_score_patched"]
    - merged["cosine_style_score_base"]
)

print("\n===== COSINE SUMMARY =====")
print("Overall Baseline:", merged["cosine_style_score_base"].mean())
print("Overall Patched :", merged["cosine_style_score_patched"].mean())
print("Overall Delta   :", merged["cosine_delta"].mean())

print("\nPer Test Delta:")
print(
    merged.groupby("test")["cosine_delta"].mean()
)

merged.to_csv(args.out_csv, index=False)
print(f"\nSaved: {args.out_csv}")