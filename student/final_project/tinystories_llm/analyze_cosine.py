import argparse
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--baseline", required=True)
ap.add_argument("--patched", required=True)
ap.add_argument("--out_csv", default="cosine_comparison.csv")
args = ap.parse_args()

df_base = pd.read_csv(args.baseline)
df_patch = pd.read_csv(args.patched)

if "cosine_style_score" not in df_base.columns:
    raise ValueError("Baseline missing cosine_style_score")

if "cosine_style_score" not in df_patch.columns:
    raise ValueError("Patched missing cosine_style_score")

# Ensure same length
if len(df_base) != len(df_patch):
    raise ValueError("Baseline and patched CSV must have same number of rows")

merged = pd.DataFrame()
merged["cosine_base"] = df_base["cosine_style_score"]
merged["cosine_patched"] = df_patch["cosine_style_score"]
merged["cosine_delta"] = (
    merged["cosine_patched"] - merged["cosine_base"]
)

print("\n===== COSINE SUMMARY =====")
print("Baseline mean :", merged["cosine_base"].mean())
print("Patched mean  :", merged["cosine_patched"].mean())
print("Delta mean    :", merged["cosine_delta"].mean())

merged.to_csv(args.out_csv, index=False)
print(f"\nSaved: {args.out_csv}")