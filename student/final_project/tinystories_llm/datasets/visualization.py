import pandas as pd
import matplotlib.pyplot as plt

# Load your comparison CSVs
single = pd.read_csv("datasets/single_cos_comparison.csv")
multi = pd.read_csv("datasets/multi_cos_comparison.csv")
distractor = pd.read_csv("datasets/distractor_cos_comparison.csv")

def mean_barplot(df, title):
    base_mean = df["cosine_style_score_base"].mean()
    patched_mean = df["cosine_style_score_patched"].mean()

    plt.figure()
    plt.bar(["Baseline", "Patched"], [base_mean, patched_mean])
    plt.title(title)
    plt.ylabel("Mean Cosine Similarity")
    plt.show()

def delta_histogram(df, title, bins=20):
    # If cosine_delta is already in your csv, this uses it.
    # If not, it computes it.
    if "cosine_delta" not in df.columns:
        df["cosine_delta"] = df["cosine_style_score_patched"] - df["cosine_style_score_base"]

    plt.figure()
    plt.hist(df["cosine_delta"], bins=bins)
    plt.title(title)
    plt.xlabel("Cosine Delta (Patched - Baseline)")
    plt.ylabel("Frequency")
    plt.show()

# 1) Mean cosine bar charts
mean_barplot(single, "Single Prompt Mean Cosine Similarity")
mean_barplot(multi, "Multi-Turn Mean Cosine Similarity")
mean_barplot(distractor, "Distractor Mean Cosine Similarity")

# 2) Delta distribution histograms (most useful for multi-turn)
delta_histogram(multi, "Multi-Turn Cosine Delta Distribution (Patched - Baseline)")
delta_histogram(distractor, "Distractor Cosine Delta Distribution (Patched - Baseline)")
delta_histogram(single, "Single Prompt Cosine Delta Distribution (Patched - Baseline)")