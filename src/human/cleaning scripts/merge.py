import pandas as pd

# -------- CONFIG --------
FILES = [
    "steingroever2015data_exp1_stop_max_profit.csv",
    "steingroever2015data_exp2.csv",
    "steingroever2015data_exp3.csv",
]

OUT_PATH = "steingroever2015data_merged.csv"
# ------------------------

dfs = []

for path in FILES:
    df = pd.read_csv(path)

    # Drop only if they exist (safe + explicit)
    cols_to_drop = [c for c in ["instructions", "protocol"] if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # Add provenance column
    df["source_file"] = path

    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)

combined.to_csv(OUT_PATH, index=False)

print(f"Saved merged CSV: {OUT_PATH}")
print("\nColumns:")
print(list(combined.columns))
print("\nRow counts by source_file:")
print(combined["source_file"].value_counts())
