import pandas as pd

CSV_PATH = "steingroever2015data_exp1.csv"

OUT_FILES = {
    "exp1_100": "steingroever2015data_exp1_100.csv",
    "exp1_stop_profit": "steingroever2015data_exp1_stop_max_profit.csv",
    "exp1_stop_2500": "steingroever2015data_exp1_stop_2500.csv",
}

def extract_instructions(text: str) -> str:
    return text.split("<<", 1)[0].strip().lower()

def classify(instr: str) -> str | None:
    # Fixed horizon
    if "100" in instr and "trial" in instr:
        return "exp1_100"

    # Free horizon
    if "until you are told to stop" in instr:
        if "maximize" in instr or "profit" in instr:
            return "exp1_stop_profit"
        if "2500" in instr:
            return "exp1_stop_2500"

    return None  # anything unexpected

df = pd.read_csv(CSV_PATH)

df["instructions"] = df["text"].apply(extract_instructions)
df["condition"] = df["instructions"].apply(classify)

# Report counts (VERY important)
print("Condition counts:")
print(df["condition"].value_counts(dropna=False))
print()

# Save each condition separately
for condition, out_path in OUT_FILES.items():
    sub = df[df["condition"] == condition].drop(columns=["instructions", "condition"])
    sub.to_csv(out_path, index=False)
    print(f"Saved {len(sub)} rows → {out_path}")

# Optional: inspect any unclassified rows
unclassified = df[df["condition"].isna()]
if len(unclassified) > 0:
    print("\n⚠️ Unclassified rows:", len(unclassified))
