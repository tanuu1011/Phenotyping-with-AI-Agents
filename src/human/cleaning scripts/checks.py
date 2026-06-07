import pandas as pd
import re

# -------- CONFIG --------
CSV_PATH = "steingroever2015data_merged.csv"
# ------------------------

df = pd.read_csv(CSV_PATH)

""" def extract_instructions(text: str) -> str:

    #Extract everything before gameplay starts.
    #Gameplay is identified by the first occurrence of '<<'.

    return text.split("<<", 1)[0].strip()

def normalize_decks(text: str) -> str:

    #Normalize deck labels so only structure is compared.

    return re.sub(
        r"labeled [A-Z](?:, [A-Z])*, and [A-Z]",
        "labeled <DECKS>",
        text
    )

def main():
    df = pd.read_csv(CSV_PATH)

    # Extract and normalize instruction blocks
    df["instructions_norm"] = (
        df["text"]
        .apply(extract_instructions)
        .apply(normalize_decks)
    )

    # Count rows per variant
    counts = df["instructions_norm"].value_counts()

    print(f"CSV: {CSV_PATH}")
    print(f"Total rows: {len(df)}")
    print(f"Number of instruction variants: {len(counts)}\n")

    for i, (instr, count) in enumerate(counts.items(), start=1):
        print("=" * 80)
        print(f"VARIANT {i} — {count} rows")
        print(instr)

if __name__ == "__main__":
    main()
"""


def count_games(text: str) -> int:
    """
    Count how many game choices appear in the transcript.
    Each game is marked by << ... >>.
    """
    return len(re.findall(r"<<.*?>>", text, flags=re.DOTALL))

df["n_games"] = df["text"].apply(count_games)

# Quick sanity check
print(df[["experiment", "participant", "n_games"]].head())

# Optional summary
print("\nSummary statistics:")
print(df["n_games"].describe())