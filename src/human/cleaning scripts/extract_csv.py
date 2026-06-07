from datasets import load_dataset
import pandas as pd

# Load Psych-101
ds = load_dataset("marcelbinz/Psych-101", split="train")

# Target experiments
experiments = [
    "steingroever2015data/exp1.csv",
    "steingroever2015data/exp2.csv",
    "steingroever2015data/exp3.csv",
]

for exp in experiments:
    # Filter rows for this experiment
    subset = ds.filter(lambda x: x["experiment"] == exp)

    # Convert directly to DataFrame
    df = pd.DataFrame({
        "experiment": subset["experiment"],
        "participant": subset["participant"],
        "text": subset["text"],
    })

    # Output filename
    out_name = exp.replace("/", "_")
    df.to_csv(out_name, index=False)

    print(f"Saved {out_name} ({len(df)} rows)")
