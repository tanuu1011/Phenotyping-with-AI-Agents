import re
import pandas as pd

TRIAL_PATTERN = re.compile(
    r"You press\s*<<([A-Z])>>\.\s*You win\s*([0-9]+(?:\.[0-9]+)?)\$\s*and lose\s*([0-9]+(?:\.[0-9]+)?)\$\.",
    re.IGNORECASE
)


def parse_human_file(in_csv: str, out_csv: str, loan: float = 2000.0):
    df = pd.read_csv(in_csv)

    required = {"experiment", "participant", "text"}  # <-- add experiment
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{in_csv} missing columns: {missing}")

    rows = []

    for _, r in df.iterrows():

        # Extract experiment number here
        experiment_str = r["experiment"]
        experiment_num = int(re.search(r'exp(\d+)', experiment_str).group(1))

        session_id = int(r["participant"])
        text = str(r["text"])
        matches = TRIAL_PATTERN.findall(text)

        # Build trial rows
        trial_rows = []
        for t, (deck, win_s, lose_s) in enumerate(matches, start=1):
            choice = deck.upper().strip()
            won = float(win_s)
            loss = -float(lose_s)
            trial_rows.append((t, choice, won, loss))

        # Reconstruct totals
        total = loan
        for (t, choice, won, loss) in trial_rows:
            total += (won + loss)
            rows.append({
                "experiment": experiment_num,            # <-- add to output
                "session_id": session_id,
                "trial_number": t,
                "choice": choice,
                "won": won,
                "loss": loss,
                "total": total,
            })

        if len(trial_rows) != 100:
            print(f"Warning: {experiment_num} participant {session_id} has {len(trial_rows)} parsed trials")

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    print(f"Saved {out_csv} with {len(out)} rows")

if __name__ == "__main__":
    #parse_human_file("steingroever2015data_100.csv", "human_variant1_parsed.csv")
    #parse_human_file("steingroever2015data_stop_max_profit.csv", "human_variant2_parsed.csv")
    parse_human_file("steingroever2015data_stop_2500.csv", "human_variant3_parsed.csv")