import os, json, random, csv, re
import pandas as pd
from pandas.errors import EmptyDataError
from openai import OpenAI


filepath ="deepseek_igt_stop_max_profit.csv"

# -----------------------------
# ENV
# -----------------------------
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"), 
    base_url="https://api.deepseek.com"
)


# -----------------------------
# ENSURE FILE ENDS WITH NEWLINE
# -----------------------------
def ensure_trailing_newline(path):
    if not os.path.exists(path):
        return
    with open(path, "rb+") as f:
        f.seek(0, os.SEEK_END)
        if f.tell() == 0:
            return
        f.seek(-1, os.SEEK_END)
        if f.read(1) != b"\n":
            f.write(b"\n")

# -----------------------------
# IGT ENVIRONMENT
# -----------------------------
class IowaGamblingTask:
    def __init__(self):
        self.decks = {
            # Bad / disadvantageous decks
            "Q": {"reward": 100.0, "loss_prob": 0.5, "loss": -250.0},   # frequent losses
            "K": {"reward": 100.0, "loss_prob": 0.1, "loss": -1250.0},  # rare large loss

            # Good / advantageous decks
            "S": {"reward": 50.0, "loss_prob": 0.5, "loss": -50.0},     # frequent small losses
            "D": {"reward": 50.0, "loss_prob": 0.1, "loss": -250.0},    # rare moderate loss
        }

    def draw(self, deck: str):
        config = self.decks[deck]
        win = config["reward"]
        loss = config["loss"] if random.random() < config["loss_prob"] else 0.0
        return win, loss

# -----------------------------
# SAFE JSON EXTRACTION
# -----------------------------
def extract_json_from_text(text: str) -> dict:
    # Pull the first {...} block even if wrapped in ```json ... ```
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in model response:\n{text}")
    return json.loads(match.group(0))

# -----------------------------
# DEEPSEEK CHOICE
# -----------------------------
def deepseek_choose_deck(state: dict) -> str:
    instructions = ( 
    "You see in front of you four decks of cards labeled Q, K, S, and D.\n" \
    "You get a loan of 2000$ of play money.\n"
    "You have to select one card at a time, from any of the four decks, until you are told to stop.\n"
    "You select a card from a deck by pressing the corresponding key.\n"
    "After turning a card, you win some money, the amount varies with the deck.\n"
    "You sometimes also have to pay a penalty, which also varies with the deck.\n"
    "Your goal is to maximize profit on the loan of the play money.\n"
    "\n" "On each trial, choose exactly ONE deck.\n"
    "The deck must be one of: Q, K, S, or D.\n"
    "Return ONLY valid JSON.\n"
    "The JSON must contain exactly one key named \"deck\".\n"
    "Example format (structure only): {\"deck\": \"Q\"}\n"
    "Do not include any text before or after the JSON.\n" 
    )

    prompt = instructions + "\n\nState:\n" + json.dumps(state)

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    content = response.choices[0].message.content.strip()

    try:
        data = extract_json_from_text(content)
    except Exception:
        raise ValueError(f"Invalid JSON from DeepSeek:\n{content}")

    # be tolerant if it ever returns {"choice":"Q"} again
    deck = data.get("deck", data.get("choice", None))
    if deck not in ("Q", "K", "S", "D"):
        raise ValueError(f"Invalid deck returned: {deck}\nRaw:\n{content}")

    return deck

# -----------------------------
# GET NEXT SESSION ID
# -----------------------------
def get_next_session_id(path):
    # If file doesn't exist -> first session
    if not os.path.exists(path):
        return 1

    # If file exists but is empty (0 bytes) -> first session
    if os.path.getsize(path) == 0:
        return 1

    try:
        df = pd.read_csv(path)
    except EmptyDataError:
        return 1  # file has no parsable content (e.g., blank lines)

    # If no rows or no session_id column -> first session
    if len(df) == 0 or "session_id" not in df.columns:
        return 1

    return int(df["session_id"].max()) + 1

# -----------------------------
# RUN MULTIPLE SESSIONS
# -----------------------------
def run_sessions(
    n_sessions=30,
    n_trials=100,
    output_csv="deepseek_igt_stop_max_profit.csv"
):
    # Fix newline issues BEFORE appending
    ensure_trailing_newline(output_csv)

    next_session = get_next_session_id(output_csv)

    file_exists = os.path.exists(output_csv)
    file_empty = (not file_exists) or (os.path.getsize(output_csv) == 0)

    fieldnames = ["session_id", "trial_number", "choice", "won", "loss", "total"]

    with open(output_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if file_empty:
            writer.writeheader()

        for s in range(n_sessions):
            session_id = next_session + s
            print(f"\n=== Running session {session_id} ===")

            task = IowaGamblingTask()
            total = 2000.0
            history = []

            for t in range(n_trials):
                state = {
                    "trial": t,
                    "current_total": total,
                    "recent_history": history[-20:]
                }

                choice = deepseek_choose_deck(state)
                win, loss = task.draw(choice)
                total += win + loss

                history.append({"trial": t, "choice": choice, "win": win, "loss": loss})

                writer.writerow({
                    "session_id": session_id,
                    "trial_number": t + 1,
                    "choice": choice,
                    "won": win,
                    "loss": loss,
                    "total": total
                })

    print(f"\nAdded {n_sessions} new sessions to {output_csv}")

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    run_sessions(n_sessions=200, n_trials=100, output_csv=filepath)

