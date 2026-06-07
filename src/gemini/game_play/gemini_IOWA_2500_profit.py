import os, json, random, csv, re, asyncio
import pandas as pd
from pandas.errors import EmptyDataError
from google import genai
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
filepath = "gemini_igt_2500_profit.csv"

# -----------------------------
# ENV
# -----------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY1")
MODEL = "gemini-3-flash-preview"

client = genai.Client(api_key=GEMINI_API_KEY)


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
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in model response:\n{text}")
    return json.loads(match.group(0))

# -----------------------------
# GEMINI CHOICE — with async
# -----------------------------
async def gemini_choose_deck(state: dict) -> str:
    instructions = (
    "You see in front of you four decks of cards labeled Q, K, S, and D.\n" \
    "You get a loan of 2000$ of play money.\n"
    "You have to select one card at a time, from any of the four decks, until you are told to stop.\n"
    "You select a card from a deck by pressing the corresponding key.\n"
    "After turning a card, you win some money, the amount varies with the deck.\n"
    "You sometimes also have to pay a penalty, which also varies with the deck.\n"
    "Your goal is to try to finish with at least 2500$.\n"
    "\n" "On each trial, choose exactly ONE deck.\n"
    "The deck must be one of: Q, K, S, or D.\n"
    "Return ONLY valid JSON.\n"
    "The JSON must contain exactly one key named \"deck\".\n"
    "Example format (structure only): {\"deck\": \"Q\"}\n"
    "Do not include any text before or after the JSON.\n"
    )

    prompt = instructions + "\n\nState:\n" + json.dumps(state)

    # run_in_executor lets the blocking Gemini SDK call run in a
    # background thread so other sessions can keep making progress
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: client.models.generate_content(model=MODEL, contents=prompt)
    )

    content = response.text.strip()

    try:
        data = extract_json_from_text(content)
    except Exception:
        raise ValueError(f"Invalid JSON from Gemini:\n{content}")

    deck = data.get("deck", data.get("choice", None))
    if deck not in ("Q", "K", "S", "D"):
        raise ValueError(f"Invalid deck returned: {deck}\nRaw:\n{content}")

    return deck

# -----------------------------
# GET NEXT SESSION ID
# -----------------------------
def get_next_session_id(path):
    if not os.path.exists(path):
        return 1
    if os.path.getsize(path) == 0:
        return 1
    try:
        df = pd.read_csv(path)
    except EmptyDataError:
        return 1
    if len(df) == 0 or "session_id" not in df.columns:
        return 1
    return int(df["session_id"].max()) + 1

# -----------------------------
# RUN A SINGLE SESSION — with async
# -----------------------------
async def run_single_session(session_id, n_trials, semaphore, writer, write_lock):
    async with semaphore:  # only max_concurrent sessions run at once
        print(f"\n=== Running session {session_id} ===")

        task = IowaGamblingTask()
        total = 2000.0
        history = []
        rows = []

        for t in range(n_trials):
            state = {
                "trial": t,
                "current_total": total,
                "recent_history": history[-20:]
            }

            choice = await gemini_choose_deck(state)
            win, loss = task.draw(choice)
            total += win + loss

            history.append({"trial": t, "choice": choice, "win": win, "loss": loss})
            rows.append({
                "session_id": session_id,
                "trial_number": t + 1,
                "choice": choice,
                "won": win,
                "loss": loss,
                "total": total
            })

        # Write this session's rows to the file.
        # write_lock makes sure two sessions don't write at the exact same time.
        async with write_lock:
            writer.writerows(rows)

        print(f"=== Session {session_id} done ===")

# -----------------------------
# RUN MULTIPLE SESSIONS — with async
# -----------------------------
async def run_sessions(
    n_sessions=50,
    n_trials=100,
    output_csv="gemini_igt_stop_max_profit.csv",
    max_concurrent=10        # how many sessions run in parallel
):
    ensure_trailing_newline(output_csv)
    next_session = get_next_session_id(output_csv)

    file_exists = os.path.exists(output_csv)
    file_empty = (not file_exists) or (os.path.getsize(output_csv) == 0)

    fieldnames = ["session_id", "trial_number", "choice", "won", "loss", "total"]

    semaphore = asyncio.Semaphore(max_concurrent)
    write_lock = asyncio.Lock()  # prevents garbled writes when sessions finish simultaneously

    with open(output_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if file_empty:
            writer.writeheader()

        tasks = [
            run_single_session(next_session + s, n_trials, semaphore, writer, write_lock)
            for s in range(n_sessions)
        ]

        await asyncio.gather(*tasks)  # runs all 100, but semaphore caps it at 10 at once

    print(f"\nAdded {n_sessions} new sessions to {output_csv}")

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    asyncio.run(run_sessions(
        n_sessions=50,
        n_trials=100,
        output_csv=filepath,
        max_concurrent=10
    ))