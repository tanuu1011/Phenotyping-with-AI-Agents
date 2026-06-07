import os, json, random, csv, re, asyncio
import pandas as pd
from pandas.errors import EmptyDataError
from google import genai
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# -----------------------------
# ENV
# -----------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY2")
MODEL = "gemini-3-flash-preview"

client = genai.Client(api_key=GEMINI_API_KEY)


# -----------------------------
# PERSONALITY PERSONAS
# These are grounded in IGT psychology literature
# -----------------------------
PERSONAS = {
    "baseline": "",  # No persona

    "risk_averse": (
        "You are a highly cautious and risk-averse decision-maker. "
        "You are extremely sensitive to losses — losing money feels significantly worse to you than gaining the same amount feels good. "
        "When you have experienced a penalty from a deck, you strongly avoid it in the future. "
        "You prefer decks that feel safe and predictable, even if their rewards are smaller. "
        "You would rather win a little consistently than risk a big loss for a bigger reward.\n\n"
    ),

    "risk_seeking": (
        "You are a highly impulsive and reward-seeking decision-maker. "
        "You are strongly attracted to decks that offer the highest immediate rewards. "
        "When a deck gives you a big win, you feel compelled to return to it again and again. "
        "You tend to downplay or ignore penalties — the excitement of a big reward outweighs the pain of a loss. "
        "You act on instinct and find it difficult to resist high-reward options even after repeated bad outcomes.\n\n"
    ),

    "exploratory": (
        "You are a curious and exploratory decision-maker. "
        "Rather than committing to one deck early, you like to sample from all available options to understand what each one offers. "
        "You believe you need enough information from all decks before you can make a good long-term decision. "
        "You actively try decks you haven't chosen recently to see if their behaviour has changed or to fill gaps in your knowledge. "
        "You are comfortable with uncertainty and see exploration as a strategy rather than a mistake.\n\n"
    ),

    "disengaged": (
        "You are feeling disengaged and unmotivated. "
        "You find it difficult to concentrate on patterns or learn from past outcomes. "
        "Wins and losses feel equally unimportant to you — nothing feels particularly rewarding or discouraging. "
        "You make choices without much deliberation and don't feel a strong drive to improve your performance. "
        "Your selections tend to be inconsistent, as you struggle to maintain a strategy across trials.\n\n"
    ),

    "sham_topic": (
        "You are a museum archivist specializing in cataloguing ancient pottery. "
        "You value clear record-keeping and careful documentation. "
        "This background is unrelated to the card task; it should not change your strategy.\n\n"
    ),

    "sham_style": (
        "You are a helpful assistant who communicates in extremely concise, technical language. "
        "Use no emotion and no storytelling. Be consistent and minimal. "
        "This is only about communication style, not decision strategy.\n\n"
    ),
}


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
async def gemini_choose_deck(state: dict, persona: str = "baseline") -> str:
    persona_text = PERSONAS.get(persona, "")
    
    instructions = (
        "You see in front of you four decks of cards labeled Q, K, S, and D.\n"
        "You get a loan of 2000$ of play money.\n"
        "You have to select one card at a time, from any of the four decks, until you are told to stop.\n"
        "You select a card from a deck by pressing the corresponding key.\n"
        "After turning a card, you win some money, the amount varies with the deck.\n"
        "You sometimes also have to pay a penalty, which also varies with the deck.\n"
        "Your goal is to maximize profit on the loan of the play money.\n"
        "\n"
        "On each trial, choose exactly ONE deck.\n"
        "The deck must be one of: Q, K, S, or D.\n"
        "Return ONLY valid JSON.\n"
        "The JSON must contain exactly one key named \"deck\".\n"
        "Example format (structure only): {\"deck\": \"Q\"}\n"
        "Do not include any text before or after the JSON.\n"
    )

    prompt = persona_text + instructions + "\n\nState:\n" + json.dumps(state)

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
# RUN A SINGLE SESSION — with async and persona
# -----------------------------
async def run_single_session(session_id, persona,n_trials, semaphore, writer, write_lock):
    async with semaphore:  # only max_concurrent sessions run at once
        print(f"\n=== Running session {session_id} | persona: {persona} ===")

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

            choice = await gemini_choose_deck(state, persona=persona)
            win, loss = task.draw(choice)
            total += win + loss

            history.append({"trial": t, "choice": choice, "win": win, "loss": loss})
            rows.append({
                "session_id": session_id,
                "persona": persona,
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
# GET PROGRESS FOR A PERSONA
# Returns (next_session_id, sessions_already_done)
# -----------------------------
def get_persona_progress(path, persona):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return 1, 0
    try:
        df = pd.read_csv(path)
    except EmptyDataError:
        return 1, 0
    if len(df) == 0 or "session_id" not in df.columns or "persona" not in df.columns:
        return 1, 0

    persona_df = df[df["persona"] == persona]
    if len(persona_df) == 0:
        # Persona not started yet (session IDs continue from overall max)
        next_id = int(df["session_id"].max()) + 1
        return next_id, 0

    sessions_done = persona_df["session_id"].nunique()
    next_id = int(df["session_id"].max()) + 1  # always continue from global max to avoid ID clashes
    return next_id, sessions_done


# -----------------------------
# RUN MULTIPLE SESSIONS — with async and persona
# -----------------------------
async def run_sessions(
    persona="baseline",
    n_sessions=50,
    n_trials=100,
    output_csv=None,
    max_concurrent=10        # how many sessions run in parallel
):
    if output_csv is None:
        output_csv = f"gemini_igt_{persona}.csv"

    ensure_trailing_newline(output_csv)
    next_session, sessions_done = get_persona_progress(output_csv, persona)

    remaining_sessions = n_sessions - sessions_done

    if remaining_sessions <= 0:
        print(f"Persona '{persona}' already has {sessions_done}/{n_sessions}sessions, so skipping.")
        return
    
    print(f"Persona '{persona}': {sessions_done} done, {remaining_sessions} remaining. Starting from session {next_session}.")

    file_exists = os.path.exists(output_csv)
    file_empty = (not file_exists) or (os.path.getsize(output_csv) == 0)

    fieldnames = ["session_id", "persona", "trial_number", "choice", "won", "loss", "total"]

    semaphore = asyncio.Semaphore(max_concurrent)
    write_lock = asyncio.Lock()  # prevents garbled writes when sessions finish simultaneously

    with open(output_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if file_empty:
            writer.writeheader()

        tasks = [
            run_single_session(next_session + s, persona, n_trials, semaphore, writer, write_lock)
            for s in range(remaining_sessions)
        ]

        await asyncio.gather(*tasks)  # runs all 100, but semaphore caps it at 10 at once

    print(f"\Finished {remaining_sessions} sessions for persona {persona} to {output_csv}")

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    for persona in PERSONAS.keys():
        asyncio.run(run_sessions(
            persona=persona,
            n_sessions=50,
            n_trials=100,
            output_csv=f"gemini_igt_{persona}.csv",
            max_concurrent=10
        ))