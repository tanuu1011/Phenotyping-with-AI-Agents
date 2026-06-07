import os, json, random, csv, re
import pandas as pd
from pandas.errors import EmptyDataError
from openai import OpenAI


# -----------------------------
# ENV
# -----------------------------
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"), 
    base_url="https://api.deepseek.com"
)


# -----------------------------
# PERSONAS
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
# DEEPSEEK CHOICE
# -----------------------------
def deepseek_choose_deck(state: dict, persona_name: str) -> str:
    persona_text = PERSONAS[persona_name]

    game_instructions = ( 
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

    # Persona comes FIRST, then game instructions
    prompt = persona_text + "\n\n" + game_instructions + "\n\nState:\n" + json.dumps(state)

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "Follow the user's behavioural instructions consistently."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        timeout=30
    )

    content = response.choices[0].message.content.strip()

    try:
        data = extract_json_from_text(content)
    except Exception:
        raise ValueError(f"Invalid JSON from DeepSeek:\n{content}")

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
# RUN MULTIPLE SESSIONS -- now with persona column in CSV
# -----------------------------
def run_sessions(
    persona: str = "baseline",
    n_sessions: int = 50,
    n_trials: int = 100,
    output_csv: str = None
):
    if output_csv is None:
        output_csv = f"deepseek_igt_{persona}.csv"

    ensure_trailing_newline(output_csv)
    next_session = get_next_session_id(output_csv)

    file_exists = os.path.exists(output_csv)
    file_empty = (not file_exists) or (os.path.getsize(output_csv) == 0)

    # Note: we now save the persona in each row so you can
    # combine all CSVs into one file and still know which is which
    fieldnames = ["session_id", "persona", "trial_number", "choice", "won", "loss", "total"]

    with open(output_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if file_empty:
            writer.writeheader()

        for s in range(n_sessions):
            session_id = next_session + s
            print(f"\n=== Running session {session_id} | persona: {persona} ===")

            task = IowaGamblingTask()
            total = 2000.0
            history = []

            for t in range(n_trials):
                print(f"Session {session_id}, Trial {t}")
                state = {
                    "trial": t,
                    "current_total": total,
                    "recent_history": history[-20:]
                }

                choice = deepseek_choose_deck(state, persona)
                win, loss = task.draw(choice)
                total += win + loss

                history.append({"trial": t, "choice": choice, "win": win, "loss": loss})

                writer.writerow({
                    "session_id": session_id,
                    "persona":    persona,
                    "trial_number": t + 1,
                    "choice":     choice,
                    "won":        win,
                    "loss":       loss,
                    "total":      total
                })

    print(f"\nFinished {n_sessions} sessions for persona '{persona}' → {output_csv}")

# -----------------------------
# RUN ALL PERSONAS
# -----------------------------
if __name__ == "__main__":
    for persona_name in PERSONAS:
        run_sessions(
            persona=persona_name,
            n_sessions=50,
            n_trials=100)