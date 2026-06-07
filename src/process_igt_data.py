import pandas as pd
import os
import shutil

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Inputs are inside src/deepseek/game_play and src/gemini/game_play
BASE_SRC = BASE_DIR

# Outputs go to project_root/data/iowa_models
BASE_DST = os.path.join(BASE_DIR, "..", "data", "iowa_models")

MODELS = ["deepseek", "gemini"]

VARIANT_FILES = {
    "igt_stop_100_trials": ("variant1", 1),
    "igt_stop_max_profit": ("variant2", 2),
    "igt_2500_profit": ("variant3", 3),
}

PERSONA_FILES = [
    "baseline",
    "exploratory",
    "disengaged",
    "risk_averse",
    "risk_seeking",
    "sham_style",
    "sham_topic",
]

# True = move copied raw files into archive instead of deleting forever
ARCHIVE_INSTEAD_OF_DELETE = False


# -----------------------------
# HELPERS
# -----------------------------
def move_or_delete(src_file, model):
    if ARCHIVE_INSTEAD_OF_DELETE:
        archive_dir = os.path.join(BASE_SRC, model, "game_play", "archive")
        os.makedirs(archive_dir, exist_ok=True)

        archive_path = os.path.join(archive_dir, os.path.basename(src_file))

        # If same file already exists in archive, overwrite it
        if os.path.exists(archive_path):
            os.remove(archive_path)

        shutil.move(src_file, archive_path)
        print(f"Moved source to archive: {archive_path}")
    else:
        os.remove(src_file)
        print(f"Deleted source: {src_file}")


# -----------------------------
# MAIN FUNCTION
# -----------------------------
def process_model(model):
    print(f"\nProcessing {model}...")

    src_dir = os.path.join(BASE_SRC, model, "game_play")
    dst_dir = os.path.join(BASE_DST, model)

    os.makedirs(dst_dir, exist_ok=True)

    combined_list = []
    persona_combined_list = []

    # -----------------------------
    # VARIANT FILES
    # -----------------------------
    for key, (variant_name, variant_num) in VARIANT_FILES.items():
        src_file = os.path.join(src_dir, f"{model}_{key}.csv")
        dst_file = os.path.join(dst_dir, f"{model}_igt_{variant_name}.csv")

        if not os.path.exists(src_file):
            print(f"Missing: {src_file}")
            continue

        df = pd.read_csv(src_file)

        df["variant"] = variant_num
        df["model"] = model

        df.to_csv(dst_file, index=False)
        print(f"Saved: {dst_file} ({len(df)} rows)")

        combined_list.append(df)

        move_or_delete(src_file, model)

    # -----------------------------
    # COMBINED VARIANT FILE
    # -----------------------------
    if combined_list:
        combined_df = pd.concat(combined_list, ignore_index=True)
        combined_path = os.path.join(dst_dir, f"{model}_igt_combined.csv")

        combined_df.to_csv(combined_path, index=False)
        print(f"Combined saved: {combined_path} ({len(combined_df)} rows)")

    # -----------------------------
    # PERSONA FILES
    # -----------------------------
    for persona in PERSONA_FILES:
        src_file = os.path.join(src_dir, f"{model}_igt_{persona}.csv")
        dst_file = os.path.join(dst_dir, f"{model}_igt_{persona}.csv")

        if not os.path.exists(src_file):
            print(f"Missing persona: {src_file}")
            continue

        df = pd.read_csv(src_file)

        df["persona"] = persona
        df["model"] = model

        df.to_csv(dst_file, index=False)
        print(f"Saved persona: {dst_file} ({len(df)} rows)")

        persona_combined_list.append(df)

        move_or_delete(src_file, model)

    # -----------------------------
    # PERSONA COMBINED FILE
    # -----------------------------
    if persona_combined_list:
        persona_df = pd.concat(persona_combined_list, ignore_index=True)
        persona_combined_path = os.path.join(
            dst_dir, f"{model}_igt_persona_combined.csv"
        )

        persona_df.to_csv(persona_combined_path, index=False)
        print(
            f"Persona combined saved: {persona_combined_path} "
            f"({len(persona_df)} rows)"
        )


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    print("Running from:", os.getcwd())
    print("BASE_SRC:", BASE_SRC)
    print("BASE_DST:", BASE_DST)

    for model in MODELS:
        process_model(model)

    print("\nDone.")