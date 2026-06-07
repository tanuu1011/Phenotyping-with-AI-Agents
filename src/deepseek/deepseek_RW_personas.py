# Fit a Rescorla–Wagner + softmax model to each DeepSeek persona run.
# Input CSV should contain:
# session_id, persona, trial_number, choice, won, loss
#
# Output CSV will contain one row per run with fitted:
# alpha, beta, nll, converged

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ------------------ model helpers ------------------
def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)  # numerical stability
    ex = np.exp(x)
    return ex / np.sum(ex)


CHOICE_MAP = {"Q": 0, "K": 1, "S": 2, "D": 3}


def rw_nll(params, choices, rewards):
    alpha, beta = params

    # reject invalid parameters
    if not (0.0 < alpha < 1.0 and beta > 0.0):
        return 1e12

    V = np.zeros(4, dtype=float)  # values for Q,K,S,D
    nll = 0.0

    for a, r in zip(choices, rewards):
        probs = softmax(beta * V)
        p = max(probs[a], 1e-12)
        nll -= np.log(p)

        # Rescorla-Wagner update for chosen deck only
        V[a] += alpha * (r - V[a])

    return nll


def fit_rw_one_run(run_df: pd.DataFrame, beta_max=100.0, reward_scale=50.0):
    run_df = run_df.sort_values("trial_number")

    choices = (
        run_df["choice"]
        .astype(str)
        .str.strip()
        .str.upper()
        .map(CHOICE_MAP)
        .to_numpy()
    )

    if np.any(pd.isna(choices)):
        bad = run_df.loc[pd.isna(choices), "choice"].unique()
        raise ValueError(f"Bad choice values found: {bad}. Expected only Q/K/S/D.")

    choices = choices.astype(int)

    # total reward on each trial = won + loss
    rewards = (
        run_df["won"].astype(float).to_numpy()
        + run_df["loss"].astype(float).to_numpy()
    )
    rewards = rewards / float(reward_scale)

    x0 = np.array([0.3, 1.0])
    bounds = [(1e-4, 1 - 1e-4), (1e-4, beta_max)]

    res = minimize(
        rw_nll,
        x0,
        args=(choices, rewards),
        method="L-BFGS-B",
        bounds=bounds,
    )
    return res


# ------------------ main pipeline ------------------
def main(
    in_csv="../../data/iowa_models/deepseek/deepseek_igt_persona_combined.csv",
    out_csv="../../data/iowa_models/deepseek/deepseek_rw_persona_results.csv",
    beta_max=100.0,
    reward_scale=50.0,
):
    df = pd.read_csv(in_csv)

    needed = {"session_id", "persona", "trial_number", "choice", "won", "loss"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    df["persona"] = df["persona"].astype(str).str.strip()
    df["model"] = "deepseek"

    # unique run id from (persona, session_id)
    # this avoids collisions if session_id restarts within persona
    df["run_id"] = df.groupby(["persona", "session_id"]).ngroup()

    results = []
    for run_id, g in df.groupby("run_id"):
        session_id = int(g["session_id"].iloc[0])
        persona = str(g["persona"].iloc[0])

        res = fit_rw_one_run(g, beta_max=beta_max, reward_scale=reward_scale)

        results.append({
            "run_id": int(run_id),
            "model": "deepseek",
            "session_id": session_id,
            "persona": persona,
            "n_trials": int(len(g)),
            "alpha": float(res.x[0]),
            "beta": float(res.x[1]),
            "nll": float(res.fun),
            "converged": bool(res.success),
        })

    results_df = pd.DataFrame(results).sort_values(["persona", "session_id"])
    results_df.to_csv(out_csv, index=False)

    tol = 1e-3
    n_at_bound = (results_df["beta"] >= beta_max - tol).sum()

    print(f"Runs at beta upper bound: {n_at_bound}/{len(results_df)}")
    print(f"Proportion at bound: {n_at_bound/len(results_df):.2%}")
    print(f"Maximum beta observed: {results_df['beta'].max():.3f}")
    print(f"Saved: {out_csv}\n")

    print("Counts per persona:")
    print(results_df["persona"].value_counts().sort_index(), "\n")

    print("Parameter summary by persona:")
    print(results_df.groupby("persona")[["alpha", "beta", "nll"]].describe())


if __name__ == "__main__":
    main()