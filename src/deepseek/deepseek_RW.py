
# Code fits a Rescorla–Wagner learning model with a softmax choice rule to each DeepSeek run.
# The goal is to estimate two parameters for each run:
# alpha = learning rate (how strongly the model updates its expectations)
# beta  = inverse temperature (how deterministic the choices are)

# IMPORTANT FIX: sessions are uniquely identified by (variant, session_id)
# because session_id often restarts at 1 in each CSV. Without this fix,
# runs from different variants could accidentally be merged together.

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ------------------ model helpers ------------------
def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)  # numerical stability
    ex = np.exp(x)
    return ex / np.sum(ex)

# Map deck labels to integer indices so they can be used in arrays.
# Q,K,S,D are the four decks in the Iowa Gambling Task style setup.
CHOICE_MAP = {"Q": 0, "K": 1, "S": 2, "D": 3}


# nll = negative log-likelihood function (lower value = better model fit)
# The optimizer will search for alpha and beta values that minimize this.
#
# Parameter meaning:
# 0 < alpha < 1
#   alpha = learning rate
#   alpha = 0   -> no learning
#   alpha = 1   -> value completely replaced by latest reward
#
# beta > 0
#   beta controls how strongly the model prefers higher-valued options
#   small beta  -> more random behaviour
#   large beta  -> more deterministic behaviour

def rw_nll(params, choices, rewards):

    alpha, beta = params

    # reject invalid parameter values by returning a large penalty
    if not (0.0 < alpha < 1.0 and beta > 0.0):   
        return 1e12        # big penalty if params outside range

    V = np.zeros(4, dtype=float)  # values for [Q,K,S,D]

    # accumulator for negative log likelihood across trials
    nll = 0.0

    # iterate through trials sequentially
    # a = action (deck chosen)
    # r = reward obtained
    for a, r in zip(choices, rewards):
        probs = softmax(beta * V)      # small beta, more random,  big beta, more decisive
        p = max(probs[a], 1e-12)       # a = actual choice
        nll -= np.log(p)               # if the model predicted the choice with high probability,
                                       # the penalty added here will be small

        # Rescorla–Wagner update for chosen deck only
        # prediction error = (actual reward - expected value)
        # value update: V_new = V_old + alpha * prediction_error
        V[a] += alpha * (r - V[a])

    return nll      # TOTAL FIT ERROR


def fit_rw_one_run(run_df: pd.DataFrame, beta_max=50.0, reward_scale=50.0):   # scaling reward by 50
    run_df = run_df.sort_values("trial_number")

    # choices: Q/K/S/D -> 0/1/2/3
    choices = run_df["choice"].astype(str).str.strip().str.upper().map(CHOICE_MAP).to_numpy()
    if np.any(pd.isna(choices)):
        bad = run_df.loc[pd.isna(choices), "choice"].unique()
        raise ValueError(f"Bad choice values found: {bad}. Expected only Q/K/S/D.")

    choices = choices.astype(int)

    # IMPORTANT: loss column is already negative 
    # So trial reward is won + loss
    rewards = (run_df["won"].astype(float) + run_df["loss"].astype(float)).to_numpy()
    rewards = rewards / float(reward_scale)

    x0 = np.array([0.3, 1.0])  # initial alpha, beta (starting guess)

    # parameter bounds
    # alpha must stay between 0 and 1
    # beta is constrained to a positive range
    bounds = [(1e-4, 1 - 1e-4), (1e-4, beta_max)]

    # run numerical optimisation to find best alpha and beta
    res = minimize(
        rw_nll,            # objective function
        x0,                # starting parameter guess
        args=(choices, rewards),   # data passed to objective function
        method="L-BFGS-B",         # optimisation algorithm
        bounds=bounds              # enforce parameter limits
    )
    return res



# ------------------ main pipeline ------------------
def main(
    in_csv="../../data/iowa_models/deepseek/deepseek_igt_combined.csv",
    out_csv="../../data/iowa_models/deepseek/deepseek_rw_results.csv",
    beta_max=100.0,
    reward_scale=50.0,  # scale because rewards are around +/-50; makes beta more stable
):
    df = pd.read_csv(in_csv)

    # quick checks
    needed = {"session_id", "trial_number", "choice", "won", "loss", "variant"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # ensure variant is clean int
    df["variant"] = pd.to_numeric(df["variant"], errors="raise").astype(int)

    # FIX: create a globally unique run id from (variant, session_id)
    # this prevents runs from different variants but same session_id
    # from being incorrectly merged together
    df["run_id"] = df.groupby(["variant", "session_id"]).ngroup()

    results = []
    for run_id, g in df.groupby("run_id"):
        variant = int(g["variant"].iloc[0])
        session_id = int(g["session_id"].iloc[0])

        res = fit_rw_one_run(g, beta_max=beta_max, reward_scale=reward_scale)

        # store the results
        results.append({
            "run_id": int(run_id),
            "session_id": session_id,
            "variant": variant,
            "n_trials": int(len(g)),      # number of trials in this run
            "alpha": float(res.x[0]),     # fitted learning rate
            "beta": float(res.x[1]),      # fitted inverse temperature
            "nll": float(res.fun),        # final negative log-likelihood
            "converged": bool(res.success),  # whether optimizer converged
        })
        
    results_df = pd.DataFrame(results).sort_values(["variant", "session_id"])
    results_df.to_csv(out_csv, index=False)

    tol = 1e-3
    n_at_bound = (results_df["beta"] >= beta_max - tol).sum()

    print(f"Runs at beta upper bound: {n_at_bound}/{len(results_df)}")
    print(f"Proportion at bound: {n_at_bound/len(results_df):.2%}")
    print(f"Maximum beta observed: {results_df['beta'].max():.3f}")

    print(f"Saved: {out_csv}\n")
    print("Counts per variant (should include 3):")
    print(results_df["variant"].value_counts().sort_index(), "\n")

    print("Parameter summary by variant:")
    print(results_df.groupby("variant")[["alpha", "beta", "nll"]].describe())

if __name__ == "__main__":
    main()