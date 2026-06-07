
# Code fits a Rescorla–Wagner learning model with a softmax choice rule to each human run.
# The goal is to estimate two parameters for each run:
# alpha = learning rate (how strongly the model updates its expectations)
# beta  = inverse temperature (how deterministic the choices are)

# IMPORTANT FIX: sessions are uniquely identified by (variant, session_id, experiment)
# because session_id often restarts at 1 in each CSV. Without this fix,
# runs from different variants could accidentally be merged together.

# Unique Deck choices handled with n_actions = no. of unique decks present in given run

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ------------------ model helpers ------------------
def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)  # numerical stability
    ex = np.exp(x)
    return ex / np.sum(ex)


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

def rw_nll(params, choices, rewards, n_actions):

    alpha, beta = params

    # reject invalid parameter values by returning a large penalty
    if not (0.0 < alpha < 1.0 and beta > 0.0):   
        return 1e12        # big penalty if params outside range

    V = np.zeros(n_actions, dtype=float)  # values for [Q,K,S,D]

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


def fit_rw_one_run(run_df: pd.DataFrame, beta_max=50.0, reward_scale=50.0):
    run_df = run_df.sort_values("trial_number").copy()

    # clean labels
    run_df["choice_clean"] = run_df["choice"].astype(str).str.strip().str.upper()

    # DYNAMIC MAPPING: each run can use different letters
    # find the unique deck labels used in THIS run only
    # example: ['D', 'H', 'J', 'V']
    unique_choices = sorted(run_df["choice_clean"].unique())

    # create a mapping from deck labels to integer indices
    # example:
    # {'D': 0, 'H': 1, 'J': 2, 'V': 3}
    choice_map = {choice: i for i, choice in enumerate(unique_choices)}

    # replace deck letters with integer action indices
    # example choices sequence:
    # D, V, D, J -> 0, 3, 0, 2
    choices = run_df["choice_clean"].map(choice_map).astype(int).to_numpy()
    # n_actions = number of available decks in this run (4)
    n_actions = len(unique_choices)

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
        args=(choices, rewards, n_actions),   # data passed to objective function
        method="L-BFGS-B",         # optimisation algorithm
        bounds=bounds              # enforce parameter limits
    )

    # res        = optimizer result object
    # choice_map = mapping used in this run
    # n_actions  = number of decks/options in this run
    return res, choice_map, n_actions



# ------------------ main pipeline ------------------
def main(
    in_csv="../../data/iowa_human/combined_igt_human.csv",
    out_csv="../../data/iowa_human/human_rw_results.csv",
    beta_max=100.0,
    reward_scale=50.0,  # scale because rewards are around +/-50; makes beta more stable
):
    df = pd.read_csv(in_csv)

    # quick checks
    needed = {"experiment","session_id", "trial_number", "choice", "won", "loss", "variant"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # ensure variant is clean int
    df["variant"] = pd.to_numeric(df["variant"], errors="raise").astype(int)
    df["experiment"] = pd.to_numeric(df["experiment"], errors="raise").astype(int)

    results = []

    # FIX: create a globally unique run id from (variant, session_id)
    # this prevents runs from different variants but same session_id
    # from being incorrectly merged together
    # one run = one participant/session in one experiment and one variant
    grouped = df.groupby(["experiment", "session_id", "variant"])

    
    for (experiment, session_id, variant), g in grouped:
        n_unique = g["choice"].astype(str).str.strip().str.upper().nunique()

        res, choice_map, n_actions = fit_rw_one_run(
            g,
            beta_max=beta_max,
            reward_scale=reward_scale
        )

        # store the results
        results.append({
            "experiment": int(experiment),
            "session_id": int(session_id),
            "variant": int(variant),
            "n_trials": int(len(g)),
            "n_unique_choices": int(n_unique),
            "n_actions": int(n_actions),
            "action_labels": ",".join(choice_map.keys()),
            "alpha": float(res.x[0]),
            "beta": float(res.x[1]),
            "nll": float(res.fun),
            "converged": bool(res.success),
        })
        
    results_df = pd.DataFrame(results).sort_values(["experiment", "variant", "session_id"])
    results_df.to_csv(out_csv, index=False)

    tol = 1e-3
    n_at_bound = (results_df["beta"] >= beta_max - tol).sum()

    print(f"Runs at beta upper bound: {n_at_bound}/{len(results_df)}")
    print(f"Proportion at bound: {n_at_bound/len(results_df):.2%}")
    print(f"Maximum beta observed: {results_df['beta'].max():.3f}")

    print(f"Saved: {out_csv}\n")

    print("Counts per variant:")
    print(results_df["variant"].value_counts().sort_index(), "\n")

    print("Counts per experiment:")
    print(results_df["experiment"].value_counts().sort_index(), "\n")

    print("Runs with only one unique choice:")
    print((results_df["n_unique_choices"] == 1).sum(), "\n")

    print("Parameter summary by variant:")
    print(results_df.groupby("variant")[["alpha", "beta", "nll"]].describe())


if __name__ == "__main__":
    main()