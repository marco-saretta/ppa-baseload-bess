"""Risk measures on scenario profits, in numpy.

The model needs them for the disagreement points (TODO 5 in model.py): without the contract
the profits are data, not variables, so their CVaR is a plain numpy computation. It must be
the same CVaR as the one the Gurobi constraints encode (TODO 1-4), otherwise w = u - d compares
two different things.

No for loops: use broadcasting, @ and numpy reductions.
"""

import numpy as np


def cvar(x: np.ndarray, p: np.ndarray, alpha: float) -> float:
    """CVaR of the profits x: the expected profit in the worst (1 - alpha) share of scenarios.

    Args:
        x: profit in each scenario, EUR, shape (S,)
        p: probability of each scenario, shape (S,), sums to 1
        alpha: confidence level, 0 <= alpha < 1

    Returns:
        CVaR, EUR

    TODO 5a - Implement the CVaR with the formula of TODO 1-2 in model.py, where VaR was a
      decision variable:
        CVaR = max over VaR of  VaR - 1 / (1 - alpha) * sum_s p[s] * max(VaR - x[s], 0)
      Here there is no solver to choose VaR, but you do not need one: the best VaR is always
      one of the scenario profits x[s]. So try every x[s] as VaR at the same time:
        - x[:, None] - x[None, :] is a matrix of shape (S, S): row = candidate VaR, column = scenario
        - np.maximum(..., 0) gives VaR_aux for every candidate and scenario
        - @ p takes the expectation over the scenarios for every candidate
        - np.max picks the best candidate
      Test it by hand before using it in the model. With x = [-10, 0, 10] and p = [1/3, 1/3, 1/3]:
        alpha = 0    -> 0    (the expectation)
        alpha = 1/3  -> -5   (mean of the worst two)
        alpha = 2/3  -> -10  (the worst one)
    """
    # Every scenario profit is a candidate VaR: rows = candidates, columns = scenarios.
    # VaR_aux[c, s] = max(x[c] - x[s], 0): how far scenario s falls below candidate c
    var_aux = np.maximum(x[:, None] - x[None, :], 0)  # shape (S, S)
    # The CVaR formula for every candidate at once, then keep the best one
    values = x - 1 / (1 - alpha) * (var_aux @ p)  # shape (S,)
    return float(np.max(values))


def risk_adjusted(x: np.ndarray, p: np.ndarray, alpha: float, beta: float) -> float:
    """The utility the model maximises, for the profits x:
        (1 - beta) * E[x] + beta * CVaR_alpha[x]

    Args:
        x: profit in each scenario, EUR, shape (S,)
        p: probability of each scenario, shape (S,)
        alpha: confidence level of the CVaR
        beta: weight of the CVaR, 0 = risk neutral

    Returns:
        risk-adjusted profit, EUR

    TODO 5b - One line with cvar() from TODO 5a. Check: beta = 0 gives the expectation p @ x.
    """
    return (1 - beta) * float(p @ x) + beta * cvar(x, p, alpha)