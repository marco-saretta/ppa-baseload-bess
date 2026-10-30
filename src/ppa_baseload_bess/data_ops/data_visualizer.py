from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd

from ..utils.logger import get_logger
from hydra.experimental.callback import Callback 

log = get_logger(__name__)


# --- Plot style ---------------------------
plt.rcParams["axes.spines.top"] = False  # no box line on top
plt.rcParams["axes.spines.right"] = False  # no box line on the right
plt.rcParams["axes.grid"] = True  # light grid behind the data
plt.rcParams["grid.color"] = "#e6e5e1"
plt.rcParams["grid.linewidth"] = 0.8
plt.rcParams["axes.axisbelow"] = True  # grid behind the bars and lines, not on top

BLUE = "#2a78d6"  # gain
RED = "#e34948"  # loss
GRAY = "#52514e"  # reference lines and their text
ORANGE = "#eb6834"  # buyer (load)  


class DataVisualizer:
    def __init__(self, cfg, model) -> None:
        self.cfg = cfg
        self.model = model
        self.has_solution = model.m.SolCount > 0

        self.out_dir = Path(self.cfg.paths.results) / self.cfg.simulations.label

        if not self.has_solution:
            log.warning("Vasiliki there is no solution, so I will be skipping the plots")
            return

        self.read_results()
        self.plot_earnings_G_over_time()
        self.plot_earnings_G_per_scenario()
        self.plot_surplus_vs_M()  

    def read_results(self) -> None:
        """Load the values created in DataExporter.postprocess()"""
        self.earnings = pd.read_csv(self.out_dir / "earnings_G.csv", index_col="hour")
        self.totals   = pd.read_csv(self.out_dir / "earnings_G_per_scenario.csv", index_col="scenario")["earnings_G_MEUR"]
        self.summary  = pd.read_csv(self.out_dir / "plot_summary.csv", index_col="key").iloc[:, 0]
        self.sweep = pd.read_csv(self.out_dir / "surplus_vs_M.csv", index_col="M")
        self.sweep_summary = pd.read_csv(self.out_dir / "surplus_summary.csv", index_col="key").iloc[:, 0]
    
    def plot_earnings_G_over_time(self) -> None:
        """Graph 1: generator earnings per time step over the horizon, one line per scenario."""
        fig, ax = plt.subplots(figsize=(10, 4.5))

        # One line per scenario column, like the notebook's loop over alpha values
        for scenario in self.earnings.columns:
            ax.plot(self.earnings.index, self.earnings[scenario], linewidth=0.8, label=scenario, zorder=3)

        # Zero line: below it the CfD costs the generator more than its spot sales earn
        ax.axhline(0, color="#0b0b0b", linewidth=0.8, zorder=2)

        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Generator earnings per time step (EUR)")
        ax.set_title(f"Generator earnings over time ({self.cfg.simulations.label})", loc="left", fontsize=11)
        ax.legend(frameon=False, loc="upper right", ncol=5, fontsize=8)
        fig.tight_layout()
        self.save(fig, "earnings_over_time")
    
    def plot_earnings_G_per_scenario(self) -> None: 
        """Graph 2: total generator earnings per scenario, worst first (the notebook's bar_plot)."""
        totals = self.totals  # MEUR, sorted worst first
        expected = self.summary["Expected Earnings"]  # MEUR
        colors = [BLUE if p >= 0 else RED for p in totals]  # blue = gain, red = loss

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(totals.index, totals, color=colors, width=0.62, zorder=3)

        # Expected earnings as a dashed line with its value, and the zero line
        ax.axhline(expected, color=GRAY, linestyle="--", linewidth=1.2, zorder=4)
        ax.text(0, expected, f" E[profit]={expected:,.3f}", va="bottom", ha="left", fontsize=9, color=GRAY)
        ax.axhline(0, color="#0b0b0b", linewidth=0.8, zorder=2)

        # Two-line title with the negotiated contract
        ax.set_title(
            f"Generator earnings per scenario ({self.cfg.simulations.label})\n"
            f"S* = {self.summary['S']:.2f} EUR/MWh, M* = {self.summary['M']:.2f} MW",
            fontsize=11,
            loc="left",
        )
        ax.set_ylabel("Generator earnings (MEUR)")
        ax.set_xlabel("Scenario (worst to best)")
        ax.tick_params(axis="x", labelrotation=45)
        fig.tight_layout()
        self.save(fig, "earnings_per_scenario")

    def plot_surplus_vs_M(self) -> None:
        """Panels A, B, C: total gain, each party's gain and the risk removed, as functions of M."""
        sweep, s = self.sweep, self.sweep_summary
        low, high = s["flat_M_low"], s["flat_M_high"]  # flat top of W, MW
        M_star, S_star = self.summary["M"], self.summary["S"]

        # Three panels on top of each other, sharing the M axis
        fig, axes = plt.subplots(3, 1, figsize=(9, 12), sharex=True)
        for ax in axes:
            ax.axvspan(low, high, color="#e6e5e1", zorder=1)  # grey band: the flat top of W
            ax.axhline(0, color="#52514e", linewidth=0.8, zorder=2)
            ax.set_ylabel("kEUR")

        # --- A: total gain W(M) ---
        ax = axes[0]
        ax.plot(sweep.index, sweep["W"], color="#0b0b0b", linewidth=2, zorder=3)
        ax.scatter([M_star], [s["W_solver"]], color="#0b0b0b", s=60, edgecolor="white", zorder=4)
        ax.text(
            (low + high) / 2, s["W_max"],
            f"flat top: M from {low:.2f} to {high:.2f} MW, W = {s['W_max']:,.1f} kEUR\n",
            ha="center", va="bottom", fontsize=9,
        )
        ax.text((low + high) / 2, s["W_max"] * 0.8, f"solver: M = {M_star:.2f}", ha="center", fontsize=9, color=GRAY)
        ax.set_title("A. Total gain W(M) = w_G + w_L (no solver: the strike cancels)", loc="left", fontsize=11)

        # --- B: each party's gain with the strike fixed at S* ---
        ax = axes[1]
        ax.plot(sweep.index, sweep["w_G"], color=BLUE, linewidth=2, label="generator", zorder=3)
        ax.plot(sweep.index, sweep["w_L"], color=ORANGE, linewidth=2, label="load", zorder=3)
        ax.plot(sweep.index, sweep["W"], color="#0b0b0b", linestyle="--", linewidth=1, label="total W", zorder=3)
        for name, column, color in [("generator", "w_G", BLUE), ("load", "w_L", ORANGE)]:
            M_peak = s[f"M_peak_{column[-1]}"]
            ax.scatter([M_peak], [sweep.loc[M_peak, column]], color=color, s=60, edgecolor="white", zorder=4)
            ax.text(M_peak, sweep.loc[M_peak, column], f"  {name} peak, M = {M_peak:.2f}", va="bottom", fontsize=9)
        ax.set_title(f"B. Each party's gain with the strike fixed at S = {S_star:.2f} EUR/MWh", loc="left", fontsize=11)

        # --- C: risk removed from each party ---
        ax = axes[2]
        ax.plot(sweep.index, sweep["risk_G"], color=BLUE, linewidth=2, zorder=3)
        ax.plot(sweep.index, sweep["risk_L"], color=ORANGE, linewidth=2, zorder=3)
        ax.plot(sweep.index, sweep["W"], color="#0b0b0b", linestyle="--", linewidth=1, zorder=3)
        for name, column, color in [("generator", "risk_G", BLUE), ("load", "risk_L", ORANGE)]:
            M_pref = s[f"M_pref_{column[-1]}"]
            ax.scatter([M_pref], [sweep.loc[M_pref, column]], color=color, s=60, edgecolor="white", zorder=4)
            ax.text(M_pref, sweep.loc[M_pref, column], f"  {name} preferred size, M = {M_pref:.2f}", va="bottom", fontsize=9)
        ax.set_title("C. Risk removed from each party, beta_i x [R_i(M) - R_i(0)], R = CVaR - E", loc="left", fontsize=11)
        ax.set_xlabel("Baseload volume M (MW)")

        # One legend for the whole figure, below the panels
        handles, labels = axes[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
        fig.tight_layout(rect=[0, 0.03, 1, 1])  # leave room at the bottom for the legend
        self.save(fig, "surplus_vs_M")

    def save(self, fig, name: str) -> None:
        """Save as results/<label>/<name>.png (replaces the notebook's plt.show()) and free the memory."""
        out_path = self.out_dir / f"{name}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        log.info(f"Saved {out_path}")
        
def plot_beta_heatmaps(S_grid, M_grid, out_path) -> None:
    """Strike price (top) and volume (bottom) for every pair of risk aversions.

    S_grid, M_grid: tables with beta_G as rows and beta_L as columns; NaN = no solution.
    """
    # Blue (low) -> light -> red (high); squares without a solution in grey
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#d3d1c7")

    fig, axes = plt.subplots(2, 1, figsize=(6, 10))
    panels = [(S_grid, "Strike price S* (EUR/MWh)"), (M_grid, "Contract volume M* (MW)")]

    for ax, (grid, label) in zip(axes, panels):
        # One colored square per (beta_L, beta_G); masked_invalid makes NaN "bad" -> grey
        image = ax.imshow(np.ma.masked_invalid(grid.to_numpy()), cmap=cmap, origin="lower")
        fig.colorbar(image, ax=ax, label=label)

        # The beta values as tick labels; origin="lower" puts beta_G = 0 at the bottom
        ax.set_xticks(range(len(grid.columns)), grid.columns)
        ax.set_yticks(range(len(grid.index)), grid.index)
        ax.set_xlabel("Load risk aversion β_L")
        ax.set_ylabel("Generator risk aversion β_G")
        ax.set_title(label, loc="left", fontsize=11)
        ax.grid(False)  # no grid lines across the squares

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


class SweepHeatmapCallback(Callback):
    """Hydra calls this before the first and after the last run of a --multirun sweep.

    Registered in config/sweep/beta_grid.yaml under hydra.callbacks.
    The callback runs in the folder the command was started from (the repo root), so results/ is relative.
    """

    def __init__(self, sweep_name: str) -> None:
        self.table_path = Path("results") / f"sweep_{sweep_name}.csv"  # filled by DataExporter.export_sweep_row()
        self.plot_path = Path("results") / f"{sweep_name}_heatmaps.png"

    def on_multirun_start(self, config, **kwargs) -> None:
        # Start from an empty table, otherwise rows from an earlier sweep would be mixed in
        self.table_path.unlink(missing_ok=True)

    def on_multirun_end(self, config, **kwargs) -> None:
        table = pd.read_csv(self.table_path)

        # From one row per run to two grids: beta_G as rows, beta_L as columns
        S_grid = table.pivot(index="beta_G", columns="beta_L", values="S")
        M_grid = table.pivot(index="beta_G", columns="beta_L", values="M")

        plot_beta_heatmaps(S_grid, M_grid, self.plot_path)
        log.info(f"Saved {self.plot_path}")
