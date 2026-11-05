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

        self.read_results()
        self.plot_earnings_G_over_time()
        self.plot_earnings_G_per_scenario()
        self.plot_surplus_vs_M()  
        self.plot_positions() 
        self.plot_preferences() 

        if not self.has_solution:
                    log.warning("Vasiliki there is no solution, so I will be skipping the plots")
                    return
        

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

    def plot_positions(self) -> None:
        """[Task 2] Generator production vs. load consumption per scenario, with the contracted energy."""
        pos = pd.read_csv(self.out_dir / "scenarios.csv", index_col="scenario").sort_values("E_G_MWh")

        fig, ax = plt.subplots(figsize=(9, 4.5))
        idx = np.arange(len(pos))  # bar positions
        w = 0.36  # two bars per scenario, as in the notebook
        ax.bar(idx - w / 2, pos["E_G_MWh"], width=w, color=BLUE, label="generator production", zorder=3)
        ax.bar(idx + w / 2, pos["E_L_MWh"], width=w, color=ORANGE, label="load consumption", zorder=3)

        # Contracted energy: in weeks where it is above the blue bar, the generator buys the shortfall at spot
        if "contract_MWh" in pos.columns:
            ax.axhline(pos["contract_MWh"].iloc[0], color=GRAY, linestyle="--", linewidth=1.2,
                       label="contracted baseload H·M*", zorder=4)

        ax.set_xticks(idx)
        ax.set_xticklabels(pos.index, rotation=45)
        ax.set_ylabel("Energy over the horizon (MWh)")
        ax.set_xlabel("Scenario (sorted by production)")
        ax.set_title(f"Energy positions per scenario ({self.cfg.simulations.label})", loc="left", fontsize=11)
        ax.legend(frameon=False, loc="upper left")
        fig.tight_layout()
        self.save(fig, "positions")

    def plot_preferences(self) -> None:
        """[Task 3] Tail risk R_i(M) = CVaR - E of each party against the volume; the peak is its preferred volume.

        Two panels with their own scale: the load's R is usually much larger than the generator's.
        """
        R = pd.read_csv(self.out_dir / "preferences.csv", index_col="M")
        s = pd.read_csv(self.out_dir / "value_ratio.csv", index_col="key").iloc[:, 0]
        M_star = None
        if self.has_solution:
            M_star = pd.read_csv(self.out_dir / "plot_summary.csv", index_col="key").iloc[:, 0]["M"]

        fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
        panels = [(axes[0], "G", "Generator", BLUE), (axes[1], "L", "Load", ORANGE)]
        for ax, side, name, color in panels:
            ax.plot(R.index, R[f"R_{side}"], color=color, linewidth=2, zorder=3)

            # Preferred volume: marker at the peak; a band if the top is flat
            low, high, height = s[f"M_pref_{side}"], s[f"M_pref_{side}_high"], s[f"R_{side}_at_pref"]
            if high > low:
                ax.axvspan(low, high, color=color, alpha=0.12, zorder=1)
            ax.scatter([low], [height], color=color, s=60, edgecolor="white", zorder=4)
            ax.text(low, height, f"  preferred M = {low:.2f} MW", va="bottom", fontsize=9)

            # Agreed volume of this run
            if M_star is not None:
                ax.axvline(M_star, color=GRAY, linestyle="--", linewidth=1.2, zorder=2,
                           label=f"agreed M* = {M_star:.2f} MW")
                ax.legend(frameon=False, loc="lower left")

            ax.set_ylabel(f"R_{side}(M) = CVaR − E (kEUR)")
            ax.set_title(f"{name}: tail risk vs. contract volume (closer to 0 = less risk)", loc="left", fontsize=11)

        axes[1].set_xlabel("Baseload volume M (MW)")
        fig.suptitle(
            f"Preferred volumes ({self.cfg.simulations.label}): ER = {s['energy_ratio']:.2f}, VR = {s['value_ratio']:.2f}, "
            f"capture prices G / L = {s['capture_G']:.1f} / {s['capture_L']:.1f} EUR/MWh",
            fontsize=11,
        )
        fig.tight_layout()
        self.save(fig, "preferences")

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

def plot_tau_lines(table, x, x_label, out_path) -> None:
    """[Task 1] Contract volume (top) and strike price (bottom) as a function of a bargaining power.

    table: one row per run (from DataExporter.sweep_row), sorted by x.
    """
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    style = {"color": GRAY, "linewidth": 1.8, "marker": "o", "markersize": 3.5, "zorder": 3}

    # 1st graph: volume
    axes[0].plot(table[x], table["M"], **style)
    axes[0].set_ylabel("M* (MW)")
    axes[0].set_title("Contract volume M* (risk aversion as in the base case)", loc="left", fontsize=11)

    # 2nd graph: strike price
    axes[1].plot(table[x], table["S"], **style)
    axes[1].set_ylabel("S* (EUR/MWh)")
    axes[1].set_title("Strike price S* (risk aversion as in the base case)", loc="left", fontsize=11)
    axes[1].set_xlabel(x_label)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


class SweepLineCallback(Callback):
    """[Task 1] Hydra calls this before the first and after the last run of a one-parameter sweep.

    Draws plot_tau_lines once for every column in x, each with its own label, from the same sweep table:
    results/<sweep_name>_lines_<x>.png
    Registered in config/sweep/tau.yaml under hydra.callbacks.
    The callback runs in the folder the command was started from (the repo root), so results/ is relative.
    """

    def __init__(self, sweep_name: str, x, x_label) -> None:
        self.sweep_name = sweep_name
        self.x = list(x)  # columns for the horizontal axis, e.g. ["tau", "tau_L"]
        self.x_label = list(x_label)  # one label per column, in the same order
        self.table_path = Path("results") / f"sweep_{sweep_name}.csv"  # filled by DataExporter.sweep_row()

    def on_multirun_start(self, config, **kwargs) -> None:
        self.table_path.unlink(missing_ok=True)  # start from an empty table

    def on_multirun_end(self, config, **kwargs) -> None:
        table = pd.read_csv(self.table_path)
        for x, x_label in zip(self.x, self.x_label):  # one figure per perspective
            out_path = Path("results") / f"{self.sweep_name}_lines_{x}.png"
            plot_tau_lines(table.sort_values(x), x, x_label, out_path)
            log.info(f"Saved {out_path}")


def plot_earnings_fan(table, x, x_label, out_path) -> None:
    """[Risk aversion fan charts] Each side's earnings over the scenarios, against one swept parameter.

    Line: median. Dark band: P25-P75. Light band: P5-P95. Dotted: P5, P50, P95 without a contract.
    table: one row per run (from DataExporter.sweep_row), sorted by x.
    """
    fig, axes = plt.subplots(2, 1, figsize=(5, 7), sharex=True)
    panels = [(axes[0], "G", BLUE, "Generator earnings (MEUR)"), (axes[1], "L", ORANGE, "Load earnings (MEUR)")]

    for ax, side, color, ylabel in panels:
        ax.fill_between(table[x], table[f"{side}_p5"], table[f"{side}_p95"], color=color, alpha=0.2,
                        linewidth=0, label="P5–P95", zorder=2)
        ax.fill_between(table[x], table[f"{side}_p25"], table[f"{side}_p75"], color=color, alpha=0.45,
                        linewidth=0, label="P25–P75", zorder=3)
        ax.plot(table[x], table[f"{side}_p50"], color=color, linewidth=2, label="median", zorder=4)

        # Without a contract: the same in every run, so the first row is enough
        for q in [5, 50, 95]:
            ax.axhline(table[f"{side}_nc_p{q}"].iloc[0], color=GRAY, linestyle=":", linewidth=1, zorder=1,
                       label="no contract (P5, P50, P95)" if q == 5 else None)
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, loc="best", fontsize=8)

    axes[0].set_title("Baseload", fontsize=11)
    axes[1].set_xlabel(x_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

class SweepFanCallback(Callback):
    """[Risk aversion fan charts] Hydra calls this before the first and after the last run of a sweep.

    Draws results/<sweep_name>_earnings_vs_<x>.png from results/sweep_<sweep_name>.csv.
    fixed: keep only the runs with these values, e.g. {"beta_G": 0.5} selects one row of the beta grid.
    The callback runs in the folder the command was started from (the repo root), so results/ is relative.
    """

    def __init__(self, sweep_name: str, x: str, x_label: str, fixed=None) -> None:
        self.x = x  # column of the sweep table on the horizontal axis, e.g. "beta_L"
        self.x_label = x_label
        self.fixed = dict(fixed or {})  # e.g. {"beta_G": 0.5}
        self.table_path = Path("results") / f"sweep_{sweep_name}.csv"  # filled by DataExporter.sweep_row()
        self.plot_path = Path("results") / f"{sweep_name}_earnings_vs_{x}.png"

    def on_multirun_start(self, config, **kwargs) -> None:
        self.table_path.unlink(missing_ok=True)  # start from an empty table

    def on_multirun_end(self, config, **kwargs) -> None:
        table = pd.read_csv(self.table_path)
        for column, value in self.fixed.items():  # keep one slice of the grid
            table = table[np.isclose(table[column], value)]
        plot_earnings_fan(table.sort_values(self.x), self.x, self.x_label, self.plot_path)
        log.info(f"Saved {self.plot_path}")

def plot_position_sweep(table, x, x_label, out_path) -> None:
    """[Task 2] Average energy positions (top) and the shares hedged (bottom) against one swept parameter."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    style = {"linewidth": 1.8, "marker": "o", "markersize": 3.5, "zorder": 3}

    # A. Who is bigger: production, consumption and the energy fixed by the contract
    ax = axes[0]
    ax.plot(table[x], table["E_G_MWh"], color=BLUE, label="generator production", **style)
    ax.plot(table[x], table["E_L_MWh"], color=ORANGE, label="load consumption", **style)
    ax.plot(table[x], table["contract_MWh"], color=GRAY, linestyle="--", label="contracted H·M*", **style)
    ax.set_ylabel("Energy over the horizon (MWh)")
    ax.set_title("A. Energy positions (average over scenarios)", loc="left", fontsize=11)
    ax.legend(frameon=False, loc="upper left")

    # B. How much of each position the contract covers (100% = the whole position)
    ax = axes[1]
    ax.plot(table[x], table["hedge_share_G"], color=BLUE, label="share of production sold", **style)
    ax.plot(table[x], table["hedge_share_L"], color=ORANGE, label="share of consumption bought", **style)
    ax.axhline(100, color=GRAY, linestyle=":", linewidth=1, zorder=2)  # above: more than the whole position
    ax.set_ylabel("Hedged (%)")
    ax.set_title("B. Share of each position covered by the contract", loc="left", fontsize=11)
    ax.legend(frameon=False, loc="upper right")
    ax.set_xlabel(x_label)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


class SweepPositionCallback(Callback):
    """[Task 2] Hydra calls this before the first and after the last run of the load-size sweep.

    Draws results/<sweep_name>_positions.png from results/sweep_<sweep_name>.csv.
    The callback runs in the folder the command was started from (the repo root), so results/ is relative.
    """

    def __init__(self, sweep_name: str, x: str, x_label: str) -> None:
        self.x = x  # column of the sweep table on the horizontal axis, e.g. "load_mw"
        self.x_label = x_label
        self.table_path = Path("results") / f"sweep_{sweep_name}.csv"  # filled by DataExporter.sweep_row()
        self.plot_path = Path("results") / f"{sweep_name}_positions.png"

    def on_multirun_start(self, config, **kwargs) -> None:
        self.table_path.unlink(missing_ok=True)  # start from an empty table

    def on_multirun_end(self, config, **kwargs) -> None:
        table = pd.read_csv(self.table_path).sort_values(self.x)
        plot_position_sweep(table, self.x, self.x_label, self.plot_path)
        log.info(f"Saved {self.plot_path}")

def value_ratio_display(table):
    """[Task 3] The small table of step 5: one row per load size, the agreed M* for each beta_L as columns.

    Only selects, reshapes and rounds for display: every number comes from DataExporter.
    """
    one_per_load = table.drop_duplicates("load_mw").set_index("load_mw").sort_index()  # ratios don't depend on beta_L
    agreed = table.pivot(index="load_mw", columns="beta_L", values="M")  # rows: load, columns: beta_L
    agreed.columns = [f"M* (β_L={b})" for b in agreed.columns]
    display = pd.concat(
        [one_per_load[["energy_ratio", "value_ratio", "M_pref_G", "M_pref_L"]], agreed, one_per_load[["case"]]],
        axis=1,
    ).rename(columns={"energy_ratio": "ER", "value_ratio": "VR", "M_pref_G": "M_pref G", "M_pref_L": "M_pref L"})
    display.index.name = "Load (MW)"
    return display.round(2)


def plot_value_ratio_table(display, beta_G, out_path) -> None:
    """[Task 3] Draw the step-5 table as an image."""
    fig, ax = plt.subplots(figsize=(11, 0.45 * len(display) + 1.2))
    ax.axis("off")  # a table, not a chart
    cells = ax.table(
        cellText=display.reset_index().values,
        colLabels=[display.index.name, *display.columns],
        loc="center",
        cellLoc="center",
    )
    cells.auto_set_font_size(False)
    cells.set_fontsize(9)
    cells.scale(1, 1.3)  # a little more height per row
    ax.set_title(f"Value ratio and preferred volumes (β_G = {beta_G})", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_value_ratio(table, out_path) -> None:
    """[Task 3] A: energy and value ratio vs. load size. B: preferred and agreed volumes vs. load size."""
    one_per_load = table.drop_duplicates("load_mw").sort_values("load_mw")
    style = {"linewidth": 1.8, "marker": "o", "markersize": 3.5, "zorder": 3}

    fig, axes = plt.subplots(2, 1, figsize=(8, 9), sharex=True)

    # A. Who is bigger, in MWh and in EUR; the gap is the effect of the solar profile
    ax = axes[0]
    ax.plot(one_per_load["load_mw"], one_per_load["energy_ratio"], color=GRAY, label="energy ratio ER", **style)
    ax.plot(one_per_load["load_mw"], one_per_load["value_ratio"], color="#0b0b0b", label="value ratio VR", **style)
    ax.axhline(1, color=GRAY, linestyle=":", linewidth=1, zorder=2)  # equal size
    ax.set_ylabel("Load / generator")
    ax.set_title("A. Energy ratio vs. value ratio", loc="left", fontsize=11)
    ax.legend(frameon=False, loc="upper left")

    # B. Each party's preferred volume, and the agreed volume for each beta_L
    ax = axes[1]
    ax.plot(one_per_load["load_mw"], one_per_load["M_pref_G"], color=BLUE, label="generator's preferred M", **style)
    ax.plot(one_per_load["load_mw"], one_per_load["M_pref_L"], color=ORANGE, label="load's preferred M", **style)
    grays = ["#b4b2a9", "#52514e", "#0b0b0b"]  # light -> dark: low -> high beta_L
    for (beta_L, group), color in zip(table.sort_values("load_mw").groupby("beta_L"), grays):
        ax.plot(group["load_mw"], group["M"], color=color, linestyle="--", linewidth=1.4, marker="s",
                markersize=3.5, label=f"agreed M*, β_L = {beta_L}", zorder=4)
    ax.set_xlabel("Average load of the buyer (MW)")
    ax.set_ylabel("Baseload volume (MW)")
    ax.set_title("B. Preferred volumes and the agreed volume", loc="left", fontsize=11)
    ax.legend(frameon=False, loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


class ValueRatioCallback(Callback):
    """[Task 3] Hydra calls this before the first and after the last run of the value-ratio sweep.

    Draws results/<name>.png and results/<name>_table.png, prints the table in the log,
    and reports the exporter's checks.
    The callback runs in the folder the command was started from (the repo root), so results/ is relative.
    """

    def __init__(self, sweep_name: str) -> None:
        self.table_path = Path("results") / f"sweep_{sweep_name}.csv"  # filled by DataExporter.sweep_row()
        self.plot_path = Path("results") / f"{sweep_name}.png"
        self.table_plot_path = Path("results") / f"{sweep_name}_table.png"

    def on_multirun_start(self, config, **kwargs) -> None:
        self.table_path.unlink(missing_ok=True)  # start from an empty table

    def on_multirun_end(self, config, **kwargs) -> None:
        table = pd.read_csv(self.table_path)

        display = value_ratio_display(table)
        log.info("Value ratio table:\n" + display.to_string())
        plot_value_ratio_table(display, table["beta_G"].iloc[0], self.table_plot_path)
        log.info(f"Saved {self.table_plot_path}")

        plot_value_ratio(table, self.plot_path)
        log.info(f"Saved {self.plot_path}")

        # Step-5 checks written by the exporter
        if not table["M_between_prefs"].dropna().astype(bool).all():
            log.warning(f"Some agreed volumes lie outside the preferred range: see {self.table_path}")
        if not table["VR_agrees_with_prefs"].astype(bool).all():
            log.warning("VR and the preferred volumes disagree for some loads (expected only near VR = 1)")
        if table[["M_pref_G_at_edge", "M_pref_L_at_edge"]].to_numpy().any():
            log.warning("Some preferred volumes are at the edge of the M grid: widen the grid")