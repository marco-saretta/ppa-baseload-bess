"""Print S, M and the gains of every sensitivity run in one table."""

from pathlib import Path

import pandas as pd

rows = []

# Every run wrote its results to a folder results/sens_<name>/
for folder in sorted(Path("results").glob("sens_*")):
    file = folder / "variables.csv"

    # Infeasible runs have no solution, so no variables.csv
    if not file.exists():
        print(f"{folder.name}: no solution")
        continue

    # variables.csv has two columns: variable, value -> turn it into a dict {name: value}
    values = pd.read_csv(file).set_index("variable")["value"].to_dict()

    rows.append(
        {
            "run": folder.name,
            "S": values["S"],
            "M": values["M"],
            "w_G": values["w_G"],
            "w_L": values["w_L"],
        }
    )

# One row per run
table = pd.DataFrame(rows)
table["total"] = table["w_G"] + table["w_L"]  # the surplus created by the contract
table["share_G"] = table["w_G"] / table["total"]  # the generator's part of it

print(table.round(2))