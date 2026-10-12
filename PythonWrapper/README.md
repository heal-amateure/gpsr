# gpsr – Python wrapper

Python wrapper for the GPSR symbolic-regression library (.NET), based on [pythonnet](https://pythonnet.github.io/).

## Requirements

- [uv](https://docs.astral.sh/uv/)
- .NET 10 SDK (for building the DLLs) or the .NET 10 runtime (to run them)

## Setup

From the repository root:

```sh
# 1. Build the GPSR DLLs into the package (incl. dependencies such as ILGPU)
dotnet publish src/Core/Core.csproj -c Release -o PythonWrapper/src/gpsr/lib

# 2. Create the venv and install the dependencies
cd PythonWrapper
uv sync
```

Rerun step 1 whenever the C# code changes. Alternatively, set `GPSR_LIB_DIR` to a folder that contains the DLLs.

## Usage

```python
import numpy as np
from gpsr import GPSRegressor, plot_result

model = GPSRegressor(generations=30, population_size=300, random_state=42)
result = model.fit(X_train, y_train, feature_names=["r", "R"], target_name="V")

print(result.expression)        # best formula (infix)
print(result.train.scores)      # NMSE, RMSE, MAE, MRE, Pearson R/R², LD

y_hat = model.predict(X_new)   # best program on new data (no target needed, invalid rows -> NaN)

test = model.evaluate(X_test, y_test)
plot_result(test, title="Test", expression=result.expression)
```

Note: `evaluate()` re-ranks the population on the given data, so afterwards `predict()` uses the best program on *that* data.

`Ctrl+C` during `fit()` stops the run early. The best result found so far is kept (`result.cancelled == True`).

Main parameters (defaults follow `src/Runner/Program.cs`): `generations`, `population_size`, `symbol_count`, `nesting_depth`, `crossover_rate`, `mutation_rate`, `elites`, `metric` (`"NMSE"`, `"RMSE"`, `"MAE"`, `"MRE"`, `"PearsonR"`, `"PearsonR2"`, `"LD"`), `minimize_complexity` (multi-objective NSGA-II), `functions` (names from `GPSR.Core.Functions`), `use_constants`, `mutators`, `parallel`, `random_state`, `verbose`.

## Version

The package version comes from Nerdbank.GitVersioning (`../version.json`), so it matches the NuGet package, written in PEP 440 syntax (`1.0.2-alpha` → `1.0.2a0`, non-public builds get `+g<commit>`).
`pyproject.toml` only holds the placeholder `0.0.0`; CI sets the real version before building. Locally:

```sh
python scripts/nbgv_version.py           # print the version
python scripts/nbgv_version.py --apply   # write it into pyproject.toml (don't commit)
```

`gpsr.__version__` returns the installed version.

## Commands

```sh
uv run examples/torus.py   # example: finds V = 2π²·R·r²
uv run pytest              # tests
uv build                   # wheel incl. DLLs in dist/
```
