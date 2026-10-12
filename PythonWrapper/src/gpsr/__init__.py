"""Python wrapper for the GPSR symbolic-regression library (.NET)."""

from importlib.metadata import version as _version

from .regressor import METRICS, EvaluationResult, FitResult, GPSRegressor, Scores

__version__ = _version("gpsr")  # from pyproject.toml, which CI fills from nbgv
__all__ = ["GPSRegressor", "FitResult", "EvaluationResult", "Scores", "METRICS", "plot_result"]


def plot_result(*args, **kwargs):
    # Lazy import so matplotlib is only loaded when plotting.
    from .plotting import plot_result as _plot

    return _plot(*args, **kwargs)
