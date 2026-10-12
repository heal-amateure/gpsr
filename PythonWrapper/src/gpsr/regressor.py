"""Pythonic wrapper around ``GPSR.Core.Algorithm.Fit``."""

from __future__ import annotations

import ctypes
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from . import _runtime

METRICS = ("PearsonR", "PearsonR2", "NMSE", "RMSE", "MRE", "MAE", "LD")
_MAXIMIZED_METRICS = {"PearsonR", "PearsonR2"}

DEFAULT_FUNCTIONS = (
    "Addition",
    "Subtraction",
    "Multiplication",
    "AnalyticQuotient",
    "ProtectedLogarithm",
    "ProtectedExponential",
    "Sine",
    "Cosine",
    "Tangent",
    "HyperbolicTangent",
    "Pi",
)
DEFAULT_MUTATORS = ("MutateReplaceSubtree", "MutateTerminateSubtree")


@dataclass
class Scores:
    nmse: float
    rmse: float
    mae: float
    mre: float
    pearson_r: float
    pearson_r2: float
    ld: float

    def as_dict(self) -> dict[str, float]:
        return dict(self.__dict__)


@dataclass
class EvaluationResult:
    """Scores and predictions of the best program on one data set."""

    scores: Scores
    y_true: np.ndarray = field(repr=False)
    y_pred: np.ndarray = field(repr=False)


@dataclass
class FitResult:
    expression: str
    expression_rpn: str
    evaluations: int
    runtime_seconds: float
    cancelled: bool
    train: EvaluationResult


class GPSRegressor:
    """Symbolic regression with the GPSR genetic-programming algorithm.

    Example::

        model = GPSRegressor(generations=50, population_size=500)
        result = model.fit(X, y, feature_names=["r", "R"])
        print(result.expression)
    """

    def __init__(
        self,
        generations: int = 100,
        population_size: int = 500,
        symbol_count: int = 25,
        nesting_depth: int = 8,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.25,
        elites: int = 1,
        metric: str = "NMSE",
        minimize_complexity: bool = True,
        functions: Sequence[str] = DEFAULT_FUNCTIONS,
        use_constants: bool = True,
        mutators: Sequence[str] = DEFAULT_MUTATORS,
        optimization_iterations: int = 10,
        perform_simplification: bool = False,
        parallel: bool = True,
        max_degree_of_parallelism: int = -1,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
        self.generations = generations
        self.population_size = population_size
        self.symbol_count = symbol_count
        self.nesting_depth = nesting_depth
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elites = elites
        self.metric = metric
        self.minimize_complexity = minimize_complexity
        self.functions = tuple(functions)
        self.use_constants = use_constants
        self.mutators = tuple(mutators)
        self.optimization_iterations = optimization_iterations
        self.perform_simplification = perform_simplification
        self.parallel = parallel
        self.max_degree_of_parallelism = max_degree_of_parallelism
        self.random_state = random_state
        self.verbose = verbose

        self.algorithm_ = None
        self.task_ = None
        self.feature_names_: list[str] | None = None
        self.target_name_: str | None = None
        self.result_: FitResult | None = None

    # ------------------------------------------------------------------ public

    def fit(
        self,
        X,
        y,
        feature_names: Sequence[str] | None = None,
        target_name: str = "y",
    ) -> FitResult:
        """Run ``Algorithm.Fit`` on (X, y). Ctrl+C stops the run early and keeps the best result."""
        _runtime.load()
        from System.Collections.Generic import HashSet, List
        from System.Threading import CancellationTokenSource
        from GPSR.Core import Algorithm, EvaluationMetric, OptimizationDirection, OptimizationTarget, Task
        from GPSR.Utils import FastRandom

        X, y, feature_names = self._check_xy(X, y, feature_names)
        self.feature_names_ = feature_names
        self.target_name_ = target_name
        if target_name in feature_names:
            raise ValueError(f"target_name {target_name!r} collides with a feature name")

        dataset = self._to_dataset(X, y)

        targets = HashSet[str]()
        targets.Add(target_name)  # first entry = target variable
        if self.minimize_complexity:
            targets.Add(OptimizationTarget.Complexity)

        direction = (
            OptimizationDirection.Maximize
            if self.metric in _MAXIMIZED_METRICS
            else OptimizationDirection.Minimize
        )
        task = Task(
            "GPSR",
            _net_list(str, feature_names),
            targets,
            getattr(EvaluationMetric, self.metric),
            direction,
            None,
        )
        task.VariableLimitsDict = dataset.GetDoubleSetLimits()

        seed = (
            int(np.random.default_rng().integers(0, 2**31 - 1))
            if self.random_state is None
            else int(self.random_state)
        )
        alg = Algorithm(
            FastRandom(seed),
            self.generations,
            self.population_size,
            self.symbol_count,
            self.nesting_depth,
            float(self.crossover_rate),
            float(self.mutation_rate),
            self.elites,
        )
        self._configure(alg, seed)

        cts = CancellationTokenSource()
        cancelled = False
        start = time.perf_counter()
        net_task = alg.Fit(task, dataset, cts.Token)
        try:
            # Poll so that Ctrl+C reaches Python while .NET is computing.
            while not net_task.IsCompleted:
                net_task.Wait(100)
        except KeyboardInterrupt:
            cancelled = True
            cts.Cancel()
            net_task.Wait()
        runtime = time.perf_counter() - start

        if net_task.IsFaulted:
            raise RuntimeError(f"GPSR fit failed: {net_task.Exception}")

        self.algorithm_ = alg
        self.task_ = task
        try:
            alg.ComputeScores()
        except Exception as exc:  # cancelled before the initial population was complete
            raise RuntimeError("Fit was cancelled before any result was available.") from exc

        self.result_ = FitResult(
            expression=alg.BestProgram,
            expression_rpn=alg.BestProgramRPN,
            evaluations=alg.EvaluationCount,
            runtime_seconds=runtime,
            cancelled=cancelled,
            train=self._best_evaluation(),
        )
        return self.result_

    def evaluate(self, X, y) -> EvaluationResult:
        """Evaluate the final population on new data and return the best program's scores.

        Note: the C# side re-ranks the population on this data, so the "best program"
        reported here can differ from the one in ``fit()``'s result.
        """
        self._check_fitted()
        X, y, _ = self._check_xy(X, y, self.feature_names_)
        self.algorithm_.ComputeScores(self._to_dataset(X, y))
        return self._best_evaluation()

    def predict(self, X) -> np.ndarray:
        """Predict with the best program of the last ``fit()``/``evaluate()``. Invalid rows yield NaN."""
        self._check_fitted()
        X, _, _ = self._check_xy(X, None, self.feature_names_)
        return _to_numpy(self.algorithm_.Predict(self._to_dataset(X)))

    @property
    def expression(self) -> str:
        self._check_fitted()
        return self.result_.expression

    # ----------------------------------------------------------------- helpers

    def _configure(self, alg, seed: int) -> None:
        import clr
        from System import Delegate, Func
        from System.Collections.Generic import List
        from GPSR.Core import Algorithm, Function, Functions, RPN, Symbol, Terminal
        from GPSR.Core.Operators import Mutation

        try:
            alg.SelectedNonterminals = _net_list(Function, [getattr(Functions, f) for f in self.functions])
        except AttributeError as exc:
            raise ValueError(f"Unknown function in {self.functions}") from exc

        terminals = [Terminal.Variable] + ([Terminal.Constant] if self.use_constants else [])
        alg.SelectedTerminals = _net_list(Terminal, terminals)

        # Build real .NET delegates (no Python callbacks in the hot loop).
        mutator_type = Func[Algorithm, RPN[Symbol], RPN[Symbol]]
        mutation_cls = clr.GetClrType(Mutation)
        mutators = List[mutator_type]()
        for name in self.mutators:
            method = mutation_cls.GetMethod(name)
            if method is None:
                raise ValueError(f"Unknown mutator {name!r}")
            mutators.Add(Delegate.CreateDelegate(clr.GetClrType(mutator_type), method))
        alg.Mutators = mutators

        alg.LogStatistics = self.verbose
        alg.UseParallelization = self.parallel
        alg.UseDeterministicParallelization = True
        alg.DeterministicSeed = seed
        alg.PerformSimplification = self.perform_simplification
        alg.OptimizationIterations = self.optimization_iterations
        alg.MaxDegreeOfParallelism = self.max_degree_of_parallelism

    def _check_xy(self, X, y, feature_names):
        if feature_names is None and hasattr(X, "columns"):  # pandas DataFrame
            feature_names = [str(c) for c in X.columns]
        X = np.ascontiguousarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y is not None:
            y = np.ascontiguousarray(y, dtype=np.float64).ravel()
        if X.ndim != 2 or (y is not None and X.shape[0] != y.shape[0]):
            raise ValueError(f"X must be (n_samples, n_features) matching y; got {X.shape} and {None if y is None else y.shape}")
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(X.shape[1])]
        feature_names = list(feature_names)
        if len(feature_names) != X.shape[1]:
            raise ValueError("len(feature_names) must equal the number of columns of X")
        return X, y, feature_names

    def _to_dataset(self, X: np.ndarray, y: np.ndarray | None = None):
        from System.Collections.Generic import Dictionary, List
        from System import Double
        from GPSR.Data import DataSet, ISeries, Series

        ds = DataSet("python", None)
        ds.Name = "python"
        ds.Series = Dictionary[str, ISeries]()
        columns = [(name, X[:, i]) for i, name in enumerate(self.feature_names_)]
        if y is not None:
            columns.append((self.target_name_, y))
        for name, values in columns:
            s = Series[Double]()
            s.Name = name
            s.Values = List[Double](_to_net_array(values))
            ds.Series.Add(name, s)
        return ds

    def _best_evaluation(self) -> EvaluationResult:
        alg = self.algorithm_
        return EvaluationResult(
            scores=Scores(
                nmse=alg.BestProgramNMSE,
                rmse=alg.BestProgramRMSE,
                mae=alg.BestProgramMAE,
                mre=alg.BestProgramMRE,
                pearson_r=alg.BestProgramPearsonR,
                pearson_r2=alg.BestProgramPearsonR2,
                ld=alg.BestProgramLD,
            ),
            y_true=_to_numpy(alg.BestProgramTrueResults),
            y_pred=_to_numpy(alg.BestProgramEstimatedResults),
        )

    def _check_fitted(self) -> None:
        if self.result_ is None:
            raise RuntimeError("Call fit() first.")


def _net_list(item_type, items):
    from System.Collections.Generic import List

    result = List[item_type]()
    for item in items:
        result.Add(item)
    return result


def _to_net_array(values: np.ndarray):
    """Copy a float64 numpy array into a System.Double[] (bulk copy, no per-element marshalling)."""
    from System import Array, Double, IntPtr
    from System.Runtime.InteropServices import Marshal

    values = np.ascontiguousarray(values, dtype=np.float64)
    arr = Array.CreateInstance(Double, len(values))
    if len(values):
        Marshal.Copy(IntPtr(values.ctypes.data), arr, 0, len(values))
    return arr


def _to_numpy(values) -> np.ndarray:
    """Copy a System.Double[] or List<double> into a numpy array."""
    from System import Array, IntPtr
    from System.Runtime.InteropServices import Marshal

    arr = values if isinstance(values, Array) else values.ToArray()
    out = np.empty(arr.Length, dtype=np.float64)
    if arr.Length:
        Marshal.Copy(arr, 0, IntPtr(out.ctypes.data_as(ctypes.c_void_p).value), arr.Length)
    return out
