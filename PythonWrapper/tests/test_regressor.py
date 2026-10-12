import numpy as np
import pytest

from gpsr import GPSRegressor


@pytest.fixture
def data():
    rng = np.random.default_rng(1)
    X = rng.uniform(-2, 2, size=(300, 2))
    y = 3 * X[:, 0] + X[:, 1] ** 2
    return X, y


def test_fit_returns_expression_and_scores(data):
    X, y = data
    model = GPSRegressor(generations=3, population_size=50, random_state=1)
    result = model.fit(X, y, feature_names=["a", "b"])

    assert result.expression
    assert result.evaluations > 0
    assert np.isfinite(result.train.scores.nmse)
    assert result.train.y_pred.shape == y.shape
    np.testing.assert_allclose(result.train.y_true, y)


def test_evaluate_on_new_data(data):
    X, y = data
    model = GPSRegressor(generations=2, population_size=30, random_state=1)
    model.fit(X[:200], y[:200])
    test = model.evaluate(X[200:], y[200:])
    assert test.y_pred.shape == (100,)


def test_rejects_unknown_metric():
    with pytest.raises(ValueError):
        GPSRegressor(metric="foo")


def test_predict_matches_training_estimates(data):
    X, y = data
    model = GPSRegressor(generations=3, population_size=50, random_state=1)
    result = model.fit(X, y)
    np.testing.assert_allclose(model.predict(X), result.train.y_pred, rtol=1e-9, equal_nan=True)
