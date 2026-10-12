"""Find the torus volume V = 2 * pi^2 * R * r^2 from synthetic data.

Run with:  uv run examples/torus.py
"""

import matplotlib.pyplot as plt
import numpy as np

from gpsr import GPSRegressor, plot_result

rng = np.random.default_rng(0)
n = 2000
r = rng.uniform(0.5, 2.0, n)
R = r + rng.uniform(0.5, 3.0, n)
V = 2 * np.pi**2 * R * r**2
X = np.column_stack([r, R])

split = int(0.8 * n)
model = GPSRegressor(generations=30, population_size=300, random_state=42, verbose=True)

print("Starting GPSR... (Ctrl+C stops early)")
result = model.fit(X[:split], V[:split], feature_names=["r", "R"], target_name="V")
test = model.evaluate(X[split:], V[split:])

print()
print(f"Expression:  {result.expression}")
print(f"Evaluations: {result.evaluations}, runtime {result.runtime_seconds:.2f}s"
      + (" (cancelled)" if result.cancelled else ""))
print(f"Train:       {result.train.scores}")
print(f"Test:        {test.scores}")

plot_result(result.train, title="Training", expression=result.expression)
plot_result(test, title="Test")
plt.show()
