# Parameter Search

`ssqp::param_search` (`ssqpy.solvers.search` in Python) provides online parameter identification for
[dynamics](costs-constraints-dynamics.md#dynamics) that expose tunable parameters (e.g. friction, mass, or other
physical coefficients embedded in a custom `StageDynamics`), plus a small utility for searching over SSQP
hyperparameters.

## Dynamics parameter search

Both searchers take (state, control, resulting next-state) data points and use Armijo line search to update a
dynamics model's internal parameters so its predictions better match observed transitions. This is useful for
adapting a model online (e.g. as in the `examples/cloud_pendulum` adaptive-model examples) or from offline rollout
data.

```python
from ssqpy.solvers import search

searcher = search.DynamicsParamGradientDescent(dynamics, min_data_size=100)

updated = searcher.addData(x_prev, u, x_actual)  # returns whether parameters changed
```

* **`DynamicsParamGradientDescent`**: accumulates data points and performs a gradient-descent update once at least
  `min_data_size` points have been collected (`setMinimumDataSize` to change this later).
* **`DynamicsParamGaussNewton`**: maintains a sliding window of `window_size` data points and performs a
  Gauss-Newton update against them. `setWindowSize(window_size, prune_last=False)` resizes the window; when
  shrinking, `prune_last=True` drops the most recent points, while the default instead prunes the most *similar*
  points, keeping the window's data diverse.

Both share `addData(x, u, x_next)` and `clearData()`, and both take `armijo_beta`/`armijo_c1` to control the Armijo
line search used for the parameter update step.

## Hyperparameter search

`HyperParamSearch` searches over a set of candidate `ssqpParams` across a set of `Model` problems to find the
configuration that performs best:

```python
hp_search = search.HyperParamSearch(candidate_params_list, model_list)
best_params = hp_search.search()
```

or build it up incrementally with `addHyperparams(params)` / `addProblem(model)` before calling `search()`.
