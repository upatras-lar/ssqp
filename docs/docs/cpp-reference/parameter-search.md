# Parameter Search

`ssqp::param_search` provides online dynamics parameter identification and SSQP hyperparameter search. See
[Parameter Search](../concepts/parameter-search.md) in Concepts for the conceptual explanation. The reference below
is generated directly from `include/param_search/search.hpp`'s own doc comments.

## `dynamics_params::Search`

The abstract base for dynamics parameter searchers (namespace `ssqp::param_search::dynamics_params`):

<!-- doxy:start level=2 -->
::: doxy.ssqp.class
    name: ssqp::param\_search::dynamics\_params::Search

<!-- doxy:end -->

### `GradientDescent`

<!-- doxy:start level=3 -->
::: doxy.ssqp.class
    name: ssqp::param\_search::dynamics\_params::GradientDescent

<!-- doxy:end -->

### `GaussNewton`

<!-- doxy:start level=3 -->
::: doxy.ssqp.class
    name: ssqp::param\_search::dynamics\_params::GaussNewton

<!-- doxy:end -->

## `HyperParamSearch`

Declared directly in `ssqp::param_search` (not under `dynamics_params`):

<!-- doxy:start level=2 -->
::: doxy.ssqp.class
    name: ssqp::param\_search::HyperParamSearch

<!-- doxy:end -->
