# QP Solvers

`ssqp::solvers::QPSolver` is the abstract interface all QP backends implement. See
[QP Solvers](../concepts/qp-solvers.md) in Concepts for the conceptual explanation. The reference below is generated
directly from the headers' own doc comments.

<!-- doxy:start level=1 -->
::: doxy.ssqp.class
    name: ssqp::solvers::QPSolver

<!-- doxy:end -->

## `admm_qp`

SSQP's own ADMM implementation; the default backend, no extra dependencies.

<!-- doxy:start level=2 -->
::: doxy.ssqp.class
    name: ssqp::solvers::admm\_qp

<!-- doxy:end -->

## `hpipm_qp`

Wraps [HPIPM](https://github.com/giaf/hpipm) via [BLASFEO](https://github.com/giaf/blasfeo). Requires building with
`-DWITH_HPIPM=ON` (see [Installation](../getting-started/installation.md#cmake-options)):

<!-- doxy:start level=2 -->
::: doxy.ssqp.class
    name: ssqp::solvers::hpipm\_qp

<!-- doxy:end -->

## `piqp_qp`

Wraps [PIQP](https://github.com/PREDICT-EPFL/piqp) with its multistage KKT solver, which uses
[BLASFEO](https://github.com/giaf/blasfeo). Requires building with `-DSSQP_WITH_PIQP=ON` (see
[Installation](../getting-started/installation.md#cmake-options)):

<!-- doxy:start level=2 -->
::: doxy.ssqp.class
    name: ssqp::solvers::piqp\_qp

<!-- doxy:end -->

## `lqr`

A Riccati-recursion solver used internally by `admm_qp` for the unconstrained (equality-only) structure of the QP
at each iteration:

<!-- doxy:start level=2 -->
::: doxy.ssqp.class
    name: ssqp::solvers::lqr

<!-- doxy:end -->
