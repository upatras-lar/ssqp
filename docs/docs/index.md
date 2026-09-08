---
hide:
  - navigation
---

# SSQP

<video autoplay loop muted playsinline style="float: left; width: 340px; margin: 0.25rem 1.5rem 1rem 0; border-radius: 8px;">
  <source src="assets/go2_multi_jump.mp4" type="video/mp4">
</video>

SSQP is an implementation of a **S**tructure-exploiting **S**equential **Q**uadratic **P**rogramming (SQP) Non-Linear
Problem (NLP) solver that works natively on manifolds, specifically tailored to Optimal Control Problems (OCPs) of the following form:

$$
\begin{split}
    \min_{\boldsymbol{x}, \boldsymbol{u}} \sum_{k=0}^{H-1} \ell_k(\boldsymbol{x}_k, \boldsymbol{u}_k) + \ell_H(\boldsymbol{x}_H) &\\
    \text{subject to} \\
    \boldsymbol{x}_{k + 1} = \boldsymbol{f}_k(\boldsymbol{x}_k, \boldsymbol{u}_k)&, 0 \leq k < H \\
    \boldsymbol{c}_k(\boldsymbol{x}_k, \boldsymbol{u}_k) \geq \boldsymbol{0}&, 0 \leq k < H \\
    \boldsymbol{c}_H(\boldsymbol{x}_H) \geq \boldsymbol{0} &
\end{split}
$$

Where:

* $\boldsymbol{x}$ and $\boldsymbol{u}$ are the optimization variables corresponding to the system state and controls,
* $H$ is the optimization horizon,
* $\ell_k(\cdot) \in \mathcal{R}$ is the cost function for optimization stage $k$,
* $\boldsymbol{f}_k(\cdot) \in \mathcal{R}^{n_x}$ is the dynamics function for optimization stage $k$, and
* $\boldsymbol{c}_k(\cdot) \in \mathcal{R}^{n_c}$ is the constraints function for optimization stage $k$.

This problem formulation splits the OCP into multiple stages up to a desired time horizon and allows the user to
assign costs and constraints for each of those stages.

$\boldsymbol{x}$ is not required to be a flat Euclidean vector: SSQP works natively on **manifolds**, so a state
can include components like a floating-base orientation (an $SO(3)$/$SE(3)$ element) alongside ordinary vector
components, with linearization and integration carried out through the correct tangent-space operators throughout.
See [Manifold-valued states](concepts/costs-constraints-dynamics.md#manifold-valued-states) for details.

At each iteration, SSQP quadratizes the cost and linearizes the dynamics/constraints around the current guess to
build a Quadratic Program (QP), exploits the stagewise (banded) structure of that QP to solve it efficiently, and
uses the QP solution as an update step, governed by a line search. See [The SSQP Solver](concepts/solver.md) for
details.

<div style="clear: both;"></div>

## Where to start

<div class="grid cards" markdown>

- **[Installation](getting-started/installation.md)**: build SSQP and its Python bindings from source
- **[Quickstart](getting-started/quickstart.md)**: solve a minimal OCP with the Python API
- **[FAQ](faq.md)**: frequently asked questions about basic usage and concepts
- **[Concepts](concepts/model.md)**: the `Model`, costs, constraints, dynamics, solvers and MPC
- **[Examples](examples.md)**: trajectory generation and MPC examples shipped with the repository
- **[Python API Reference](reference/index.md)**: generated reference for the `ssqpy` bindings

</div>

## Repository layout

* [`examples/`](https://github.com/upatras-lar/ssqp/tree/main/examples): examples using SSQP to construct and solve
  OCPs for robotics tasks: offline trajectory generation, and real-time Model Predictive Control (MPC) both in
  simulation and on real systems.
* [`include/`](https://github.com/upatras-lar/ssqp/tree/main/include) and
  [`src/`](https://github.com/upatras-lar/ssqp/tree/main/src): the C++ header and source directories for the
  library.
* [`python/`](https://github.com/upatras-lar/ssqp/tree/main/python): the library's Python binding code (the
  `ssqpy` module), built using [nanobind](https://github.com/wjakob/nanobind).
* [`tests/`](https://github.com/upatras-lar/ssqp/tree/main/tests): unit tests for library functionality.
* [`external/`](https://github.com/upatras-lar/ssqp/tree/main/external): external dependencies of the library.

!!! note
    SSQP's building process and documentation are a work in progress.
