# SSQP
SSQP is an implementation of a Sequential Quadratic Programming (SQP) Non-Linear Problem (NLP) solver, specifically tailored to Optimal Control Problems (OCPs) of the following form:

$$\Large
\begin{split}
    \min_{\boldsymbol{x}, \boldsymbol{u}} \sum_{k=0}^{H-1} \ell_k(\boldsymbol{x}_k, \boldsymbol{u}_k) + \ell_H(\boldsymbol{x}_H) &\\
    \text{subject to} \
    \boldsymbol{x}_{k + 1} = \boldsymbol{f}_k(\boldsymbol{x}_k, \boldsymbol{u}_k)&, 0 \leq k < H \\
    \boldsymbol{c}_k(\boldsymbol{x}_k, \boldsymbol{u}_k) \geq \boldsymbol{0}&, 0 \leq k < H \\
    \boldsymbol{c}_H(\boldsymbol{x}_H) \geq \boldsymbol{0} &
\end{split}
$$

Where:

* $\boldsymbol{x}$ and $\boldsymbol{u}$ are the optimization variables corresponding to the system state and controls,
* $H$ is the optimization horizon,
* $\ell_k(\cdot) \in \mathcal{R}$ is the cost function for optimization stage k,
* $\boldsymbol{f}_k(\cdot) \in \mathcal{R}^{n_x}$ is the dynamics function for optimization stage k and
* $\boldsymbol{c}_k(\cdot) \in \mathcal{R}^{n_c}$ is the constraints function for optimization stage k.

This problem formulation splits the OCP into multiple stages up to a desired time horizon and allows the user to assign costs and constraints for each of those stages.

# Repository
Within the SSQP repository, the following subdirectories can be found:

* `examples/`: A directory containing multiple examples utilizing SSQP to construct and solve OCPs for various robotics-related tasks for trajectory generation, as well as real-time Model Predictive Control (MPC) for simulations and real systems.
* `include/` and `src/`: The header and source directories for the library, respectively.
* `python/`: The directory defining the library's Python binding code, using [nanobind](https://github.com/wjakob/nanobind).
* `tests/`: A directory containing unit tests for certain functionalities of the library.
* `external/`: Inclusive directory for all external dependencies of the library (whether optional or not).

Currently, the project is built using the provided `Makefile`. Use `make` to build the C++ examples and `make python` to build the Python library. A Dockerfile is also provided in the `.ci` directory. Note that the SSQP building process is a work in progress (and relatively ad hoc as of now).

### Benchmarking
To execute the benchmarking scripts used to compare the performance of the forward and inverse dynamics formulations for our solver, once the _ssqpy_ (the Python bindings for our library) is built, follow the instructions found in the _README_ in `examples/`, as well as in `examples/benchmarks`.