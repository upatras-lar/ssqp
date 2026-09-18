# Benchmarks

[`benchmarks/`](https://github.com/upatras-lar/ssqp/tree/main/benchmarks) contains two independent sets of
benchmarks, covering two different questions:

* **[Solver](benchmarks-solver.md)**: how does SSQP compare against other established OCP solvers (Fatrop, Ipopt, Aligator,
  mim_solvers/CSQP) on the same robotics problems, each one solved through a shared problem definition so every
  solver attempts the same costs and constraints?
* **[Solver Modes](benchmarks-solver-modes.md)**: within SSQP itself, how does the choice of dynamics formulation
  (`ForwardDynamics`, `InverseDynamics`, `ReducedInverseDynamics`, see [Model](concepts/model.md#solver-mode))
  affect convergence on a given problem?

## Why we run these

Neither set of benchmarks exists to produce a single headline number. The Solver benchmarks exist to give a rough,
honest sense of where SSQP sits relative to the state of the art on realistic robotics OCPs, rather than only ever
being evaluated in isolation. The Solver Mode benchmarks exist for a more practical reason: SSQP supports several
ways to formulate the same underlying problem, and picking well between them matters for how fast and how
reliably a given problem actually solves, so we wanted data to back that choice rather than intuition alone.

## General conclusions

Across the Solver benchmarks, SSQP is competitive with the other solvers on this shared problem set; see the "On
fairness" note on the [Solver Benchmarks](benchmarks-solver.md) page for why that's phrased carefully rather than as a
claim of being faster or higher-quality than any specific one of them.

Across the Solver Mode benchmarks, the clearest pattern is with floating-base, legged problems:
`ReducedInverseDynamics` tends to converge in fewer iterations and less time than `ForwardDynamics` or
`InverseDynamics` there (see the [Go2 Multi Jump](benchmarks-solver-modes.md#go2-multi-jump) and [G1
Jump](benchmarks-solver-modes.md#g1-jump) tables). It also tends to produce solutions of higher quality.
Both sets of plots go into the actual per-problem numbers; this page is only meant as a starting point before reading 
either in full.

<div class="grid cards" markdown>

- **[Solver Benchmarks](benchmarks-solver.md)**: SSQP vs. other OCP solvers, problem by problem
- **[Solver Mode Benchmarks](benchmarks-solver-modes.md)**: SSQP's own dynamics modes, compared against each other

</div>
