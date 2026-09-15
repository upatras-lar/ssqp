# Benchmarks

[`benchmarks/solver/`](https://github.com/upatras-lar/ssqp/tree/main/benchmarks/solver) compares SSQP against a
handful of other OCP solvers on the same set of problems. Every solver formulates the identical problem through
[`ProblemWrapper`](https://github.com/upatras-lar/ssqp/blob/main/benchmarks/solver/ProblemWrapper.py), an abstract
interface (costs, constraints, `finalize`, `solve`, `save`) with one subclass per backend translating those calls
into that backend's own API, so a given problem stays defined once while every solver attempts the same costs and
constraints on it.

Six solvers are compared:

* `ssqp-admm` / `ssqp-hpipm`: SSQP itself, using the two per-stage QP backends described in
  [QP Solvers](concepts/qp-solvers.md)
* `fatrop` / `ipopt`: formulated through [CasADi](https://web.casadi.org/)'s `Opti` stack, solved with the
  respective plugin
* `aligator`: [Aligator](https://github.com/Simple-Robotics/aligator)
* `mim_solvers`: [mim_solvers](https://github.com/machines-in-motion/mim_solvers)' CSQP, built on
  [Crocoddyl](https://github.com/loco-3d/crocoddyl)

All six are run to the same nominal tolerance (`1e-2`, see
[`benchmark_config.py`](https://github.com/upatras-lar/ssqp/blob/main/benchmarks/solver/benchmark_config.py)), even
though that tolerance bounds a different internal quantity for each (SSQP's per-stage KKT/constraint residual,
Fatrop/Ipopt's scaled KKT error, Aligator's primal-dual residual, CSQP's convergence gap), since reimplementing
every solver's own termination criterion for an exact match isn't practical.

!!! note "On fairness"
    A fully fair, apples-to-apples comparison across six independent solver codebases is close to impossible: each
    has its own convergence criterion (see above), its own QP backend and warm-starting strategy, its own default
    tuning, and its own years of separate engineering investment behind it. These benchmarks are not a claim that
    SSQP is faster or higher-quality than any of the others; the goal is to give a rough sense of where SSQP sits
    relative to the state of the art on a shared set of problems, not to crown a winner.

## Problem sizes

The problems range from Pendubot's long, low-dimensional swing-up to Go2 Frog's short but heavily constrained,
contact-switching locomotion. Sizes below are per SSQP's own direct-transcription formulation (state = generalized
position + velocity, control = generalized torque plus one force variable per active contact); a range means the
value changes over the horizon with the contact schedule.

| Problem | Horizon | State Dimension | Control Dimension | Constraints per Stage | NLP size (vars / constraints) |
| --- | --- | --- | --- | --- | --- |
| [Pendubot](#pendubot) | 2500 steps (2.5 s) | 4 | 2 | 6 (terminal: 4) | 15,004 / 15,004 |
| [Go2 Standing](#go2-standing) | 100 steps (1 s) | 37 | 30 | 72 (terminal: 27) | 6,737 / 7,227 |
| [Go2 Jump](#go2-jump) | 150 steps (1.5 s) | 37 | 18–27\* | 51–78 (terminal: 42) | 9,367 / 10,932 |
| [Go2 Wheelie](#go2-wheelie) | 200 steps (2 s) | 37 | 24–30\* | 66–84 (terminal: 39) | 12,957 / 15,405 |
| [Go2 Frog](#go2-frog) | 300 steps (3 s) | 37 | 18–30\* | 48–84 (terminal: 30) | 18,841 / 21,342 |
| [Panda Circle](#panda-circle) | 400 steps (4 s) | 18 | 9 | 30 (terminal: 21) | 10,818 / 12,021 |

\* Varies with the number of feet in contact at each stage: every active contact adds a 3D force variable to that
stage's control vector, so the dimension shrinks during flight phases and grows back during stance.

## Reading the plots

Each plot bars four quantities per solver, side by side:

* **Runtime**, in milliseconds
* **Iterations**, the outer solve loop's iteration count
* **Final objective**, the converged cost
* **Constraint violation**, the worst single bound violation at any stage (position, velocity, torque, friction,
  and, for the direct-transcription solvers, the dynamics-transition defect), against a dashed line marking the
  shared tolerance

A fast, few-iteration solve is not a meaningful win if the trajectory it returns is infeasible, which is what the
constraint violation panel is for. A solver that fails to converge for a given problem is labeled "failed" rather
than silently omitted or plotted at zero.

## Running the benchmarks yourself

See the [`solver/` README](https://github.com/upatras-lar/ssqp/blob/main/benchmarks/solver/README.md) for the full
setup (a Docker image is provided, since several of these solvers pull in their own heavy dependency chains).
Once set up:

```bash
python run_benchmark.py [problem] -s [solver]
```

`plot_benchmarks.py` then renders one problem's `data/<solver>/<problem>.json` results into the plots shown below:

```bash
python plot_benchmarks.py pendubot
```

## Pendubot

<video autoplay loop muted playsinline style="display: block; margin: 0 auto; width: 420px; max-width: 100%; border-radius: 8px;">
  <source src="../assets/benchmark_pendubot.mp4" type="video/mp4">
</video>

The [Pendubot Swing-Up](examples/pendubot.md) problem: an underactuated double pendulum swinging from hanging to
upright, actuated at the shoulder joint only.

<img src="../assets/pendubot_benchmark.png" style="display: block; margin: 0 auto; max-width: 70%; border-radius: 8px;">

## Go2 Standing

<video autoplay loop muted playsinline style="display: block; margin: 0 auto; width: 420px; max-width: 100%; border-radius: 8px;">
  <source src="../assets/benchmark_go2_standing.mp4" type="video/mp4">
</video>

The Go2 quadruped holding a fixed standing configuration, all four feet in contact throughout, with no other motion
target. A short, contact-heavy but otherwise static regulation problem.

<img src="../assets/go2_standing_benchmark.png" style="display: block; margin: 0 auto; max-width: 70%; border-radius: 8px;">

## Go2 Jump

<video autoplay loop muted playsinline style="display: block; margin: 0 auto; width: 420px; max-width: 100%; border-radius: 8px;">
  <source src="../assets/benchmark_go2_jump.mp4" type="video/mp4">
</video>

The [Go2 Jump](examples/go2-jump.md) problem: stance, flight, stance, with all four feet leaving the ground for a
scheduled window in the middle and landing back at a specified footprint. In this case, the front left foot has _no contact_ and, therefore, should always remain in air.

<img src="../assets/go2_jump_benchmark.png" style="display: block; margin: 0 auto; max-width: 70%; border-radius: 8px;">

## Go2 Wheelie

<video autoplay loop muted playsinline style="display: block; margin: 0 auto; width: 420px; max-width: 100%; border-radius: 8px;">
  <source src="../assets/benchmark_go2_wheelie.mp4" type="video/mp4">
</video>

The Go2 rearing up on its rear two feet: the rear feet stay in stance for the whole horizon, while the front two
lift off for a window in the middle and come back down at the end. Unlike [Go2 Jump](#go2-jump), there's no target
trajectory for the freed feet; the standing-configuration cost is what pulls them (and the rest of the robot) back
down once the window closes.

<img src="../assets/go2_wheelie_benchmark.png" style="display: block; margin: 0 auto; max-width: 70%; border-radius: 8px;">

## Go2 Frog

<video autoplay loop muted playsinline style="display: block; margin: 0 auto; width: 420px; max-width: 100%; border-radius: 8px;">
  <source src="../assets/benchmark_go2_frog.mp4" type="video/mp4">
</video>

A longer, four-hop bounding motion 2 meters forward, with staggered per-leg liftoff/landing and brief full-flight
windows between hops rather than Go2 Jump's single synchronized flight phase. The same motion also appears, run
directly rather than through the benchmark harness, as [Go2 Frog](gallery.md#go2-frog) in the Gallery.

<img src="../assets/go2_frog_benchmark.png" style="display: block; margin: 0 auto; max-width: 70%; border-radius: 8px;">

## Panda Circle

<video autoplay loop muted playsinline style="display: block; margin: 0 auto; width: 420px; max-width: 100%; border-radius: 8px;">
  <source src="../assets/benchmark_panda_circle.mp4" type="video/mp4">
</video>

The Franka Panda tracing a circular end effector path in task space via frame-pose tracking costs, additionally
constrained to a rectangular window inscribed in that circle, same as
[Panda Circle (Trajectory Generation)](examples/panda-circle-trajgen.md), reformulated here in
`ForwardDynamics` mode (torque-actuated) so every backend solves an equivalent problem.

<img src="../assets/panda_circle_benchmark.png" style="display: block; margin: 0 auto; max-width: 70%; border-radius: 8px;">
