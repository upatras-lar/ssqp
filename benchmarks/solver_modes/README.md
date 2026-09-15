# SSQP Benchmarks
The scripts contained in this directory serve the purpose of comparing the performance of the forward and inverse dynamics formulations for different types of OCPs.

We perform these comparisons for offline trajectory generation (`trajgen` directory) as well as MPC (`mpc` directory).

### Executing the benchmarking scripts
Before executing any of the benchmarking scripts, run the `setup.py` script. This will create all required data directories, if not already present.

Additionally, the following scripts will require a few dependencies to execute. We suggest using a Python Virual Environment to install these dependencies.

To do so, first create a virtual environment:

```
python -m venv benchmark-venv
```

After this, activate the newly created virtual environment using:

```
source benchmark-venv/bin/activate
```

Once activated, you can now install the required dependencies using PIP:

```
pip install numpy matplotlib PyQt6
```

Obviously, you will also need to have built _ssqpy_, the library's Python bindings, and have them available to Python.

Following this, you can execute benchmarking scripts for trajectory generation or MPC.

#### Trajectory Generation
In the `trajgen/` directory, you will find scripts used for benchmarking the performance of the different solver modes (forward dynamics, inverse dynamics and, where applicable, reduced inverse dynamics) for trajectory generation. Problems for the Pendubot, the Franka Emika Panda and the Go2 robots are generated in `problems.py`. We currently run these benchmarks on certain randomly generated problems for each of these systems; `go2_multi_jump` and `g1_jump` are each a single, fixed problem instead, ported from the examples gallery.

To actually perform the benchmarks, move to the `trajgen/` directory and use the `run_benchmark` script:

```
python run_benchmark.py [pendubot|panda|go2|go2_multi_jump|g1_jump] -q [admm|hpipm]
```

The `model` positional argument selects which system to benchmark, and `-q`/`--qp-solver` selects the QP backend to use (`admm` by default). By default, all dynamics modes supported by the chosen model are benchmarked in parallel; use `--modes` to run only a subset (e.g. `--modes forward inverse`). Use `-n`/`--n-problems` to change the number of problems generated (50 by default, ignored by `go2_multi_jump`/`g1_jump`), and `--horizon` to change the horizon used for the Pendubot (200 by default).

Pass `--all-problems` instead of a model name to benchmark every available model in one invocation:

```
python run_benchmark.py --all-problems -q [admm|hpipm]
```

Once the script is done executing, the resulting data files can be located at `trajgen/data/results`.

To generate plots using these files, you can use the `plot` script. Run `python plot.py -h` for a list of options for the script. Generally, you can just execute `python plot.py [model] -q [admm|hpipm]` to generate plots for some model and QP backend, such as `python plot.py pendubot` for the Pendubot, assuming the necessary data files have been created.

#### MPC
The MPC benchmarking scripts are located in the `mpc/` directory. For MPC, we utilize a predefined task for each model and assess the solver's capability to perform that task in real time.

As a simulator, we use our own model's dynamics, but perform _RK4_ integration (instead of the Euler integration used by the solver).

##### Pendubot
For the Pendubot, we start at the hanging (downward) configuration and the task is to perform a swing and reach the upward configuration, as well as remain at that configuration. There is also the ability to apply a certain perturbation at some step.

The Pendubot benchmarking script, `pendubot`, is located in the `mpc/` directory. After moving to that directory, run `python pendubot.py -h` for a list of options for the script.

A sample execution of the script is the following:

```
python pendubot.py -m forward -s 1000 --disturbance-step 500 --disturbance-within 1.0 --disturbance-duration 20
```

This executes MPC using forward dynamics, for a total of 1000 integration steps. It applies a uniform random disturbance, sampled from $\mathcal{N}(0, 1)$, at step 500, lasting 20 integration steps.

The resulting plot will be saved to `mpc/data/plots`.

##### Panda
For the Panda, the MPC task we define is to perform a circlular motion around the _yz_-plane with the end-effector, while keeping the _x_-coordinate constant. We also define constraints forcing the end-effector to remain within an axis-aligned square inscribed within the circle.

The script can be executed as follows:

```
python panda.py -m forward
```

And similarly for inverse mode.

Note that by tuning the cost weights and the solver parameters, it is possible to improve the resulting behavior of the solver.
