# SSQP Benchmarks
This directory contains scripts used to benchmark various aspects for our solver.

Currently, the scripts contained in this directory serve the purpose of comparing the performance of the forward and inverse dynamics formulations for different types of OCPs.

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
pip install matplotlib
pip install numpy
```

Obviously, you will also need to have built _ssqpy_, the library's Python bindings, and have them available to Python.

Following this, you can execute benchmarking scripts for trajectory generation or MPC.

#### Trajectory Generation
In the `trajgen/` directory, you will find various scripts used for benchmarking the performance of forward and inverse dynamics for trajectory generation. We currently run these benchmarks on certain randomly generated problems for the Pendubot, the Franka Emika Panda and the Go2 robots.

For the former two (Pendubot and Panda), you will require a parameter file in order to benchmark. In the `trajgen/data/params` repository, we already provide some sample parameter files for these systems, for both forward and inverse dynamics. If you'd instead like to perform parameter search on your own, you can use the `panda_param_search` and `pendubot_param_search` scripts. If you'd further like to customize which parameters are included in the parameter set to search through, look through these scripts and add these parameter values to the appropriate parameter set generation functions.

To actually perform the benchmarks, you can use the following scripts:

* `pendubot_benchmark`
* `panda_benchmark`
* `go2_benchmark`

If you'd like to change the number of problems generated for each system, use the `N_problems` variable declared at the start of each of these scripts.

Once each of these scripts is done executing, the resulting data files can be located at `trajgen/data/results`.

To generate plots using these files, you can use the `plot` script. Run `python plot.py -h` for a list of options for the script. Generally, you can just execute `python plot.py -m [model]` to generate plots for some model, such as `python plot.py -m pendubot` for the Pendubot, assuming the necessary data files have been created.

#### MPC
The MPC benchmarking scripts are located in the `mpc/` directory. For MPC, we utilize a predefined task for each model and assess the solver's capability to perform that task in real time.

As a simulator, we use our own model's dynamics, but perform _RK4 _ integration (instead of the Euler integration used by the solver).

##### Pendubot
For the Pendubot, we start at the hanging (downward) configuration and the task is to perform a swing and reach the upward configuration, as well as remain at that configuration. There is also the ability to apply a certain perturbation at some step.

The Pendubot benchmarking script, `pendubot`, is located in the `mpc/` directory. Run `python pendubot.py -h` for a list of options for the script.

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
