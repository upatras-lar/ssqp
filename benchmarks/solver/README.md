### Solver Benchmarks
This directory aims to benchmark the performance of our SSQP solver compared to other state of the art, similar optimal control solvers.

A subdirectory can be found for each solver we use for benchmarking. Inside, we formulate each problem using the Python interface of the corresponding solver, attempting to get as close a representation to the original problem as we can, such that all solvers tackle the same OCPs.

#### Docker
The following command can be used to build a docker container, in which the `run_benchmarks` script can be executed for every solver.

```
docker build --build-arg NPROC=[n] -t solver-benchmark .
```

Replace `[n]` with the number of CPU cores to use for building dependencies from source.

After building, run the container using the command below:

```
docker run --rm -it -v $PWD:/workspace -v $PWD/../../examples/assets/urdf:/workspace/urdf:ro -v $PWD/../../build/bindings:/workspace/bindings:ro solver-benchmark:latest
```

#### `run_benchmarks.py` script
Once inside the container, use the following syntax to execute a benchmark:

```
python run_benchmarks.py [problem] -s [solver]
```

Replace `[problem]` with the benchmarking problem to run, as well as `[solver]` with the solver to use to solve the problem.

The `-h` command line argument can be used for a list of all available problem and solver options.

#### Defining a custom problem
Using the abstract interface defined in `ProblemWrapper.py`, any problem can be defined in a way such that it can be benchmarked with any supported solver.

To define a custom problem, you must write the problem using the problem wrapper. After this, import it from the `run_benchmark` script and add it wherever necessary, similar to how existing problems are added. It can then be executed with any of the supported solvers.