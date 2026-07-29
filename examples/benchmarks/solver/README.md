### Solver Benchmarks
This directory aims to benchmark the performance of our SSQP solver compared to other state of the art, similar optimal control solvers.

A subdirectory can be found for each solver we use for benchmarking. Inside, we formulate each problem using the Python interface of the corresponding solver, attempting to get as close a representation to the original problem as we can, such that all solvers tackle the same OCPs.

#### SSQP
Firstly, for SSQP, we formulate the problems using our Python API. Since these are trajectory generation problems, the scripts can be executed in the same manner as all other Python trajectory generation examples, by building the Python interface for SSQP and executing the `setup_env.sh` script.

#### Other Solvers
For each other solver, we provide a subdirectory containing the scripts formulating each of our benchmarking OCPs using that solver's API. Each such subdirectory also contains a docker file, using which a container can be built to execute the scripts.

### Casadi Formulations
In the `casadi_formulations` subdirectory, we use CasADi and the Pinocchio CasADi interface to formulate each problem using CasADi. This is done because multiple solvers can then use those formulations (IPOPT and FATROP).

To execute scripts in this directory, build the docker image using the following command:

```
docker build --build-arg NPROC=${n} -t casadi-formulations .
```

Replace `${n}` with the number of cores you want to assign for each build.

Once the build completes, you will now have a docker image `casadi-formulations`. You may run this docker image from the `casadi_formulations` subdirectory, using:

```
docker run --rm -it -v $PWD:/workspace -v $PWD/../../../assets/urdf:/workspace/urdf:ro casadi-formulations
```

Inside this docker, you will be able to execute each script using `python script.py`.

### FATROP
To execute the FATROP benchmarks, use the `casadi-formulations` docker in the `cassadi_formulations` subdirectory. There, each script benchmarking FATROP is named `{problem}_fatrop.py` and, upon execution, will create a file `{problem}_fatrop.json`. This is a trajectory file compatible with the visualizer, so you can use the visualizer in `/examples/visualizer` to then view the generated trajectory.

### IPOPT
To execute the IPOPT benchmarks, follow the same procedure as that of FATROP. You may use the same docker image, `casadi-formulations`. Execute the scripts titled `{problem}_ipopt.py`, each of which will create `{problem}_ipopt.json` as a trajectory output file.

### Aligator
For the Aligator solver, we provide the subdirectory `aligator` with its own docker file. Build this using:

```
docker build --build-arg NPROC=${n} -t aligator-dev .
```

Replace `${n}` with the number of cores to use for parallel building. Run the docker image generated from the `aligator` subdirectory using:

```
docker run --rm -it -v $PWD:/workspace -v $PWD/../../../assets/urdf:/workspace/urdf:ro aligator-dev
```

Within, you can now execute any of the scripts using `python {problem}.py`. Each of these will generate `{problem}.json` as a trajectory output file.

### mim_solvers
To execute the mim_solvers benchmarks, build the docker image in the `mim_solvers` subdirectory.

```
docker build --build-arg NPROC=${n} -t mim-solvers-dev .
```

And run using:

```
docker run --rm -it -v $PWD:/workspace -v $PWD/../../../assets/urdf:/workspace/urdf:ro mim-solvers-dev
```