# How to run the integration tests

## On Daint
I recomend getting on an interactive session where you can directly run the integration tests for all configurtions (in particular types of parallelization).
```bash
srun --pty --partition=debug --account=xxxx bash
```

Then, you can run the integration tests sequentially using:
```bash
python runner.py
```

Further configurations are available in the `runner.py` file.

## On Fritz
I recomend getting on an interactive session where you can directly run the integration tests for all configurtions (in particular types of parallelization).
```bash
salloc -N 1 --partition=spr2tb --time=00:30:00
```

Then, you can run the integration tests sequentially using:
```bash
srun python runner.py
```

## Data of the tests
The tests do not use any data file of the repository. Each test generates its data from scratch in `tests/integration/_generated/` (not tracked) and compares the results of DALIA to the values used to generate the data:
- the AR tests (`gar1`, `gar2`, `par1`) run the `generate_data.py` script of the matching example,
- the other tests use the generators of `data_generators.py` (regular planar mesh, P1 finite element matrices, latent fields sampled from the prior of the submodels).

With MPI, rank 0 generates the data and the other ranks wait for it.
