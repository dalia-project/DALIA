# DALIA testing folder
The DALIA testing suite is in construction. Integration tests are not yet available, for now you can refer to the examples provided in the `examples/` directory for testing of the entire pipeline.


## How to run tests

The tests can either be run directly using `pytest` or through the provided `runner.sh` script. The `runner.sh` script allows for more convenient selection of test categories and backends. 

In a "functionnal" environment, on a cluster with the appropriate modules loaded, and with a working conda environment activated, you can run the tests as follows:
- Directly using: `./runner.sh`
- Check available options: `./runner.sh --help`.

## Continuous integration

The workflow `.github/workflows/tests.yml` runs on every pull request and on every push to `main` and `dev`. On a GitHub hosted runner (CPU only, NumPy backend) it runs, sequentially and with 2 MPI processes:
- the unit and component integration tests: `pytest tests`,
- the integration tests: `python tests/integration/runner.py`, which exits with an error if a test fails.

The GPU backend (CuPy) is not covered by the workflow, it has to be tested on a machine with a GPU using `ARRAY_MODULE=cupy`.

## Tests status

| Reference | Status | Reason |
| --------- | ------ | ------ |
| `component_integration/solvers/sparse_solvers/sequential/test_selected_inversion()` | Not Implemented | Not Implemented                                                 |
| `component_integration/solvers/sparse_solvers/sequential/test_factorize()` | Limited (cannot check for numerical correctness) | LU decomposition instead of Cholesky due to `scipy` limitations |
| `component_integration/solvers/structured_solvers/distributed/test_factorize()` | Limited (cannot check for numerical correctness) | Distributed factorization is not numerically equal to sequential reference |

