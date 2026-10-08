

# DALIA
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)  

---

Python implementation of the methodology of Integrated Nested Laplace Approximations (INLA), putting the accent on portability, modularity and performance.

If you want to get involved in the development of DALIA, please feel free to contact us directly.

## Installation
Detailed installation instructions are provided in [install.md](./install.md).


## Testing

The testing suite is described in detail in [tests/README.md](./tests/README.md). It relies on `pytest` and can be run either directly or through the provided `runner.sh` script.

## Examples

Some examples are provided with running scripts. The examples are being tracked using `git-lfs`, to download them, run the following commands:
```bash
git lfs pull
git lfs checkout
```

You can then navigate in the `examples/` directory and run the given examples. For example, to run a Gaussian Spatio-Temporal model, you can run:
```bash
python gst_small/run.py
```

Additionaly, *slurms* scripts to run the examples on different HPC clusters are provided.

## Benchmarks

... work in progress


# Citing DALIA

The main DALIA paper describing its high performance computing strategies is available through the following reference:

``` bibtex
@inproceedings{10.1145/3712285.3759832,
      author = {Gaedke-Merzh\"{a}user, Lisa and Maillou, Vincent and Rodriguez Avellaneda, Fernando and Schenk, Olaf and Moraga, Paula and Luisier, Mathieu and Ziogas, Alexandros Nikolaos and Rue, H\r{a}vard},
      title = {Accelerated Spatio-Temporal Bayesian Modeling for Multivariate Gaussian Processes},
      year = {2025},
      isbn = {9798400714665},
      publisher = {Association for Computing Machinery},
      address = {New York, NY, USA},
      url = {https://doi.org/10.1145/3712285.3759832},
      doi = {10.1145/3712285.3759832},
      abstract = {Multivariate Gaussian processes (GPs) offer a powerful probabilistic framework to represent complex interdependent phenomena. They pose, however, significant computational challenges in high-dimensional settings, which frequently arise in spatio-temporal applications. We present DALIA, a highly scalable framework for performing Bayesian inference tasks on spatio-temporal multivariate GPs, based on the methodology of integrated nested Laplace approximations. Our approach relies on a sparse inverse covariance matrix formulation of the GP, puts forward a GPU-accelerated block-dense approach, and introduces a hierarchical, triple-layer, distributed-memory parallel scheme. We showcase weak-scaling performance surpassing the state of the art by two orders of magnitude on a model whose parameter space is 8 \texttimes{} larger and measure strong-scaling speedups of three orders of magnitude when running on 496 GH200 superchips on the Alps supercomputer. Applying DALIA to an air pollution study over northern Italy spanning 48 days, we showcase refined spatial resolutions over the aggregated pollutant measurements.},
      booktitle = {Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis},
      pages = {949–972},
      numpages = {24},
      keywords = {Large-Scale Bayesian Inference, Spatio-Temporal Modeling, Distributed Memory Computing},
      location = {},
      series = {SC '25}
}
```

If you are using the *Serinv* solver for Spatio-Temporal modeling, please also cite the following reference:

``` bibtex
@inproceedings{11186484,
      author = { Maillou, Vincent and Gaedke-Merzhauser, Lisa and Ziogas, Alexandros Nikolaos and Schenk, Olaf and Luisier, Mathieu },
      booktitle = { 2025 IEEE International Conference on Cluster Computing (CLUSTER) },
      title = {{ Parallel Selected Inversion of Block-Tridiagonal with Arrowhead Matrices }},
      year = {2025},
      volume = {},
      ISSN = {},
      pages = {1-12},
      abstract = { The inversion of structured sparse matrices is a fundamental yet computationally and memory-intensive task in many scientific applications, such as Bayesian statistical modeling and material science. In certain cases, only particular entries of the full inverse are required. This has motivated the development of so-called selected inversion algorithms (SIA), capable of computing only specific elements of the full inverse. Currently, most SIA implementations are restricted to shared-/distributed-memory CPU architectures or to single GPUs. Here, we introduce novel numerical methods to perform the parallel selected inversion and Cholesky decomposition of positive-definite, block-tridiagonal with arrowhead matrices. A distributed memory, GPU-accelerated implementation of our approach is presented and integrated into the structured solver library Serinv. We demonstrate its performance on synthetic and real datasets from statistical air temperature prediction models and achieve CPU (GPU) speedups of up to $2.6 \times(71.4 \times)$ over the SIA of the PARDISO library and up to $14 \times(380.9 \times)$ over the MUMPS library, when scaling to 16 processes. },
      keywords = {Materials science and technology;Temperature distribution;Computational modeling;Graphics processing units;Linear algebra;Predictive models;Libraries;Supercomputers;Sparse matrices;Parallel algorithms},
      doi = {10.1109/CLUSTER59342.2025.11186484},
      url = {https://doi.ieeecomputersociety.org/10.1109/CLUSTER59342.2025.11186484},
      publisher = {IEEE Computer Society},
      address = {Los Alamitos, CA, USA},
      month =sep
}
```