# How to install DALIA on one of the default clusters or your personal machine
DALIA has been developed on several (super)computing infrastructures. Since the beginning of the project, our goal has been to provide a seamless experience for users regardless of the underlying hardware and software stack.

Because full generalization is not always possible, we provide a simplified procedure for multiple supercomputing infrastructures (with different hardware) to make installing DALIA on a new cluster as easy as possible.

## Purpose
Base `conda` environments for DALIA are provided in the `dalia/envs` directory. These environments contain all the Python packages on which DALIA relies. Hardware-specific packages (e.g., `cupy` for GPU compute, `mpi4py` for multiprocessing) are not included; they can be installed as extensions of the base environment. We provide base environments for both `x86` and `aarch64` architectures.

These environments are used in the CI/CD pipelines for the respective clusters and are also available for users to quickly set up DALIA on these clusters or any other machine.

## Supported Clusters
We currently use three different clusters for the testing and development of DALIA:

| Organization | Cluster | Arch                              | Description                                                    | Memory                     | Nodes                        |
| ------------ | ------- | --------------------------------- | -------------------------------------------------------------- | -------------------------- | ---------------------------- |
| [FAU](https://doc.nhr.fau.de/)          | [Fritz](https://doc.nhr.fau.de/clusters/fritz/)   | x86 (Sapphire Rapids)             | 2x Intel Xeon Platinum 8470 per node<br>(2x52-cores @ 2.0 GHz) | Up to 2TB DDR5             | 64x 8470, 992x 8360Y         |
| [FAU](https://doc.nhr.fau.de/)          | [Alex](https://doc.nhr.fau.de/clusters/alex/)    | x86 (AMD EPYC 7713)<br>Ampere<br> | 8x Nvidia A100 per node<br>(2x 64-cores CPU + 8 GPUs)          | Up to 80GB HBM2            | 18x A100 80GB, 20x A100 40GB |
| [CSCS](https://www.cscs.ch/)         | [Daint](https://docs.cscs.ch/clusters/daint/)   | ARM (Grace)<br>Hopper             | 4x Nvidia GH200 per node<br>(72-cores CPU + 1 GPU)             | 128GB LPDDR5X<br>96GB HBM3 | 1022                         |


## Environments Configuration Matrix
DALIA is designed to work across a wide variety of hardware and software stacks (e.g., GPU acceleration, distributed memory, vendor-specific libraries). The table below summarizes the supported configurations and the corresponding pre-configured `conda` environments.

|              | No Comm              | Host MPI           | GPU-Aware MPI      | xCCL               |
| :----------- | :------------------- | :----------------- | :----------------- | :----------------- |
| CPU (x86)    | *dalia_base_fritz* | *dalia_hmpi_fritz* | NA                 | NA                 |
| GPU (NVIDIA) | *dalia_base_alex*  | NA  | NA  | *dalia_xccl_alex*  |
| GPU (NVIDIA) | *dalia_base_daint* | NA | NA | *dalia_xccl_daint* |
| CPU (ARM)    | x                    | x                  | x                  | x                  |
| GPU (AMD)    | x                    | x                  | x                  | x                  |

The `dalia_base` environment contains all the dependencies needed to run DALIA on a single node without any communication library (e.g., MPI, xCCL) and without GPU support. This environment includes only hardware-independent Python dependencies. In `dalia/scripts/` we provide interactive installers that can create these `dalia_base` environments for you and, when applicable, extend them to support multi-node communication with MPI or xCCL and/or GPU acceleration.

**Notes:** 
- On the Daint and Alex clusters, CuPy is installed using wheels that include NCCL. Therefore, NCCL is available whenever CuPy is installed. For this reason, no standalone GPU-aware MPI environments are provided.
- On Daint, the `dalia_base_daint` environment allows one to run on the ARM CPU. However, we do not provide any conda environment specifically targeted for a general ARM CPU-cluster.

# Detailed Instructions
## On Fritz@FAU
### a) Installation
1. Clone the repositories to your workspace:    
    ```bash
    mv /my/install/path
    git clone https://github.com/dalia-project/DALIA
    git clone https://github.com/vincent-maillou/serinv # Optional, recommended for ST-modeling
    ```
2. Source the `fritz_fau_utils.sh` script to access the install utilities:
    ```bash
    cd DALIA/
    source scripts/fritz_fau_utils.sh
    ```
3. Load the required environment modules:
    ```bash
    fritz_load_modules
    ```
4. Create the conda environment:
    ```bash
    fritz_create_conda_env --dalia-path=path/to/DALIA --serinv-path=path/to/serinv --install-mpi4py --dev-mode
    ```
    Notes:
    - This example installs all optional dependencies and uses developer mode. You can also run the installer in interactive mode by simply running `fritz_create_conda_env`.
    - Developer mode (`--dev-mode`) keeps the most performant conda environment available, as well as all environments created along the way. This ensures that during development DALIA can be tested against all supported configurations.
    - The created environment is activated automatically at the end of the installation.
5. Activate the conda environment:
    ```bash
    fritz_activate_conda_env
    ```
    Note: This function will try to activate the most performant environment available on the cluster. You can also activate a specific environment by providing the `--env="desired_environment"` argument to the function.

### b) Usage
Right after installation, you can use DALIA using the loaded modules and activated conda environment.

However, in general, in future shell sessions you will need to:
1. Source the `fritz_fau_utils.sh` script:
    ```bash
    source /path/to/DALIA/scripts/fritz_fau_utils.sh
    ```
2. Load the required environment modules:
    ```bash
    fritz_load_modules
    ```
3. Activate the conda environment:
    ```bash
    fritz_activate_conda_env
    ```

You will then be ready to use DALIA.

### c) Verify Installation
There is currently two ways to verify that DALIA has been installed correctly:
1. Run the provided test suite. In an interactive session: `salloc -N 1 --time=00:30:00` (you might need to wait to get the allocation), you can (after activated the correct conda environment) run the test suite: `./path/to/DALIA/tests/runner.sh`.
2. Run one of the provided examples, you can check the `/path/to/DALIA/examples/run_example_fritz_fau.sh` script for an example on how to submit a job on Fritz.

## On Alex@FAU
### a) Installation

1. Clone the repositories to your workspace:    
    ```bash
    mv /my/install/path
    git clone https://github.com/dalia-project/DALIA
    git clone https://github.com/vincent-maillou/serinv # Optional, recommended for ST-modeling
    ```
2. Source the `alex_fau_utils.sh` script to access the install utilities:
    ```bash
    cd DALIA/
    source scripts/alex_fau_utils.sh
    ```
3. Load the required environment modules:
    ```bash
    alex_load_modules
    ```
4. Create the conda environment:
    ```bash
    alex_create_conda_env --dalia-path=path/to/DALIA --serinv-path=path/to/serinv --install-mpi4py --dev-mode
    ```
    Notes:
    - This example installs all optional dependencies and uses developer mode. You can also run the installer in interactive mode by simply running `alex_create_conda_env`.
    - Developer mode (`--dev-mode`) keeps the most performant conda environment available, as well as all environments created along the way. This ensures that during development DALIA can be tested against all supported configurations.
    - The created environment is activated automatically at the end of the installation.
5. Activate the conda environment:
    ```bash
    alex_activate_conda_env
    ```
    Note: This function will try to activate the most performant environment available on the cluster. You can also activate a specific environment by providing the `--env` argument to the function.

### b) Usage
Right after installation, you can use DALIA using the loaded modules and activated conda environment.

However, in general, in future shell sessions you will need to:
1. Source the `alex_fau_utils.sh` script:
    ```bash
    source /path/to/DALIA/scripts/alex_fau_utils.sh
    ```
2. Load the required environment modules:
    ```bash
    alex_load_modules
    ```
3. Activate the conda environment:
    ```bash
    alex_activate_conda_env
    ```

You will then be ready to use DALIA.

### c) Verify Installation
There is currently two ways to verify that DALIA has been installed correctly:
1. Run the provided test suite. In an interactive session: `salloc --gres=gpu:a100:1 --time=0:30:00` (you might need to wait to get the allocation), you can (after activated the correct conda environment) run the test suite: `./path/to/DALIA/tests/runner.sh`.
2. Run one of the provided examples, you can check the `/path/to/DALIA/examples/run_example_alex_fau.sh` script for an example on how to submit a job on Alex.


## On Daint@CSCS
### Preamble

In order to successfully install DALIA or even clone the repository on Daint, you will need to install `git-lfs` (Git Large File Storage) in your user space (see "Notes on `git-lfs`" for details). We provide a gist utility script alongside the following installation instructions:

```bash
mv /my/install/path
git clone https://gist.github.com/vincent-maillou/d2d38937f7aafbf0cee98c65cf5cfbca # Get the installation script
cd d2d38937f7aafbf0cee98c65cf5cfbca/ # Move into the script directory

chmod u+x daint_install_git_lfs.sh # Render the script executable
./daint_install_git_lfs.sh # Run the installation script
export PATH="$HOME/.local/bin:$PATH" # Add git-lfs to your PATH

which git-lfs # Verify the installation
```

Git-lfs will now be installed and available in your user space. Its path have been added to your `.bashrc` file, so it will be available in future shell sessions.


### a) Installation
1. Clone the repositories to your workspace:    
    ```bash
    mv /my/install/path
    git clone https://github.com/dalia-project/DALIA
    git clone https://github.com/vincent-maillou/serinv # Optional, recommended for ST-modeling
    ```

2. Source the `daint_cscs_utils.sh` script to access the install utilities:
    ```bash
    cd DALIA/
    source scripts/daint_cscs_utils.sh
    ```

3. Use the `daint_install_conda` utility to install Miniconda in your user space (if not already installed):
    ```bash
    daint_install_conda --yes
    ```
    **Notes:**
    - This step is only required if you do not already have `conda` installed.
    - The `--yes` option automatically confirms the installation prompts.
    - Additional information about the installer options can be found by running `daint_install_conda --help`.

4. Source the installed conda environment:
    ```bash
    source ~/miniconda3/etc/profile.d/conda.sh
    ```
    **Notes:**
    - This is needed only once; in subsequent shell sessions conda will be initialized automatically through your `.bashrc` file.

5. Install the programming environment (`uenv`):
    ```bash
    daint_install_uenv
    ```

6. Start the programming environment:
    ```bash
    daint_start_uenv
    ```
    **Notes:**
    - This script assumes that you have successfully installed the programming environment in the previous step. If a programming environment is already active, it will be stopped and replaced with a new one.

7. Source the `daint_cscs_utils.sh` script again to access the install utilities:
    ```bash
    source scripts/daint_cscs_utils.sh
    ```
    **Notes:**
    - This is needed because starting the programming environment spawns a new shell session in which previously sourced scripts are no longer available.

8. Load the required environment modules:
    ```bash
    daint_load_modules
    ```

9. Create the conda environment:
    ```bash
    daint_create_conda_env --dalia-path=path/to/DALIA --serinv-path=path/to/serinv --install-mpi4py --dev-mode
    ```

10. Activate the conda environment:
    ```bash
    daint_activate_conda_env
    ```
    Note: This function tries to activate the most performant environment available on the cluster. You can also activate a specific environment by providing the `--env` argument.

### b) Usage
Right after installation, you can use DALIA using the loaded modules and activated conda environment.

However, in general, in future shell sessions you will need to:
1. Source the `daint_cscs_utils.sh` script:
    ```bash
    source /path/to/DALIA/scripts/daint_cscs_utils.sh
    ```
2. Start the programming environment:
    ```bash
    daint_start_uenv
    ```
3. Source the `daint_cscs_utils.sh` script again (`uenv` is starting a new shell) to access the install utilities:
    ```bash
    source /path/to/DALIA/scripts/daint_cscs_utils.sh
    ```
4. Load the required environment modules:
    ```bash
    daint_load_modules
    ```
5. Activate the conda environment:
    ```bash
    daint_activate_conda_env
    ```

You will then be ready to use DALIA.

### c) Verify Installation
There is currently two ways to verify that DALIA has been installed correctly:
1. Run the provided test suite. Directly from the front node (after activating the correct conda environment) you can run the test suite: `./path/to/DALIA/tests/runner.sh`.
2. Run one of the provided examples, you can check the `/path/to/DALIA/examples/run_example_daint_cscs.sh` script for an example on how to submit a job on Daint.

## On a Personal Machine
### a) Installation
Given a working installation of `git` and `conda`, you can install DALIA on your personal machine as follows:

1. Clone the repositories to your workspace:
    ```bash
    mv /my/install/path
    git clone https://github.com/dalia-project/DALIA
    git clone https://github.com/vincent-maillou/serinv # Optional, recommended for ST-modeling
    ```

2. Based on your micro-architecture (x86 or aarch64), create the conda environment using the provided installer scripts:
- For x86 architecture:
    ```bash
    conda env create --name "env_name" -f path/to/DALIA/envs/dalia_base_x86.yml
    ```
- For aarch64 architecture:
    ```bash
    conda env create --name "env_name" -f path/to/DALIA/envs/dalia_base_aarch64.yml
    ```

3. Activate the conda environment:
    ```bash
    conda activate env_name
    ```

4. Install DALIA in editable mode:
    ```bash
    cd path/to/DALIA
    pip install -e .
    ```

5. Install Serinv (optional, recommended for ST-modeling) in editable mode:
    ```bash
    cd path/to/serinv
    pip install -e .
    ```

### b) Verify Installation
They are currently two ways to verify that DALIA has been installed correctly:
1. Run the provided test suite: `./path/to/DALIA/tests/runner.sh`.
2. Run one of the provided examples, for example: `python /path/to/DALIA/examples/gr/run.py`

## Notes on `git-lfs`

### Overview
DALIA uses `git-lfs` (Git Large File Storage) to manage large files such as example datasets. Files tracked by `git-lfs` are stored as pointers in the repository and downloaded on-demand rather than during the initial clone. This is particularly beneficial for large binary files (images, audio files, datasets) that don't compress well.

### Availability by Cluster

#### Fritz and Alex (FAU)
- `git-lfs` is available by default
- Location: `/usr/bin/git-lfs`
- No additional installation required

#### Daint (CSCS)
- `git-lfs` is **not** provided by the system
- Must be installed manually in your user space
- Installation script available at: https://gist.github.com/vincent-maillou/d2d38937f7aafbf0cee98c65cf5cfbca

### Important Note on Repository Cloning
Without `git-lfs` installed on Daint, `git clone` will fail because it automatically triggers `git-lfs checkout` during the checkout process. This is why the Daint installation instructions include a `git-lfs` installation step in the preamble (see section "On Daint@CSCS > Preamble" for detailed instructions).

## Other Informations

- After installing packages with conda, it is recommended to run `conda clean --all` to free up disk space.
- These installation procedures and `conda` environments have been tested on Linux systems based on both `x86` and `aarch64` architectures.
- If the `sqlite` module does not work properly, forcing the following version might help:
    ```bash
    conda install conda-forge::sqlite=3.45.3
    ```
- CuPy switched NCCL to lazy import, which means NCCL must be imported explicitly with `from cupy.cuda import nccl` before it is available in `cupy.cuda.nccl`.
