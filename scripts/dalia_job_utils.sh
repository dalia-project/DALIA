#!/bin/bash

dalia_set_perfenv() {
    echo "dalia_set_perfenv: setting up DALIA performance environment variables."
    export ARRAY_MODULE=numpy # numpy or cupy
    export MPI_CUDA_AWARE=0 # 0 or 1
    export USE_NCCL=0 # 0 or 1
    export MPICH_GPU_SUPPORT_ENABLED=0 # 0 or 1
    echo "DALIA Environment Configuration:"
    echo "  - ARRAY_MODULE: ${ARRAY_MODULE}"
    echo "  - MPI_CUDA_AWARE: ${MPI_CUDA_AWARE}"
    echo "  - USE_NCCL: ${USE_NCCL}"
    echo "  - MPICH_GPU_SUPPORT_ENABLED: ${MPICH_GPU_SUPPORT_ENABLED}"
    echo ""
}

dalia_print_job_config() {
    echo "dalia_print_job_config: printing job configuration."
    echo "SLURM Job Configuration:"
    echo "  - Job Name: ${SLURM_JOB_NAME}"
    echo "  - Job ID: ${SLURM_JOB_ID}"
    echo "  - Nodes: ${SLURM_NNODES}"
    echo "  - Tasks per node: ${SLURM_NTASKS_PER_NODE}"
    echo "  - Total tasks: ${SLURM_NTASKS}"
    echo "  - CPUs per task: ${SLURM_CPUS_PER_TASK}"
    if nvidia-smi &> /dev/null; then
        echo "  - GPUs per task: 1"
    else
        echo "  - GPUs per task: 0 (seemingly no GPU available)"
    fi
    echo "  - Time limit: ${SLURM_TIMELIMIT}"
    echo "  - Partition: ${SLURM_JOB_PARTITION}"
    echo "  - Account: ${SLURM_JOB_ACCOUNT}"
}
