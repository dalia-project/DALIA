#!/bin/bash

alex_load_modules() {
    echo "alex_load_modules: loading Alex system modules."
    
    # Check if module command is available
    if ! command -v module &> /dev/null; then
        echo "   Error: 'module' command not found. Please ensure you are on a system with environment modules."
        return 1
    fi
    
    # Purge any existing modules
    echo "   Purging existing modules..."
    module purge 2>/dev/null || {
        echo "   Warning: Failed to purge modules (this may be normal on some systems)."
    }

    # Load required modules
    echo "   Loading required modules: mkl/2023.2.0 gcc/12.1.0 cuda/12.9.0 openmpi/4.1.6-nvhpc23.7-cuda12  python"
    module load mkl/2023.2.0 gcc/12.1.0 openmpi/4.1.6-nvhpc23.7-cuda12 cuda/12.9.0 python || {
        echo "   Error: Failed to load required modules."
        echo "   Available modules:"
        module avail 2>&1 | head -20
        echo "   (output truncated - use 'module avail' for full list)"
        return 1
    }
    
    echo "   Successfully loaded all required modules."
    return 0
}

alex_check_modules() {
    echo "alex_check_modules: checking if required modules are loaded."
    
    local required_modules=("mkl/2023.2.0" "gcc/12.1.0" "cuda/12.9.0" "openmpi/4.1.6-nvhpc23.7-cuda12" "python")
    local missing_modules=()
    
    # Get list of currently loaded modules
    local loaded_modules=$(module list 2>&1 | grep -E "mkl|gcc|openmpi|cuda|nvhpc|python")
    
    # Check each required module
    for module in "${required_modules[@]}"; do
        if ! echo "$loaded_modules" | grep -q "$module"; then
            missing_modules+=("$module")
        fi
    done
    
    if [ ${#missing_modules[@]} -eq 0 ]; then
        echo "   All required modules are loaded."
        return 0
    else
        echo "   Error: Missing required modules: ${missing_modules[*]}"
        echo "   Please run 'alex_load_modules' first."
        return 1
    fi
}

alex_create_conda_env_help() {
    echo "alex_create_conda_env: Create DALIA conda environment for Alex supercomputer"
    echo ""
    echo "Usage:"
    echo "  alex_create_conda_env [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --dalia-path=PATH       Path to DALIA repository root directory"
    echo "  --serinv-path=PATH      Path to serinv repository (automatically installs serinv)"
    echo "  --install-mpi4py        Install mpi4py and create enhanced environment"
    echo "  --dev-mode              Keep intermediate environments (use with --install-mpi4py)"
    echo ""
    echo "Examples:"
    echo "  # Interactive mode (GPU support included by default)"
    echo "  alex_create_conda_env"
    echo ""
    echo "  # Non-interactive mode with all options"
    echo "  alex_create_conda_env --dalia-path=/path/to/dalia --serinv-path=/path/to/serinv --install-mpi4py --dev-mode"
    echo ""
    echo "  # Install base with GPU support only"
    echo "  alex_create_conda_env --dalia-path=/path/to/dalia"
    echo ""
    echo "Note: GPU support via cupy is installed by default for Alex cluster."
    echo "      If parameters are not provided, the function will prompt interactively."
}


alex_create_conda_env() {
    echo "alex_create_conda_env: creating DALIA conda environment for Alex."
    
    # Parse command line arguments
    local dalia_path=""
    local serinv_path=""
    local install_mpi4py_flag=""
    local dev_mode_flag=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dalia-path=*)
                dalia_path="${1#*=}"
                shift
                ;;
            --serinv-path=*)
                serinv_path="${1#*=}"
                shift
                ;;
            --install-mpi4py)
                install_mpi4py_flag="y"
                shift
                ;;
            --dev-mode)
                dev_mode_flag="y"
                shift
                ;;
            --help|-h)
                alex_create_conda_env_help
                return 0
                ;;
            *)
                echo "   Warning: Unknown parameter '$1' ignored."
                echo "   Use --help for usage information."
                shift
                ;;
        esac
    done
    
    # 0. Deactivate any currently active conda environment
    echo "   Deactivating any currently active conda environment..."
    conda deactivate 2>/dev/null || true
    
    # 1. Check that the needed modules are loaded
    if ! alex_check_modules; then
        return 1
    fi
    
    # 2. Get DALIA repository path
    if [[ -z "$dalia_path" ]]; then
        echo "   Please enter the path to the DALIA root repository:"
        read -r dalia_path
    else
        echo "   Using provided DALIA path: ${dalia_path}"
    fi
    
    # Expand tilde and remove trailing slash
    dalia_path=$(eval echo "${dalia_path}")
    dalia_path="${dalia_path%/}"
    
    # 3. Validate DALIA repository path
    if [[ ! -d "$dalia_path" ]]; then
        echo "   Error: Directory '${dalia_path}' does not exist."
        return 1
    fi
    
    if [[ ! -f "${dalia_path}/pyproject.toml" ]]; then
        echo "   Error: '${dalia_path}' does not appear to be a DALIA repository (missing pyproject.toml)."
        return 1
    fi
    
    if [[ ! -d "${dalia_path}/src/dalia" ]]; then
        echo "   Error: '${dalia_path}' does not contain the DALIA source code (missing src/dalia/)."
        return 1
    fi
    
    # 4. Locate the conda environment file
    local env_file="${dalia_path}/envs/dalia_base_x86.yml"
    if [[ ! -f "$env_file" ]]; then
        echo "   Error: Conda environment file not found at '${env_file}'."
        return 1
    fi
    
    echo "   Found DALIA repository at: ${dalia_path}"
    echo "   Found conda environment file at: ${env_file}"
    
    # Check if environment already exists
    local env_name="dalia_base_alex"
    if conda env list | grep -q "^${env_name} "; then
        echo "   Warning: Conda environment '${env_name}' already exists."
        echo "   Do you want to remove and recreate it? (y/N): "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            echo "   Removing existing environment..."
            conda env remove -n "$env_name" -y || {
                echo "   Error: Failed to remove existing environment."
                return 1
            }
        else
            echo "   Skipping environment creation. Using existing environment."
        fi
    fi
    
    # Create conda environment from YAML file (only if it doesn't exist or was removed)
    if ! conda env list | grep -q "^${env_name} "; then
        echo "   Creating conda environment from '${env_file}'..."
        conda env create --name "$env_name" -f "$env_file" || {
            echo "   Error: Failed to create conda environment from '${env_file}'."
            return 1
        }
    fi
    
    # 5. Activate the conda environment
    echo "   Activating conda environment..."
    if ! alex_activate_conda_env --env="$env_name"; then
        return 1
    fi
    
    # 6. Install DALIA from the repository
    echo "   Installing DALIA in development mode..."
    cd "$dalia_path" || {
        echo "   Error: Failed to change directory to '${dalia_path}'."
        return 1
    }
    
    python -m pip install --no-deps --editable . || {
        echo "   Error: Failed to install DALIA in development mode."
        return 1
    }
    
    # 7. Optional: Install serinv structured sparse solver
    echo ""
    local install_serinv=""
    
    if [[ -n "$serinv_path" ]]; then
        # Serinv path provided via command line - install automatically
        echo "   Serinv path provided via --serinv-path. Installing serinv automatically..."
        install_serinv="y"
    else
        # Interactive mode - ask user
        echo "   Spatio-temporal problems in DALIA can leverage the 'serinv' structured sparse solver for improved performance."
        echo "   Do you want to install the serinv structured sparse solver? (y/N): "
        read -r install_serinv
        
        if [[ "$install_serinv" =~ ^[Yy]$ ]]; then
            echo "   Please enter the path to the serinv root repository:"
            read -r serinv_path
        fi
    fi
    
    if [[ "$install_serinv" =~ ^[Yy]$ ]]; then
        # Expand tilde and remove trailing slash
        serinv_path=$(eval echo "${serinv_path}")
        serinv_path="${serinv_path%/}"
        
        # Validate serinv repository path
        if [[ ! -d "$serinv_path" ]]; then
            echo "   Error: Directory '${serinv_path}' does not exist."
            echo "   Skipping serinv installation."
        elif [[ ! -f "${serinv_path}/pyproject.toml" ]] && [[ ! -f "${serinv_path}/setup.py" ]]; then
            echo "   Error: '${serinv_path}' does not appear to be a valid Python package (missing pyproject.toml or setup.py)."
            echo "   Skipping serinv installation."
        else
            echo "   Installing serinv from '${serinv_path}' in development mode..."
            cd "$serinv_path" || {
                echo "   Error: Failed to change directory to '${serinv_path}'."
                echo "   Skipping serinv installation."
            }
            
            if python -m pip install --no-deps --editable .; then
                echo "   Successfully installed serinv in development mode."
            else
                echo "   Warning: Failed to install serinv. You may need to install it manually later."
            fi
            
            # Return to DALIA directory
            cd "$dalia_path" || true
        fi
    else
        echo "   Skipping serinv installation."
    fi

    # 8. Install cupy with GPU support (default for Alex cluster)
    echo ""
    echo "   Installing cupy with GPU support using SLURM job (default for Alex cluster)..."
    
    # Install cupy using SLURM job on GPU partition
    echo "   Submitting SLURM job to install cupy on GPU partition..."
    local job_script=$(mktemp)
    cat > "$job_script" <<'SLURM_EOF'
#!/bin/bash -l
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
conda activate ENV_NAME_PLACEHOLDER
python -m pip install cupy-cuda12x --no-cache-dir
SLURM_EOF
    
    # Replace the environment name placeholder
    sed -i "s/ENV_NAME_PLACEHOLDER/${env_name}/g" "$job_script"
    
    # Submit the job and capture job ID
    local job_output=$(sbatch --partition=a40 --nodes=1 --gres=gpu:a40:1 --time=00:05:00 --job-name="cupy_install_${env_name}" "$job_script")
    local job_id=$(echo "$job_output" | grep -oE '[0-9]+')
    
    # Clean up temporary script
    rm -f "$job_script"
    
    if [[ -n "$job_id" ]]; then
        echo "   SLURM job submitted successfully with ID: ${job_id}"
        echo "   Waiting for cupy installation to complete..."
        
        # Wait for job completion
        local job_status=""
        local wait_count=0
        local max_wait=120  # Maximum wait time in 5-second intervals (10 minutes)
        
        while [[ "$wait_count" -lt "$max_wait" ]]; do
            job_status=$(squeue -j "$job_id" -h -o "%T" 2>/dev/null || echo "COMPLETED")
            
            case "$job_status" in
                "RUNNING")
                    echo "   Job ${job_id} is running... (${wait_count}/${max_wait})"
                    ;;
                "PENDING")
                    echo "   Job ${job_id} is pending... (${wait_count}/${max_wait})"
                    ;;
                "COMPLETED"|"")
                    echo "   Job ${job_id} completed successfully."
                    break
                    ;;
                "FAILED"|"CANCELLED"|"TIMEOUT")
                    echo "   Error: Job ${job_id} failed with status: ${job_status}"
                    break
                    ;;
                *)
                    echo "   Job ${job_id} status: ${job_status} (${wait_count}/${max_wait})"
                    ;;
            esac
            
            sleep 5
            ((wait_count++))
        done
        
        # Check final job status
        if [[ "$wait_count" -ge "$max_wait" ]]; then
            echo "   Warning: Timeout waiting for job completion. Please check job status manually: squeue -j ${job_id}"
            echo "   You can also check the job output with: scontrol show job ${job_id}"
            # Clean up SLURM job output files after timeout
            echo "   Cleaning up SLURM job files..."
            rm -f "cupy_install_${env_name}.o${job_id}" "cupy_install_${env_name}.e${job_id}" 2>/dev/null || true
        elif [[ "$job_status" == "COMPLETED" || -z "$job_status" ]]; then
            # Removed cupy instalaltion check as on front node there is no Cuda capable device.
            echo "   CuPy installation completed."

            # Clean up SLURM job output files
            echo "   Cleaning up SLURM job files..."
            rm -f "cupy_install_${env_name}.o${job_id}" "cupy_install_${env_name}.e${job_id}" 2>/dev/null || true
        else
            echo "   Error: CuPy installation job failed. Please check SLURM logs."
            echo "   Continuing with base environment without GPU support..."
            echo "   You can install cupy manually later by submitting a GPU job."
            # Clean up SLURM job output files after failure
            echo "   Cleaning up SLURM job files..."
            rm -f "cupy_install_${env_name}.o${job_id}" "cupy_install_${env_name}.e${job_id}" 2>/dev/null || true
        fi
    else
        echo "   Error: Failed to submit SLURM job for cupy installation."
        echo "   Continuing with base environment without GPU support..."
        echo "   You can install cupy manually later by submitting a GPU job."
    fi
    
    # 9. Optional: Install mpi4py and create enhanced environment
    echo ""
    local install_mpi4py=""
    
    if [[ -n "$install_mpi4py_flag" ]]; then
        # MPI4py installation requested via command line
        echo "   MPI4py installation requested via --install-mpi4py. Installing automatically..."
        install_mpi4py="y"
    else
        # Interactive mode - ask user
        echo "   MPI support can be added through mpi4py for improved parallel performance."
        echo "   Do you want to install mpi4py and create an enhanced environment? (y/N): "
        read -r install_mpi4py
    fi
    
    if [[ "$install_mpi4py" =~ ^[Yy]$ ]]; then
        echo "   Creating enhanced environment with mpi4py support..."
        
        # Determine the enhanced environment name
        local mpi_enhanced_env_name="dalia_xccl_alex"  # Always GPU + MPI for Alex
        echo "   Creating environment with GPU and mpi4py support..."
        
        # Deactivate current environment
        echo "   Deactivating current environment to create MPI-enhanced version..."
        conda deactivate 2>/dev/null || true
        
        # Remove enhanced environment if it already exists
        if conda env list | grep -q "^${mpi_enhanced_env_name} "; then
            echo "   Removing existing MPI-enhanced environment..."
            conda env remove -n "$mpi_enhanced_env_name" -y || {
                echo "   Warning: Failed to remove existing MPI-enhanced environment."
            }
        fi
        
        # Clone the current environment
        if conda create --name "$mpi_enhanced_env_name" --clone "$env_name" -y; then
            echo "   Successfully created MPI-enhanced environment."
            
            # Activate the enhanced environment
            echo "   Activating MPI-enhanced environment '${mpi_enhanced_env_name}'..."
            if alex_activate_conda_env --env="$mpi_enhanced_env_name"; then
                # Install mpi4py in the enhanced environment
                echo "   Installing mpi4py with OpenMPI support in enhanced environment..."
                cd "$dalia_path" || true
                if MPICC=$(which mpicc) pip install --no-cache-dir mpi4py; then
                    echo "   Successfully installed mpi4py in MPI-enhanced environment."
                    
                    # Determine whether to keep base environment
                    local keep_base=""
                    if [[ -n "$install_mpi4py_flag" ]]; then
                        # Command line mode - use dev_mode_flag to decide
                        if [[ -n "$dev_mode_flag" ]]; then
                            keep_base="y"
                            echo "   Developer mode enabled via --dev-mode. Keeping base environment."
                        else
                            keep_base="n"
                            echo "   Default mode: removing base environment to keep only enhanced version."
                        fi
                    else
                        # Interactive mode - ask user
                        echo ""
                        echo "   Do you want to keep the base environment 'dalia_base_alex' for development without MPI? (y/N): "
                        read -r keep_base
                    fi
                    
                    if [[ ! "$keep_base" =~ ^[Yy]$ ]]; then
                        echo "   Removing base environment 'dalia_base_alex'..."
                        conda env remove -n "dalia_base_alex" -y || {
                            echo "   Warning: Failed to remove base environment."
                        }
                    else
                        echo "   Keeping base environment for development without MPI."
                    fi
                    
                    env_name="$mpi_enhanced_env_name"  # Update env_name for final message
                else
                    echo "   Error: Failed to install mpi4py in MPI-enhanced environment."
                    echo "   Removing broken MPI-enhanced environment and reverting to base environment..."
                    
                    # Deactivate the enhanced environment
                    conda deactivate 2>/dev/null || true
                    
                    # Remove the broken enhanced environment
                    conda env remove -n "$mpi_enhanced_env_name" -y || {
                        echo "   Warning: Failed to remove broken MPI-enhanced environment."
                    }
                    
                    # Reactivate the base environment
                    if alex_activate_conda_env --env="dalia_base_alex"; then
                        echo "   Reverted to base environment 'dalia_base_alex'."
                        env_name="dalia_base_alex"  # Reset env_name to base environment
                        echo "   You can install mpi4py manually later with: MPICC=\$(which mpicc) pip install --no-cache-dir mpi4py"
                    else
                        echo "   Warning: Failed to reactivate base environment."
                    fi
                fi
            else
                echo "   Warning: Failed to activate MPI-enhanced environment."
            fi
        else
            echo "   Error: Failed to create MPI-enhanced environment."
            echo "   Continuing with base environment..."
        fi
    else
        echo "   Skipping mpi4py installation."
    fi

    # 10. Final success message    
    echo ""
    echo "   Success! DALIA conda environment '${env_name}' has been created and configured."
    echo "   Repository path: ${dalia_path}"
    if [[ "$install_serinv" =~ ^[Yy]$ ]] && [[ -d "$serinv_path" ]]; then
        echo "   Serinv path: ${serinv_path}"
    fi
    echo "   Base environment includes GPU support via cupy (default for Alex cluster)."
    if [[ "$install_mpi4py" =~ ^[Yy]$ ]]; then
        if [[ "$env_name" == *"xccl"* ]]; then
            echo "   Enhanced environment with mpi4py and NCCL support created."
        fi
    fi
    echo "   To use a specific environment in the future, run: alex_activate_conda_env --env=\"alex_env_name\""
    echo "   To use the most performant environment you have available, just run: alex_activate_conda_env"
    
    return 0
}



alex_activate_conda_env() {
    echo "alex_activate_conda_env: activating DALIA conda environment."
    
    # Parse command line arguments
    local specified_env=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --env=*)
                specified_env="${1#*=}"
                shift
                ;;
            --help|-h)
                echo "alex_activate_conda_env: Activate DALIA conda environment"
                echo ""
                echo "Usage:"
                echo "  alex_activate_conda_env [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --env=NAME              Specific environment name to activate"
                echo "  --help/-h               Show this help message"
                echo ""
                echo "Examples:"
                echo "  alex_activate_conda_env                        # Auto-select best available"
                echo "  alex_activate_conda_env --env=dalia_base_alex  # Activate specific environment"
                return 0
                ;;
            *)
                echo "   Warning: Unknown parameter '$1' ignored."
                echo "   Use --help for usage information."
                shift
                ;;
        esac
    done
    
    # Check if conda is available
    if ! command -v conda &> /dev/null; then
        echo "   Error: conda command not found. Please ensure conda is installed and in PATH."
        return 1
    fi
    
    # Safely deactivate current environment
    conda deactivate 2>/dev/null || true

    # Define available environments in order of preference (most performant first)
    local env_priorities=("dalia_xccl_alex" "dalia_base_alex")
    local env_name=""
    
    # If environment name is provided as argument, use it directly
    if [[ -n "$specified_env" ]]; then
        env_name="$specified_env"
        echo "   Using specified conda environment '${env_name}'..."
        
        # Validate that the specified environment exists
        if ! conda env list | grep -q "^${env_name} "; then
            echo "   Error: Conda environment '${env_name}' does not exist."
            echo "   Available environments:"
            conda env list
            return 1
        fi
    else
        # Check which environments are available and select the most performant one
        echo "   Checking available DALIA conda environments..."
        local available_envs=$(conda env list 2>/dev/null | grep -E "^(dalia_xccl_alex|dalia_base_alex) " | awk '{print $1}')
        
        for preferred_env in "${env_priorities[@]}"; do
            if echo "$available_envs" | grep -q "^${preferred_env}$"; then
                env_name="$preferred_env"
                echo "   Selected most performant available environment: '${env_name}'"
                break
            fi
        done
        
        # Fallback if no DALIA environments found
        if [[ -z "$env_name" ]]; then
            echo "   Warning: No DALIA-specific environments found. Available environments:"
            conda env list
            echo "   Falling back to 'base' environment..."
            env_name="base"
        fi
    fi

    echo "   Activating conda environment '${env_name}'..."
    conda activate ${env_name} || {
        echo "   Error: Failed to activate conda environment '${env_name}'"
        echo "   Please check that the environment exists and conda is properly configured."
        return 1
    }
    
    # Verify activation was successful
    local current_env=$(conda info --envs | grep '\*' | awk '{print $1}')
    if [[ "$current_env" == "$env_name" ]]; then
        echo "   Successfully activated conda environment '${env_name}'"
    else
        echo "   Warning: Environment activation may not have been successful."
        echo "   Expected: ${env_name}, Current: ${current_env}"
    fi
    
    return 0
}

alex_set_perfenv() {
    echo "alex_set_perfenv: setting performance environment variables for Alex."

    unset SLURM_EXPORT_ENV

    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
    export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
}