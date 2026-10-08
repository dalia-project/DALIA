#!/bin/bash

daint_install_conda_help() {
    echo "daint_install_conda: Install Miniconda3 for the user"
    echo ""
    echo "Usage:"
    echo "  daint_install_conda [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --install-path=PATH    Custom installation path (default: \$HOME/miniconda3)"
    echo "  --no-init              Skip shell initialization"
    echo "  --init-all             Initialize for all detected shells (bash, zsh)"
    echo "  --yes                  Skip confirmation prompts"
    echo "  --help, -h             Show this help message"
    echo ""
    echo "Examples:"
    echo "  daint_install_conda                                   # Install to \$HOME/miniconda3"
    echo "  daint_install_conda --install-path=/custom/path       # Install to custom location"
    echo "  daint_install_conda --no-init                         # Install without shell init"
    echo "  daint_install_conda --yes                             # Skip all prompts"
    echo ""
    echo "Return codes:"
    echo "  0 - Success"
    echo "  1 - Existing installation detected"
    echo "  2 - Download failed"
    echo "  3 - Installation failed"
    echo "  4 - Insufficient disk space"
    echo "  5 - Network connectivity issues"
}

daint_install_conda() {
    echo "daint_install_conda: Installing Miniconda3 for the user."
    
    # Parse command line arguments
    local install_path="$HOME/miniconda3"
    local no_init=""
    local init_all=""
    local auto_confirm=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --install-path=*)
                install_path="${1#*=}"
                shift
                ;;
            --no-init)
                no_init="y"
                shift
                ;;
            --init-all)
                init_all="y"
                shift
                ;;
            --yes)
                auto_confirm="y"
                shift
                ;;
            --help|-h)
                daint_install_conda_help
                return 0
                ;;
            *)
                echo "   Warning: Unknown parameter '$1' ignored."
                echo "   Use --help for usage information."
                shift
                ;;
        esac
    done
    
    # Expand tilde and remove trailing slash
    install_path="${install_path/#\~/$HOME}"
    install_path="${install_path%/}"
    
    echo "   Target installation path: ${install_path}"
    echo ""
    
    # =============================================================================
    # 1. PRE-INSTALLATION CHECKS
    # =============================================================================
    
    echo "   Step 1/7: Checking for existing Miniconda/Conda installations..."
    local existing_installation=""
    
    # Check if installation directory exists
    if [[ -d "$install_path" ]]; then
        echo "   Error: Directory '${install_path}' already exists."
        existing_installation="directory"
    fi
    
    # Check if conda is in PATH
    if command -v conda &> /dev/null; then
        local conda_location=$(which conda 2>/dev/null)
        echo "   Error: conda command found in PATH at: ${conda_location}"
        existing_installation="path"
    fi
    
    # Check .bashrc for conda initialization
    if [[ -f "$HOME/.bashrc" ]] && grep -q "# >>> conda initialize >>>" "$HOME/.bashrc"; then
        echo "   Error: Conda initialization block found in ~/.bashrc"
        existing_installation="bashrc"
    fi
    
    # Check .bash_profile for conda references
    if [[ -f "$HOME/.bash_profile" ]] && grep -qi "conda\|miniconda\|anaconda" "$HOME/.bash_profile"; then
        echo "   Error: Conda references found in ~/.bash_profile"
        existing_installation="bash_profile"
    fi
    
    # Check .zshrc if it exists
    if [[ -f "$HOME/.zshrc" ]] && grep -q "# >>> conda initialize >>>" "$HOME/.zshrc"; then
        echo "   Error: Conda initialization block found in ~/.zshrc"
        existing_installation="zshrc"
    fi
    
    # If any existing installation found, exit with instructions
    if [[ -n "$existing_installation" ]]; then
        echo ""
        echo "   =========================================="
        echo "   EXISTING CONDA INSTALLATION DETECTED"
        echo "   =========================================="
        echo "   An existing conda/miniconda installation was detected on your system."
        echo "     a) If this installation is working, you may continue..."
        echo "     b) If this installation is broken, please manually uninstall it and re-running this script:"
        echo "       1. Remove the installation directory:"
        if [[ -d "$install_path" ]]; then
            echo "       rm -rf ${install_path}"
        fi
        echo ""
        echo "       2. Remove conda initialization from shell configuration files:"
        if [[ -f "$HOME/.bashrc" ]] && grep -q "# >>> conda initialize >>>" "$HOME/.bashrc"; then
            echo "      Edit ~/.bashrc and remove the section between:"
            echo "      '# >>> conda initialize >>>' and '# <<< conda initialize <<<'"
        fi
        if [[ -f "$HOME/.bash_profile" ]] && grep -qi "conda\|miniconda\|anaconda" "$HOME/.bash_profile"; then
            echo "      Edit ~/.bash_profile and remove conda-related lines"
        fi
        if [[ -f "$HOME/.zshrc" ]] && grep -q "# >>> conda initialize >>>" "$HOME/.zshrc"; then
            echo "      Edit ~/.zshrc and remove the section between:"
            echo "      '# >>> conda initialize >>>' and '# <<< conda initialize <<<'"
        fi
        echo ""
        echo "       3. Start a new terminal session or run: source ~/.bashrc"
        echo ""
        echo "       4. Re-run this installer: daint_install_conda"
        echo ""
        return 1
    fi
    
    echo "   No existing conda installation detected."
    echo ""
    
    # =============================================================================
    # 2. VALIDATE INSTALLATION PATH
    # =============================================================================
    
    echo "   Step 2/7: Validating installation path..."
    
    # Get parent directory
    local parent_dir=$(dirname "$install_path")
    
    # Check if parent directory exists
    if [[ ! -d "$parent_dir" ]]; then
        echo "   Error: Parent directory '${parent_dir}' does not exist."
        return 3
    fi
    
    # Check write permissions
    if [[ ! -w "$parent_dir" ]]; then
        echo "   Error: No write permission for parent directory '${parent_dir}'."
        return 3
    fi
    
    echo "   Installation path is valid."
    echo ""
    
    # =============================================================================
    # 3. CHECK DISK SPACE
    # =============================================================================
    
    echo "   Step 3/7: Checking available disk space..."
    
    # Get available space in KB
    local available_space=$(df -k "$parent_dir" | tail -1 | awk '{print $4}')
    local required_space=5242880  # 5 GB in KB
    
    if [[ $available_space -lt $required_space ]]; then
        local available_gb=$((available_space / 1024 / 1024))
        echo "   Error: Insufficient disk space. Available: ${available_gb}GB, Required: 5GB"
        return 4
    fi
    
    local available_gb=$((available_space / 1024 / 1024))
    echo "   Sufficient disk space available: ${available_gb}GB"
    echo ""
    
    # =============================================================================
    # 4. CHECK NETWORK CONNECTIVITY AND DOWNLOAD MINICONDA
    # =============================================================================
    
    echo "   Step 4/7: Downloading Miniconda installer..."
    
    # Detect system architecture
    local arch=$(uname -m)
    local installer_name=""
    
    case "$arch" in
        x86_64)
            installer_name="Miniconda3-latest-Linux-x86_64.sh"
            ;;
        aarch64|arm64)
            installer_name="Miniconda3-latest-Linux-aarch64.sh"
            ;;
        *)
            echo "   Error: Unsupported architecture: ${arch}"
            echo "   Supported architectures: x86_64, aarch64"
            return 3
            ;;
    esac
    
    echo "   Detected architecture: ${arch}"
    echo "   Installer: ${installer_name}"
    
    # Set download URL
    local download_url="https://repo.anaconda.com/miniconda/${installer_name}"
    local installer_path="/tmp/${installer_name}"
        
    # Test network connectivity
    echo "   Testing connectivity to repo.anaconda.com..."
    if ! curl -s --connect-timeout 10 --max-time 15 "https://repo.anaconda.com" > /dev/null; then
        if ! wget -q --timeout=10 --tries=1 --spider "https://repo.anaconda.com" 2>/dev/null; then
            echo "   Error: Cannot reach repo.anaconda.com. Please check your network connection."
            echo "   If you're behind a proxy, set http_proxy and https_proxy environment variables."
            return 5
        fi
    fi
    echo "   Network connectivity OK"
    
    # Remove existing partial download if present
    if [[ -f "$installer_path" ]]; then
        echo "   Removing existing installer file..."
        rm -f "$installer_path"
    fi
    
    # Download installer
    echo "   Downloading from: ${download_url}"
    echo "   This may take a few minutes..."
    
    # Try curl first, then wget
    local download_success=0
    if command -v curl &> /dev/null; then
        if curl -L -o "$installer_path" "$download_url" 2>&1 | grep -v "^#"; then
            download_success=1
        fi
    elif command -v wget &> /dev/null; then
        if wget -O "$installer_path" "$download_url"; then
            download_success=1
        fi
    else
        echo "   Error: Neither curl nor wget is available for downloading."
        return 2
    fi
    
    if [[ $download_success -eq 0 ]]; then
        echo "   Error: Failed to download Miniconda installer."
        rm -f "$installer_path"
        return 2
    fi
    
    # Verify download
    if [[ ! -f "$installer_path" ]]; then
        echo "   Error: Installer file not found after download."
        return 2
    fi
    
    local file_size=$(stat -c%s "$installer_path" 2>/dev/null || stat -f%z "$installer_path" 2>/dev/null)
    if [[ $file_size -lt 50000000 ]]; then  # Less than 50MB indicates a problem
        echo "   Error: Downloaded file is too small (${file_size} bytes). Download may be incomplete."
        rm -f "$installer_path"
        return 2
    fi
    
    echo "   Successfully downloaded installer ($(($file_size / 1024 / 1024))MB)"
    echo ""
    
    # =============================================================================
    # 5. CONFIRM INSTALLATION
    # =============================================================================
    
    if [[ -z "$auto_confirm" ]]; then
        echo "   =========================================="
        echo "   Ready to install Miniconda3"
        echo "   =========================================="
        echo "   Installation path: ${install_path}"
        echo "   Architecture: ${arch}"
        echo "   Installer size: $(($file_size / 1024 / 1024))MB"
        echo ""
        echo -n "   Proceed with installation? [y/N]: "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "   Installation cancelled by user."
            rm -f "$installer_path"
            return 0
        fi
    fi
    
    # =============================================================================
    # 6. RUN INSTALLER
    # =============================================================================
    
    echo ""
    echo "   Step 5/7: Running Miniconda installer..."
    echo "   This may take several minutes..."
    
    # Run installer in batch mode
    if bash "$installer_path" -b -p "$install_path"; then
        echo "   Successfully installed Miniconda3 to: ${install_path}"
    else
        echo "   Error: Miniconda installation failed."
        rm -f "$installer_path"
        return 3
    fi
    
    echo ""
    
    # =============================================================================
    # 7. POST-INSTALLATION CONFIGURATION
    # =============================================================================
    
    echo "   Step 6/7: Configuring conda..."
    
    # Verify installation
    if [[ ! -f "${install_path}/bin/conda" ]]; then
        echo "   Error: conda executable not found at ${install_path}/bin/conda"
        rm -f "$installer_path"
        return 3
    fi
    
    # Get conda version
    local conda_version=$("${install_path}/bin/conda" --version 2>&1)
    echo "   Installed: ${conda_version}"
    
    # Initialize conda for shell(s)
    if [[ -z "$no_init" ]]; then
        echo "   Initializing conda for shell(s)..."
        
        # Always initialize bash
        if "${install_path}/bin/conda" init bash > /dev/null 2>&1; then
            echo "   Initialized conda for bash"
        else
            echo "   Warning: Failed to initialize conda for bash"
        fi
        
        # Initialize other shells if requested
        if [[ -n "$init_all" ]]; then
            # Initialize zsh if .zshrc exists
            if [[ -f "$HOME/.zshrc" ]]; then
                if "${install_path}/bin/conda" init zsh > /dev/null 2>&1; then
                    echo "   Initialized conda for zsh"
                else
                    echo "   Warning: Failed to initialize conda for zsh"
                fi
            fi
        fi
    else
        echo "   Skipping shell initialization (--no-init flag specified)"
    fi
    
    echo ""
    
    # =============================================================================
    # 8. CLEANUP
    # =============================================================================
    
    echo "   Step 7/7: Cleaning up..."
    rm -f "$installer_path"
    echo "   Removed temporary installer file"
    echo ""
    
    # =============================================================================
    # 9. FINAL INSTRUCTIONS
    # =============================================================================
    
    echo "   =========================================="
    echo "   SUCCESS: Miniconda3 Installed!"
    echo "   =========================================="
    echo "   Installation path: ${install_path}"
    echo "   Conda version: ${conda_version}"
    echo ""
    
    if [[ -z "$no_init" ]]; then
        echo "   To activate conda, run ONE of the following:"
        echo "   1. Start a new terminal session, OR"
        echo "   2. Run: source ~/.bashrc"
        echo ""
        echo "   After activation, verify the installation with:"
        echo "      conda --version"
        echo "      conda info"
    else
        echo "   Shell initialization was skipped."
        echo "   To use conda, you can:"
        echo "   1. Manually initialize: ${install_path}/bin/conda init bash"
        echo "   2. Or activate manually: eval \"\$(${install_path}/bin/conda shell.bash hook)\""
    fi
    echo ""
    
    return 0
}

daint_install_git_lfs() {
    # This script will install and configure git-lfs on Daint@ALPS as it is not 
    # available by default in the system.

    # 1. Move to your home directory
    cd $HOME || { echo "Could not change to home directory"; exit 1; }

    # 2. Download git-lfs for ARM64
    wget https://github.com/git-lfs/git-lfs/releases/download/v3.7.1/git-lfs-linux-arm64-v3.7.1.tar.gz || { echo "Could not download git-lfs"; exit 1; }

    # 3. Extract the downloaded tarball
    tar -xvf git-lfs-linux-arm64-v3.7.1.tar.gz || { echo "Could not extract git-lfs tarball"; exit 1; }

    # 4. Move in the extracted directory and make the installer executable
    cd git-lfs-3.7.1 || { echo "Could not change to git-lfs directory"; exit 1; }
    chmod +x install.sh || { echo "Could not make installer executable"; exit 1; }

    # 5. Change the installer prefix to your home directory
    sed -i 's|^prefix="/usr/local"$|prefix="$HOME/.local"|' install.sh || { echo "Could not modify installer prefix"; exit 1; }

    # 6. Make the .local/bin directory if it does not exist
    mkdir -p "$HOME/.local/bin" || { echo "Could not create .local/bin directory"; exit 1; }

    # 7. Run the installer
    ./install.sh || { echo "Could not install git-lfs"; exit 1; }

    # 8. Add .local/bin to your PATH if not already present
    if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
        export PATH="$HOME/.local/bin:$PATH"
    fi

    # 9 Add .local/bin to your PATH in .bashrc for future sessions
    if ! grep -q 'export PATH="$HOME/.local/bin:$PATH"' "$HOME/.bashrc"; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
    fi

    # 10. Verify the installation
    if command -v git-lfs &> /dev/null; then
        echo "git-lfs installed successfully!"
    else
        echo "git-lfs installation failed"
        exit 1
    fi
}

daint_install_uenv() {
    # Documentation: https://docs.cscs.ch/software/uenv/
    echo "daint_install_uenv: setting up Daint.Alps virtual environment."

    if uenv image ls | grep -q "prgenv-gnu/25.6:v2"; then
        echo "   prgenv-gnu/25.6:v2 image already exists, skipping pull"
    else
        echo "   Pulling prgenv-gnu/25.6:v2 image..."
        uenv image pull prgenv-gnu/25.6:v2 || {
            echo "   Error: Failed to pull prgenv-gnu/25.6:v2 image"
            return 1
        }
    fi

    return 0
}

daint_start_uenv() {
    # Documentation: https://docs.cscs.ch/software/uenv/
    echo "daint_start_uenv: starting Daint.Alps uenv environment."

    # Stop any existing uenv session
    uenv stop 2>/dev/null || echo "   (No existing uenv to stop)"
    
    # Start new uenv session
    echo "   Starting uenv with prgenv-gnu/25.6:v2..."
    echo "   WARNING: This is gonna start a new shell session, if you want to"
    echo "   use other functions from this script, you need to source it again."
    uenv start --view=modules prgenv-gnu/25.6:v2
}

daint_load_modules() {
    echo "daint_load_modules: loading Daint system modules."
    
    # Check if we're already in a uenv session
    if ! uenv status &>/dev/null; then
        echo "Error: Not in a uenv session. Modules can only be loaded within a uenv session."
        return 1
    fi

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
    echo "   Loading required modules: cuda gcc meson ninja nccl cray-mpich cmake openblas aws-ofi-nccl netlib-scalapack"
    module load cuda gcc meson ninja nccl cray-mpich cmake openblas aws-ofi-nccl netlib-scalapack || {
        echo "   Error: Failed to load required modules."
        echo "   Available modules:"
        module avail 2>&1 | head -20
        echo "   (output truncated - use 'module avail' for full list)"
        return 1
    }

    # Check for CUDA_HOME environment variable
    if [[ -z "$CUDA_HOME" ]]; then
        echo "   Error: CUDA_HOME not set, please ensure the CUDA module properly sets CUDA_HOME or manually set CUDA_HOME to your CUDA installation directory"
        return 1
    fi

    # Set CUDA environment variables
    export CUDA_DIR=$CUDA_HOME
    export CUDA_PATH=$CUDA_HOME
    export CPATH=$CUDA_HOME/include:$CPATH
    export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

    # Set NCCL environment variables (dynamically find the nccl installation)
    export NCCL_ROOT=$(ls -d /user-environment/linux-neoverse_v2/nccl-* 2>/dev/null | head -1)
    if [[ -z "$NCCL_ROOT" ]]; then
        echo "   Error: NCCL not found in /user-environment/linux-neoverse_v2/"
        return 1
    fi
    export NCCL_LIB_DIR=$NCCL_ROOT/lib
    export NCCL_INCLUDE_DIR=$NCCL_ROOT/include

    export CPATH=$NCCL_ROOT/include:$CPATH
    export CFLAGS="-I$NCCL_INCLUDE_DIR":$CFLAGS
    export LDFLAGS="-L$NCCL_LIB_DIR":$LDFLAGS
    export LIBRARY_PATH=$NCCL_LIB_DIR:$LIBRARY_PATH
    export LD_LIBRARY_PATH=$NCCL_LIB_DIR:$LD_LIBRARY_PATH
    
    echo "   Successfully loaded all required modules."
    return 0
}

daint_check_modules() {
    echo "daint_check_modules: checking if required modules are loaded."
    
    local required_modules=("cuda" "gcc" "meson" "ninja" "nccl" "cray-mpich" "cmake" "openblas" "aws-ofi-nccl" "netlib-scalapack")
    local missing_modules=()
    
    # Get list of currently loaded modules
    local loaded_modules=$(module list 2>&1 | grep -E "cuda|gcc|meson|ninja|nccl|cray-mpich|cmake|openblas|aws-ofi-nccl|netlib-scalapack")
    
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
        echo "   Please run 'daint_load_modules' first."
        return 1
    fi
}

daint_create_conda_env_help() {
    echo "daint_create_conda_env: Create DALIA conda environment for Daint supercomputer"
    echo ""
    echo "Usage:"
    echo "  daint_create_conda_env [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --dalia-path=PATH       Path to DALIA repository root directory"
    echo "  --serinv-path=PATH      Path to serinv repository (automatically installs serinv)"
    echo "  --install-mpi4py        Install mpi4py and create enhanced environment"
    echo "  --dev-mode              Keep intermediate environments (use with --install-mpi4py)"
    echo ""
    echo "Examples:"
    echo "  # Interactive mode (GPU support included by default)"
    echo "  daint_create_conda_env"
    echo ""
    echo "  # Non-interactive mode with all options"
    echo "  daint_create_conda_env --dalia-path=/path/to/dalia --serinv-path=/path/to/serinv --install-mpi4py --dev-mode"
    echo ""
    echo "  # Install base with GPU support only"
    echo "  daint_create_conda_env --dalia-path=/path/to/dalia"
    echo ""
    echo "Note: GPU support via cupy is installed by default for Daint cluster."
    echo "      If parameters are not provided, the function will prompt interactively."
}

daint_create_conda_env() {
    echo "daint_create_conda_env: creating DALIA conda environment for Daint."
    
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
                daint_create_conda_env_help
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
    if ! daint_check_modules; then
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
    dalia_path="${dalia_path/#~/$HOME}"
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
    local env_file="${dalia_path}/envs/dalia_base_aarch64.yml"
    if [[ ! -f "$env_file" ]]; then
        echo "   Error: Conda environment file not found at '${env_file}'."
        return 1
    fi
    
    echo "   Found DALIA repository at: ${dalia_path}"
    echo "   Found conda environment file at: ${env_file}"
    
    # Check if environment already exists
    local env_name="dalia_base_daint"
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
    if ! daint_activate_conda_env --env="$env_name"; then
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
        serinv_path="${serinv_path/#~/$HOME}"
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
    
    # 8. Install cupy with GPU support (default for Daint cluster)
    echo "   Installing cupy for GPU support..."

    if python -m pip install cupy-cuda12x --no-cache-dir; then
        # Check if CuPy is working
        if python -c "import cupy; A = cupy.random.rand(10,10)" &> /dev/null; then
            echo "   Successfully installed cupy for GPU support."
            echo "   CuPy configuration:"
            python -c "import cupy; cupy.show_config()"
        else
            echo "   Warning: Could not verify CuPy installation. Please test it manually."
        fi
    else
        echo "   Warning: Failed to install cupy. You may need to install it manually later."
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
        local mpi_enhanced_env_name="dalia_xccl_daint"  # Always GPU + MPI for Daint, NCCL is installed with CuPy.
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
            if daint_activate_conda_env --env="$mpi_enhanced_env_name"; then
                # Install mpi4py in the enhanced environment
                echo "   Installing mpi4py with OpenMPI support in enhanced environment..."
                cd "$dalia_path" || true
                if MPICC=$(which mpicc) python -m pip install --no-cache-dir --no-binary=mpi4py mpi4py; then
                    echo "   Successfully installed mpi4py in MPI-enhanced environment."

                    # Update the libcxx given Daint NCCL modules requirements
                    conda update libstdcxx-ng
                    conda install -c conda-forge libstdcxx-ng

                    # Try to import nccl to ensure it's available
                    echo "   Verifying NCCL installation in MPI-enhanced environment..."
                    if python -c "from cupy.cuda import nccl; nccl.get_unique_id()"; then
                        echo "   NCCL is available in MPI-enhanced environment."
                    else
                        echo "   Warning: NCCL does not seem to be available in MPI-enhanced environment."
                    fi
                    
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
                        echo "   Do you want to keep the base environment 'dalia_base_daint' for development without MPI? (y/N): "
                        read -r keep_base
                    fi
                    
                    if [[ ! "$keep_base" =~ ^[Yy]$ ]]; then
                        echo "   Removing base environment 'dalia_base_daint'..."
                        conda env remove -n "dalia_base_daint" -y || {
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
                    if daint_activate_conda_env --env="dalia_base_daint"; then
                        echo "   Reverted to base environment 'dalia_base_daint'."
                        env_name="dalia_base_daint"  # Reset env_name to base environment
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
    echo "   Base environment includes GPU support via cupy (default for Daint cluster)."
    if [[ "$install_mpi4py" =~ ^[Yy]$ ]]; then
        if [[ "$env_name" == *"xccl"* ]]; then
            echo "   Enhanced environment with mpi4py and NCCL support created."
        fi
    fi
    echo "   To use a specific environment in the future, run: daint_activate_conda_env --env=\"daint_env_name\""
    echo "   To use the most performant environment you have available, just run: daint_activate_conda_env"
    
    return 0
}

daint_activate_conda_env() {
    echo "daint_activate_conda_env: activating DALIA conda environment."
    
    # Parse command line arguments
    local specified_env=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --env=*)
                specified_env="${1#*=}"
                shift
                ;;
            --help|-h)
                echo "daint_activate_conda_env: Activate DALIA conda environment"
                echo ""
                echo "Usage:"
                echo "  daint_activate_conda_env [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --env=NAME              Specific environment name to activate"
                echo "  --help/-h               Show this help message"
                echo ""
                echo "Examples:"
                echo "  daint_activate_conda_env                        # Auto-select best available"
                echo "  daint_activate_conda_env --env=dalia_base_daint # Activate specific environment"
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
    local env_priorities=("dalia_xccl_daint" "dalia_base_daint")
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
        local available_envs=$(conda env list 2>/dev/null | grep -E "^(dalia_xccl_daint|dalia_base_daint) " | awk '{print $1}')
        
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
    local current_env=$(conda info --envs | grep '\*' |  grep 'dalia*' | awk '{print $1}')
    if [[ "$current_env" == "$env_name" ]]; then
        echo "   Successfully activated conda environment '${env_name}'"
    else
        echo "   Warning: Environment activation may not have been successful."
        echo "   Expected: ${env_name}, Current: ${current_env}"
    fi
    
    return 0
}

daint_set_perfenv() {
    echo "daint_set_perfenv: setting performance environment variables for Daint."

    set -e
    
    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
    export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
    export MPICH_GPU_SUPPORT_ENABLED=0

    # NCCL Performance Configuration
    # More can be found: https://docs.cscs.ch/software/communication/nccl/#using-nccl
    # This forces NCCL to use the libfabric plugin, enabling full use of the
    # Slingshot network. If the plugin can not be found, applications will fail to
    # start. With the default value, applications would instead fall back to e.g.
    # TCP, which would be significantly slower than with the plugin. More information
    # about `NCCL_NET` can be found at:
    # https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-net
    export NCCL_NET="AWS Libfabric"
    # Use GPU Direct RDMA when GPU and NIC are on the same NUMA node. More
    # information about `NCCL_NET_GDR_LEVEL` can be found at:
    # https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-net-gdr-level-formerly-nccl-ib-gdr-level
    export NCCL_NET_GDR_LEVEL=PHB
    export NCCL_CROSS_NIC=1
    # Starting with nccl 2.27 a new protocol (LL128) was enabled by default, which
    # typically performs worse on Slingshot. The following disables that protocol.
    export NCCL_PROTO=^LL128
    # These `FI` (libfabric) environment variables have been found to give the best
    # performance on the Alps network across a wide range of applications. Specific
    # applications may perform better with other values.
    export FI_CXI_DEFAULT_CQ_SIZE=131072
    export FI_CXI_DEFAULT_TX_SIZE=16384
    export FI_CXI_DISABLE_HOST_REGISTER=1
    export FI_CXI_RX_MATCH_MODE=software
    export FI_MR_CACHE_MONITOR=userfaultfd

    export FI_CXI_RDZV_GET_MIN=0
    export FI_CXI_RDZV_THRESHOLD=0
    export FI_CXI_RDZV_EAGER_SIZE=0
}