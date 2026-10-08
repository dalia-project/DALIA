#!/bin/bash

# DALIA Test Runner Script
# Automatically detects available backends and runs appropriate tests
# Compatible with bash, zsh, tcsh, and csh shells

# =============================================================================
# Shell Detection and Compatibility Functions
# =============================================================================

detect_shell() {
    # Detect the current shell type
    if [ -n "$BASH_VERSION" ]; then
        echo "bash"
    elif [ -n "$ZSH_VERSION" ]; then
        echo "zsh"
    elif [ -n "$tcsh" ]; then
        echo "tcsh"
    elif [ -n "$version" ]; then
        echo "tcsh"  # tcsh sets $version variable
    else
        # Fallback: check $SHELL variable
        case "$SHELL" in
            *bash*) echo "bash" ;;
            *zsh*) echo "zsh" ;;
            *tcsh*) echo "tcsh" ;;
            *csh*) echo "csh" ;;
            *) echo "unknown" ;;
        esac
    fi
}

set_env_var() {
    # Set environment variable using appropriate shell syntax
    local var_name="$1"
    local var_value="$2"
    local shell_type=$(detect_shell)
    
    case "$shell_type" in
        bash|zsh|sh)
            export "$var_name=$var_value"
            ;;
        tcsh|csh)
            setenv "$var_name" "$var_value"
            ;;
        *)
            # Try both methods as fallback
            export "$var_name=$var_value" 2>/dev/null || setenv "$var_name" "$var_value" 2>/dev/null
            ;;
    esac
}

print_message() {
    # Print formatted message with consistent styling
    local message="$1"
    local type="$2"  # INFO, SUCCESS, WARNING, ERROR
    
    case "$type" in
        SUCCESS) echo "✓ $message" ;;
        WARNING) echo "⚠ $message" ;;
        ERROR) echo "✗ $message" ;;
        *) echo "• $message" ;;
    esac
}

# =============================================================================
# Argument Parsing Functions
# =============================================================================

show_help() {
    # Display help information
    echo "DALIA Test Runner Script"
    echo "========================"
    echo
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "OPTIONS:"
    echo "  --unit                Run only unit tests (unit/ directory)"
    echo "  --component-integration  Run only component integration tests (component_integration/ directory)"
    echo "  --cpu                 Run only CPU backend tests (NumPy)"
    echo "  --gpu                 Run only GPU backend tests (CuPy)"
    echo "  --mpi                 Run only MPI distributed tests"
    echo "  --yes                 Skip confirmation prompt (for automated testing)"
    echo "  --help, -h            Show this help message"
    echo
    echo "EXAMPLES:"
    echo "  $0                    Run all available tests"
    echo "  $0 --unit --cpu       Run unit tests on CPU backend only"
    echo "  $0 --mpi --gpu --yes  Run MPI tests on GPU backend without confirmation"
    echo "  $0 --component-integration  Run component integration tests on all available backends"
    echo
}

parse_arguments() {
    # Parse command-line arguments and set global flags
    RUN_UNIT=0
    RUN_COMPONENT_INTEGRATION=0
    RUN_CPU=0
    RUN_GPU=0
    RUN_MPI=0
    AUTO_CONFIRM=0
    RUN_ALL=1  # Default to running all tests
    BACKEND_SPECIFIED=0  # Track if any backend flag was specified
    
    while [ $# -gt 0 ]; do
        case "$1" in
            --unit)
                RUN_UNIT=1
                RUN_ALL=0
                ;;
            --component-integration)
                RUN_COMPONENT_INTEGRATION=1
                RUN_ALL=0
                ;;
            --cpu)
                RUN_CPU=1
                RUN_ALL=0
                BACKEND_SPECIFIED=1
                ;;
            --gpu)
                RUN_GPU=1
                RUN_ALL=0
                BACKEND_SPECIFIED=1
                ;;
            --mpi)
                RUN_MPI=1
                RUN_ALL=0
                BACKEND_SPECIFIED=1
                ;;
            --yes)
                AUTO_CONFIRM=1
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                echo "Use --help for usage information."
                exit 1
                ;;
        esac
        shift
    done
    
    # If no specific test type was selected, run all
    if [ $RUN_ALL -eq 1 ]; then
        RUN_UNIT=1
        RUN_COMPONENT_INTEGRATION=1
        RUN_CPU=1
        RUN_GPU=1
        RUN_MPI=1
    # If test directories were specified but no backends, run on all backends
    elif [ $BACKEND_SPECIFIED -eq 0 ] && [ $RUN_ALL -eq 0 ]; then
        RUN_CPU=1
        RUN_GPU=1
        RUN_MPI=1
    fi
}

# =============================================================================
# Backend Detection Functions
# =============================================================================

check_gpu_availability() {
    # Check if NVIDIA GPU is available via nvidia-smi
    if command -v nvidia-smi >/dev/null 2>&1; then
        if nvidia-smi >/dev/null 2>&1; then
            local gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
            print_message "GPU detected: $gpu_count NVIDIA GPU(s) available" "SUCCESS"
            return 0
        else
            print_message "nvidia-smi found but not working properly" "WARNING"
            return 1
        fi
    else
        print_message "nvidia-smi not found - no NVIDIA GPU detected" "INFO"
        return 1
    fi
}

check_cupy_installation() {
    # Check if CuPy is installed and working in the current environment
    print_message "Checking CuPy installation and functionality..." "INFO"
    
    # Try to import and test CuPy
    python -c "
import sys
try:
    import cupy as cp
    # Test basic CUDA operation
    test_array = cp.array([1, 2, 3])
    result = cp.sum(test_array)
    print('CuPy test successful: sum([1,2,3]) =', result.get())
    sys.exit(0)
except ImportError:
    print('CuPy not installed')
    sys.exit(1)
except Exception as e:
    print('CuPy installed but not working:', str(e))
    sys.exit(2)
" 2>/dev/null
    
    local cupy_status=$?
    case $cupy_status in
        0)
            print_message "CuPy is installed and working correctly" "SUCCESS"
            return 0
            ;;
        1)
            print_message "CuPy is not installed" "WARNING"
            return 1
            ;;
        2)
            print_message "CuPy is installed but not functioning (check CUDA drivers)" "WARNING"
            return 1
            ;;
        *)
            print_message "Unable to test CuPy installation" "ERROR"
            return 1
            ;;
    esac
}

check_mpi_installation() {
    # Check if MPI is available (srun/mpirun/mpiexec and mpi4py)
    print_message "Checking MPI installation and functionality..." "INFO"
    
    # Check for srun, mpirun or mpiexec command
    if ! command -v srun >/dev/null 2>&1 && ! command -v mpirun >/dev/null 2>&1 && ! command -v mpiexec >/dev/null 2>&1; then
        print_message "srun/mpirun/mpiexec not found - MPI not available" "WARNING"
        return 1
    fi
    
    # Determine which MPI launcher to use (priority: srun, mpirun, mpiexec)
    if command -v srun >/dev/null 2>&1; then
        MPI_LAUNCHER="srun"
    elif command -v mpirun >/dev/null 2>&1; then
        MPI_LAUNCHER="mpirun"
    elif command -v mpiexec >/dev/null 2>&1; then
        MPI_LAUNCHER="mpiexec"
    fi
    
    # Try to import and test mpi4py
    $MPI_LAUNCHER -n 1 python -c "
import sys
try:
    import mpi4py
    from mpi4py import MPI
    # Test basic MPI functionality
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print('mpi4py test successful: rank', rank, 'of', size)
    sys.exit(0)
except ImportError:
    print('mpi4py not installed')
    sys.exit(1)
except Exception as e:
    print('mpi4py installed but not working:', str(e))
    sys.exit(2)
" 2>/dev/null
    
    local mpi_status=$?
    case $mpi_status in
        0)
            print_message "MPI is installed and working correctly" "SUCCESS"
            return 0
            ;;
        1)
            print_message "mpi4py is not installed" "WARNING"
            return 1
            ;;
        2)
            print_message "mpi4py is installed but not functioning" "WARNING"
            return 1
            ;;
        *)
            print_message "Unable to test MPI installation" "ERROR"
            return 1
            ;;
    esac
}

determine_available_backends() {
    # Determine which backends are available for testing
    local backends=""
    local has_gpu=0
    local has_mpi=0
    
    print_message "Detecting available backends..." "INFO" >&2
    echo >&2
    
    # NumPy is always available
    print_message "NumPy backend: Always available" "SUCCESS" >&2
    backends="numpy"
    
    # Check for GPU + CuPy
    if check_gpu_availability >&2; then
        if check_cupy_installation >&2; then
            print_message "CuPy backend: Available" "SUCCESS" >&2
            backends="$backends cupy"
            has_gpu=1
        else
            print_message "CuPy backend: Unavailable (CuPy not working)" "WARNING" >&2
        fi
    else
        print_message "CuPy backend: Unavailable (No GPU detected)" "WARNING" >&2
    fi
    
    # Check for MPI
    if check_mpi_installation >&2; then
        print_message "MPI backend: Available" "SUCCESS" >&2
        backends="$backends mpi"
        has_mpi=1
        
        # Check MPI + GPU combination
        if [ $has_gpu -eq 1 ]; then
            print_message "MPI + GPU backend: Available" "SUCCESS" >&2
            backends="$backends mpi-gpu"
        else
            print_message "MPI + GPU backend: Unavailable (No GPU)" "WARNING" >&2
        fi
    else
        print_message "MPI backend: Unavailable (MPI not working)" "WARNING" >&2
    fi
    
    echo "$backends"
}

# =============================================================================
# Test Execution Functions
# =============================================================================

choose_tests_to_run() {
    # Display which tests will be run based on available backends
    local backends="$1"
    local mpi_available=0
    
    echo "Test execution plan:"
    
    # Show test directories
    local test_dirs=$(get_test_directories)
    if [ -n "$test_dirs" ]; then
        echo "  Test directories: $test_dirs"
    else
        echo "  Test directories: all"
    fi
    
    # Check if MPI is actually available
    if echo "$backends" | grep -q "mpi"; then
        mpi_available=1
    fi
    
    # Show backends
    if [ $RUN_CPU -eq 1 ] && echo "$backends" | grep -q "numpy"; then
        if [ $RUN_MPI -eq 1 ] && [ $mpi_available -eq 1 ]; then
            echo "  - CPU backend (NumPy) - Serial and MPI tests (2 processes)"
        else
            echo "  - CPU backend (NumPy) - Serial tests"
        fi
    fi
    
    if [ $RUN_GPU -eq 1 ] && echo "$backends" | grep -q "cupy"; then
        if [ $RUN_MPI -eq 1 ] && [ $mpi_available -eq 1 ]; then
            echo "  - GPU backend (CuPy) - Serial and MPI tests (2 processes)"
        else
            echo "  - GPU backend (CuPy) - Serial tests"
        fi
    fi
    
    if [ $RUN_MPI -eq 1 ] && [ $RUN_CPU -eq 0 ] && [ $RUN_GPU -eq 0 ]; then
        if [ $mpi_available -eq 1 ]; then
            echo "  - CPU backend (NumPy) - MPI tests only (2 processes)"
            if echo "$backends" | grep -q "cupy"; then
                echo "  - GPU backend (CuPy) - MPI tests only (2 processes)"
            fi
        else
            echo "  - MPI tests requested but MPI is not available"
        fi
    fi
}

get_test_directories() {
    # Determine which test directories to run based on command-line arguments
    local test_dirs=""
    
    if [ $RUN_UNIT -eq 1 ] && [ $RUN_COMPONENT_INTEGRATION -eq 1 ]; then
        test_dirs="unit/ component_integration/"
    elif [ $RUN_UNIT -eq 1 ]; then
        test_dirs="unit/"
    elif [ $RUN_COMPONENT_INTEGRATION -eq 1 ]; then
        test_dirs="component_integration/"
    fi
    
    echo "$test_dirs"
}

run_numpy_tests() {
    # Run tests with NumPy backend
    local test_dirs="$1"  # Optional: specific test directories to run
    
    echo "Running testing suite on CPU backend (NumPy)..."
    echo "==============================================="
    
    set_env_var "ARRAY_MODULE" "numpy"
    
    # Determine which MPI launcher to use (priority: srun, mpirun, mpiexec)
    if command -v srun >/dev/null 2>&1; then
        MPI_LAUNCHER="srun -n 1"
    elif command -v mpirun >/dev/null 2>&1; then
        MPI_LAUNCHER="mpirun -n 1"
    elif command -v mpiexec >/dev/null 2>&1; then
        MPI_LAUNCHER="mpiexec -n 1"
    else # No MPI available
        MPI_LAUNCHER=""
    fi

    # Run pytest
    if command -v pytest >/dev/null 2>&1; then
        if [ -n "$test_dirs" ]; then
            $MPI_LAUNCHER pytest $test_dirs -v
        else
            $MPI_LAUNCHER pytest . -v
        fi
        local exit_code=$?
        return $exit_code
    else
        print_message "pytest not found - cannot run tests" "ERROR"
        return 1
    fi
}

run_cupy_tests() {
    # Run tests with CuPy backend
    local test_dirs="$1"  # Optional: specific test directories to run
    
    echo "Running testing suite on GPU backend (CuPy)..."
    echo "==============================================="
    
    set_env_var "ARRAY_MODULE" "cupy"
    
    # Determine which MPI launcher to use (priority: srun, mpirun, mpiexec)
    if command -v srun >/dev/null 2>&1; then
        MPI_LAUNCHER="srun -n 1"
    elif command -v mpirun >/dev/null 2>&1; then
        MPI_LAUNCHER="mpirun -n 1"
    elif command -v mpiexec >/dev/null 2>&1; then
        MPI_LAUNCHER="mpiexec -n 1"
    else # No MPI available
        MPI_LAUNCHER=""
    fi

    # Run pytest
    if command -v pytest >/dev/null 2>&1; then
        if [ -n "$test_dirs" ]; then
            $MPI_LAUNCHER pytest $test_dirs -v
        else
            $MPI_LAUNCHER pytest . -v
        fi
        local exit_code=$?
        return $exit_code
    else
        print_message "pytest not found - cannot run tests" "ERROR"
        return 1
    fi
}

run_mpi_numpy_tests() {
    # Run MPI tests with NumPy backend
    local test_dirs="$1"  # Optional: specific test directories to run
    
    echo "Running MPI testing suite on CPU backend (NumPy)..."
    echo "===================================================="
    
    set_env_var "ARRAY_MODULE" "numpy"
    
    # Determine which MPI launcher to use (priority: srun, mpirun, mpiexec)
    if command -v srun >/dev/null 2>&1; then
        MPI_LAUNCHER="srun"
    elif command -v mpirun >/dev/null 2>&1; then
        MPI_LAUNCHER="mpirun"
    elif command -v mpiexec >/dev/null 2>&1; then
        MPI_LAUNCHER="mpiexec"
    fi

    # Run with 2 processes using the MPI launcher determined during backend detection
    echo "Running MPI tests with 2 processes using $MPI_LAUNCHER..."
    if command -v pytest >/dev/null 2>&1; then
        if [ -n "$test_dirs" ]; then
            $MPI_LAUNCHER -n 2 pytest --with-mpi $test_dirs -v
        else
            $MPI_LAUNCHER -n 2 pytest --with-mpi . -v
        fi
        if [ $? -ne 0 ]; then
            print_message "MPI tests with 2 processes failed" "ERROR"
            return 1
        else
            print_message "MPI tests with 2 processes completed successfully" "SUCCESS"
            return 0
        fi
    else
        print_message "pytest not found - cannot run MPI tests" "ERROR"
        return 1
    fi
}

run_mpi_cupy_tests() {
    # Run MPI tests with CuPy backend
    local test_dirs="$1"  # Optional: specific test directories to run
    
    echo "Running MPI testing suite on GPU backend (CuPy)..."
    echo "==================================================="
    
    set_env_var "ARRAY_MODULE" "cupy"
    
    # Determine which MPI launcher to use (priority: srun, mpirun, mpiexec)
    if command -v srun >/dev/null 2>&1; then
        MPI_LAUNCHER="srun"
    elif command -v mpirun >/dev/null 2>&1; then
        MPI_LAUNCHER="mpirun"
    elif command -v mpiexec >/dev/null 2>&1; then
        MPI_LAUNCHER="mpiexec"
    fi

    # Run with 2 processes using the MPI launcher determined during backend detection
    echo "Running MPI + GPU tests with 2 processes using $MPI_LAUNCHER..."
    if command -v pytest >/dev/null 2>&1; then
        if [ -n "$test_dirs" ]; then
            $MPI_LAUNCHER -n 2 pytest --with-mpi $test_dirs -v
        else
            $MPI_LAUNCHER -n 2 pytest --with-mpi . -v
        fi
        if [ $? -ne 0 ]; then
            print_message "MPI + GPU tests with 2 processes failed" "ERROR"
            return 1
        else
            print_message "MPI + GPU tests with 2 processes completed successfully" "SUCCESS"
            return 0
        fi
    else
        print_message "pytest not found - cannot run MPI tests" "ERROR"
        return 1
    fi
}

# =============================================================================
# Main Execution Logic
# =============================================================================

main() {
    # Main function that orchestrates the test execution
    local shell_type=$(detect_shell)
    
    # Parse command-line arguments
    parse_arguments "$@"
    
    echo "=============================================="
    echo "             DALIA Tests Runner"
    echo "=============================================="
    echo "Detected shell: $shell_type"
    echo "Current directory: $(pwd)"
    echo
    
    # Detect available backends
    local available_backends=$(determine_available_backends)
    
    # Show test plan
    choose_tests_to_run "$available_backends"
    
    # Ask for confirmation unless --yes was specified
    if [ $AUTO_CONFIRM -eq 0 ]; then
        echo -n "Proceed with test execution? [y/N]: "
        read confirmation
        case "$confirmation" in
            [yY]|[yY][eE][sS])
                echo "Starting tests..."
                ;;
            *)
                echo "Test execution cancelled by user."
                exit 0
                ;;
        esac
    else
        echo "Auto-confirmation enabled. Starting tests..."
    fi
    
    echo
    local overall_success=0
    local test_dirs=$(get_test_directories)
    
    # Run serial tests if requested
    if [ $RUN_CPU -eq 1 ] || [ $RUN_GPU -eq 1 ]; then
        echo "===== SERIAL TESTS ====="
        
        # Run NumPy tests if CPU backend is requested
        if [ $RUN_CPU -eq 1 ] && echo "$available_backends" | grep -q "numpy"; then
            if ! run_numpy_tests "$test_dirs"; then
                overall_success=1
            fi
            echo
        fi
        
        # Run CuPy tests if GPU backend is requested and available
        if [ $RUN_GPU -eq 1 ] && echo "$available_backends" | grep -q "cupy"; then
            if ! run_cupy_tests "$test_dirs"; then
                overall_success=1
            fi
            echo
        elif [ $RUN_GPU -eq 1 ]; then
            print_message "GPU backend requested but not available" "WARNING"
        fi
    fi
    
    # Run MPI tests if requested and available
    if [ $RUN_MPI -eq 1 ]; then
        if echo "$available_backends" | grep -q "mpi"; then
            echo "===== MPI TESTS ====="
            
            # Run MPI + NumPy tests (always run MPI tests on CPU if MPI is available)
            if ! run_mpi_numpy_tests "$test_dirs"; then
                overall_success=1
            fi
            echo
            
            # Run MPI + CuPy tests if GPU is also available and requested
            if [ $RUN_GPU -eq 1 ] && echo "$available_backends" | grep -q "mpi-gpu"; then
                if ! run_mpi_cupy_tests "$test_dirs"; then
                    overall_success=1
                fi
                echo
            fi
        else
            print_message "MPI backend requested but not available - skipping MPI tests" "WARNING"
        fi
    fi
    
    # Final summary
    echo "=============================================="
    if [ $overall_success -eq 0 ]; then
        echo "     All tests completed successfully!"
    else
        echo "  Some tests failed - check output above"
    fi
    echo "=============================================="
    
    exit $overall_success
}

# =============================================================================
# Script Entry Point
# =============================================================================

# Check if script is being sourced or executed
if [ "${BASH_SOURCE[0]}" = "${0}" ] 2>/dev/null || [ "${(%):-%x}" = "${0}" ] 2>/dev/null; then
    # Script is being executed directly
    main "$@"
fi

