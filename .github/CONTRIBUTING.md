# Contributing to DALIA

Thank you for your interest in contributing to DALIA! This document provides guidelines and instructions for contributing to the project.

## Before You Start

When modifying the code, ensure that the following are still working:
- All existing tests pass
- Code follows the established style guidelines
- Documentation is updated as needed
- New features include appropriate tests

## General Coding Guidelines

We follow the NumPy/CuPy coding style guidelines, which are derived from the [PEP8](https://peps.python.org/pep-0008/) style guide.

### Development Environment Setup

1. **Install pre-commit hooks**: To ensure correct formatting, install and use `pre-commit`:
   ```bash
   pip install pre-commit
   pre-commit install
   ```

2. **Code formatting**: We use automated tools to maintain code quality. The pre-commit hooks will automatically check your code before each commit.

### Contribution workflow

The DALIA repository uses a dual-branch workflow with `main` and `dev` branches:

1. **Development**: New features are developed from the `dev` branch and merged via pull request back to it
2. **Release**: When a release is ready, changes are merged via pull request from the `dev` branch to the `main` branch

#### How to contribute:

1. **Fork** the DALIA repository
2. **Create a new branch** from the `dev` branch (not `main`)
3. **Develop** your feature or fix on your branch
4. **Create a pull request** to merge your changes into the `dev` branch of the DALIA repository

> **Note**: Always create your feature branches from `dev`, not `main`, to ensure your changes can be properly integrated.

### Guidelines for commit messages

Use descriptive commit messages with one of the following prefixes:

#### Core Development
- `STATS` : new feature or change related to statistical modeling
- `SLVR` : new feature or change related to solvers or numerical methods
- `API`: an (incompatible) API change
- `DEP`: deprecate something, or remove a deprecated object
- `ENH`: enhancement
- `BUG`: bug fix
- `CI`: continuous integration

#### Documentation and Testing
- `DOC`: documentation
- `TST`: addition or modification of tests
- `EXPL`: changes related to examples or tutorials
- `DEV`: development tool or utility

#### Code Quality and Maintenance
- `MAINT`: maintenance commit (refactoring, typos, etc.)
- `REV`: revert an earlier commit
- `STY`: style fix (whitespace, PEP8)
- `TYP`: static typing

#### Build and Release
- `BLD`: change related to building DALIA or its dependencies
- `REL`: related to releasing DALIA

#### Work in Progress
- `WIP`: work in progress, do not merge

**Example**: `ENH: Add new spatial kernel implementation`




## Testing Guidelines

DALIA testing relies on the [pytest](https://pytest.org/) framework.

Since DALIA is designed to be as performant as possible given the available hardware (HW) backends, the testing suite is designed to separate the testing of HW-agnostic code from the testing of HW-specific code.

Tests are located in the `tests/` directory. We support three levels of tests:

- **`unit/`**: These tests are designed to test HW-agnostic code and use pytest's `mock` feature to mock the backends
- **`component_integration/`**: These tests are designed to test HW-specific code and leverage specific backend features for validation
- **`integration/`**: These tests are designed to test the full DALIA stack and are full end-to-end tests that should be run on real hardware setups

For more detailed information, see the `tests/README.md` file.

## Documentation Guidelines

All functions and classes should be documented using the [NumPy docstring format](https://numpydoc.readthedocs.io/en/latest/format.html).

### Requirements:
- **Public functions**: Must have complete docstrings with parameters, returns, and examples
- **Classes**: Must document the class purpose, attributes, and key methods
- **Modules**: Should have module-level docstrings explaining their purpose

### Example:
```python
def example_function(param1: int, param2: str) -> bool:
    """
    Brief description of the function.

    Parameters
    ----------
    param1 : int
        Description of param1.
    param2 : str
        Description of param2.

    Returns
    -------
    bool
        Description of return value.
    """
    pass
```


## Examples and Tutorials Guidelines

Examples and tutorials should be placed in the `examples/` directory. They should be self-contained and demonstrate clear use cases of DALIA workflows.

### Requirements for each example:
- **README file**: Must explain the use case, expected output, and how to run the example,
- **Self-contained**: Should include all necessary data files or data generation scripts,
- **Clear documentation**: Code should be well-commented and easy to follow,
- **Tested**: Examples should be verified to work with the current DALIA version.

