# Contributing to SpinDoctor

Thank you for your interest in contributing to SpinDoctor! This document provides
guidelines and instructions for contributing to the project.

## Code of Conduct

We expect all contributors to be respectful and considerate in all project
interactions, keeping the environment welcoming and inclusive for everyone.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:

   ```bash
   git clone https://github.com/your-username/rms-spindoctor.git
   cd rms-spindoctor
   ```

3. Create a virtual environment and install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. Before pushing changes, run the full check suite (lint, types, tests, docs):

   ```bash
   ./scripts/run-all-checks.sh
   ```

## Development Workflow

1. Create a new branch for your feature or bugfix:

   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b bugfix/issue-number
   ```

2. Make your changes, following our coding standards
3. Write or update tests as necessary
4. Run the tests to ensure they pass:

   ```bash
   pytest
   ```

5. Commit your changes with a descriptive message:

   ```bash
   git commit -m "Add feature: description of your changes"
   ```

6. Push your branch to your fork:

   ```bash
   git push origin feature/your-feature-name
   ```

7. Open a Pull Request on GitHub

## Coding Standards

We follow these standards for all code contributions:

* **Python Style**: Follow PEP 8
* **Type Hints**: Use type hints for all function parameters and return
  values
* **Docstrings**: Document all classes and methods with docstrings following
  the Google style
* **Testing**: Include unit tests for new functionality
* **Compatibility**: Ensure compatibility with Python 3.11+

Example of a well-formatted function:

```python
def calculate_offset(
    image: NDArrayFloatType, model: NDArrayFloatType
) -> tuple[float, float]:
    """Calculate the offset between an image and a model.

    Parameters:
        image: The observed image as a NumPy array
        model: The theoretical model as a NumPy array

    Returns:
        A tuple containing the (v, u) offset in pixels
    """
    # Implementation here
    return v_offset, u_offset
```

## Pull Request Process

1. Ensure all tests pass
2. Update documentation if necessary
3. Make sure your code is properly formatted and passes both ruff and mypy
4. Request a review from a maintainer
5. Address any feedback from reviewers

The maintainers will merge your PR once it meets all requirements.

## Testing

We use pytest for testing. To run the tests:

```bash
pytest
```

The default `addopts = ["-m", "not integration and not postgres"]` skips
integration tests (most need real PDS3 holdings + SPICE kernels and are
slow). The simulator integration tests, which also carry the
`sim_integration` marker, need no external data, and CI runs them on
every pull request.
To include them, override the marker filter:

```bash
pytest -m ""              # full suite, including integration
pytest -m integration     # only integration
pytest -m sim_integration # only the simulator integration tests
```

`scripts/run-all-checks.sh -i` (or `--integration`) does the same for
the all-checks runner.

For more verbose output:

```bash
pytest -v
```

To run a specific test file:

```bash
pytest tests/test_specific_file.py
```

## Documentation

We use Sphinx for documentation. To build the docs:

```bash
cd docs
make html
```

The generated documentation will be in `docs/_build/html`.

When adding new features, please update the relevant documentation:

* Update docstrings for new functions and classes
* Add examples if appropriate
* Update the user guide or developer guide if necessary

## Reporting Issues

If you find a bug or have a suggestion for improvement:

1. Check if the issue already exists in the GitHub issue tracker
2. If not, create a new issue with:
   * A clear, descriptive title
   * A detailed description of the issue
   * Steps to reproduce (for bugs)
   * Your environment information (Python version, OS, etc.)
   * Any relevant logs or screenshots

Thank you for contributing to SpinDoctor!
