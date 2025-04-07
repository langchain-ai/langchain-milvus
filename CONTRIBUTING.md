# Contributing to LangChain Milvus

Thank you for your interest in contributing to the LangChain Milvus integration! This document provides guidelines and instructions for contributing to this project.

## Development Environment Setup

This project uses Poetry for dependency management. Make sure you have Poetry installed. See the [Poetry installation documentation](https://python-poetry.org/docs/#installation) for installation instructions.

Then install the dependencies:

```bash
cd libs/milvus
poetry install
```

## Pull Request Process

1. Fork the repository and create your branch from `main`.
2. Make your changes.
3. Run tests and linting to ensure your code meets the project's standards.
4. Update documentation if necessary.
5. Submit a pull request.


## Development Workflow

### Running Tests

We use pytest for testing. To run tests:

```bash
# Run all unit tests
make test

# Run specific test file
make test TEST_FILE=tests/unit_tests/specific_test.py

# Run specific test method
make test TEST_FILE=tests/unit_tests/specific_test.py::test_method_name

# Run integration tests
make integration_tests

# Run tests in watch mode (automatically rerun on file changes)
make test_watch
```

### Code Quality

We maintain code quality through linting and formatting:

#### Linting

```bash
# Lint all Python files
make lint

# Lint only changed files compared to master
make lint_diff

# Lint only the package code
make lint_package

# Lint only the tests
make lint_tests
```

#### Formatting

```bash
# Format all Python files
make format

# Format only changed files compared to master
make format_diff
```

#### Spell Checking

```bash
# Check for spelling errors
make spell_check

# Fix spelling errors
make spell_fix
```

#### Import Checking

```bash
# Check imports in the package
make check_imports
```


### Managing Dependencies with Poetry

If you want to modify project dependencies, you can refer to the [Poetry document](https://python-poetry.org/docs/managing-dependencies/). Here are some common commands to manage dependencies:

#### Adding Dependencies

```bash
# Add a regular dependency
poetry add package-name

# Add a development dependency
poetry add --group dev package-name

# Add a dependency with specific version
poetry add package-name==1.2.3

# Add a dependency with version constraints
poetry add "package-name>=1.0.0,<2.0.0"
```

#### Updating Dependencies

To update the project's dependencies:

```bash
# Update all dependencies
poetry update

# Update a specific dependency
poetry update package-name
```

#### Locking Dependencies

After adding or updating dependencies, Poetry automatically updates the `poetry.lock` file. If you need to manually generate or update the lock file:

```bash
# Generate/update the lock file without installing dependencies
poetry lock

# Update the lock file and install dependencies
poetry lock --no-update
```


## Code of Conduct

Please follow the [LangChain code of conduct](https://python.langchain.com/docs/contributing/) when participating in this project.

## Questions?

If you have any questions, feel free to open an issue or reach out to the maintainers.

Thank you for your contributions!