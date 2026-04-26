# Contributing to Attograd

Thank you for your interest in contributing to Attograd! This document provides guidelines and instructions for contributing to this project.

## Setting Up Development Environment

1. Fork the repository on GitHub.
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/attograd.git
   cd attograd
   ```

3. Install the package in development mode with all development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

4. Set up pre-commit hooks (optional):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Project Structure

```
attograd/
├── attograd/           # Main package
│   ├── __init__.py     # Package initialization
│   ├── tensor.py       # Core tensor implementation
│   ├── loss_functions.py # Loss functions
│   ├── cuda/           # CUDA acceleration
│   │   └── ...
│   ├── nn/             # Neural network layers
│   │   └── ...
│   └── viz/            # Visualization tools
│       └── ...
├── examples/           # Example usage
│   └── ...
├── tests/              # Unit tests
│   └── ...
├── docs/               # Documentation
│   └── ...
├── setup.py            # Package setup script
├── README.md           # Project readme
└── LICENSE             # License file
```

## Development Workflow

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes, following the code style guidelines below.

3. Write tests for your changes if applicable.

4. Run the tests to ensure they pass:
   ```bash
   pytest
   ```

5. Commit your changes:
   ```bash
   git commit -m "Description of your changes"
   ```

6. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

7. Create a pull request from your branch to the original repository.

## Code Style Guidelines

- Follow PEP 8 style guide for Python code.
- Use docstrings (Google style) for all functions, classes, and modules.
- Keep lines under 100 characters.
- Write clear, descriptive variable and function names.

## Testing

- Write unit tests for any new functionality.
- Ensure all tests pass before submitting a pull request.
- Aim for high test coverage in your code.

## Documentation

- Update documentation for any new features or changes to existing functionality.
- Include usage examples where appropriate.

## Reporting Issues

If you find a bug or have a suggestion for improvement:

1. Check if the issue already exists in the issue tracker.
2. If not, create a new issue, providing a clear description and, if possible, steps to reproduce.

## License

By contributing to Attograd, you agree that your contributions will be licensed under the project's MIT License.
