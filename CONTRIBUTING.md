# Contributing to AirCompSim

Thank you for your interest in contributing to AirCompSim! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please be respectful and constructive in all interactions.

## Getting Started

### Development Environment

1. **Fork and clone** the repository:
   ```bash
   git clone https://github.com/yourusername/EnergyEfficientAirComputing.git
   cd EnergyEfficientAirComputing
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode**:
   ```bash
   pip install -e ".[dev]"
   ```

4. **Set up pre-commit hooks** (optional but recommended):
   ```bash
   pre-commit install
   ```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Changes

- Follow the existing code style
- Add type hints to all functions
- Include docstrings (Google style)
- Write tests for new functionality

### 3. Run Quality Checks

```bash
# Format code
black aircompsim/ tests/

# Run linter
ruff check aircompsim/ tests/

# Type checking
mypy aircompsim/

# Run tests
pytest tests/ -v
```

### 4. Commit Changes

Use clear, descriptive commit messages:

```bash
git commit -m "feat: add battery temperature modeling"
git commit -m "fix: correct energy calculation for hover mode"
git commit -m "docs: update README with new examples"
```

### 5. Submit Pull Request

- Push your branch and create a PR
- Provide a clear description of changes
- Reference any related issues

## Code Style

### Python Style Guide

- Follow PEP 8 conventions
- Use `black` for formatting (line length: 100)
- Use `ruff` for linting

### Type Hints

All functions must have type annotations:

```python
def compute_energy(distance: float, velocity: float) -> float:
    """Compute energy consumption."""
    return distance * velocity * 0.05
```

### Docstrings

Use Google-style docstrings:

```python
def select_server(
    self,
    task: Task,
    servers: List[Server],
) -> SchedulingDecision:
    """Select the best server for task processing.
    
    Args:
        task: Task to be scheduled.
        servers: Available servers.
        
    Returns:
        Scheduling decision with selected server.
        
    Raises:
        ValueError: If no servers available.
    """
```

## Testing

### Writing Tests

- Place tests in `tests/unit/` or `tests/integration/`
- Use descriptive test names
- Test edge cases

```python
def test_flight_energy_negative_distance_raises():
    """Test that negative distance raises ValueError."""
    model = EnergyModel()
    with pytest.raises(ValueError):
        model.compute_flight_energy(distance=-10, velocity=5)
```

### Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=aircompsim

# Specific file
pytest tests/unit/test_energy.py -v
```

## Project Structure

```
aircompsim/
├── core/           # Simulation engine
├── entities/       # Domain entities
├── energy/         # Energy modeling
├── drl/            # Deep RL algorithms
├── mobility/       # Movement models
├── visualization/  # Plotting utilities
└── config/         # Configuration management
```

## Pull Request Checklist

- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] New code has type hints
- [ ] New code has docstrings
- [ ] Tests added for new functionality
- [ ] Documentation updated if needed

## Questions?

Open an issue or reach out to the maintainers.
