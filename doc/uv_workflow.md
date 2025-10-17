# UV Workflow Guide

## What is uv?

`uv` is a fast Python package manager written in Rust. It handles dependency management and virtual environments automatically, making Python project setup much faster than traditional pip/venv workflows.

## Key Concepts

### Automatic Environment Management
Unlike traditional Python workflows where you manually create and activate virtual environments, `uv` automatically creates and manages a `.venv` directory for you. You don't need to activate it manually.

### Project-based Dependencies
Dependencies are defined in `pyproject.toml` instead of `requirements.txt`. This is the modern Python standard (PEP 621).

## Common Workflows

### Initial Setup
```bash
# Clone the project
git clone <repo-url>
cd greek_forge

# Install dependencies (creates .venv automatically)
uv sync

# Install with dev dependencies
uv sync --extra dev
```

### Running Commands
```bash
# uv run automatically uses the project's .venv
uv run python src/models/train.py
uv run pytest
uv run jupyter notebook

# Or activate the environment manually if preferred
source .venv/bin/activate  # On Unix/macOS
# Then run commands normally: python src/models/train.py
```

### Adding Dependencies
```bash
# Add a new package
uv add requests

# Add a dev dependency
uv add --dev black

# This automatically updates pyproject.toml and uv.lock
```

### Removing Dependencies
```bash
uv remove requests
```

### Updating Dependencies
```bash
# Update all packages
uv lock --upgrade

# Then sync the environment
uv sync
```

### Syncing After Pulling Changes
```bash
# After git pull, sync to install new dependencies
uv sync
```

## Benefits Over pip/venv

1. **Speed**: 10-100x faster than pip
2. **Automatic environment management**: No manual venv creation/activation needed
3. **Lockfile**: `uv.lock` ensures reproducible installs
4. **Modern standards**: Uses `pyproject.toml` (PEP 621)

## Files to Know

- `pyproject.toml`: Project metadata and dependencies
- `uv.lock`: Locked versions for reproducibility (commit this)
- `.venv/`: Virtual environment (automatically created, don't commit)

## Pro Tips

- Always use `uv run` to execute commands - it ensures you're using the project environment
- Commit `uv.lock` to version control for reproducible builds
- `.venv` is in `.gitignore` - never commit it
- `uv sync` is idempotent - safe to run multiple times
