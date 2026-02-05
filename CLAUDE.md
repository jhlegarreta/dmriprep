# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

dMRIPrep is a robust, analysis-agnostic preprocessing pipeline for diffusion MRI (dMRI) data. It's part of the NiPreps (Neuroimaging Preprocessing) ecosystem and automatically adapts best-practice workflows to diverse dMRI datasets.

- **Documentation**: https://www.nipreps.org/dmriprep
- **Docker Hub**: https://hub.docker.com/r/nipreps/dmriprep
- **Python**: 3.10, 3.11, 3.12, 3.13

## Common Commands

### Testing
```bash
# Run all tests with coverage
pytest dmriprep -svx

# Run tests in parallel
pytest dmriprep -n auto

# Run specific test file
pytest dmriprep/cli/tests/test_parser.py -v

# Run tests via tox (all Python versions)
tox

# Specific Python version
tox -e py312
```

### Code Quality
```bash
# Check style
tox -e style

# Auto-fix style issues
tox -e style-fix

# Spellcheck
tox -e spellcheck

# Or directly with ruff
ruff check dmriprep
ruff format dmriprep
```

### Building
```bash
# Install in development mode
pip install -e ".[dev,test]"

# Or with pixi (preferred for full environment)
pixi install -e editable

# Build distributions
python -m build
```

### Documentation
```bash
# Build docs with pixi
pixi run -e docs build-docs

# Live preview
pixi run -e docs docs  # serves at localhost:8000
```

## Architecture

### Core Components

- **`dmriprep/cli/`** - Command-line interface
  - `run.py` - Main entry point (`dmriprep` command)
  - `parser.py` - Argument parsing
  - `workflow.py` - CLI workflow orchestration

- **`dmriprep/config/`** - Runtime configuration system (ToML-based, cross-process communication via filesystem)

- **`dmriprep/workflows/`** - Nipype workflow definitions
  - `base.py` - Main dMRIPrep workflow
  - `dwi/base.py` - DWI preprocessing pipeline
  - `dwi/eddy.py` - Eddy current correction

- **`dmriprep/interfaces/`** - Custom Nipype interfaces
  - `bids.py` - BIDS-specific interfaces
  - `images.py` - Image processing
  - `vectors.py` - Gradient/vector handling
  - `reports.py` - Report generation

- **`dmriprep/utils/`** - Utility functions

### Key Dependencies

- **Nipype** - Workflow engine
- **NiWorkflows/SMRIPrep/SDCFlows** - Shared NiPreps components
- **DIPY** - Diffusion imaging processing
- **PyBIDS** - BIDS dataset handling

## Code Style

- Line length: 99 characters
- Quotes: Single quotes
- Linter: Ruff (replaces flake8/black/isort)
- Docstrings: NumPy style

## Testing Notes

- Tests are located within the `dmriprep/` package (not a separate `tests/` directory)
- Some tests require DIPY test data, cached in `~/.cache/data/` or `$DMRIPREP_TESTS_DATA`
- TemplateFlow templates cached in `~/.cache/templateflow/`
- `conftest.py` provides `dipy_test_data` fixture for Sherbrooke HARDI dataset

## Configuration System

The `dmriprep/config/` module provides singleton configuration management:
- ToML-based storage
- Cross-process communication via filesystem
- Sections: `environment`, `execution`, `workflow`, `nipype`, `loggers`
