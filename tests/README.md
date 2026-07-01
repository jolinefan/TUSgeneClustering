# Test Suite

Unit and integration tests for the TUS Gene clustering analysis pipeline.

## Running Tests

```bash
# All tests (39 total, ~40s)
python -m pytest tests/ -v

# Clustering tests only (19 tests)
python -m pytest tests/test_clustering.py -v

# Visualization tests only (20 tests)
python -m pytest tests/test_visualization.py -v

# Quick smoke test
python -m pytest tests/ -v -x  # Stop on first failure
```

## Validation Script

For numerical validation of pipeline outputs (not unit tests):

```bash
python scripts/validate_results.py
```

This runs 24 integration checks on actual pipeline outputs:
- Data dimensions and z-score properties
- PCA variance and reproducibility
- Cluster label validity and silhouette ranges
- Spot-checks of saved CSV values
- Config consistency

## Adding New Tests

1. Add test functions to existing test classes, or
2. Create new test classes for new functionality
3. Use fixtures for shared setup (see `@pytest.fixture`)
4. Mark slow tests with `@pytest.mark.slow`

## Dependencies

```bash
pip install pytest
```
