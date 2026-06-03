Relativistic Fluid Solver (prototype)

Quick start

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

2. Run tests:

```bash
pytest -q
```

3. Run simple example:

```bash
python examples/run_simple.py
```

What I changed

- Implemented batched `metric_inverse` and `sqrt_det_g` using backend-native linear algebra with small regularization.
- Added small unit tests in `tests/test_metric_ops.py` to validate conformal metrics and random SPD matrices.
- Added a minimal example `examples/run_simple.py` to exercise a tiny 1D simulation loop.

Next steps

- Add boundary-condition abstraction and a robust Poisson solver.
- Add verification benchmarks (Sod, relativistic shock tube).
- Vectorize inner loops and ensure full `xp` backend parity for PyTorch.
