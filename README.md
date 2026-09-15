# multidimensional-chemE-index

**Name vs content.** The repo is titled like a ChemE index. The code is a **prototype N-dimensional relativistic fluid / 3+1 ADM sketch**, not an index formula and not a finished Navier–Stokes solver.

## Authorship (locked)

This was an agent-forcing run: correct continuum / GR / fluid equations were fed in and the model was pushed toward an N-dimensional relativistic Navier–Stokes skeleton. That is a specification and scientific-management artifact. It is **not** a claim of hand-coded or thoroughly hand-debugged numerics. Treat the Python as generated under constraint, then spot-checked.

## What is actually in the tree

One module: `numpy_fluid_core.py` (~40 kB).

| Piece | Status |
|---|---|
| `XPBackend` | NumPy always; Torch if CUDA is up. CPU Torch falls back to NumPy. |
| `MetricGeometry` | Conformal `g_ij = ψ² δ_ij`, batched `inv` / `sqrt(det g)`, Christoffel, Ricci, Laplace–Beltrami |
| `ADMState` | lapse, shift, γ_ij, K_ij, mass, v, p, ρ₀, ε |
| `RelativisticStressEnergyTensor` | Perfect-fluid T (γ, enthalpy) |
| `ConstraintChecker` | Hamiltonian residual only |
| `TimeIntegrator` + Euler + RK4 | Metric, K, ρ₀, v, ε RHSs |
| `PressureProjector` | Jacobi Poisson; not wired as the main path |
| Tests | **Metric inverse / conformal det only** (`tests/test_metric_ops.py`) |
| Example | 8-cell 1-D, 3 Euler steps, prints mean Hamiltonian residual |

Default constants are SI (`c = 2.997…e8`, `G = 6.67e-11`). EOS is `P = (Γ-1) ρ₀ ε` with `Γ = 5/3`.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python examples/run_simple.py
```

## What this does not carry

- A ChemE index.
- A verified relativistic Navier–Stokes code. No Sod, no relativistic shock tube (listed as next steps and still next steps).
- Stable GR in 1-D: constraint source uses `1/(n_dim-1)`, which is undefined at `n_dim=1` — and that is the shipped example dimension.
- Metric-consistent special-relativistic helpers: `_velocity_magnitude_squared` is Euclidean `δ_ij`, not `g_ij`.
- Vectorized inner loops. Dimension loops are Python `for`.
- Repo hygiene: `__pycache__` is committed; no `.gitignore`.

The metric tests passing means conformal `det` / `inv` algebra is sane. It does not mean the ADM RHS or the fluid step is correct.

## Next (from the original README, still true)

- Boundary-condition object and a real Poisson solver.
- Sod / relativistic shock-tube verification.
- `xp` parity on CPU Torch, not only CUDA.
