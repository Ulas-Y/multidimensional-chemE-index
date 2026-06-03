import numpy as np
from numpy_fluid_core import XPBackend, MetricGeometry, ADMState, RelativisticFluidSimulation

# Create a tiny 1D uniform grid with trivial metric
n_cells = 8
psi = np.ones(n_cells)

# simple fields
lapse = np.ones(n_cells)
shift = np.zeros((n_cells, 1))
spatial_metric = MetricGeometry(1).metric_from_conformal(psi)
extrinsic_curv = np.zeros((n_cells, 1, 1))

mass = np.ones(n_cells)
velocity = np.zeros((n_cells, 1))
pressure = np.ones(n_cells) * 1e-3
rest_density = np.ones(n_cells)
internal_energy = np.ones(n_cells) * 1e-3

sim = RelativisticFluidSimulation(
    n_dim=1,
    lapse=lapse,
    shift=shift,
    spatial_metric=spatial_metric,
    extrinsic_curv=extrinsic_curv,
    mass=mass,
    velocity=velocity,
    pressure=pressure,
    rest_density=rest_density,
    internal_energy=internal_energy,
    integrator_type='euler'
)

for step in range(3):
    sim.step(1e-3)
    print(f"Step {step}: ham_res = {sim.check_constraints()['hamiltonian_residual'].mean()}")
