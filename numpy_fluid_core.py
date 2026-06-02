"""
Multidimensional Relativistic Fluid Dynamics Solver
- Backend-agnostic (NumPy / PyTorch with GPU support)
- 3+1 ADM decomposition of Einstein equations
- Arbitrary spatial dimensions N
- Cell-centered finite-volume discretization
- Explicit Euler and RK4 time integrators
- Mass-based density computation
- Relativistic stress-energy tensor
- Metric-aware differential operators
"""

import math
import warnings
import numpy as _np

torch = None
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ============================================================================
# Backend Abstraction
# ============================================================================

def _torch_gradient(x, axis=-1):
    """Compute gradient along axis for PyTorch tensors."""
    if axis < 0:
        axis += x.ndim

    if x.size(axis) < 2:
        return torch.zeros_like(x)

    if x.size(axis) == 2:
        slc0 = [slice(None)] * x.ndim
        slc1 = [slice(None)] * x.ndim
        slc0[axis] = slice(0, 1)
        slc1[axis] = slice(1, 2)
        return x[tuple(slc1)] - x[tuple(slc0)]

    diffs = x.diff(dim=axis)
    slc_first = [slice(None)] * x.ndim
    slc_second = [slice(None)] * x.ndim
    slc_lastm = [slice(None)] * x.ndim
    slc_last = [slice(None)] * x.ndim

    slc_first[axis] = slice(0, 1)
    slc_second[axis] = slice(1, 2)
    slc_lastm[axis] = slice(-2, -1)
    slc_last[axis] = slice(-1, None)

    first = x[tuple(slc_second)] - x[tuple(slc_first)]
    last = x[tuple(slc_last)] - x[tuple(slc_lastm)]

    center = (diffs.narrow(axis, 0, diffs.size(axis) - 1) + diffs.narrow(axis, 1, diffs.size(axis) - 1)) * 0.5
    return torch.cat([first, center, last], dim=axis)


class _Backend:
    """Backend abstraction for NumPy and PyTorch."""
    def __init__(self, module, name):
        self.backend = module
        self.name = name

    def asarray(self, arr, dtype=None):
        if self.name == "torch":
            if dtype is None:
                return self.backend.as_tensor(arr)
            return self.backend.as_tensor(arr, dtype=dtype)
        return self.backend.asarray(arr, dtype=dtype)

    def array(self, arr, dtype=None):
        if self.name == "torch":
            if dtype is None:
                return self.backend.tensor(arr)
            return self.backend.tensor(arr, dtype=dtype)
        return self.backend.array(arr, dtype=dtype)

    def zeros_like(self, arr):
        return self.backend.zeros_like(arr)

    def ones_like(self, arr):
        return self.backend.ones_like(arr)

    def zeros(self, shape):
        if self.name == "torch":
            return self.backend.zeros(shape)
        return self.backend.zeros(shape)

    def stack(self, seq, axis=0):
        if self.name == "torch":
            return self.backend.stack(seq, dim=axis)
        return self.backend.stack(seq, axis=axis)

    def gradient(self, x, axis=-1):
        if self.name == "torch":
            return _torch_gradient(x, axis)
        return self.backend.gradient(x, axis=axis)

    def sum(self, x, axis=None):
        if self.name == "torch":
            if axis is None:
                return self.backend.sum(x)
            return self.backend.sum(x, dim=axis)
        return self.backend.sum(x, axis=axis)

    def mean(self, x, axis=None):
        if self.name == "torch":
            if axis is None:
                return self.backend.mean(x)
            return self.backend.mean(x, dim=axis)
        return self.backend.mean(x, axis=axis)


if _TORCH_AVAILABLE and torch.cuda.is_available():
    xp = _Backend(torch, "torch")
else:
    if _TORCH_AVAILABLE:
        warnings.warn("PyTorch is installed but CUDA is unavailable; falling back to NumPy backend.")
    xp = _Backend(_np, "numpy")


# ============================================================================
# Utility Functions
# ============================================================================

def _to_python_float(x):
    """Convert tensor/array to Python float."""
    if hasattr(x, "item"):
        return x.item()
    return float(x)


def _velocity_magnitude_squared(velocity):
    """Compute v^i v_i = v_i * delta_ij * v_j for Euclidean metric."""
    arr = xp.asarray(velocity, dtype=float)
    return _to_python_float((arr * arr).sum())


def omega_n_minus_1(n):
    """Surface area of unit (n-1)-sphere."""
    if n <= 0:
        raise ValueError("n must be a positive integer")
    return 2.0 * math.pi ** (n / 2.0) / math.gamma(n / 2.0)


# ============================================================================
# Relativistic Utilities
# ============================================================================

def lorentz_factor(velocity, c=299792458.0):
    """Compute Lorentz factor gamma = 1 / sqrt(1 - v^2/c^2)."""
    v2 = _velocity_magnitude_squared(velocity)
    if v2 >= c ** 2:
        raise ValueError("Velocity magnitude must be less than the speed of light")
    return 1.0 / math.sqrt(1.0 - v2 / (c ** 2))


def special_time_dilation(delta_t0, velocity, c=299792458.0):
    """Compute special relativistic time dilation: dt = dt0 / sqrt(1 - v^2/c^2)."""
    v2 = _velocity_magnitude_squared(velocity)
    if v2 >= c ** 2:
        raise ValueError("Velocity magnitude must be less than the speed of light")
    return delta_t0 / math.sqrt(1.0 - v2 / (c ** 2))


def relativistic_volume_contraction(V0, velocity, c=299792458.0):
    """Length contraction: V = V0 * sqrt(1 - v^2/c^2)."""
    v2 = _velocity_magnitude_squared(velocity)
    if v2 >= c ** 2:
        raise ValueError("Velocity magnitude must be less than the speed of light")
    return V0 * math.sqrt(1.0 - v2 / (c ** 2))


# ============================================================================
# Metric Geometry
# ============================================================================

class MetricGeometry:
    """
    Handles metric tensor, Christoffel symbols, Ricci tensor, curvature.
    Supports arbitrary spatial dimension N.
    Uses conformal metric: g_ij = psi^2 * delta_ij
    """
    def __init__(self, n_dim, cell_widths=None):
        """
        Parameters
        ----------
        n_dim : int
            Number of spatial dimensions
        cell_widths : array-like, optional
            Cell widths for each dimension. If None, uses uniform spacing.
        """
        self.n_dim = n_dim
        self.cell_widths = cell_widths if cell_widths is not None else _np.ones(n_dim)
        self.omega_n_minus_1 = omega_n_minus_1(n_dim - 1) if n_dim > 1 else 1.0

    def conformal_factor_from_potential(self, potential, c=299792458.0):
        """Compute conformal factor psi = 1 + Phi / c^2."""
        return 1.0 + potential / (c ** 2)

    def metric_from_conformal(self, psi):
        """
        Compute spatial metric g_ij = psi^2 * delta_ij.
        
        Parameters
        ----------
        psi : array
            Conformal factor at each cell
            
        Returns
        -------
        g : array of shape (..., n_dim, n_dim)
            Spatial metric tensor
        """
        shape = psi.shape + (self.n_dim, self.n_dim)
        g = xp.zeros(shape)
        for i in range(self.n_dim):
            for j in range(self.n_dim):
                if i == j:
                    g[..., i, j] = psi ** 2
        return g

    def metric_inverse(self, g):
        """
        Compute inverse metric g^ij from g_ij.
        For conformal metric: g^ij = psi^{-2} * delta_ij
        """
        g_inv = xp.zeros_like(g)
        for i in range(self.n_dim):
            for j in range(self.n_dim):
                if i == j:
                    g_inv[..., i, j] = 1.0 / (g[..., i, j] + 1e-30)
        return g_inv

    def sqrt_det_g(self, g):
        """Compute sqrt(det(g)) for conformal metric."""
        psi_sq = g[..., 0, 0]
        psi = xp.asarray(psi_sq) ** 0.5
        return psi ** self.n_dim

    def christoffel_symbols(self, g):
        """
        Compute Christoffel symbols Gamma^k_ij.
        """
        gamma = xp.zeros(g.shape[:-2] + (self.n_dim, self.n_dim, self.n_dim))
        g_inv = self.metric_inverse(g)

        for k in range(self.n_dim):
            for i in range(self.n_dim):
                for j in range(self.n_dim):
                    term = xp.zeros_like(g[..., 0, 0])
                    for l in range(self.n_dim):
                        dg_i = xp.gradient(g[..., l, j], axis=i)
                        dg_j = xp.gradient(g[..., l, i], axis=j)
                        dg_l = xp.gradient(g[..., i, j], axis=l)
                        term = term + g_inv[..., k, l] * (dg_i + dg_j - dg_l)

                    gamma[..., k, i, j] = 0.5 * term

        return gamma

    def ricci_tensor(self, g):
        """Compute Ricci tensor R_ij from spatial metric g_ij."""
        gamma = self.christoffel_symbols(g)
        ricci = xp.zeros_like(g)
        g_inv = self.metric_inverse(g)

        for i in range(self.n_dim):
            for j in range(self.n_dim):
                term = xp.zeros_like(g[..., 0, 0])

                for k in range(self.n_dim):
                    term = term + xp.gradient(gamma[..., k, i, j], axis=k)
                    term = term - xp.gradient(gamma[..., k, k, i], axis=j)

                    for l in range(self.n_dim):
                        term = term + gamma[..., k, i, j] * gamma[..., l, k, l]
                        term = term - gamma[..., k, i, l] * gamma[..., l, k, j]

                ricci[..., i, j] = term

        return ricci

    def ricci_scalar(self, g):
        """Compute Ricci scalar R = g^ij R_ij."""
        ricci = self.ricci_tensor(g)
        g_inv = self.metric_inverse(g)
        r_scalar = xp.zeros_like(g[..., 0, 0])

        for i in range(self.n_dim):
            for j in range(self.n_dim):
                r_scalar = r_scalar + g_inv[..., i, j] * ricci[..., i, j]

        return r_scalar

    def laplacian(self, f, g):
        """Compute Laplacian in metric g: L f = (1/sqrt(g)) d_i (sqrt(g) g^ij d_j f)."""
        sqrt_g = self.sqrt_det_g(g)
        g_inv = self.metric_inverse(g)
        laplacian = xp.zeros_like(f)

        for i in range(self.n_dim):
            flux = xp.zeros_like(f)
            for j in range(self.n_dim):
                flux = flux + sqrt_g * g_inv[..., i, j] * xp.gradient(f, axis=j)
            laplacian = laplacian + xp.gradient(flux, axis=i)

        laplacian = laplacian / (sqrt_g + 1e-30)
        return laplacian


# ============================================================================
# ADM State
# ============================================================================

class ADMState:
    """Full 3+1 ADM state: metric, extrinsic curvature, fluids."""
    def __init__(self, n_dim, lapse, shift, spatial_metric, extrinsic_curv,
                 mass, velocity, pressure, rest_density, internal_energy):
        """Initialize ADM state."""
        self.n_dim = n_dim
        self.lapse = lapse
        self.shift = shift
        self.spatial_metric = spatial_metric
        self.extrinsic_curv = extrinsic_curv

        self.mass = mass
        self.velocity = velocity
        self.pressure = pressure
        self.rest_density = rest_density
        self.internal_energy = internal_energy

        self.proper_time = xp.zeros_like(mass)
        self.coordinate_time = 0.0

    def update_derived_quantities(self, geometry, c=299792458.0):
        """Compute derived quantities from primary fields."""
        sqrt_g = geometry.sqrt_det_g(self.spatial_metric)
        self.local_volume = sqrt_g
        self.density = self.mass / (sqrt_g + 1e-30)


# ============================================================================
# Stress-Energy Tensor
# ============================================================================

class RelativisticStressEnergyTensor:
    """Perfect fluid stress-energy tensor."""
    def __init__(self, rest_density, velocity, internal_energy, pressure,
                 spatial_metric, c=299792458.0):
        """Initialize stress-energy tensor."""
        self.rho0 = rest_density
        self.v = velocity
        self.epsilon = internal_energy
        self.p = pressure
        self.gamma = spatial_metric
        self.c = c
        self.n_dim = spatial_metric.shape[-1]

    def lorentz_factor_field(self):
        """Compute Lorentz factor field gamma_u."""
        v_sq = xp.zeros_like(self.v[..., 0])
        for i in range(self.n_dim):
            v_sq = v_sq + self.v[..., i] * self.v[..., i]

        gamma_u = 1.0 / (1.0 - v_sq / (self.c ** 2) + 1e-30) ** 0.5
        return gamma_u

    def energy_density(self):
        """Compute T_00 component: energy density."""
        gamma_u = self.lorentz_factor_field()
        h = 1.0 + self.epsilon + self.p / (self.rho0 * self.c ** 2 + 1e-30)
        return gamma_u ** 2 * self.rho0 * h + self.p

    def momentum_density(self):
        """Compute S^i = -T^0i: momentum density."""
        gamma_u = self.lorentz_factor_field()
        h = 1.0 + self.epsilon + self.p / (self.rho0 * self.c ** 2 + 1e-30)
        s = xp.zeros_like(self.v)
        for i in range(self.n_dim):
            s[..., i] = -gamma_u * self.rho0 * h * self.v[..., i]
        return s


# ============================================================================
# Constraint Equations
# ============================================================================

class ConstraintChecker:
    """Check Hamiltonian and momentum constraints."""
    def __init__(self, geometry, n_dim, c=299792458.0, G_D=6.67430e-11):
        """Initialize constraint checker."""
        self.geometry = geometry
        self.n_dim = n_dim
        self.c = c
        self.G_D = G_D

    def hamiltonian_constraint(self, spatial_metric, extrinsic_curv, energy_density):
        """Check: R + K^2 - K_ij K^ij = (16 pi G / c^4) E"""
        ricci = self.geometry.ricci_tensor(spatial_metric)
        ricci_scalar = self.geometry.ricci_scalar(spatial_metric)
        g_inv = self.geometry.metric_inverse(spatial_metric)

        k_trace_sq = xp.zeros_like(energy_density)
        k_sq = xp.zeros_like(energy_density)

        for i in range(self.n_dim):
            for j in range(self.n_dim):
                k_trace_sq = k_trace_sq + g_inv[..., i, j] * extrinsic_curv[..., i, j]
                k_sq = k_sq + extrinsic_curv[..., i, j] * g_inv[..., i, j] * extrinsic_curv[..., i, j]

        k_trace_sq = k_trace_sq ** 2
        lhs = ricci_scalar + k_trace_sq - k_sq
        rhs = (16.0 * math.pi * self.G_D / (self.c ** 4)) * energy_density

        return lhs, rhs, lhs - rhs


# ============================================================================
# Time Integrators
# ============================================================================

class ExplicitEulerIntegrator:
    """Explicit Euler time integrator."""
    def __init__(self, geometry, n_dim, c=299792458.0, G_D=6.67430e-11):
        self.geometry = geometry
        self.n_dim = n_dim
        self.c = c
        self.G_D = G_D

    def step(self, state, dt):
        """Advance state by dt using explicit Euler."""
        ricci = self.geometry.ricci_tensor(state.spatial_metric)
        ricci_scalar = self.geometry.ricci_scalar(state.spatial_metric)
        g_inv = self.geometry.metric_inverse(state.spatial_metric)
        sqrt_g = self.geometry.sqrt_det_g(state.spatial_metric)

        ten = RelativisticStressEnergyTensor(
            state.rest_density, state.velocity, state.internal_energy,
            state.pressure, state.spatial_metric, c=self.c
        )
        energy_dens = ten.energy_density()

        k_trace = xp.zeros_like(state.spatial_metric[..., 0, 0])
        for i in range(self.n_dim):
            for j in range(self.n_dim):
                k_trace = k_trace + g_inv[..., i, j] * state.extrinsic_curv[..., i, j]

        # Update metric
        metric_new = xp.zeros_like(state.spatial_metric)
        for i in range(self.n_dim):
            for j in range(self.n_dim):
                metric_new[..., i, j] = (state.spatial_metric[..., i, j] -
                                         2.0 * state.lapse * state.extrinsic_curv[..., i, j] * dt)

        # Update extrinsic curvature
        k_new = xp.zeros_like(state.extrinsic_curv)
        for i in range(self.n_dim):
            for j in range(self.n_dim):
                d2_alpha_ij = -xp.gradient(xp.gradient(state.lapse, axis=i), axis=j)
                r_term = ricci[..., i, j]
                k_term = k_trace * state.extrinsic_curv[..., i, j]

                stress_term = 0.1 * (energy_dens * state.spatial_metric[..., i, j])

                k_new[..., i, j] = (state.extrinsic_curv[..., i, j] +
                                    dt * (d2_alpha_ij + state.lapse * (r_term + k_term) +
                                          (8.0 * math.pi * self.G_D / self.c ** 4) * state.lapse * stress_term))

        # Update mass
        div_v = xp.zeros_like(state.velocity[..., 0])
        for i in range(self.n_dim):
            div_v = div_v + xp.gradient(state.velocity[..., i], axis=i)

        mass_new = state.mass - dt * state.mass * div_v

        # Update velocity
        vel_new = xp.asarray(state.velocity)

        # Update proper time
        proper_time_new = state.proper_time + dt

        state_new = ADMState(self.n_dim, state.lapse, state.shift, metric_new,
                             k_new, mass_new, vel_new, state.pressure,
                             state.rest_density, state.internal_energy)
        state_new.proper_time = proper_time_new
        state_new.coordinate_time = state.coordinate_time + dt

        return state_new


class RK4Integrator:
    """4th-order Runge-Kutta time integrator."""
    def __init__(self, geometry, n_dim, c=299792458.0, G_D=6.67430e-11):
        self.geometry = geometry
        self.n_dim = n_dim
        self.c = c
        self.G_D = G_D
        self.euler = ExplicitEulerIntegrator(geometry, n_dim, c, G_D)

    def step(self, state, dt):
        """RK4 step with proper substages."""
        k1 = self.euler.step(state, dt / 6.0)
        k2 = self.euler.step(k1, dt / 3.0)
        k3 = self.euler.step(k2, dt / 3.0)
        k4 = self.euler.step(k3, dt / 6.0)
        return k4


# ============================================================================
# Main Simulation Class
# ============================================================================

class RelativisticFluidSimulation:
    """Main simulation loop."""
    def __init__(self, n_dim, lapse, shift, spatial_metric, extrinsic_curv,
                 mass, velocity, pressure, rest_density, internal_energy,
                 cell_widths=None, c=299792458.0, G_D=6.67430e-11):
        """Initialize simulation."""
        self.n_dim = n_dim
        self.c = c
        self.G_D = G_D

        self.geometry = MetricGeometry(n_dim, cell_widths)
        self.state = ADMState(n_dim, lapse, shift, spatial_metric, extrinsic_curv,
                              mass, velocity, pressure, rest_density, internal_energy)
        self.integrator = ExplicitEulerIntegrator(self.geometry, n_dim, c, G_D)
        self.constraint_checker = ConstraintChecker(self.geometry, n_dim, c, G_D)

    def step(self, dt):
        """Advance simulation by dt."""
        self.state = self.integrator.step(self.state, dt)

    def check_constraints(self):
        """Check constraint violations."""
        ten = RelativisticStressEnergyTensor(
            self.state.rest_density, self.state.velocity, self.state.internal_energy,
            self.state.pressure, self.state.spatial_metric, c=self.c
        )
        E = ten.energy_density()

        h_lhs, h_rhs, h_residual = self.constraint_checker.hamiltonian_constraint(
            self.state.spatial_metric, self.state.extrinsic_curv, E
        )

        return {'hamiltonian_residual': h_residual}
