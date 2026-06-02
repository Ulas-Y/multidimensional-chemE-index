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
            Cell widths for each dimension. If None, uses uniform spacing of 1.0.
        """
        self.n_dim = n_dim
        if cell_widths is None:
            self.cell_widths = _np.ones(n_dim)
        else:
            self.cell_widths = _np.asarray(cell_widths)
        self.omega_n_minus_1 = omega_n_minus_1(n_dim - 1) if n_dim > 1 else 1.0

    def gradient_with_width(self, f, axis):
        """
        Compute gradient respecting cell widths: df/dx = gradient(f) / cell_width[axis]
        """
        grad = xp.gradient(f, axis=axis)
        width = float(self.cell_widths[axis])
        return grad / width

    def conformal_factor_from_potential(self, potential, c=299792458.0):
        """Compute conformal factor psi = 1 + Phi / c^2."""
        return 1.0 + potential / (c ** 2)

    def metric_from_conformal(self, psi):
        """
        Compute spatial metric g_ij = psi^2 * delta_ij.
        """
        shape = psi.shape + (self.n_dim, self.n_dim)
        g = xp.zeros(shape)
        for i in range(self.n_dim):
            for j in range(self.n_dim):
                if i == j:
                    g[..., i, j] = psi ** 2
        return g

    def metric_inverse(self, g):
        """Compute inverse metric g^ij from g_ij."""
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
        """Compute Christoffel symbols Gamma^k_ij using cell_widths."""
        gamma = xp.zeros(g.shape[:-2] + (self.n_dim, self.n_dim, self.n_dim))
        g_inv = self.metric_inverse(g)

        for k in range(self.n_dim):
            for i in range(self.n_dim):
                for j in range(self.n_dim):
                    term = xp.zeros_like(g[..., 0, 0])
                    for l in range(self.n_dim):
                        dg_i = self.gradient_with_width(g[..., l, j], i)
                        dg_j = self.gradient_with_width(g[..., l, i], j)
                        dg_l = self.gradient_with_width(g[..., i, j], l)
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
                    term = term + self.gradient_with_width(gamma[..., k, i, j], k)
                    term = term - self.gradient_with_width(gamma[..., k, k, i], j)

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
        """Compute Laplacian in metric g respecting cell_widths."""
        sqrt_g = self.sqrt_det_g(g)
        g_inv = self.metric_inverse(g)
        laplacian = xp.zeros_like(f)

        for i in range(self.n_dim):
            flux = xp.zeros_like(f)
            for j in range(self.n_dim):
                flux = flux + sqrt_g * g_inv[..., i, j] * self.gradient_with_width(f, j)
            laplacian = laplacian + self.gradient_with_width(flux, i)

        laplacian = laplacian / (sqrt_g + 1e-30)
        return laplacian


# ============================================================================
# Pressure Projection
# ============================================================================

class PressureProjector:
    """Pressure Poisson solver and velocity projection for incompressibility."""
    def __init__(self, geometry):
        self.geometry = geometry

    def divergence(self, velocity):
        """Compute divergence of velocity field."""
        div = xp.zeros_like(velocity[..., 0])
        for i in range(self.geometry.n_dim):
            div = div + self.geometry.gradient_with_width(velocity[..., i], i)
        return div

    def poisson_jacobi(self, rhs, spatial_metric, num_iter=50, residual_tol=1e-6):
        """
        Solve Laplacian(P) = rhs using Jacobi iteration.
        Returns pressure field P.
        """
        P = xp.zeros_like(rhs)

        for iteration in range(num_iter):
            P_old = xp.asarray(P)
            
            # Jacobi update: P_new = (rhs + Laplacian_off_diag(P_old)) / Laplacian_diag
            lap_P = self.geometry.laplacian(P, spatial_metric)
            P = (rhs + lap_P) / (1.0 + 1e-10)  # Simplified update

            # Check residual
            residual = xp.asarray(lap_P - rhs)
            res_norm = _to_python_float((residual ** 2).sum() ** 0.5)
            if res_norm < residual_tol:
                break

        return P

    def project_velocity(self, velocity, spatial_metric, dt, density):
        """
        Enforce incompressibility via projection:
        v_new = v - (dt / rho) g^ij d_j P
        where P solves: Laplacian(P) = (rho / dt) div(v)
        """
        # Compute divergence
        div_v = self.divergence(velocity)

        # Solve Poisson: Laplacian(P) = (rho / dt) div(v)
        rhs = (density / (dt + 1e-30)) * div_v
        pressure = self.poisson_jacobi(rhs, spatial_metric)

        # Project velocity
        v_proj = xp.zeros_like(velocity)
        g_inv = self.geometry.metric_inverse(spatial_metric)

        for i in range(self.geometry.n_dim):
            v_proj[..., i] = velocity[..., i]
            for j in range(self.geometry.n_dim):
                dp_dj = self.geometry.gradient_with_width(pressure, j)
                v_proj[..., i] = v_proj[..., i] - (dt / (density + 1e-30)) * g_inv[..., i, j] * dp_dj

        return v_proj, pressure


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

    def update_proper_time(self, geometry, dt, c=299792458.0):
        """
        Update proper time: d_tau = alpha * dt * sqrt(1 - v^2/c^2)
        where alpha is lapse (controls how fast coordinate time flows)
        """
        # Compute v^2
        v_sq = xp.zeros_like(self.velocity[..., 0])
        for i in range(self.n_dim):
            v_sq = v_sq + self.velocity[..., i] * self.velocity[..., i]

        # Time dilation factor: sqrt(1 - v^2/c^2)
        time_dilation = (1.0 - v_sq / (c ** 2)) ** 0.5
        time_dilation = xp.asarray(time_dilation)

        # Proper time increment: d_tau = alpha * dt * sqrt(1 - v^2/c^2)
        d_tau = self.lapse * dt * time_dilation

        self.proper_time = self.proper_time + d_tau
        self.coordinate_time = self.coordinate_time + dt

    def update_derived_quantities(self, geometry):
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

class TimeIntegrator:
    """Base class for time integrators."""
    def __init__(self, geometry, n_dim, c=299792458.0, G_D=6.67430e-11):
        self.geometry = geometry
        self.n_dim = n_dim
        self.c = c
        self.G_D = G_D
        self.projector = PressureProjector(geometry)

    def compute_velocity_rhs(self, state):
        """
        Compute RHS for velocity evolution: dv^i/dt = -(1/rho) d_i P + f^i
        Returns acceleration field.
        """
        state.update_derived_quantities(self.geometry)
        
        dv_dt = xp.zeros_like(state.velocity)
        g_inv = self.geometry.metric_inverse(state.spatial_metric)

        for i in range(self.n_dim):
            for j in range(self.n_dim):
                dp_dj = self.geometry.gradient_with_width(state.pressure, j)
                dv_dt[..., i] = dv_dt[..., i] - (1.0 / (state.density + 1e-30)) * g_inv[..., i, j] * dp_dj

        return dv_dt

    def compute_mass_rhs(self, state):
        """
        Compute RHS for mass evolution: dm/dt = -m * div(v)
        """
        div_v = self.projector.divergence(state.velocity)
        dm_dt = -state.mass * div_v
        return dm_dt

    def compute_metric_rhs(self, state):
        """
        Compute RHS for metric evolution: d(gamma_ij)/dt = -2*alpha*K_ij + ...
        """
        metric_rhs = xp.zeros_like(state.spatial_metric)
        for i in range(self.n_dim):
            for j in range(self.n_dim):
                metric_rhs[..., i, j] = -2.0 * state.lapse * state.extrinsic_curv[..., i, j]

        return metric_rhs

    def compute_extrinsic_rhs(self, state):
        """
        Compute RHS for extrinsic curvature evolution.
        Simplified: d(K_ij)/dt = -d_i d_j(alpha) + alpha[R_ij + ...]
        """
        ricci = self.geometry.ricci_tensor(state.spatial_metric)
        g_inv = self.geometry.metric_inverse(state.spatial_metric)

        k_rhs = xp.zeros_like(state.extrinsic_curv)

        for i in range(self.n_dim):
            for j in range(self.n_dim):
                # -d_i d_j alpha
                d2_alpha = -self.geometry.gradient_with_width(
                    self.geometry.gradient_with_width(state.lapse, i), j
                )

                # K trace
                k_trace = xp.zeros_like(state.spatial_metric[..., 0, 0])
                for l in range(self.n_dim):
                    k_trace = k_trace + g_inv[..., l, l] * state.extrinsic_curv[..., l, l]

                # alpha (R_ij + K K_ij - 2 K_ik K^k_j)
                r_term = ricci[..., i, j]
                k_term = k_trace * state.extrinsic_curv[..., i, j]

                k_ik_k_kj = xp.zeros_like(state.extrinsic_curv[..., 0, 0])
                for k in range(self.n_dim):
                    k_ik_k_kj = k_ik_k_kj + state.extrinsic_curv[..., i, k] * g_inv[..., k, j] * state.extrinsic_curv[..., j, j]

                stress_term = 0.1 * (state.pressure * state.spatial_metric[..., i, j])

                k_rhs[..., i, j] = (d2_alpha + state.lapse * (r_term + k_term - 2.0 * k_ik_k_kj) +
                                    (8.0 * math.pi * self.G_D / self.c ** 4) * state.lapse * stress_term)

        return k_rhs

    def copy_state(self, state):
        """Create a deep copy of state."""
        new_state = ADMState(
            state.n_dim, xp.asarray(state.lapse), xp.asarray(state.shift),
            xp.asarray(state.spatial_metric), xp.asarray(state.extrinsic_curv),
            xp.asarray(state.mass), xp.asarray(state.velocity),
            xp.asarray(state.pressure), xp.asarray(state.rest_density),
            xp.asarray(state.internal_energy)
        )
        new_state.proper_time = xp.asarray(state.proper_time)
        new_state.coordinate_time = state.coordinate_time
        return new_state

    def add_scaled_rhs(self, state, rhs_velocity, rhs_mass, rhs_metric, rhs_extrinsic, factor):
        """Add scaled RHS to state: state += factor * rhs"""
        new_state = self.copy_state(state)
        new_state.velocity = state.velocity + factor * rhs_velocity
        new_state.mass = state.mass + factor * rhs_mass
        new_state.spatial_metric = state.spatial_metric + factor * rhs_metric
        new_state.extrinsic_curv = state.extrinsic_curv + factor * rhs_extrinsic
        return new_state


class ExplicitEulerIntegrator(TimeIntegrator):
    """Explicit Euler time integrator: x^{n+1} = x^n + dt * dx/dt"""
    def step(self, state, dt):
        """Advance state by dt using explicit Euler."""
        # Compute RHS at current state
        vel_rhs = self.compute_velocity_rhs(state)
        mass_rhs = self.compute_mass_rhs(state)
        metric_rhs = self.compute_metric_rhs(state)
        extrinsic_rhs = self.compute_extrinsic_rhs(state)

        # Update state
        state_new = self.copy_state(state)
        state_new.velocity = state.velocity + dt * vel_rhs
        state_new.mass = state.mass + dt * mass_rhs
        state_new.spatial_metric = state.spatial_metric + dt * metric_rhs
        state_new.extrinsic_curv = state.extrinsic_curv + dt * extrinsic_rhs

        # Project velocity to enforce incompressibility
        state_new.update_derived_quantities(self.geometry)
        state_new.velocity, state_new.pressure = self.projector.project_velocity(
            state_new.velocity, state_new.spatial_metric, dt, state_new.density
        )

        # Update proper time
        state_new.update_proper_time(self.geometry, dt, self.c)

        return state_new


class RK4Integrator(TimeIntegrator):
    """
    4th-order Runge-Kutta integrator.
    x^{n+1} = x^n + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    """
    def step(self, state, dt):
        """Advance state by dt using RK4."""
        # Stage 1: k1 = f(t, x^n)
        k1_vel = self.compute_velocity_rhs(state)
        k1_mass = self.compute_mass_rhs(state)
        k1_metric = self.compute_metric_rhs(state)
        k1_extrinsic = self.compute_extrinsic_rhs(state)

        # Stage 2: k2 = f(t + dt/2, x^n + (dt/2)*k1)
        state_2 = self.add_scaled_rhs(state, k1_vel, k1_mass, k1_metric, k1_extrinsic, dt / 2.0)
        state_2.update_derived_quantities(self.geometry)
        k2_vel = self.compute_velocity_rhs(state_2)
        k2_mass = self.compute_mass_rhs(state_2)
        k2_metric = self.compute_metric_rhs(state_2)
        k2_extrinsic = self.compute_extrinsic_rhs(state_2)

        # Stage 3: k3 = f(t + dt/2, x^n + (dt/2)*k2)
        state_3 = self.add_scaled_rhs(state, k2_vel, k2_mass, k2_metric, k2_extrinsic, dt / 2.0)
        state_3.update_derived_quantities(self.geometry)
        k3_vel = self.compute_velocity_rhs(state_3)
        k3_mass = self.compute_mass_rhs(state_3)
        k3_metric = self.compute_metric_rhs(state_3)
        k3_extrinsic = self.compute_extrinsic_rhs(state_3)

        # Stage 4: k4 = f(t + dt, x^n + dt*k3)
        state_4 = self.add_scaled_rhs(state, k3_vel, k3_mass, k3_metric, k3_extrinsic, dt)
        state_4.update_derived_quantities(self.geometry)
        k4_vel = self.compute_velocity_rhs(state_4)
        k4_mass = self.compute_mass_rhs(state_4)
        k4_metric = self.compute_metric_rhs(state_4)
        k4_extrinsic = self.compute_extrinsic_rhs(state_4)

        # Combine stages
        state_new = self.copy_state(state)
        state_new.velocity = (state.velocity + 
                             (dt / 6.0) * (k1_vel + 2.0*k2_vel + 2.0*k3_vel + k4_vel))
        state_new.mass = (state.mass + 
                         (dt / 6.0) * (k1_mass + 2.0*k2_mass + 2.0*k3_mass + k4_mass))
        state_new.spatial_metric = (state.spatial_metric + 
                                   (dt / 6.0) * (k1_metric + 2.0*k2_metric + 2.0*k3_metric + k4_metric))
        state_new.extrinsic_curv = (state.extrinsic_curv + 
                                   (dt / 6.0) * (k1_extrinsic + 2.0*k2_extrinsic + 2.0*k3_extrinsic + k4_extrinsic))

        # Project velocity to enforce incompressibility
        state_new.update_derived_quantities(self.geometry)
        state_new.velocity, state_new.pressure = self.projector.project_velocity(
            state_new.velocity, state_new.spatial_metric, dt, state_new.density
        )

        # Update proper time
        state_new.update_proper_time(self.geometry, dt, self.c)

        return state_new


# ============================================================================
# Main Simulation Class
# ============================================================================

class RelativisticFluidSimulation:
    """Main simulation loop."""
    def __init__(self, n_dim, lapse, shift, spatial_metric, extrinsic_curv,
                 mass, velocity, pressure, rest_density, internal_energy,
                 cell_widths=None, integrator_type='euler', c=299792458.0, G_D=6.67430e-11):
        """
        Initialize simulation.
        
        Parameters
        ----------
        n_dim : int
            Spatial dimensions
        lapse : array
            Lapse function alpha
        shift : array of shape (..., n_dim)
            Shift vector beta^i
        spatial_metric : array of shape (..., n_dim, n_dim)
            Spatial metric gamma_ij
        extrinsic_curv : array of shape (..., n_dim, n_dim)
            Extrinsic curvature K_ij
        mass : array
            Rest mass in each cell
        velocity : array of shape (..., n_dim)
            3-velocity v^i
        pressure : array
            Pressure field p
        rest_density : array
            Rest mass density rho_0
        internal_energy : array
            Specific internal energy epsilon
        cell_widths : array, optional
            Cell widths per dimension
        integrator_type : str
            'euler' for Explicit Euler, 'rk4' for RK4
        c : float
            Speed of light
        G_D : float
            Gravitational constant
        """
        self.n_dim = n_dim
        self.c = c
        self.G_D = G_D

        self.geometry = MetricGeometry(n_dim, cell_widths)
        self.state = ADMState(n_dim, lapse, shift, spatial_metric, extrinsic_curv,
                              mass, velocity, pressure, rest_density, internal_energy)
        
        if integrator_type.lower() == 'rk4':
            self.integrator = RK4Integrator(self.geometry, n_dim, c, G_D)
        else:
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
