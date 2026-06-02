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
        """
        Compute sqrt(det(g)) for arbitrary metric g_ij at each grid point.
        Uses LU decomposition to compute determinant for N dimensions.
        """
        # Handle both PyTorch and NumPy backends
        if xp.name == "torch":
            det_g = torch.det(g)
        else:
            # For NumPy, compute determinant cell-by-cell
            shape = g.shape[:-2]
            det_g = xp.zeros(shape)
            if len(shape) == 0:
                # Scalar case
                det_g = _np.linalg.det(g)
            else:
                # Iterate over all cells
                it = _np.nditer(shape, flags=['multi_index'])
                for _ in it:
                    idx = it.multi_index
                    det_g[idx] = _np.linalg.det(g[idx])

        sqrt_det_g = xp.asarray((det_g + 1e-30) ** 0.5)
        return sqrt_det_g

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

    def laplacian_diagonal_and_off_diag(self, f, spatial_metric):
        """
        Decompose Laplacian(f) = diag_term + off_diag_term for Jacobi iteration.
        Returns (diagonal_part, off_diagonal_part) where:
        Laplacian(f) = diagonal_part / diag_coeff + off_diagonal_part
        """
        sqrt_g = self.geometry.sqrt_det_g(spatial_metric)
        g_inv = self.geometry.metric_inverse(spatial_metric)

        # Compute full Laplacian
        laplacian_f = self.geometry.laplacian(f, spatial_metric)

        # For simplicity, diagonal dominance approximation:
        # Compute analytical diagonal of Laplace-Beltrami for use in Jacobi
        diag_coeff = xp.zeros_like(f)
        for i in range(self.geometry.n_dim):
            dx_i = float(self.geometry.cell_widths[i])
            diag_coeff = diag_coeff - (2.0 * g_inv[..., i, i]) / (dx_i ** 2)

        return laplacian_f, diag_coeff

    def poisson_jacobi(self, rhs, spatial_metric, num_iter=100, residual_tol=1e-5):
        """
        Solve Laplacian(P) = rhs using proper Jacobi relaxation.
        P_new = (rhs - off_diag(P_old)) / diag_coeff
        """
        P = xp.zeros_like(rhs)

        # Precompute inverse metric and analytical diagonal of Laplace-Beltrami
        g_inv = self.geometry.metric_inverse(spatial_metric)
        L_diag = xp.zeros_like(rhs)
        for i in range(self.geometry.n_dim):
            dx_i = float(self.geometry.cell_widths[i])
            L_diag = L_diag - (2.0 * g_inv[..., i, i]) / (dx_i ** 2)

        for iteration in range(num_iter):
            # Compute Laplacian at current P
            lap_P = self.geometry.laplacian(P, spatial_metric)

            # Residual: Laplacian(P) - rhs
            residual = lap_P - rhs

            # Standard Jacobi relaxation (omega=1.0): exact central-diagonal correction
            P = P - residual / (L_diag + 1e-30)

            # Check convergence (L2 norm)
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
        v^2 = g_ij v^i v^j is proper metric contraction
        """
        # Construct Eulerian 3-velocity U^i = (v^i + beta^i) / alpha
        U = xp.zeros_like(self.velocity)
        for i in range(self.n_dim):
            U[..., i] = (self.velocity[..., i] + self.shift[..., i]) / (self.lapse + 1e-30)

        # Compute U^2 = g_ij U^i U^j
        U_sq = xp.zeros_like(U[..., 0])
        for i in range(self.n_dim):
            for j in range(self.n_dim):
                U_sq = U_sq + self.spatial_metric[..., i, j] * U[..., i] * U[..., j]

        # Lorentz factor field and proper time increment: d_tau = (alpha / gamma) * dt
        gamma_field = 1.0 / ((1.0 - U_sq / (c ** 2)) ** 0.5 + 1e-30)
        d_tau = (self.lapse / (gamma_field + 1e-30)) * dt

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
                 spatial_metric, lapse, shift, c=299792458.0):
        """Initialize stress-energy tensor."""
        self.rho0 = rest_density
        self.v = velocity
        self.epsilon = internal_energy
        self.p = pressure
        self.gamma = spatial_metric
        self.lapse = lapse
        self.shift = shift
        self.c = c
        self.n_dim = spatial_metric.shape[-1]

    def lorentz_factor_field(self):
        """Compute Lorentz factor field gamma = 1/sqrt(1 - U^2/c^2) using Eulerian U^i."""
        # Construct Eulerian 3-velocity U^i = (v^i + beta^i) / alpha
        U = xp.zeros_like(self.v)
        for i in range(self.n_dim):
            U[..., i] = (self.v[..., i] + self.shift[..., i]) / (self.lapse + 1e-30)

        # Compute U^2 = g_ij U^i U^j
        U_sq = xp.zeros_like(U[..., 0])
        for i in range(self.n_dim):
            for j in range(self.n_dim):
                U_sq = U_sq + self.gamma[..., i, j] * U[..., i] * U[..., j]

        gamma_u = 1.0 / ((1.0 - U_sq / (self.c ** 2)) ** 0.5 + 1e-30)
        return gamma_u

    def energy_density(self):
        """Compute T_00 component: energy density."""
        gamma_u = self.lorentz_factor_field()
        h = 1.0 + self.epsilon + self.p / (self.rho0 * self.c ** 2 + 1e-30)
        return (gamma_u ** 2) * self.rho0 * h - self.p

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
        # EOS and stabilization defaults
        self.Gamma_eos = 5.0 / 3.0
        self.sigma_visc = 0.06

    def compute_velocity_rhs(self, state, gamma_field, h):
        """
        Compute RHS for velocity evolution:
        dv^i/dt = -(1/rho) g^ij ∂_j P  [pressure gradient]
                  -(v·∇)v^i             [convective advection]
                  +(nu) ∇²v^i           [viscous diffusion]
        """
        state.update_derived_quantities(self.geometry)

        dv_dt = xp.zeros_like(state.velocity)
        g_inv = self.geometry.metric_inverse(state.spatial_metric)
        metric = state.spatial_metric
        v = state.velocity

        rho0 = state.rest_density

        # Pressure gradient term: -(1/(rho0 * h * gamma^2)) g^ij ∂_j P
        for i in range(self.n_dim):
            for j in range(self.n_dim):
                dp_dj = self.geometry.gradient_with_width(state.pressure, j)
                coeff = -1.0 / (rho0 + 1e-30) / (h + 1e-30) / (gamma_field ** 2 + 1e-30)
                dv_dt[..., i] = dv_dt[..., i] + coeff * g_inv[..., i, j] * dp_dj

        # Convective advection: -(v·∇)v^i = -Σ_j v^j ∂_j v^i
        for i in range(self.n_dim):
            for j in range(self.n_dim):
                dv_i_dj = self.geometry.gradient_with_width(v[..., i], j)
                dv_dt[..., i] = dv_dt[..., i] - v[..., j] * dv_i_dj

        # Viscous diffusion: nu * ∇^2 v^i, nu read from state if provided
        nu = getattr(state, 'viscosity', 0.0)
        if nu != 0.0:
            for i in range(self.n_dim):
                lap_vi = self.geometry.laplacian(v[..., i], metric)
                dv_dt[..., i] = dv_dt[..., i] + nu * lap_vi

        return dv_dt

    def update_pressure_from_eos(self, state):
        """Update pressure in-place from EOS: P = (Gamma - 1) rho0 epsilon."""
        Gamma = getattr(self, 'Gamma_eos', 5.0 / 3.0)
        state.pressure = (Gamma - 1.0) * state.rest_density * state.internal_energy

    def compute_lorentz_factor(self, state):
        """Compute Eulerian Lorentz factor gamma from U^i = (v^i + beta^i) / alpha."""
        U = xp.zeros_like(state.velocity)
        for i in range(self.n_dim):
            U[..., i] = (state.velocity[..., i] + state.shift[..., i]) / (state.lapse + 1e-30)

        U_sq = xp.zeros_like(U[..., 0])
        for i in range(self.n_dim):
            for j in range(self.n_dim):
                U_sq = U_sq + state.spatial_metric[..., i, j] * U[..., i] * U[..., j]

        gamma = 1.0 / ((1.0 - U_sq / (self.c ** 2)) ** 0.5 + 1e-30)
        return gamma, U

    def compute_density_rhs(self, state, gamma_field):
        """Compute RHS for rest-mass density: ∂_t ρ0 = -Σ_i v^i ∂_i ρ0 - ρ0 Θ"""
        rho0 = state.rest_density
        v = state.velocity

        # Compute ln(gamma * sqrt(det g)) and its gradients
        sqrt_g = self.geometry.sqrt_det_g(state.spatial_metric)
        ln_factor = xp.log((gamma_field + 1e-30) * (sqrt_g + 1e-30) + 1e-30)

        Theta = xp.zeros_like(rho0)
        for i in range(self.n_dim):
            dvii = self.geometry.gradient_with_width(v[..., i], i)
            dln = self.geometry.gradient_with_width(ln_factor, i)
            Theta = Theta + (dvii + v[..., i] * dln)

        adv = xp.zeros_like(rho0)
        for i in range(self.n_dim):
            drho_di = self.geometry.gradient_with_width(rho0, i)
            adv = adv + v[..., i] * drho_di

        d_rho_dt = -adv - rho0 * Theta
        return d_rho_dt

    def compute_energy_rhs(self, state, gamma_field):
        """Compute RHS for specific internal energy: ∂_t ε = -v·∇ε - (P/ρ0) Θ"""
        eps = state.internal_energy
        rho0 = state.rest_density
        P = state.pressure
        v = state.velocity

        # Compute Theta as in density RHS
        sqrt_g = self.geometry.sqrt_det_g(state.spatial_metric)
        ln_factor = xp.log((gamma_field + 1e-30) * (sqrt_g + 1e-30) + 1e-30)

        Theta = xp.zeros_like(eps)
        for i in range(self.n_dim):
            dvii = self.geometry.gradient_with_width(v[..., i], i)
            dln = self.geometry.gradient_with_width(ln_factor, i)
            Theta = Theta + (dvii + v[..., i] * dln)

        adv = xp.zeros_like(eps)
        for i in range(self.n_dim):
            deps_di = self.geometry.gradient_with_width(eps, i)
            adv = adv + v[..., i] * deps_di

        d_eps_dt = -adv - (P / (rho0 + 1e-30)) * Theta
        return d_eps_dt

    def compute_artificial_dissipation(self, field):
        """Compute simple Lax-Friedrichs type artificial dissipation for scalar or vector field."""
        # sigma_visc * Σ_i (dx_i)^2 ∂^2 φ / ∂ x_i^2
        sigma = getattr(self, 'sigma_visc', 0.06)
        # second derivatives per axis
        diss = xp.zeros_like(field)
        for i in range(self.n_dim):
            dx_i = float(self.geometry.cell_widths[i])
            second = self.geometry.gradient_with_width(self.geometry.gradient_with_width(field, i), i)
            diss = diss + (dx_i ** 2) * second
        return sigma * diss

    def compute_rhs(self, state):
        """Compute complete RHS tuple for (metric, extrinsic, rho0, velocity, eps)."""
        # 1. Update pressure from EOS before evaluating RHS
        self.update_pressure_from_eos(state)

        # 2. Compute Lorentz gamma and enthalpy
        gamma_field, U = self.compute_lorentz_factor(state)
        rho0 = state.rest_density
        eps = state.internal_energy
        P = state.pressure
        h = 1.0 + eps / (self.c ** 2) + P / (rho0 * (self.c ** 2) + 1e-30)

        # 3. Evaluate hydrodynamic RHS
        d_rho_dt = self.compute_density_rhs(state, gamma_field) + self.compute_artificial_dissipation(state.rest_density)
        d_eps_dt = self.compute_energy_rhs(state, gamma_field) + self.compute_artificial_dissipation(state.internal_energy)
        d_v_dt = self.compute_velocity_rhs(state, gamma_field, h) + self.compute_artificial_dissipation(state.velocity)

        # 4. Geometric RHS preserved
        d_g_dt = self.compute_metric_rhs(state)
        d_k_dt = self.compute_extrinsic_rhs(state)

        return d_g_dt, d_k_dt, d_rho_dt, d_v_dt, d_eps_dt

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
        Compute RHS for extrinsic curvature evolution via 3+1 ADM.
        d(K_ij)/dt = -∇_i ∇_j α + α[R_ij + K_trace K_ij - 2K_ik K^k_j] + (8πG/c⁴) α S_ij
        where S_ij = T_ij - (1/(n-1))(T - T_00)g_ij is the traceless stress tensor
        """
        ricci = self.geometry.ricci_tensor(state.spatial_metric)
        g_inv = self.geometry.metric_inverse(state.spatial_metric)

        k_rhs = xp.zeros_like(state.extrinsic_curv)

        # Compute K trace: K = g^ij K_ij
        k_trace = xp.zeros_like(state.spatial_metric[..., 0, 0])
        for l in range(self.n_dim):
            for m in range(self.n_dim):
                k_trace = k_trace + g_inv[..., l, m] * state.extrinsic_curv[..., l, m]

        # Construct Eulerian U^i and gamma and specific enthalpy h
        U = xp.zeros_like(state.velocity)
        for a in range(self.n_dim):
            U[..., a] = (state.velocity[..., a] + state.shift[..., a]) / (state.lapse + 1e-30)

        U_sq = xp.zeros_like(U[..., 0])
        for a in range(self.n_dim):
            for b in range(self.n_dim):
                U_sq = U_sq + state.spatial_metric[..., a, b] * U[..., a] * U[..., b]
        gamma = 1.0 / ((1.0 - U_sq / (self.c ** 2)) ** 0.5 + 1e-30)

        rho0 = state.rest_density
        h = 1.0 + state.internal_energy + state.pressure / (rho0 * self.c ** 2 + 1e-30)

        # Lowered index U_i = g_ik U^k
        U_lower = xp.zeros_like(U)
        for p in range(self.n_dim):
            for q in range(self.n_dim):
                U_lower[..., p] = U_lower[..., p] + state.spatial_metric[..., p, q] * U[..., q]

        # Compute spatial stress S_ij and its trace
        S = xp.zeros_like(state.spatial_metric)
        for a in range(self.n_dim):
            for b in range(self.n_dim):
                S[..., a, b] = rho0 * h * (gamma ** 2) * U_lower[..., a] * U_lower[..., b] + state.pressure * state.spatial_metric[..., a, b]

        S_trace = xp.zeros_like(rho0)
        for l in range(self.n_dim):
            for m in range(self.n_dim):
                S_trace = S_trace + g_inv[..., l, m] * S[..., l, m]

        # Energy density E = gamma^2 rho0 h - P
        E = (gamma ** 2) * rho0 * h - state.pressure

        kappa_n = 16.0 * math.pi * self.G_D / (self.c ** 4)

        for i in range(self.n_dim):
            for j in range(self.n_dim):
                # -∇_i ∇_j α (negative Hessian of lapse)
                d2_alpha = -self.geometry.gradient_with_width(
                    self.geometry.gradient_with_width(state.lapse, i), j
                )

                # Ricci + K trace term
                r_term = ricci[..., i, j]
                k_term = k_trace * state.extrinsic_curv[..., i, j]

                # K_ik K^k_j = Σ_k Σ_l K_ik g^kl K_lj
                k_ik_k_kj = xp.zeros_like(state.extrinsic_curv[..., 0, 0])
                for k in range(self.n_dim):
                    for l in range(self.n_dim):
                        k_ik_k_kj = (k_ik_k_kj + 
                                     state.extrinsic_curv[..., i, k] * g_inv[..., k, l] * 
                                     state.extrinsic_curv[..., l, j])

                # Traceless material source: S_ij - (1/(n-1)) g_ij (S_trace - E)
                stress_term = S[..., i, j] - (1.0 / (self.n_dim - 1)) * state.spatial_metric[..., i, j] * (S_trace - E)

                k_rhs[..., i, j] = (d2_alpha + state.lapse * (r_term + k_term - 2.0 * k_ik_k_kj) + kappa_n * state.lapse * stress_term)

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

    def add_scaled_rhs(self, state, rhs_velocity, rhs_rest_density, rhs_metric, rhs_extrinsic, rhs_internal_energy, factor):
        """Add scaled RHS to state: state += factor * rhs for full primitive set"""
        new_state = self.copy_state(state)
        new_state.velocity = state.velocity + factor * rhs_velocity
        # evolve rest_density and internal energy
        new_state.rest_density = state.rest_density + factor * rhs_rest_density
        new_state.internal_energy = state.internal_energy + factor * rhs_internal_energy
        new_state.spatial_metric = state.spatial_metric + factor * rhs_metric
        new_state.extrinsic_curv = state.extrinsic_curv + factor * rhs_extrinsic
        return new_state


class ExplicitEulerIntegrator(TimeIntegrator):
    """Explicit Euler time integrator: x^{n+1} = x^n + dt * dx/dt"""
    def step(self, state, dt):
        """Advance state by dt using explicit Euler."""
        # Compute RHS for full system
        d_g_dt, d_k_dt, d_rho_dt, d_v_dt, d_eps_dt = self.compute_rhs(state)

        # Update state primitives
        state_new = self.copy_state(state)
        state_new.velocity = state.velocity + dt * d_v_dt
        state_new.rest_density = state.rest_density + dt * d_rho_dt
        state_new.internal_energy = state.internal_energy + dt * d_eps_dt
        state_new.spatial_metric = state.spatial_metric + dt * d_g_dt
        state_new.extrinsic_curv = state.extrinsic_curv + dt * d_k_dt

        # Update derived quantities and pressure from EOS
        state_new.update_derived_quantities(self.geometry)
        self.update_pressure_from_eos(state_new)

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
        k1_g, k1_k, k1_rho, k1_v, k1_eps = self.compute_rhs(state)

        # Stage 2: k2 = f(t + dt/2, x^n + (dt/2)*k1)
        state_2 = self.add_scaled_rhs(state, k1_v, k1_rho, k1_g, k1_k, k1_eps, dt / 2.0)
        state_2.update_derived_quantities(self.geometry)
        self.update_pressure_from_eos(state_2)
        k2_g, k2_k, k2_rho, k2_v, k2_eps = self.compute_rhs(state_2)

        # Stage 3: k3 = f(t + dt/2, x^n + (dt/2)*k2)
        state_3 = self.add_scaled_rhs(state, k2_v, k2_rho, k2_g, k2_k, k2_eps, dt / 2.0)
        state_3.update_derived_quantities(self.geometry)
        self.update_pressure_from_eos(state_3)
        k3_g, k3_k, k3_rho, k3_v, k3_eps = self.compute_rhs(state_3)

        # Stage 4: k4 = f(t + dt, x^n + dt*k3)
        state_4 = self.add_scaled_rhs(state, k3_v, k3_rho, k3_g, k3_k, k3_eps, dt)
        state_4.update_derived_quantities(self.geometry)
        self.update_pressure_from_eos(state_4)
        k4_g, k4_k, k4_rho, k4_v, k4_eps = self.compute_rhs(state_4)

        # Combine stages
        state_new = self.copy_state(state)
        state_new.velocity = (state.velocity + (dt / 6.0) * (k1_v + 2.0*k2_v + 2.0*k3_v + k4_v))
        state_new.rest_density = (state.rest_density + (dt / 6.0) * (k1_rho + 2.0*k2_rho + 2.0*k3_rho + k4_rho))
        state_new.internal_energy = (state.internal_energy + (dt / 6.0) * (k1_eps + 2.0*k2_eps + 2.0*k3_eps + k4_eps))
        state_new.spatial_metric = (state.spatial_metric + (dt / 6.0) * (k1_g + 2.0*k2_g + 2.0*k3_g + k4_g))
        state_new.extrinsic_curv = (state.extrinsic_curv + (dt / 6.0) * (k1_k + 2.0*k2_k + 2.0*k3_k + k4_k))

        # Update derived quantities and pressure from EOS
        state_new.update_derived_quantities(self.geometry)
        self.update_pressure_from_eos(state_new)

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
            self.state.pressure, self.state.spatial_metric, self.state.lapse, self.state.shift, c=self.c
        )
        E = ten.energy_density()

        h_lhs, h_rhs, h_residual = self.constraint_checker.hamiltonian_constraint(
            self.state.spatial_metric, self.state.extrinsic_curv, E
        )

        return {'hamiltonian_residual': h_residual}
