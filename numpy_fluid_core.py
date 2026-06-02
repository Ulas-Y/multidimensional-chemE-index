import warnings
import numpy as _np

torch = None
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def _torch_gradient(x, axis=-1):
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

    def stack(self, seq, axis=0):
        if self.name == "torch":
            return self.backend.stack(seq, dim=axis)
        return self.backend.stack(seq, axis=axis)

    def gradient(self, x, axis=-1):
        if self.name == "torch":
            return _torch_gradient(x, axis)
        return self.backend.gradient(x, axis=axis)


if _TORCH_AVAILABLE and torch.cuda.is_available():
    xp = _Backend(torch, "torch")
else:
    if _TORCH_AVAILABLE:
        warnings.warn("PyTorch is installed but CUDA is unavailable; falling back to NumPy backend.")
    xp = _Backend(_np, "numpy")


class FluidState:
    def __init__(self, velocity, pressure, density=1.0, viscosity=1.0, body_force=None, velocity_dot=None):
        self.velocity = xp.asarray(velocity, dtype=float)
        self.pressure = xp.asarray(pressure, dtype=float)
        self.density = float(density)
        self.viscosity = float(viscosity)
        self.dim = self.velocity.shape[0]
        if body_force is None:
            self.body_force = xp.zeros_like(self.velocity)
        else:
            self.body_force = xp.asarray(body_force, dtype=float)
        if velocity_dot is None:
            self.velocity_dot = xp.zeros_like(self.velocity)
        else:
            self.velocity_dot = xp.asarray(velocity_dot, dtype=float)

    def divergence(self):
        return sum(xp.gradient(self.velocity[i], axis=i) for i in range(self.dim))

    def pressure_gradient(self):
        return xp.stack([xp.gradient(self.pressure, axis=i) for i in range(self.dim)], axis=0)

    def laplacian_component(self, i):
        return sum(xp.gradient(xp.gradient(self.velocity[i], axis=j), axis=j) for j in range(self.dim))

    def viscous_term(self):
        return xp.stack([self.laplacian_component(i) for i in range(self.dim)], axis=0)

    def incompressibility_correction(self):
        div = self.divergence()
        return xp.stack([xp.gradient(div, axis=i) for i in range(self.dim)], axis=0)

    def convective_acceleration(self):
        accel = xp.zeros_like(self.velocity)
        for i in range(self.dim):
            for j in range(self.dim):
                accel[i] += self.velocity[j] * xp.gradient(self.velocity[i], axis=j)
        return accel

    def local_acceleration(self):
        return self.velocity_dot

    def acceleration(self):
        return self.local_acceleration() + self.convective_acceleration()


class MultidimensionalFluidEquation:
    def __init__(self, state: FluidState):
        self.state = state

    def right_hand_side(self):
        pressure_term = -self.state.pressure_gradient()
        viscous_term = self.state.viscosity * self.state.viscous_term()
        body_force = self.state.density * self.state.body_force
        correction = (1.0 / 3.0) * self.state.viscosity * self.state.incompressibility_correction()
        return pressure_term + viscous_term + body_force + correction

    def momentum_residual(self):
        lhs = self.state.density * self.state.acceleration()
        rhs = self.right_hand_side()
        return lhs - rhs
