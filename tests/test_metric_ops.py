import numpy as np
from numpy_fluid_core import MetricGeometry, XPBackend


def test_conformal_metric_det_inv():
    xp = XPBackend.select('numpy')
    mg = MetricGeometry(n_dim=2, cell_widths=[1.0, 1.0])

    # simple conformal factor across 3 cells
    psi = np.array([1.2, 0.8, 1.5])
    g = mg.metric_from_conformal(psi)

    # sqrt(det(g)) for conformal metric should be psi**n_dim
    sqrt_det = mg.sqrt_det_g(g)
    expected = psi ** mg.n_dim
    assert np.allclose(sqrt_det, expected, rtol=1e-6, atol=1e-12)

    g_inv = mg.metric_inverse(g)
    # verify g @ g_inv == I for each cell
    for i in range(g.shape[0]):
        prod = g[i].dot(g_inv[i])
        assert np.allclose(prod, np.eye(2), rtol=1e-6, atol=1e-12)


def test_batched_det_inv_random():
    xp = XPBackend.select('numpy')
    mg = MetricGeometry(n_dim=2)

    rng = np.random.RandomState(0)
    mats = []
    for _ in range(5):
        A = rng.randn(2, 2)
        m = A.T.dot(A) + 0.1 * np.eye(2)
        mats.append(m)

    g = np.stack(mats, axis=0)
    det_np = np.linalg.det(g)

    sqrt_det = mg.sqrt_det_g(g)
    assert np.allclose(sqrt_det, np.sqrt(det_np), rtol=1e-6)

    g_inv = mg.metric_inverse(g)
    for i in range(g.shape[0]):
        assert np.allclose(g[i].dot(g_inv[i]), np.eye(2), rtol=1e-6)
