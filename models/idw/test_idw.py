import numpy as np
from idw_model import idw


def make_data(T=4, N=8, M=30, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.uniform(0, 100, (N, 2)),
            rng.uniform(40, 100, (T, N)),
            rng.uniform(0, 100, (M, 2)))

# test that the output shape is correct
def test_output_shape():
    pos, lev, q = make_data(T=4, N=8, M=30)
    assert idw(pos, lev, q).shape == (4, 30)

# test that predictions are exact at the phone positions
def test_exact_at_phone_positions():
    pos, lev, _ = make_data()
    np.testing.assert_allclose(idw(pos, lev, pos), lev, atol=1e-6)

# test that predictions are within the range of measured levels
def test_within_measured_range():
    pos, lev, q = make_data(N=12, M=200)
    out = idw(pos, lev, q)
    assert out.max() <= lev.max() + 1e-9 and out.min() >= lev.min() - 1e-9

# test that predictions are closer to the nearest phone than the furthest phone
def test_single_phone():
    pos, lev, q = make_data(T=3, N=1, M=10)
    np.testing.assert_allclose(idw(pos, lev, q), np.repeat(lev, 10, axis=1))
