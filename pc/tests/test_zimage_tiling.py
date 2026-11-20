"""Z-Image tiling helpers (pure functions, no model or GPU needed)."""
import itertools

import numpy as np
from PIL import Image
from unified import zimage


def test_positions_cover_edges():
    pos = zimage._positions(3000, 1024, 96)
    assert pos[0] == 0
    assert pos[-1] + 1024 == 3000
    assert all(b > a for a, b in itertools.pairwise(pos))


def test_positions_single_tile():
    assert zimage._positions(800, 1024, 96) == [0]


def test_ramp_is_one_in_the_middle():
    w = zimage._ramp(200, 96, left_edge=False, right_edge=False)
    assert abs(w[100] - 1.0) < 1e-6
    assert w[0] < 0.02 and w[-1] < 0.02


def test_ramp_keeps_outer_edges_flat():
    w = zimage._ramp(200, 96, left_edge=True, right_edge=True)
    assert np.allclose(w, 1.0)


def test_ramp_survives_tiles_smaller_than_overlap():
    w = zimage._ramp(32, 96, left_edge=False, right_edge=False)
    assert w.shape == (32,)
    assert w[0] < 0.02 and w[-1] < 0.02


def test_pad_to_multiple():
    img = Image.new("RGB", (100, 70))
    out = zimage._pad_to_multiple(img)
    assert out.size == (112, 80)
    assert zimage._pad_to_multiple(Image.new("RGB", (96, 64))).size == (96, 64)
