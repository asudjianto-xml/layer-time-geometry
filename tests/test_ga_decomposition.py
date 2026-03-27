"""Tests for GA decomposition and curvature — synthetic data, no GPU needed."""

import numpy as np
import pytest
from scipy.linalg import expm

from layer_time_ga.algebra import Bivector, bivector_from_skew, rotor_from_orthogonal
from layer_time_ga.decomposition import (
    versor_decompose,
    extract_rotor_field,
    extract_bivector_field,
    extract_metric_field,
)
from layer_time_ga.curvature import (
    holonomy_rotor,
    holonomy_scalar_map,
    commutator_field,
    commutator_plane_decomposition,
)
from layer_time_ga.capacity import ga_capacity_profile


def make_synthetic_hidden_states(L: int = 6, T: int = 4, k: int = 16,
                                 seed: int = 42) -> np.ndarray:
    """Create synthetic whitened hidden states with structure.

    Simulates a simple model: each layer rotates and stretches the
    previous layer's representation, with small random noise.
    """
    rng = np.random.default_rng(seed)
    H = np.zeros((L, T, k))

    # Layer 0: random
    H[0] = rng.standard_normal((T, k))

    # Each subsequent layer applies a rotation + stretch + noise
    for l in range(1, L):
        # Random rotation (small angle)
        A = rng.standard_normal((k, k)) * 0.1
        A = 0.5 * (A - A.T)
        R = expm(A)
        # Small stretch
        stretch = 1.0 + 0.05 * rng.standard_normal(k)
        for t in range(T):
            H[l, t] = R @ (stretch * H[l - 1, t]) + 0.01 * rng.standard_normal(k)

    return H


class TestVersorDecomposition:
    def setup_method(self):
        self.H = make_synthetic_hidden_states()

    def test_basic_decomposition(self):
        vd = versor_decompose(self.H, l=1)
        assert vd.rotor is not None
        assert vd.metric is not None
        assert vd.bivector is not None
        assert vd.layer_index == 1

    def test_rotor_is_orthogonal(self):
        vd = versor_decompose(self.H, l=2)
        U = vd.rotor.matrix
        np.testing.assert_allclose(U @ U.T, np.eye(U.shape[0]), atol=1e-6)

    def test_metric_is_symmetric(self):
        vd = versor_decompose(self.H, l=2)
        P = vd.metric
        np.testing.assert_allclose(P, P.T, atol=1e-6)

    def test_condition_number_positive(self):
        vd = versor_decompose(self.H, l=2)
        assert vd.condition_number >= 1.0

    def test_effective_rank_bounded(self):
        vd = versor_decompose(self.H, l=2)
        assert 0 < vd.effective_rank <= vd.rank

    def test_bivector_is_skew(self):
        vd = versor_decompose(self.H, l=2)
        B = vd.bivector.matrix
        np.testing.assert_allclose(B, -B.T, atol=1e-6)


class TestRotorField:
    def setup_method(self):
        self.H = make_synthetic_hidden_states()

    def test_field_length(self):
        rf = extract_rotor_field(self.H, skip_first=True)
        # L=6, skip first: transitions 1->2, 2->3, 3->4, 4->5 = 4
        assert len(rf.rotors) == 4

    def test_field_length_no_skip(self):
        rf = extract_rotor_field(self.H, skip_first=False)
        # L=6: transitions 0->1, 1->2, ..., 4->5 = 5
        assert len(rf.rotors) == 5

    def test_angles_array(self):
        rf = extract_rotor_field(self.H, skip_first=True)
        assert rf.angles.shape == (4,)
        assert np.all(rf.angles >= 0)

    def test_condition_numbers_array(self):
        rf = extract_rotor_field(self.H, skip_first=True)
        assert rf.condition_numbers.shape == (4,)
        assert np.all(rf.condition_numbers >= 1.0)

    def test_bivector_field_convenience(self):
        bivs = extract_bivector_field(self.H, skip_first=True)
        assert len(bivs) == 4
        for b in bivs:
            assert isinstance(b, Bivector)

    def test_metric_field(self):
        mf = extract_metric_field(self.H, skip_first=True)
        assert len(mf) == 4
        for m in mf:
            assert "P" in m
            assert "condition_number" in m
            assert m["condition_number"] >= 1.0


class TestHolonomy:
    def setup_method(self):
        self.H = make_synthetic_hidden_states()

    def test_single_plaquette(self):
        hr = holonomy_rotor(self.H, l=1, t=0)
        assert hr.scalar_curvature >= 0
        assert hr.layer == 1
        assert hr.token == 0

    def test_holonomy_bivector_is_skew(self):
        hr = holonomy_rotor(self.H, l=1, t=0)
        if hr.bivector is not None:
            B = hr.bivector.matrix
            np.testing.assert_allclose(B, -B.T, atol=1e-6)

    def test_scalar_map_shape(self):
        L, T, k = self.H.shape
        curv = holonomy_scalar_map(self.H)
        assert curv.shape == (L - 1, T - 1)

    def test_flat_space_low_curvature(self):
        """If all layers are identical, curvature should be near zero."""
        H_flat = np.tile(self.H[0:1], (6, 1, 1))  # repeat layer 0
        curv = holonomy_scalar_map(H_flat)
        assert curv.max() < 1e-6


class TestCommutatorField:
    def setup_method(self):
        self.H = make_synthetic_hidden_states()
        self.bivs = extract_bivector_field(self.H, skip_first=True)

    def test_commutator_norms_shape(self):
        norms = commutator_field(self.bivs)
        n = len(self.bivs)
        assert norms.shape == (n, n)

    def test_commutator_symmetric(self):
        norms = commutator_field(self.bivs)
        np.testing.assert_allclose(norms, norms.T, atol=1e-12)

    def test_diagonal_zero(self):
        norms = commutator_field(self.bivs)
        np.testing.assert_allclose(np.diag(norms), 0.0, atol=1e-12)

    def test_plane_decomposition(self):
        info = commutator_plane_decomposition(self.bivs, n_planes=2)
        assert "principal_planes" in info
        assert "total_norm" in info
        assert info["total_norm"] >= 0


class TestGACapacity:
    def setup_method(self):
        self.H = make_synthetic_hidden_states()

    def test_basic_capacity(self):
        cap = ga_capacity_profile(self.H)
        assert cap.C_acc >= 0
        assert cap.C_eff == 0.0  # no dependency provided

    def test_capacity_with_dependency(self):
        L = self.H.shape[0]
        D_layer = np.random.default_rng(42).exponential(size=L)
        cap = ga_capacity_profile(self.H, D_layer=D_layer)
        assert cap.C_acc >= 0
        assert cap.C_eff >= 0
        assert 0 <= cap.cconc <= 1.0

    def test_layer_contributions(self):
        cap = ga_capacity_profile(self.H)
        assert len(cap.layer_contributions) == len(cap.bivectors)

    def test_principal_planes(self):
        cap = ga_capacity_profile(self.H, n_planes=2)
        assert isinstance(cap.principal_planes, list)
