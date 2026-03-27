"""Tests for GA algebra primitives — round-trip and consistency checks."""

import numpy as np
import pytest
from scipy.linalg import expm, logm

from layer_time_ga.algebra import (
    Bivector,
    Rotor,
    bivector_from_skew,
    skew_from_bivector,
    rotor_from_orthogonal,
    rotor_angle,
    rotor_plane,
    rotor_compose,
    rotor_inverse,
    commutator_bivector,
    grade_decomposition,
    geometric_product_vectors,
)


# ── Helpers ───────────────────────────────────────────────────────

def random_skew(k: int, rng=None) -> np.ndarray:
    """Generate a random skew-symmetric matrix."""
    if rng is None:
        rng = np.random.default_rng(42)
    A = rng.standard_normal((k, k))
    return 0.5 * (A - A.T)


def random_orthogonal(k: int, rng=None) -> np.ndarray:
    """Generate a random orthogonal matrix with det = +1."""
    A = random_skew(k, rng=rng) * 0.3  # small angle to keep logm stable
    U = expm(A)
    if np.linalg.det(U) < 0:
        U[:, 0] *= -1
    return U


# ── Bivector tests ────────────────────────────────────────────────

class TestBivector:
    def test_from_skew_round_trip(self):
        A = random_skew(8)
        B = bivector_from_skew(A)
        A_back = skew_from_bivector(B)
        np.testing.assert_allclose(A, A_back, atol=1e-12)

    def test_skew_symmetry_enforced(self):
        # Non-symmetric input should be projected
        M = np.random.default_rng(0).standard_normal((8, 8))
        B = bivector_from_skew(M)
        np.testing.assert_allclose(B.matrix, -B.matrix.T, atol=1e-12)

    def test_norm(self):
        A = random_skew(8)
        B = bivector_from_skew(A)
        expected = np.linalg.norm(A, "fro")
        assert abs(B.norm - expected) < 1e-12

    def test_n_components(self):
        B = bivector_from_skew(random_skew(10))
        assert B.n_components == 45  # 10*9/2

    def test_components_length(self):
        B = bivector_from_skew(random_skew(6))
        assert len(B.components()) == 15  # 6*5/2

    def test_add_sub(self):
        A1 = random_skew(8)
        A2 = random_skew(8, rng=np.random.default_rng(99))
        B1 = bivector_from_skew(A1)
        B2 = bivector_from_skew(A2)
        B_sum = B1 + B2
        np.testing.assert_allclose(B_sum.matrix, A1 + A2, atol=1e-12)
        B_diff = B1 - B2
        np.testing.assert_allclose(B_diff.matrix, A1 - A2, atol=1e-12)

    def test_scalar_mul(self):
        B = bivector_from_skew(random_skew(8))
        B2 = 3.0 * B
        np.testing.assert_allclose(B2.matrix, 3.0 * B.matrix, atol=1e-12)

    def test_principal_planes(self):
        A = random_skew(8)
        B = bivector_from_skew(A)
        planes = B.principal_planes(n_planes=2)
        assert len(planes) <= 2
        for p in planes:
            assert "angle" in p
            assert "plane_vectors" in p
            assert p["angle"] >= 0


# ── Rotor tests ───────────────────────────────────────────────────

class TestRotor:
    def test_from_orthogonal_round_trip(self):
        """Rotor → matrix → rotor should give same matrix."""
        U = random_orthogonal(8)
        R = rotor_from_orthogonal(U)
        np.testing.assert_allclose(R.matrix, U, atol=1e-8)

    def test_bivector_exponential(self):
        """exp(A) where A is the bivector should recover the orthogonal matrix."""
        U = random_orthogonal(8)
        R = rotor_from_orthogonal(U, compute_bivector=True)
        U_reconstructed = expm(R.bivector.matrix)
        np.testing.assert_allclose(U_reconstructed.real, U, atol=1e-6)

    def test_identity_detection(self):
        R = rotor_from_orthogonal(np.eye(8))
        assert R.is_identity

    def test_non_identity(self):
        U = random_orthogonal(8)
        R = rotor_from_orthogonal(U)
        assert not R.is_identity

    def test_inverse(self):
        U = random_orthogonal(8)
        R = rotor_from_orthogonal(U)
        R_inv = rotor_inverse(R)
        product = R.compose(R_inv)
        np.testing.assert_allclose(product.matrix, np.eye(8), atol=1e-8)

    def test_compose_is_matrix_multiply(self):
        U1 = random_orthogonal(8)
        U2 = random_orthogonal(8, rng=np.random.default_rng(99))
        R1 = rotor_from_orthogonal(U1)
        R2 = rotor_from_orthogonal(U2)
        R12 = rotor_compose(R1, R2)
        np.testing.assert_allclose(R12.matrix, U1 @ U2, atol=1e-8)

    def test_apply_preserves_norm(self):
        U = random_orthogonal(8)
        R = rotor_from_orthogonal(U)
        v = np.random.default_rng(0).standard_normal(8)
        v_rot = R.apply(v)
        assert abs(np.linalg.norm(v_rot) - np.linalg.norm(v)) < 1e-10

    def test_angle(self):
        R = rotor_from_orthogonal(np.eye(8))
        assert R.angle < 1e-6

    def test_plane_extraction(self):
        U = random_orthogonal(8)
        R = rotor_from_orthogonal(U, compute_bivector=True)
        planes = rotor_plane(R, n_planes=2)
        assert isinstance(planes, list)


# ── Commutator tests ──────────────────────────────────────────────

class TestCommutator:
    def test_antisymmetry(self):
        B1 = bivector_from_skew(random_skew(8))
        B2 = bivector_from_skew(random_skew(8, rng=np.random.default_rng(99)))
        c12 = commutator_bivector(B1, B2)
        c21 = commutator_bivector(B2, B1)
        np.testing.assert_allclose(c12.matrix, -c21.matrix, atol=1e-12)

    def test_self_commutator_is_zero(self):
        B = bivector_from_skew(random_skew(8))
        c = commutator_bivector(B, B)
        assert c.norm < 1e-12

    def test_result_is_skew(self):
        B1 = bivector_from_skew(random_skew(8))
        B2 = bivector_from_skew(random_skew(8, rng=np.random.default_rng(99)))
        c = commutator_bivector(B1, B2)
        np.testing.assert_allclose(c.matrix, -c.matrix.T, atol=1e-12)

    def test_jacobi_identity(self):
        """[A, [B, C]] + [B, [C, A]] + [C, [A, B]] = 0."""
        rng = np.random.default_rng(42)
        A = bivector_from_skew(random_skew(6, rng=rng))
        B = bivector_from_skew(random_skew(6, rng=rng))
        C = bivector_from_skew(random_skew(6, rng=rng))
        t1 = commutator_bivector(A, commutator_bivector(B, C))
        t2 = commutator_bivector(B, commutator_bivector(C, A))
        t3 = commutator_bivector(C, commutator_bivector(A, B))
        total = t1 + t2 + t3
        assert total.norm < 1e-10


# ── Grade decomposition ──────────────────────────────────────────

class TestGradeDecomposition:
    def test_symmetric_plus_skew(self):
        M = np.random.default_rng(0).standard_normal((8, 8))
        gd = grade_decomposition(M)
        reconstructed = gd["grade_0"] + gd["grade_2"].matrix
        np.testing.assert_allclose(reconstructed, M, atol=1e-12)

    def test_symmetric_is_symmetric(self):
        M = np.random.default_rng(0).standard_normal((8, 8))
        gd = grade_decomposition(M)
        S = gd["grade_0"]
        np.testing.assert_allclose(S, S.T, atol=1e-12)

    def test_skew_is_skew(self):
        M = np.random.default_rng(0).standard_normal((8, 8))
        gd = grade_decomposition(M)
        A = gd["grade_2"].matrix
        np.testing.assert_allclose(A, -A.T, atol=1e-12)


# ── Geometric product ─────────────────────────────────────────────

class TestGeometricProduct:
    def test_parallel_vectors(self):
        a = np.array([1.0, 0, 0, 0])
        b = np.array([3.0, 0, 0, 0])
        gp = geometric_product_vectors(a, b)
        assert abs(gp["scalar"] - 3.0) < 1e-12
        assert gp["bivector"].norm < 1e-12

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0, 0, 0])
        b = np.array([0, 2.0, 0, 0])
        gp = geometric_product_vectors(a, b)
        assert abs(gp["scalar"]) < 1e-12
        assert abs(gp["bivector"].norm - 2.0 * np.sqrt(2)) < 1e-10

    def test_scalar_is_dot_product(self):
        rng = np.random.default_rng(42)
        a = rng.standard_normal(8)
        b = rng.standard_normal(8)
        gp = geometric_product_vectors(a, b)
        assert abs(gp["scalar"] - np.dot(a, b)) < 1e-10
