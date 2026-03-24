"""
Thorough tests for layer_time_geometry.py backend functions using synthetic data.
No model loading required — all tests use numpy/torch synthetic hidden states.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import pytest

import layer_time_geometry as ltg

# ── Fixtures ─────────────────────────────────────────────────────────────────

np.random.seed(42)

# Simulated dimensions: 5 layers, 8 tokens, 64 hidden dim
L, T, p = 5, 8, 64
k = 16  # whitened dim

@pytest.fixture
def raw_hidden():
    """Simulate raw hidden states (L, T, p) with realistic structure."""
    # Base + layer-dependent drift + noise
    base = np.random.randn(1, 1, p) * 5
    layer_drift = np.cumsum(np.random.randn(L, 1, p) * 0.3, axis=0)
    token_drift = np.cumsum(np.random.randn(1, T, p) * 0.1, axis=1)
    noise = np.random.randn(L, T, p) * 0.5
    return (base + layer_drift + token_drift + noise).astype(np.float32)

@pytest.fixture
def metric(raw_hidden):
    """Fitted MetricStructure from raw hidden states."""
    H_flat = raw_hidden.reshape(-1, p)
    return ltg.estimate_metric(H_flat, n_components=k)

@pytest.fixture
def whitened(raw_hidden, metric):
    """Whitened hidden states (L, T, k)."""
    return ltg.whiten(raw_hidden, metric)


# ── Section 3: Metric and Whitening ──────────────────────────────────────────

class TestMetric:
    def test_estimate_metric_shapes(self, metric):
        assert metric.mean.shape == (p,)
        assert metric.V_k.shape == (p, k)
        assert metric.eigvals_k.shape == (k,)
        assert metric.W.shape == (p, k)
        assert metric.k == k

    def test_explained_variance_positive(self, metric):
        assert 0 < metric.explained_var <= 1.0

    def test_eigvals_descending(self, metric):
        assert np.all(np.diff(metric.eigvals_k) <= 1e-10)

    def test_n_components_capped(self):
        """k should be capped at min(n_components, N-1, p)."""
        H = np.random.randn(3, p)  # only 3 samples
        m = ltg.estimate_metric(H, n_components=256)
        assert m.k == 2  # min(256, 3-1, 64) = 2


class TestWhiten:
    def test_whiten_shape(self, whitened):
        assert whitened.shape == (L, T, k)

    def test_whiten_approximately_isotropic(self, raw_hidden, metric):
        """Whitened data should have approximately unit covariance."""
        H_flat = raw_hidden.reshape(-1, p)
        W_flat = ltg.whiten(H_flat, metric)
        cov = np.cov(W_flat, rowvar=False)
        # Diagonal should be close to 1
        np.testing.assert_allclose(np.diag(cov), 1.0, atol=0.3)

    def test_whiten_centered(self, raw_hidden, metric):
        H_flat = raw_hidden.reshape(-1, p)
        W_flat = ltg.whiten(H_flat, metric)
        np.testing.assert_allclose(W_flat.mean(axis=0), 0.0, atol=0.1)


# ── Section 4: Directional-Radial Decomposition ─────────────────────────────

class TestDirectionalRadial:
    def test_decomposition_shapes(self, whitened):
        dr = ltg.decompose_direction_energy(whitened)
        assert dr.H_hat.shape == (L, T, k)
        assert dr.r.shape == (L, T)
        assert dr.u.shape == (L, T)

    def test_unit_directions(self, whitened):
        dr = ltg.decompose_direction_energy(whitened)
        norms = np.linalg.norm(dr.H_hat, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_reconstruction(self, whitened):
        """r * H_hat should reconstruct H_tilde."""
        dr = ltg.decompose_direction_energy(whitened)
        reconstructed = dr.r[..., np.newaxis] * dr.H_hat
        np.testing.assert_allclose(reconstructed, whitened, atol=1e-5)

    def test_log_radius(self, whitened):
        dr = ltg.decompose_direction_energy(whitened)
        np.testing.assert_allclose(np.exp(dr.u), dr.r, atol=1e-5)


# ── Section 5: Kernels ──────────────────────────────────────────────────────

class TestKernels:
    def test_layer_kernel_shape(self, whitened):
        K = ltg.layer_kernel(whitened, t=0)
        assert K.shape == (L, L)

    def test_layer_kernel_symmetric(self, whitened):
        K = ltg.layer_kernel(whitened, t=0)
        np.testing.assert_allclose(K, K.T, atol=1e-6)

    def test_layer_kernel_psd(self, whitened):
        K = ltg.layer_kernel(whitened, t=0)
        eigvals = np.linalg.eigvalsh(K)
        assert np.all(eigvals >= -1e-8)

    def test_temporal_kernel_shape(self, whitened):
        K = ltg.temporal_kernel(whitened, l=0)
        assert K.shape == (T, T)

    def test_temporal_kernel_symmetric(self, whitened):
        K = ltg.temporal_kernel(whitened, l=0)
        np.testing.assert_allclose(K, K.T, atol=1e-6)

    def test_spatiotemporal_kernel_shape(self, whitened):
        K = ltg.spatiotemporal_kernel(whitened)
        assert K.shape == (L * T, L * T)

    def test_diffusion_operator_row_stochastic(self, whitened):
        K = ltg.layer_kernel(whitened, t=0)
        P = ltg.diffusion_operator(K)
        row_sums = P.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


# ── Section 6-7: Interaction Operators and S+A ───────────────────────────────

class TestInteraction:
    def test_temporal_interaction_shape(self, whitened):
        M = ltg.temporal_interaction(whitened, t=0)
        assert M.shape == (L, L)

    def test_layer_interaction_shape(self, whitened):
        M = ltg.layer_interaction(whitened, l=0)
        assert M.shape == (T, T)

    def test_sa_decomposition_sum(self, whitened):
        M = ltg.layer_interaction(whitened, l=0)
        S, A = ltg.symmetric_antisymmetric(M)
        np.testing.assert_allclose(S + A, M, atol=1e-5)

    def test_symmetric_is_symmetric(self, whitened):
        M = ltg.layer_interaction(whitened, l=0)
        S, _ = ltg.symmetric_antisymmetric(M)
        np.testing.assert_allclose(S, S.T, atol=1e-10)

    def test_antisymmetric_is_antisymmetric(self, whitened):
        M = ltg.layer_interaction(whitened, l=0)
        _, A = ltg.symmetric_antisymmetric(M)
        np.testing.assert_allclose(A, -A.T, atol=1e-10)


# ── Section 8: Operator Decomposition ────────────────────────────────────────

class TestOperatorDecomposition:
    def test_layer_operator_fields(self, whitened):
        op = ltg.layer_operator(whitened, l=1)
        assert op.T_op.shape[0] == op.T_op.shape[1]  # square
        assert op.U.shape == op.T_op.shape
        assert op.P.shape == op.T_op.shape
        assert op.V.shape[0] == k
        assert op.V.shape[1] == op.rank
        assert op.rank >= 1
        assert op.singular_values is not None

    def test_polar_reconstruction(self, whitened):
        """T = U @ P"""
        op = ltg.layer_operator(whitened, l=1)
        reconstructed = op.U @ op.P
        np.testing.assert_allclose(reconstructed, op.T_op, atol=1e-4)

    def test_U_orthogonal(self, whitened):
        op = ltg.layer_operator(whitened, l=1)
        I = np.eye(op.rank)
        np.testing.assert_allclose(op.U @ op.U.T, I, atol=1e-4)

    def test_P_symmetric(self, whitened):
        op = ltg.layer_operator(whitened, l=1)
        np.testing.assert_allclose(op.P, op.P.T, atol=1e-4)


# ── Section 9: Discrete Flow ────────────────────────────────────────────────

class TestDiscreteFlow:
    def test_delta_layer_shape(self, whitened):
        dH = ltg.delta_layer(whitened)
        assert dH.shape == (L - 1, T, k)

    def test_delta_time_shape(self, whitened):
        dH = ltg.delta_time(whitened)
        assert dH.shape == (L, T - 1, k)

    def test_delta_layer_correctness(self, whitened):
        dH = ltg.delta_layer(whitened)
        expected = whitened[1:] - whitened[:-1]
        np.testing.assert_allclose(dH, expected, atol=1e-10)

    def test_delta_time_correctness(self, whitened):
        dH = ltg.delta_time(whitened)
        expected = whitened[:, 1:] - whitened[:, :-1]
        np.testing.assert_allclose(dH, expected, atol=1e-10)


# ── Section 10: Curvature ───────────────────────────────────────────────────

class TestCurvature:
    def test_curvature_shape(self, whitened):
        Omega = ltg.curvature(whitened)
        assert Omega.shape == (L - 1, T - 1)

    def test_curvature_nonnegative(self, whitened):
        Omega = ltg.curvature(whitened)
        assert np.all(Omega >= 0)

    def test_curvature_norm_passthrough(self, whitened):
        Omega = ltg.curvature(whitened)
        Omega2 = ltg.curvature_norm(Omega)
        np.testing.assert_array_equal(Omega, Omega2)

    def test_constant_field_low_curvature(self):
        """If H is a constant field (all vectors identical), curvature should be ~0."""
        c = np.random.randn(k).astype(np.float32)
        H_const = np.tile(c, (L, T, 1))
        H_const += np.random.randn(L, T, k).astype(np.float32) * 1e-6
        Omega = ltg.curvature(H_const)
        assert Omega.max() < 0.1


# ── Section 11: Geometric Algebra ────────────────────────────────────────────

class TestGeometricAlgebra:
    def test_bivector_antisymmetric(self, whitened):
        dr = ltg.decompose_direction_energy(whitened)
        B = ltg.bivector(dr.H_hat, l=0, t=0)
        assert B.shape == (k, k)
        np.testing.assert_allclose(B, -B.T, atol=1e-10)

    def test_skew_generator_antisymmetric(self, whitened):
        op = ltg.layer_operator(whitened, l=1)
        A = ltg.skew_generator(op.U)
        np.testing.assert_allclose(A, -A.T, atol=1e-4)

    def test_bivector_field_shape(self, whitened):
        dr = ltg.decompose_direction_energy(whitened)
        B = ltg.bivector_field(dr.H_hat, l=0)
        assert B.shape == (k, k)


# ── Section 12-13: Sample Geometry ───────────────────────────────────────────

class TestSampleGeometry:
    def test_sample_geometry_fields(self, whitened):
        sg = ltg.sample_geometry(whitened)
        assert sg.Omega_norms.shape == (L - 1, T - 1)
        assert isinstance(sg.difficulty, float)
        assert isinstance(sg.difficulty_total, float)
        assert isinstance(sg.directionality, float)
        assert sg.rotation_devs.shape == (L - 1,)
        assert sg.scaling_devs.shape == (L - 1,)
        assert sg.operator_ranks.shape == (L - 1,)
        assert sg.S_norms.shape == (L - 1,)
        assert sg.A_norms.shape == (L - 1,)

    def test_difficulty_consistent(self, whitened):
        sg = ltg.sample_geometry(whitened)
        expected = sg.Omega_norms.sum() / sg.Omega_norms.size
        np.testing.assert_allclose(sg.difficulty, expected, atol=1e-6)

    def test_length_robust_metrics(self, whitened):
        sg = ltg.sample_geometry(whitened)
        assert 0 <= sg.curv_concentration <= 1
        assert 0 <= sg.curv_peak_layer < L - 1
        assert sg.curv_entropy >= 0
        assert sg.R_windowed is not None

    def test_metric_side_diagnostics(self, whitened):
        sg = ltg.sample_geometry(whitened)
        assert sg.condition_numbers is not None
        assert sg.eranks is not None
        assert sg.stretching_field is not None
        assert sg.stretching_field.shape == (L - 1, T)
        assert isinstance(sg.lyapunov_max, float)
        assert isinstance(sg.stretch_concentration, float)

    def test_feature_vector_fixed_length(self, whitened):
        sg = ltg.sample_geometry(whitened)
        phi = ltg.sample_feature_vector(sg, n_eigs=5)
        assert phi.ndim == 1
        assert len(phi) > 0
        # Different T should produce same length
        H2 = whitened[:, :4, :]  # fewer tokens
        sg2 = ltg.sample_geometry(H2)
        phi2 = ltg.sample_feature_vector(sg2, n_eigs=5)
        assert len(phi) == len(phi2)  # length-invariant!

    def test_prompt_kernel_shape(self, whitened):
        sg1 = ltg.sample_geometry(whitened)
        sg2 = ltg.sample_geometry(whitened[:, :4, :])
        K = ltg.prompt_kernel([sg1, sg2], n_eigs=5)
        assert K.shape == (2, 2)
        # Symmetric
        np.testing.assert_allclose(K, K.T, atol=1e-6)
        # PSD
        eigvals = np.linalg.eigvalsh(K)
        assert np.all(eigvals >= -1e-8)


# ── Section 15: Steering Diagnostics ─────────────────────────────────────────

class TestSteeringDiagnostics:
    def test_steering_diagnostics_shapes(self, whitened):
        # Simulate a perturbation
        H_after = whitened + np.random.randn(*whitened.shape) * 0.1
        diag = ltg.steering_diagnostics(whitened, H_after)
        assert diag.angular_ratio.shape == (L, T)
        assert diag.radial_ratio.shape == (L, T)
        assert diag.delta_Omega_norms.shape == (L - 1, T - 1)
        assert isinstance(diag.delta_S_norm, float)
        assert isinstance(diag.delta_A_norm, float)
        assert isinstance(diag.R_before, float)
        assert isinstance(diag.R_after, float)

    def test_angular_radial_sum_to_one(self, whitened):
        H_after = whitened + np.random.randn(*whitened.shape) * 0.1
        diag = ltg.steering_diagnostics(whitened, H_after)
        # angular^2 + radial^2 ≈ 1 (Pythagorean)
        sum_sq = diag.angular_ratio**2 + diag.radial_ratio**2
        np.testing.assert_allclose(sum_sq, 1.0, atol=0.1)

    def test_zero_perturbation(self, whitened):
        diag = ltg.steering_diagnostics(whitened, whitened)
        np.testing.assert_allclose(diag.delta_Omega_norms, 0.0, atol=1e-6)
        np.testing.assert_allclose(diag.delta_S_norm, 0.0, atol=1e-6)
        np.testing.assert_allclose(diag.delta_A_norm, 0.0, atol=1e-6)


# ── Section 16: Generation-Time Analysis ─────────────────────────────────────

class TestGenerationGeometry:
    @pytest.fixture
    def gen_trajectory(self, metric):
        """Simulate 4 generation steps with growing T."""
        steps = []
        for n in range(4):
            T_n = T + n
            H = np.random.randn(L, T_n, p).astype(np.float32)
            steps.append(H)
        return steps

    def test_generation_geometry_shapes(self, gen_trajectory, metric):
        gg = ltg.generation_geometry(gen_trajectory, metric, device="cpu")
        assert len(gg.steps) == 4
        assert gg.difficulties.shape == (4,)
        assert gg.directionalities.shape == (4,)
        assert gg.lyapunov_exponents.shape == (4,)
        assert gg.curv_concentrations.shape == (4,)
        assert gg.last_token_norms.shape[0] == 4
        assert gg.last_token_drift.shape == (3,)  # n_steps - 1

    def test_generation_geometry_nonneg_difficulties(self, gen_trajectory, metric):
        gg = ltg.generation_geometry(gen_trajectory, metric, device="cpu")
        assert np.all(gg.difficulties >= 0)

    def test_drift_nonneg(self, gen_trajectory, metric):
        gg = ltg.generation_geometry(gen_trajectory, metric, device="cpu")
        assert np.all(gg.last_token_drift >= 0)

    def test_curvature_evolution_shape(self, gen_trajectory, metric):
        gg = ltg.generation_geometry(gen_trajectory, metric, device="cpu")
        curv = ltg.generation_curvature_evolution(gg)
        assert curv.shape == (4, L - 1)

    def test_attention_shift_shape(self, gen_trajectory, metric):
        gg = ltg.generation_geometry(gen_trajectory, metric, device="cpu")
        eranks = ltg.generation_attention_shift(gg)
        assert eranks.shape == (4, L)
        assert np.all(eranks >= 1.0)  # erank >= 1


# ── Helper function tests ───────────────────────────────────────────────────

class TestHelpers:
    def test_curv_concentration_range(self, whitened):
        Omega = ltg.curvature(whitened)
        cc = ltg._curv_concentration(Omega)
        assert 0 <= cc <= 1

    def test_curv_peak_layer_range(self, whitened):
        Omega = ltg.curvature(whitened)
        pl = ltg._curv_peak_layer(Omega)
        assert 0 <= pl < L - 1

    def test_curv_entropy_nonneg(self, whitened):
        Omega = ltg.curvature(whitened)
        ce = ltg._curv_entropy(Omega)
        assert ce >= 0

    def test_condition_number(self):
        sv = np.array([10.0, 5.0, 1.0])
        kappa = ltg._condition_number(sv)
        np.testing.assert_allclose(kappa, 10.0, atol=1e-6)

    def test_erank(self):
        sv = np.array([1.0, 1.0, 1.0])  # uniform → erank = 3
        er = ltg._erank(sv)
        np.testing.assert_allclose(er, 3.0, atol=1e-6)

    def test_stretching_field_shape(self, whitened):
        dr = ltg.decompose_direction_energy(whitened)
        S = ltg._stretching_field(dr.r)
        assert S.shape == (L - 1, T)
        assert np.all(S >= 0)

    def test_lyapunov_max(self):
        sv = np.array([2.0, 2.0, 2.0])
        lm = ltg._lyapunov_max(sv)
        np.testing.assert_allclose(lm, np.log(2.0), atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
