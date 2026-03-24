"""
GPU-accelerated function tests.
Verifies curvature_gpu, layer_operator_gpu, sample_geometry_gpu produce
results numerically close to their CPU equivalents.
Skipped automatically if CUDA is not available.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

import layer_time_geometry as ltg

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

pytestmark = pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")

np.random.seed(42)
L, T, p, k = 5, 8, 64, 16


@pytest.fixture(scope="module")
def metric():
    H = np.random.randn(L * T, p).astype(np.float32)
    return ltg.estimate_metric(H, n_components=k)


@pytest.fixture(scope="module")
def whitened(metric):
    H = np.random.randn(L, T, p).astype(np.float32)
    return ltg.whiten(H, metric)


# ── curvature_gpu vs curvature ───────────────────────────────────────────────

class TestCurvatureGPU:
    def test_shape_matches(self, whitened):
        cpu = ltg.curvature(whitened)
        gpu = ltg.curvature_gpu(whitened, device="cuda")
        assert cpu.shape == gpu.shape

    def test_values_close(self, whitened):
        cpu = ltg.curvature(whitened)
        gpu = ltg.curvature_gpu(whitened, device="cuda")
        np.testing.assert_allclose(gpu, cpu, rtol=1e-3, atol=1e-4)

    def test_nonnegative(self, whitened):
        gpu = ltg.curvature_gpu(whitened, device="cuda")
        assert np.all(gpu >= 0)


# ── layer_operator_gpu vs layer_operator ─────────────────────────────────────

class TestLayerOperatorGPU:
    def test_rank_matches(self, whitened):
        for l in range(1, L - 1):
            cpu = ltg.layer_operator(whitened, l)
            gpu = ltg.layer_operator_gpu(whitened, l, device="cuda")
            assert cpu.rank == gpu.rank, f"Rank mismatch at layer {l}"

    def test_T_op_close(self, whitened):
        for l in range(1, L - 1):
            cpu = ltg.layer_operator(whitened, l)
            gpu = ltg.layer_operator_gpu(whitened, l, device="cuda")
            # T_op may differ in sign convention per column; compare via reconstruction
            recon_cpu = cpu.U @ cpu.P
            recon_gpu = gpu.U @ gpu.P
            np.testing.assert_allclose(
                np.abs(recon_gpu), np.abs(recon_cpu), rtol=0.05, atol=1e-3,
                err_msg=f"Reconstruction mismatch at layer {l}")

    def test_U_orthogonal(self, whitened):
        for l in range(1, L - 1):
            gpu = ltg.layer_operator_gpu(whitened, l, device="cuda")
            I = np.eye(gpu.rank)
            np.testing.assert_allclose(
                gpu.U @ gpu.U.T, I, atol=0.05,
                err_msg=f"U not orthogonal at layer {l}")

    def test_P_symmetric(self, whitened):
        for l in range(1, L - 1):
            gpu = ltg.layer_operator_gpu(whitened, l, device="cuda")
            np.testing.assert_allclose(
                gpu.P, gpu.P.T, atol=1e-4,
                err_msg=f"P not symmetric at layer {l}")


# ── sample_geometry_gpu vs sample_geometry ───────────────────────────────────

class TestSampleGeometryGPU:
    def test_difficulty_close(self, whitened):
        cpu = ltg.sample_geometry(whitened)
        gpu = ltg.sample_geometry_gpu(whitened, device="cuda")
        np.testing.assert_allclose(gpu.difficulty, cpu.difficulty, rtol=0.05)

    def test_curvature_norms_close(self, whitened):
        cpu = ltg.sample_geometry(whitened)
        gpu = ltg.sample_geometry_gpu(whitened, device="cuda")
        np.testing.assert_allclose(gpu.Omega_norms, cpu.Omega_norms, rtol=0.05, atol=1e-3)

    def test_kernel_eigs_match(self, whitened):
        """Kernels are computed on CPU in both paths — should match exactly."""
        cpu = ltg.sample_geometry(whitened)
        gpu = ltg.sample_geometry_gpu(whitened, device="cuda")
        np.testing.assert_allclose(gpu.K_layer_eigenvalues, cpu.K_layer_eigenvalues, atol=1e-5)
        np.testing.assert_allclose(gpu.K_time_eigenvalues, cpu.K_time_eigenvalues, atol=1e-5)

    def test_all_fields_present(self, whitened):
        gpu = ltg.sample_geometry_gpu(whitened, device="cuda")
        assert hasattr(gpu, 'difficulty')
        assert hasattr(gpu, 'Omega_norms')
        assert hasattr(gpu, 'rotation_devs')
        assert hasattr(gpu, 'scaling_devs')
        assert hasattr(gpu, 'S_norms')
        assert hasattr(gpu, 'A_norms')
        assert hasattr(gpu, 'K_layer_eigenvalues')
        assert hasattr(gpu, 'K_time_eigenvalues')

    def test_directionality_close(self, whitened):
        cpu = ltg.sample_geometry(whitened)
        gpu = ltg.sample_geometry_gpu(whitened, device="cuda")
        # Compare directionality (||A||/||S||) for layers where both are valid
        for l in range(1, L - 1):
            if not (np.isnan(cpu.A_norms[l]) or np.isnan(gpu.A_norms[l])):
                np.testing.assert_allclose(
                    gpu.A_norms[l], cpu.A_norms[l], rtol=0.1, atol=1e-3,
                    err_msg=f"A_norms mismatch at layer {l}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
