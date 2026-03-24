"""
Tests for the layer_time package API: _compat, results, plotting.
Uses synthetic data — no model loading required.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for CI
import matplotlib.pyplot as plt

import layer_time_geometry as ltg
from layer_time._compat import resolve_device, is_gpu
from layer_time.results import AnalysisResult, ComparisonResult, SteeringResult, GenerationResult
from layer_time import plotting

# ── Fixtures ─────────────────────────────────────────────────────────────────

np.random.seed(123)
L, T, p, k = 5, 8, 64, 16


@pytest.fixture
def metric():
    H = np.random.randn(L * T, p).astype(np.float32)
    return ltg.estimate_metric(H, n_components=k)


@pytest.fixture
def whitened(metric):
    H = np.random.randn(L, T, p).astype(np.float32)
    return ltg.whiten(H, metric)


@pytest.fixture
def sample_geom(whitened):
    return ltg.sample_geometry(whitened)


@pytest.fixture
def direction_energy(whitened):
    return ltg.decompose_direction_energy(whitened)


@pytest.fixture
def analysis_result(whitened, metric, sample_geom, direction_energy):
    return AnalysisResult(
        prompt="test prompt for analysis",
        tokens=["test", " prompt", " for", " analysis", " extra", " tokens", " here", " end"],
        hidden_states=np.random.randn(L, T, p).astype(np.float32),
        whitened=whitened,
        metric=metric,
        geometry=sample_geom,
        direction_energy=direction_energy,
    )


@pytest.fixture
def steering_diag(whitened):
    H_after = whitened + np.random.randn(*whitened.shape) * 0.1
    return ltg.steering_diagnostics(whitened, H_after)


@pytest.fixture
def gen_geometry(metric):
    """Simulate GenerationGeometry with 3 steps."""
    steps_raw = [np.random.randn(L, T + i, p).astype(np.float32) for i in range(3)]
    gg = ltg.generation_geometry(steps_raw, metric, device="cpu")
    gg.token_ids = list(range(T + 2))
    gg.token_strings = [f"tok{i}" for i in range(T + 2)]
    return gg


@pytest.fixture
def gen_result(analysis_result, gen_geometry):
    """Simulate GenerationResult."""
    step_results = [analysis_result] * 3
    return GenerationResult(
        prompt="test prompt",
        generated_text="test prompt generated",
        token_strings=["test", " prompt", " gen", " erated"] + [f"tok{i}" for i in range(6)],
        token_ids=list(range(10)),
        prompt_length=2,
        trajectory=gen_geometry,
        step_results=step_results,
    )


# ── _compat tests ───────────────────────────────────────────────────────────

class TestCompat:
    def test_resolve_device_none(self):
        d = resolve_device(None)
        assert d in ("cuda", "cpu")

    def test_resolve_device_auto(self):
        d = resolve_device("auto")
        assert d in ("cuda", "cpu")

    def test_resolve_device_explicit(self):
        assert resolve_device("cpu") == "cpu"
        assert resolve_device("cuda:0") == "cuda:0"

    def test_is_gpu_true(self):
        assert is_gpu("cuda") is True
        assert is_gpu("cuda:0") is True

    def test_is_gpu_false(self):
        assert is_gpu("cpu") is False


# ── AnalysisResult tests ────────────────────────────────────────────────────

class TestAnalysisResult:
    def test_n_layers(self, analysis_result):
        assert analysis_result.n_layers == L

    def test_n_tokens(self, analysis_result):
        assert analysis_result.n_tokens == T

    def test_curvature_map(self, analysis_result):
        cm = analysis_result.curvature_map
        assert cm.shape == (L - 1, T - 1)

    def test_feature_vector(self, analysis_result):
        fv = analysis_result.feature_vector(n_eigs=5)
        assert fv.ndim == 1
        assert len(fv) > 0
        assert not np.any(np.isnan(fv))

    def test_summary_dict(self, analysis_result):
        s = analysis_result.summary()
        assert isinstance(s, dict)
        assert "prompt" in s
        assert "difficulty" in s
        assert "directionality" in s
        assert "n_layers" in s
        assert s["n_layers"] == L
        assert s["n_tokens"] == T
        assert isinstance(s["difficulty"], float)
        assert isinstance(s["lyapunov_max"], float)


# ── ComparisonResult tests ───────────────────────────────────────────────────

class TestComparisonResult:
    def test_labels(self, analysis_result):
        comp = ComparisonResult(
            results=[analysis_result, analysis_result],
            kernel=np.eye(2),
        )
        labels = comp.labels
        assert len(labels) == 2
        assert all(isinstance(l, str) for l in labels)

    def test_summaries(self, analysis_result):
        comp = ComparisonResult(
            results=[analysis_result, analysis_result],
            kernel=np.eye(2),
        )
        sums = comp.summaries()
        assert len(sums) == 2
        assert all(isinstance(s, dict) for s in sums)


# ── GenerationResult tests ───────────────────────────────────────────────────

class TestGenerationResult:
    def test_n_steps(self, gen_result):
        assert gen_result.n_steps == 3

    def test_generated_tokens(self, gen_result):
        gt = gen_result.generated_tokens
        assert isinstance(gt, list)
        assert len(gt) == len(gen_result.token_strings) - gen_result.prompt_length

    def test_difficulties(self, gen_result):
        d = gen_result.difficulties
        assert d.shape == (3,)
        assert np.all(d >= 0)

    def test_directionalities(self, gen_result):
        d = gen_result.directionalities
        assert d.shape == (3,)

    def test_drift(self, gen_result):
        d = gen_result.drift
        assert d.shape == (2,)  # n_steps - 1
        assert np.all(d >= 0)

    def test_frontier_curvature(self, gen_result):
        fc = gen_result.frontier_curvature()
        assert fc.shape[0] == 3  # n_steps
        assert fc.shape[1] == L - 1

    def test_attention_shift(self, gen_result):
        ashift = gen_result.attention_shift()
        assert ashift.shape[0] == 3
        assert ashift.shape[1] == L

    def test_summary_trajectory(self, gen_result):
        traj = gen_result.summary_trajectory()
        assert len(traj) == 3
        assert all("generation_step" in row for row in traj)
        assert all("n_tokens_so_far" in row for row in traj)
        assert traj[0]["generation_step"] == 0
        assert traj[1]["generation_step"] == 1


# ── Plotting tests ───────────────────────────────────────────────────────────

class TestPlotting:
    """Verify all plot functions execute without error and return axes."""

    def test_plot_curvature_heatmap(self, sample_geom):
        ax = plotting.plot_curvature_heatmap(sample_geom.Omega_norms)
        assert ax is not None
        plt.close('all')

    def test_plot_curvature_heatmap_with_tokens(self, sample_geom):
        tokens = [f"t{i}" for i in range(T)]
        ax = plotting.plot_curvature_heatmap(sample_geom.Omega_norms, tokens=tokens)
        assert ax is not None
        plt.close('all')

    def test_plot_operator_profile(self, sample_geom):
        ax = plotting.plot_operator_profile(sample_geom)
        assert ax is not None
        plt.close('all')

    def test_plot_energy_landscape(self, direction_energy):
        ax = plotting.plot_energy_landscape(direction_energy)
        assert ax is not None
        plt.close('all')

    def test_plot_kernel_matrix(self):
        K = np.random.randn(3, 3)
        K = K @ K.T  # PSD
        ax = plotting.plot_kernel_matrix(K, labels=["a", "b", "c"])
        assert ax is not None
        plt.close('all')

    def test_plot_stretching_field(self, sample_geom):
        S = np.random.rand(L - 1, T)
        ax = plotting.plot_stretching_field(S)
        assert ax is not None
        plt.close('all')

    def test_plot_steering_diagnostics(self, steering_diag):
        axes = plotting.plot_steering_diagnostics(steering_diag)
        assert len(axes) == 3
        plt.close('all')

    def test_plot_curvature_profile(self, sample_geom):
        ax = plotting.plot_curvature_profile(sample_geom.Omega_norms)
        assert ax is not None
        plt.close('all')

    def test_plot_directionality_profile(self, sample_geom):
        ax = plotting.plot_directionality_profile(sample_geom)
        assert ax is not None
        plt.close('all')

    def test_plot_generation_trajectory(self, gen_result):
        axes = plotting.plot_generation_trajectory(gen_result)
        assert len(axes) == 4
        plt.close('all')

    def test_plot_frontier_curvature(self, gen_result):
        ax = plotting.plot_frontier_curvature(gen_result)
        assert ax is not None
        plt.close('all')

    def test_plot_attention_shift(self, gen_result):
        ax = plotting.plot_attention_shift(gen_result)
        assert ax is not None
        plt.close('all')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
