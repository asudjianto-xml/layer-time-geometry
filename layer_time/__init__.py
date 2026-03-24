"""
layer_time — Geometric interpretability for transformer language models.

Provides a high-level API for analyzing how transformers process inputs
using the layer-time geometric framework from Sudjianto & Zhang.

Quick start::

    from layer_time import LayerTimeAnalyzer

    analyzer = LayerTimeAnalyzer("Qwen/Qwen2.5-7B")
    analyzer.fit_metric(["calibration text 1", "calibration text 2"])

    result = analyzer.analyze("The capital of France is")
    print(result.summary())
    analyzer.plot_curvature(result)

    # Generation-time tracking
    gen = analyzer.generate_and_track("The capital of France is", max_new_tokens=10)
    analyzer.plot_generation_trajectory(gen)
"""

__version__ = "0.2.0"

# Ensure layer_time_geometry is importable (handles both pip-installed and dev mode)
try:
    import layer_time_geometry
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from .analyzer import LayerTimeAnalyzer
from .results import AnalysisResult, ComparisonResult, SteeringResult, GenerationResult
from .plotting import (
    plot_curvature_heatmap,
    plot_operator_profile,
    plot_energy_landscape,
    plot_kernel_matrix,
    plot_stretching_field,
    plot_steering_diagnostics,
    plot_curvature_profile,
    plot_directionality_profile,
    plot_generation_trajectory,
    plot_frontier_curvature,
    plot_attention_shift,
)

from layer_time_geometry import (
    MetricStructure,
    SampleGeometry,
    DirectionalRadial,
    OperatorDecomposition,
    SteeringDiagnostics,
    GenerationGeometry,
)

__all__ = [
    # Main class
    "LayerTimeAnalyzer",
    # Result types
    "AnalysisResult",
    "ComparisonResult",
    "SteeringResult",
    "GenerationResult",
    # Backend dataclasses
    "MetricStructure",
    "SampleGeometry",
    "DirectionalRadial",
    "OperatorDecomposition",
    "SteeringDiagnostics",
    "GenerationGeometry",
    # Plotting — static analysis
    "plot_curvature_heatmap",
    "plot_operator_profile",
    "plot_energy_landscape",
    "plot_kernel_matrix",
    "plot_stretching_field",
    "plot_steering_diagnostics",
    "plot_curvature_profile",
    "plot_directionality_profile",
    # Plotting — generation time
    "plot_generation_trajectory",
    "plot_frontier_curvature",
    "plot_attention_shift",
]
