"""
Versor decomposition of layer transitions.

Wraps the existing polar decomposition (T = UP) from the backend and
re-expresses it in GA terms:

    T^(l) = R^(l) * P^(l)

where R^(l) is a **rotor** (even-grade Clifford element, stored as an
orthogonal matrix) and P^(l) is the **metric deformation** (symmetric
positive-definite matrix whose eigenvalues give the stretch spectrum).

The rotor carries a **bivector generator** B^(l) = log(R^(l)), which
encodes the *plane* and *angle* of rotation at layer l.

The collection {R^(0), R^(1), ..., R^(L-2)} forms the **rotor field**
on the layer axis — a discrete gauge connection whose holonomy is
the curvature.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import sys
from pathlib import Path

# Ensure backend is importable
try:
    import layer_time_geometry as backend
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import layer_time_geometry as backend

from .algebra import Bivector, Rotor, rotor_from_orthogonal, bivector_from_skew


# ── Versor decomposition of a single layer ────────────────────────

@dataclass
class VersorDecomposition:
    """GA-flavored polar decomposition of a layer transition.

    Attributes:
        rotor: The rotation part R (Rotor, wrapping orthogonal matrix U).
        metric: The stretch part P (symmetric PD matrix).
        bivector: The bivector generator B = log(R) of the rotation.
        singular_values: Eigenvalues of P (= singular values of T).
        condition_number: σ_max / σ_min of P.
        effective_rank: exp(entropy of normalized singular values).
        rank: Dimension of the token subspace in which T was estimated.
        layer_index: Which layer transition this represents.
    """
    rotor: Rotor
    metric: np.ndarray              # (r, r) symmetric PD
    bivector: Bivector
    singular_values: np.ndarray     # (r,)
    condition_number: float
    effective_rank: float
    rank: int
    layer_index: int


def _effective_rank(sv: np.ndarray) -> float:
    """Effective rank from singular value spectrum."""
    sv_pos = sv[sv > 1e-12]
    if len(sv_pos) == 0:
        return 0.0
    p = sv_pos / sv_pos.sum()
    entropy = -np.sum(p * np.log(p + 1e-30))
    return float(np.exp(entropy))


def versor_decompose(H_tilde: np.ndarray, l: int,
                     rank_thresh: float = 0.01,
                     compute_bivector: bool = True) -> VersorDecomposition:
    """Compute the versor (GA polar) decomposition of layer transition l.

    Delegates to backend.layer_operator() for the actual computation,
    then wraps the result in GA objects.

    Args:
        H_tilde: (L, T, k) whitened hidden states.
        l: Layer index (transition from l to l+1).
        rank_thresh: Singular value threshold for rank truncation.
        compute_bivector: Whether to compute the bivector generator (involves logm).

    Returns:
        VersorDecomposition for the transition l → l+1.
    """
    op = backend.layer_operator(H_tilde, l, rank_thresh=rank_thresh)
    k = H_tilde.shape[2]  # full whitened dimension

    # Wrap the orthogonal factor as a Rotor
    rotor = rotor_from_orthogonal(op.U, compute_bivector=compute_bivector)
    bivector_sub = rotor.bivector if rotor.bivector is not None else Bivector(
        matrix=np.zeros_like(op.U), dim=op.U.shape[0]
    )

    # Lift bivector from r-dimensional subspace to full k-dimensional
    # whitened space: B_full = V @ B_sub @ V^T.  This ensures all
    # bivectors live in the same k x k space so that commutators,
    # holonomy, and other cross-layer operations are well-defined
    # regardless of per-layer rank differences.
    B_sub = bivector_sub.matrix
    B_full = op.V @ B_sub @ op.V.T
    bivector = Bivector(matrix=B_full, dim=k)

    sv = op.singular_values if op.singular_values is not None else np.linalg.svd(
        op.P, compute_uv=False
    )
    kappa = float(sv[0] / (sv[-1] + 1e-12)) if len(sv) > 0 else 1.0
    erank = _effective_rank(sv)

    return VersorDecomposition(
        rotor=rotor,
        metric=op.P,
        bivector=bivector,
        singular_values=sv,
        condition_number=kappa,
        effective_rank=erank,
        rank=op.rank,
        layer_index=l,
    )


# ── Layer rotor field ─────────────────────────────────────────────

@dataclass
class LayerRotorField:
    """The rotor field across all layer transitions.

    This is the discrete gauge connection on the layer axis.
    Composing rotors around a closed loop gives the holonomy (curvature).

    Attributes:
        rotors: List of Rotor objects, one per layer transition.
        bivectors: List of Bivector generators, one per transition.
        angles: (n_transitions,) rotation angle at each layer.
        condition_numbers: (n_transitions,) condition number of P at each layer.
        effective_ranks: (n_transitions,) effective rank of P at each layer.
        metrics: List of P matrices (symmetric PD), one per transition.
        decompositions: Full VersorDecomposition list if retained.
    """
    rotors: list[Rotor]
    bivectors: list[Bivector]
    angles: np.ndarray
    condition_numbers: np.ndarray
    effective_ranks: np.ndarray
    metrics: list[np.ndarray]
    decompositions: list[VersorDecomposition]


def extract_rotor_field(H_tilde: np.ndarray,
                        skip_first: bool = True,
                        rank_thresh: float = 0.01) -> LayerRotorField:
    """Extract the full rotor field from whitened hidden states.

    Computes the versor decomposition at every layer transition and
    collects the rotors, bivectors, and metric statistics.

    Args:
        H_tilde: (L, T, k) whitened hidden states.
        skip_first: Skip the first transition (embedding → layer 1),
            which is often numerically degenerate.
        rank_thresh: Singular value threshold for rank truncation.

    Returns:
        LayerRotorField containing all per-layer GA quantities.
    """
    L = H_tilde.shape[0]
    start = 1 if skip_first else 0

    decomps = []
    for l in range(start, L - 1):
        vd = versor_decompose(H_tilde, l, rank_thresh=rank_thresh)
        decomps.append(vd)

    rotors = [vd.rotor for vd in decomps]
    bivectors = [vd.bivector for vd in decomps]
    angles = np.array([vd.rotor.angle for vd in decomps])
    kappas = np.array([vd.condition_number for vd in decomps])
    eranks = np.array([vd.effective_rank for vd in decomps])
    metrics = [vd.metric for vd in decomps]

    return LayerRotorField(
        rotors=rotors,
        bivectors=bivectors,
        angles=angles,
        condition_numbers=kappas,
        effective_ranks=eranks,
        metrics=metrics,
        decompositions=decomps,
    )


def extract_bivector_field(H_tilde: np.ndarray,
                           skip_first: bool = True,
                           rank_thresh: float = 0.01) -> list[Bivector]:
    """Extract just the bivector generators from the rotor field.

    Convenience function returning the list of Bivector objects.

    Args:
        H_tilde: (L, T, k) whitened hidden states.
        skip_first: Skip first transition.
        rank_thresh: Singular value threshold.

    Returns:
        List of Bivector objects, one per layer transition.
    """
    rf = extract_rotor_field(H_tilde, skip_first=skip_first,
                             rank_thresh=rank_thresh)
    return rf.bivectors


def extract_metric_field(H_tilde: np.ndarray,
                         skip_first: bool = True,
                         rank_thresh: float = 0.01) -> list[dict]:
    """Extract the metric (stretch) field with eigenvalue spectra.

    Args:
        H_tilde: (L, T, k) whitened hidden states.
        skip_first: Skip first transition.
        rank_thresh: Singular value threshold.

    Returns:
        List of dicts with keys:
            'P': symmetric PD matrix
            'singular_values': eigenvalue spectrum
            'condition_number': κ
            'effective_rank': erank
            'layer_index': l
    """
    rf = extract_rotor_field(H_tilde, skip_first=skip_first,
                             rank_thresh=rank_thresh)
    results = []
    for vd in rf.decompositions:
        results.append({
            "P": vd.metric,
            "singular_values": vd.singular_values,
            "condition_number": vd.condition_number,
            "effective_rank": vd.effective_rank,
            "layer_index": vd.layer_index,
        })
    return results
