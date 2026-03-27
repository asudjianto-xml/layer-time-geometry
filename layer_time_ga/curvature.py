"""
Curvature as holonomy of the rotor field.

In the GA framework, curvature measures the failure of rotors to commute
around a closed loop (plaquette) on the layer-time grid.

Given the rotor field {R^(l)} on the layer axis, the **holonomy** around
a plaquette at (l, t) is:

    R_loop = R_right(l+1,t) · R_up(l,t+1)^{-1} · R_right(l,t)^{-1} · R_up(l,t)

If R_loop = 1 (identity rotor), the grid is flat at that point.
The deviation from identity — measured by the bivector log(R_loop) —
is the curvature.

The key advantage of the GA formulation is that R_loop carries not just
a *magnitude* (how much curvature) but a *direction* (which plane is
curved).  The matrix formulation gives ||Ω - I||_F; the GA formulation
gives the full bivector of the holonomy.

This module also provides the **commutator field**: [B_i, B_j] for all
pairs of layer bivectors, with decomposition into principal planes.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

from .algebra import (
    Bivector,
    Rotor,
    bivector_from_skew,
    rotor_from_orthogonal,
    commutator_bivector,
)


# ── Holonomy ──────────────────────────────────────────────────────

@dataclass
class HolonomyResult:
    """Holonomy (curvature) at a single plaquette.

    Attributes:
        rotor: The holonomy rotor R_loop.
        bivector: The holonomy bivector log(R_loop).
        scalar_curvature: ||R_loop - I||_F (backward-compatible scalar).
        bivector_norm: ||B_loop||_F (rotation magnitude of the holonomy).
        principal_plane: Dominant plane of curvature (two unit vectors).
        layer: Layer index of the plaquette.
        token: Token index of the plaquette.
    """
    rotor: Rotor
    bivector: Bivector
    scalar_curvature: float
    bivector_norm: float
    principal_plane: Optional[dict]
    layer: int
    token: int


def _local_transport_rotor(h_from: np.ndarray, h_to: np.ndarray,
                           eps: float = 1e-8) -> Rotor:
    """Compute the local transport rotor between two hidden-state vectors.

    This is the Procrustes rotation that maps the direction of h_from
    to the direction of h_to, acting as identity on the orthogonal
    complement.  Wraps the result as a Rotor.

    Note: this does NOT include the scaling (r_to / r_from).  The rotor
    captures only the rotational part of the transport.

    Args:
        h_from: (k,) source vector.
        h_to: (k,) target vector.
        eps: numerical stability.

    Returns:
        Rotor representing the rotation from h_from direction to h_to direction.
    """
    r_from = np.linalg.norm(h_from) + eps
    r_to = np.linalg.norm(h_to) + eps
    a = h_from / r_from
    b = h_to / r_to

    cos_theta = np.clip(np.dot(a, b), -1.0, 1.0)
    v = b - cos_theta * a
    v_norm = np.linalg.norm(v)

    k = len(a)
    if v_norm < eps:
        R = np.eye(k) if cos_theta > 0 else np.eye(k) - 2.0 * np.outer(a, a)
    else:
        v = v / v_norm
        sin_theta = np.sqrt(max(1.0 - cos_theta**2, 0.0))
        R = (np.eye(k)
             + (cos_theta - 1.0) * (np.outer(a, a) + np.outer(v, v))
             + sin_theta * (np.outer(v, a) - np.outer(a, v)))

    return rotor_from_orthogonal(R, compute_bivector=False)


def holonomy_rotor(H_tilde: np.ndarray, l: int, t: int,
                   eps: float = 1e-8) -> HolonomyResult:
    """Compute the holonomy rotor at plaquette (l, t).

    Traverses the plaquette:
        (l,t) → (l+1,t) → (l+1,t+1) → (l,t+1) → (l,t)

    by composing local transport rotors around the loop.

    Args:
        H_tilde: (L, T, k) whitened hidden states.
        l: Layer index (must be < L-1).
        t: Token index (must be < T-1).
        eps: numerical stability.

    Returns:
        HolonomyResult with the loop rotor, its bivector, and diagnostics.
    """
    h_lt = H_tilde[l, t]
    h_l1t = H_tilde[l + 1, t]
    h_lt1 = H_tilde[l, t + 1]
    h_l1t1 = H_tilde[l + 1, t + 1]

    # Path 1: layer first (up), then time (right) at l+1
    R_up = _local_transport_rotor(h_lt, h_l1t, eps)
    R_right_up = _local_transport_rotor(h_l1t, h_l1t1, eps)
    # Composed: go up then right
    P1 = R_right_up.matrix @ R_up.matrix

    # Path 2: time first (right), then layer (up) at t+1
    R_right = _local_transport_rotor(h_lt, h_lt1, eps)
    R_up_right = _local_transport_rotor(h_lt1, h_l1t1, eps)
    # Composed: go right then up
    P2 = R_up_right.matrix @ R_right.matrix

    # Holonomy: difference operator (how much the two paths disagree)
    # In the rotor formulation: R_loop = P1 @ P2^{-1}
    R_loop_matrix = P1 @ P2.T  # P2^{-1} = P2^T since orthogonal

    # Ensure it's a proper rotation
    U_loop, _, Vt_loop = np.linalg.svd(R_loop_matrix)
    R_clean = U_loop @ Vt_loop
    if np.linalg.det(R_clean) < 0:
        R_clean = -R_clean

    loop_rotor = rotor_from_orthogonal(R_clean, compute_bivector=True)

    scalar_curv = float(np.linalg.norm(R_clean - np.eye(R_clean.shape[0]), "fro"))
    biv_norm = loop_rotor.bivector.norm if loop_rotor.bivector else 0.0

    # Principal plane of curvature
    principal = None
    if loop_rotor.bivector is not None and biv_norm > 1e-8:
        planes = loop_rotor.bivector.principal_planes(n_planes=1)
        if planes:
            principal = planes[0]

    return HolonomyResult(
        rotor=loop_rotor,
        bivector=loop_rotor.bivector,
        scalar_curvature=scalar_curv,
        bivector_norm=biv_norm,
        principal_plane=principal,
        layer=l,
        token=t,
    )


def holonomy_field(H_tilde: np.ndarray,
                   eps: float = 1e-8) -> list[list[HolonomyResult]]:
    """Compute the holonomy at every plaquette on the layer-time grid.

    Args:
        H_tilde: (L, T, k) whitened hidden states.
        eps: numerical stability.

    Returns:
        Nested list: result[l][t] is the HolonomyResult at plaquette (l, t).
        Dimensions: (L-1) x (T-1).
    """
    L, T, k = H_tilde.shape
    results = []
    for l in range(L - 1):
        row = []
        for t in range(T - 1):
            row.append(holonomy_rotor(H_tilde, l, t, eps=eps))
        results.append(row)
    return results


def holonomy_scalar_map(H_tilde: np.ndarray,
                        eps: float = 1e-8) -> np.ndarray:
    """Compute the scalar curvature map (backward-compatible with backend).

    Args:
        H_tilde: (L, T, k) whitened hidden states.

    Returns:
        (L-1, T-1) array of scalar curvature values.
    """
    L, T, _ = H_tilde.shape
    curv = np.zeros((L - 1, T - 1))
    for l in range(L - 1):
        for t in range(T - 1):
            hr = holonomy_rotor(H_tilde, l, t, eps=eps)
            curv[l, t] = hr.scalar_curvature
    return curv


def holonomy_bivector(hr: HolonomyResult) -> Bivector:
    """Extract the bivector from a holonomy result.

    This is the *direction* of curvature — which plane the holonomy
    rotates in.  This information is not available from the scalar
    curvature alone.

    Args:
        hr: HolonomyResult from holonomy_rotor().

    Returns:
        Bivector of the holonomy.
    """
    return hr.bivector


# ── Commutator field ──────────────────────────────────────────────

def commutator_field(bivectors: list[Bivector]) -> np.ndarray:
    """Compute the pairwise commutator norms of a list of bivectors.

    [B_i, B_j] = B_i B_j - B_j B_i

    This is the infinitesimal version of holonomy: it measures how much
    two layer rotations fail to commute.

    Args:
        bivectors: List of n Bivector objects (one per layer transition).

    Returns:
        (n, n) symmetric matrix where entry (i, j) = ||[B_i, B_j]||_F.
    """
    n = len(bivectors)
    norms = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            comm = commutator_bivector(bivectors[i], bivectors[j])
            norms[i, j] = comm.norm
            norms[j, i] = norms[i, j]
    return norms


def commutator_bivectors(bivectors: list[Bivector]) -> list[list[Optional[Bivector]]]:
    """Compute the full pairwise commutator bivectors (not just norms).

    This gives the *direction* of non-commutativity between each pair
    of layer rotations.

    Args:
        bivectors: List of n Bivector objects.

    Returns:
        (n, n) nested list where entry [i][j] is the Bivector [B_i, B_j]
        (None on the diagonal).
    """
    n = len(bivectors)
    result = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            comm = commutator_bivector(bivectors[i], bivectors[j])
            result[i][j] = comm
            result[j][i] = Bivector(matrix=-comm.matrix, dim=comm.dim)
    return result


def commutator_plane_decomposition(bivectors: list[Bivector],
                                   n_planes: int = 3) -> dict:
    """Decompose the aggregate commutator structure into principal planes.

    Sums all pairwise commutator bivectors (weighted by norm) and extracts
    the dominant planes of non-commutativity across the entire network.

    Args:
        bivectors: List of n Bivector objects.
        n_planes: Number of principal planes to extract.

    Returns:
        dict with:
            'aggregate_bivector': The sum of all |[B_i, B_j]|.
            'principal_planes': List of plane descriptors.
            'total_norm': Total commutator norm.
    """
    n = len(bivectors)
    if n < 2:
        k = bivectors[0].dim if bivectors else 0
        return {
            "aggregate_bivector": Bivector(matrix=np.zeros((k, k)), dim=k),
            "principal_planes": [],
            "total_norm": 0.0,
        }

    k = bivectors[0].dim
    aggregate = np.zeros((k, k))
    total_norm = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            comm = commutator_bivector(bivectors[i], bivectors[j])
            # Use absolute value of each component to aggregate without cancellation
            aggregate += np.abs(comm.matrix)
            total_norm += comm.norm

    # Re-skew-symmetrize (the abs broke skew-symmetry, so reconstruct)
    aggregate_skew = 0.5 * (aggregate - aggregate.T)
    agg_biv = bivector_from_skew(aggregate_skew)

    planes = agg_biv.principal_planes(n_planes=n_planes)

    return {
        "aggregate_bivector": agg_biv,
        "principal_planes": planes,
        "total_norm": total_norm,
    }


# ── Nonseparability index ───────────────────────────────────────────

def nonseparability_index(
    H_tilde: np.ndarray, eps: float = 1e-8
) -> dict:
    """Compute the nonseparability index D(s) = sum of all holonomy norms.

    This is a single scalar summarising the total amount of non-separable
    (interactive) computation in the hidden-state field.

    D(s) = 0 means layer and token operations are independent (separable).
    D(s) > 0 means the model performs genuinely interactive computation.

    Args:
        H_tilde: (L, T, k) whitened hidden states.
        eps: numerical stability.

    Returns:
        dict with:
            D_total: total nonseparability (sum of holonomy norms).
            D_mean: mean curvature per plaquette.
            holo_map: (L-1, T-1) scalar curvature at each plaquette.
            regime: str, one of 'flat', 'low', 'high', 'chaotic'.
    """
    hmap = holonomy_scalar_map(H_tilde, eps=eps)
    D_total = float(hmap.sum())
    D_mean = float(hmap.mean())

    # Classify curvature regime
    cv = float(np.std(hmap) / (D_mean + eps))  # coefficient of variation
    if D_mean < 0.01:
        regime = "flat"
    elif D_mean < 0.5:
        regime = "low"
    elif cv < 1.5:
        regime = "high"
    else:
        regime = "chaotic"

    return {
        "D_total": D_total,
        "D_mean": D_mean,
        "holo_map": hmap,
        "regime": regime,
    }
