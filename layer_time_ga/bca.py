"""
Bivector Component Analysis (BCA) for frontier trajectories.

BCA decomposes the lagged second-moment matrix M_tau into symmetric
(PCA / grade-0) and antisymmetric (BCA / grade-2) components:

    M_tau = C_tau + B_tau

where C_tau = (M + M^T)/2 is what PCA analyses and B_tau = (M - M^T)/2
is a skew-symmetric matrix (bivector) encoding directed temporal flow.

The antisymmetric component B_tau has purely imaginary eigenvalues in
conjugate pairs ±i*omega_j.  Each pair defines a principal bivector
plane of directed flow with rotation strength omega_j.

Three diagnostics summarise the directed flow:

    Vorticity   Omega = sum(omega_j)       — total directed flow
    Coherence   Psi   = omega_1 / Omega    — flow concentration
    Asymmetry   ||B|| / ||C||              — directed vs undirected

Reference: Sudjianto & Setiawan, "Bivector Component Analysis" (2026).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .algebra import Bivector, bivector_from_skew


# ── Dataclasses ───────────────────────────────────────────────────

@dataclass
class BCAResult:
    """Result of a single BCA decomposition at a fixed lag.

    Attributes:
        lag: The lag tau used.
        M_tau: (k, k) lagged cross-moment matrix.
        C_tau: (k, k) symmetric component (grade-0).
        B_tau: (k, k) antisymmetric component (grade-2 / bivector).
        bivector: The antisymmetric component as a Bivector object.
        rotation_strengths: Sorted rotation strengths omega_j (descending).
        plane_vectors: List of (q_{2j-1}, q_{2j}) pairs spanning each
            principal plane of directed flow (from real Schur decomposition).
        vorticity: Sum of rotation strengths.
        coherence: omega_1 / vorticity (fraction in dominant plane).
        asymmetry_ratio: ||B_tau||_F / ||C_tau||_F.
        norm_C: Frobenius norm of C_tau.
        norm_B: Frobenius norm of B_tau.
    """
    lag: int
    M_tau: np.ndarray
    C_tau: np.ndarray
    B_tau: np.ndarray
    bivector: Bivector
    rotation_strengths: np.ndarray
    plane_vectors: list[tuple[np.ndarray, np.ndarray]]
    vorticity: float
    coherence: float
    asymmetry_ratio: float
    norm_C: float
    norm_B: float


@dataclass
class BCALagSweep:
    """Results of sweeping BCA across multiple lags.

    Attributes:
        lags: List of lags tested.
        results: List of BCAResult, one per lag.
        asymmetry_ratios: Asymmetry ratio at each lag.
        vorticities: Vorticity at each lag.
        coherences: Coherence at each lag.
    """
    lags: list[int]
    results: list[BCAResult]
    asymmetry_ratios: np.ndarray
    vorticities: np.ndarray
    coherences: np.ndarray


@dataclass
class RollingBCAResult:
    """Results of rolling-window BCA.

    Attributes:
        window: Window size used.
        lag: Lag used within each window.
        steps: Centre decode step for each window.
        vorticities: Vorticity at each window position.
        coherences: Coherence at each window position.
        asymmetry_ratios: Asymmetry ratio at each window position.
        results: Full BCAResult at each window position.
    """
    window: int
    lag: int
    steps: np.ndarray
    vorticities: np.ndarray
    coherences: np.ndarray
    asymmetry_ratios: np.ndarray
    results: list[BCAResult]


@dataclass
class BCAPhasePortrait:
    """BCA phase portrait: frontier projections onto principal bivector planes.

    Attributes:
        global_bca: BCAResult from the combined trajectory.
        plane_vectors: List of (v1, v2) pairs spanning each principal
            bivector plane (orthonormalised).
        projections: Dict mapping prompt name to (n_steps, 2*n_planes)
            array of projections.  Columns [2i, 2i+1] are the
            coordinates on the i-th bivector plane.
    """
    global_bca: BCAResult
    plane_vectors: list[tuple[np.ndarray, np.ndarray]]
    projections: dict[str, np.ndarray]


# ── Core functions ────────────────────────────────────────────────

def _lagged_moment(F: np.ndarray, lag: int, center: bool = True) -> np.ndarray:
    """Compute centred lagged cross-moment matrix.

    M_tau = (1/n) sum_s  tilde_F_s^T  tilde_F_{s+tau}

    where tilde_F_s = F_s - mean(F) when center=True.

    Args:
        F: (n_steps, k) trajectory matrix.
        lag: Number of steps to lag.
        center: If True, subtract the mean before computing the
            lagged moment.  Centering ensures that the antisymmetric
            component reflects temporal asymmetry rather than
            persistent mean drift.

    Returns:
        (k, k) lagged cross-moment matrix.
    """
    if center:
        F = F - F.mean(axis=0, keepdims=True)
    X_t = F[:-lag]
    X_lag = F[lag:]
    n = X_t.shape[0]
    return (X_t.T @ X_lag) / n


def bca_decompose(F: np.ndarray, lag: int = 1, center: bool = True) -> BCAResult:
    """Decompose a frontier trajectory into symmetric and antisymmetric components.

    Args:
        F: (n_steps, k) trajectory matrix (e.g. layer-averaged whitened
           frontier columns).
        lag: Lag tau for the second-moment computation.
        center: If True (default), centre the trajectory before
            computing the lagged moment.

    Returns:
        BCAResult with all decomposition quantities.
    """
    M_tau = _lagged_moment(F, lag, center=center)

    C_tau = 0.5 * (M_tau + M_tau.T)
    B_tau = 0.5 * (M_tau - M_tau.T)

    norm_C = np.linalg.norm(C_tau, 'fro')
    norm_B = np.linalg.norm(B_tau, 'fro')
    asymmetry_ratio = norm_B / (norm_C + 1e-12)

    # Bivector object
    bivector = bivector_from_skew(B_tau)

    # Real Schur decomposition of B_tau to get principal planes
    from scipy.linalg import schur
    T_schur, Q = schur(B_tau, output='real')

    # Extract rotation strengths from 2x2 blocks and corresponding
    # plane vectors.  In the real Schur form of a skew-symmetric
    # matrix, each 2x2 block [[0, omega], [-omega, 0]] defines a
    # principal plane.
    omega_list = []
    plane_vectors = []
    k = B_tau.shape[0]
    i = 0
    while i < k:
        if i + 1 < k and abs(T_schur[i + 1, i]) > 1e-10:
            w = abs(T_schur[i, i + 1])
            omega_list.append(w)
            plane_vectors.append((Q[:, i].copy(), Q[:, i + 1].copy()))
            i += 2
        else:
            i += 1

    # Sort by descending rotation strength
    if omega_list:
        order = np.argsort(omega_list)[::-1]
        omega = np.array([omega_list[j] for j in order])
        plane_vectors = [plane_vectors[j] for j in order]
    else:
        omega = np.array([])
        plane_vectors = []

    vorticity = float(omega.sum())
    coherence = float(omega[0] / vorticity) if vorticity > 1e-12 else 0.0

    return BCAResult(
        lag=lag,
        M_tau=M_tau,
        C_tau=C_tau,
        B_tau=B_tau,
        bivector=bivector,
        rotation_strengths=omega,
        plane_vectors=plane_vectors,
        vorticity=vorticity,
        coherence=coherence,
        asymmetry_ratio=asymmetry_ratio,
        norm_C=norm_C,
        norm_B=norm_B,
    )


def bca_lag_sweep(
    F: np.ndarray,
    lags: Optional[list[int]] = None,
    center: bool = True,
) -> BCALagSweep:
    """Sweep BCA across multiple lags to find characteristic time scales.

    Args:
        F: (n_steps, k) trajectory matrix.
        lags: List of lags to test.  Defaults to range(1, 16).
        center: If True (default), centre the trajectory before
            computing lagged moments.

    Returns:
        BCALagSweep with per-lag results and summary arrays.
    """
    if lags is None:
        max_lag = min(15, F.shape[0] // 3)
        lags = list(range(1, max_lag + 1))

    results = [bca_decompose(F, lag=tau, center=center) for tau in lags]

    return BCALagSweep(
        lags=lags,
        results=results,
        asymmetry_ratios=np.array([r.asymmetry_ratio for r in results]),
        vorticities=np.array([r.vorticity for r in results]),
        coherences=np.array([r.coherence for r in results]),
    )


def bca_rolling(
    F: np.ndarray,
    window: int = 15,
    lag: int = 1,
    center: bool = True,
) -> RollingBCAResult:
    """Compute BCA in a rolling window along the frontier trajectory.

    Centering is performed independently within each window.

    Args:
        F: (n_steps, k) trajectory matrix.
        window: Window size in decode steps.
        lag: Lag for BCA within each window.
        center: If True (default), centre each window independently.

    Returns:
        RollingBCAResult with per-window diagnostics.
    """
    n_steps = F.shape[0]
    results = []
    steps = []

    for start in range(n_steps - window + 1):
        F_win = F[start:start + window]
        r = bca_decompose(F_win, lag=lag, center=center)
        results.append(r)
        steps.append(start + window // 2)

    return RollingBCAResult(
        window=window,
        lag=lag,
        steps=np.array(steps),
        vorticities=np.array([r.vorticity for r in results]),
        coherences=np.array([r.coherence for r in results]),
        asymmetry_ratios=np.array([r.asymmetry_ratio for r in results]),
        results=results,
    )


def bca_phase_portrait(
    trajectories: dict[str, np.ndarray],
    lag: int = 1,
    n_planes: int = 2,
    center: bool = True,
) -> BCAPhasePortrait:
    """Compute BCA phase portrait: project trajectories onto principal bivector planes.

    Fits BCA on the combined trajectory of all prompts, extracts the
    top principal bivector planes (from the real Schur decomposition),
    and projects each prompt's trajectory onto those planes.

    Args:
        trajectories: Dict mapping prompt name to (n_steps, k) array.
        lag: Lag for the global BCA decomposition.
        n_planes: Number of principal bivector planes to extract.
        center: If True (default), centre the combined trajectory.

    Returns:
        BCAPhasePortrait with plane vectors and per-prompt projections.
    """
    # Combine all trajectories
    all_F = np.vstack(list(trajectories.values()))
    global_bca = bca_decompose(all_F, lag=lag, center=center)

    # Use the plane vectors from the real Schur decomposition
    plane_vectors = global_bca.plane_vectors[:n_planes]

    # Centre each trajectory using the global mean for consistent projection
    global_mean = all_F.mean(axis=0) if center else np.zeros(all_F.shape[1])

    # Project each trajectory
    projections = {}
    for name, F in trajectories.items():
        F_c = F - global_mean if center else F
        cols = []
        for v1, v2 in plane_vectors:
            cols.append(F_c @ v1)
            cols.append(F_c @ v2)
        projections[name] = np.column_stack(cols)

    return BCAPhasePortrait(
        global_bca=global_bca,
        plane_vectors=plane_vectors,
        projections=projections,
    )


def frontier_bca(
    frontier_whitened: np.ndarray,
    lag: int = 1,
) -> BCAResult:
    """Convenience function: run BCA on a frontier array from GenerationResult.

    Averages across layers to produce a (n_steps, k) trajectory, then
    calls bca_decompose.

    Args:
        frontier_whitened: (n_steps, L, k) whitened frontier array
            (as stored in GenerationResult.frontier_whitened).
        lag: Lag for the BCA decomposition.
        center: If True (default), centre the trajectory before
            computing the lagged moment.

    Returns:
        BCAResult.
    """
    F_avg = frontier_whitened.mean(axis=1)  # (n_steps, k)
    return bca_decompose(F_avg, lag=lag)
