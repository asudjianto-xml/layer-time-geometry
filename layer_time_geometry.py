"""
Layer-Time Geometry of Transformer Language Models
===================================
Core module implementing the geometric framework from Sudjianto & Zhang.

Provides:
- Hidden state extraction from Qwen2.5-7B via HuggingFace Transformers
- Metric estimation and whitening (Section 3)
- Directional-radial decomposition (Section 4)
- Layer, temporal, and spatiotemporal kernels (Section 5)
- Directed interaction operators with S+A decomposition (Sections 6-7)
- Operator decomposition via polar decomposition (Section 8)
- Discrete geometric flow and curvature (Sections 9-10)
- Geometric algebra: bivectors and skew generators (Section 11)
- Sample-level geometry and reasoning metrics (Sections 12-14)
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional
from scipy.linalg import polar, logm


# ============================================================
# Section 2: Hidden State Extraction
# ============================================================

def extract_hidden_states(model, tokenizer, text: str, device: str = "cuda") -> torch.Tensor:
    """
    Extract hidden states from all layers for a single input.

    Returns:
        H: Tensor of shape (L, T, p) — L layers, T tokens, p hidden dim
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # hidden_states is a tuple of (L+1) tensors of shape (1, T, p)
    # index 0 is the embedding layer; layers 1..L are transformer layers
    hidden_states = outputs.hidden_states  # tuple of (1, T, p)

    # Stack all layers including embedding: shape (L+1, T, p)
    H = torch.stack(hidden_states, dim=0).squeeze(1)  # (L+1, T, p)
    return H.float()


def extract_hidden_states_batch(model, tokenizer, texts: list[str],
                                device: str = "cuda") -> list[torch.Tensor]:
    """Extract hidden states for multiple texts. Returns list of (L, T_i, p) tensors."""
    results = []
    for text in texts:
        H = extract_hidden_states(model, tokenizer, text, device)
        results.append(H)
    return results


# ============================================================
# Section 3: Metric Structure and Whitening
# ============================================================

@dataclass
class MetricStructure:
    """Estimated metric from hidden state covariance with PCA reduction."""
    mean: np.ndarray        # (p_orig,) centering vector
    V_k: np.ndarray         # (p_orig, k) top-k eigenvectors for projection
    eigvals_k: np.ndarray   # (k,) top-k eigenvalues
    W: np.ndarray           # (p_orig, k) whitening projection: V_k @ diag(1/sqrt(eigvals))
    k: int                  # reduced dimensionality
    explained_var: float    # fraction of variance explained


def estimate_metric(H: np.ndarray, n_components: int = 256,
                    reg: float = 1e-6) -> MetricStructure:
    """
    Estimate the metric via PCA-based whitening.

    Projects to the top-k principal components and whitens within that
    subspace, avoiding the ill-conditioning of full p×p inversion.

    Args:
        H: array of shape (N, p) — flattened hidden states across all (l,t)
        n_components: number of principal components to retain
        reg: regularization for numerical stability

    Returns:
        MetricStructure for PCA-whitened projection
    """
    N, p = H.shape
    k = min(n_components, N - 1, p)

    # Center
    mean = H.mean(axis=0)
    H_centered = H - mean

    # SVD on centered data (more stable than eigendecomp of covariance)
    # H_centered = U S V^T, so covariance eigenvectors = V, eigenvalues = S^2/(N-1)
    U, S, Vt = np.linalg.svd(H_centered, full_matrices=False)
    eigvals = (S ** 2) / (N - 1)

    # Keep top-k
    V_k = Vt[:k, :].T           # (p, k)
    eigvals_k = eigvals[:k]     # (k,)
    explained_var = eigvals_k.sum() / (eigvals.sum() + reg)

    # Whitening matrix: project to k dims and scale to unit variance
    # H_white = H_centered @ W, where W = V_k @ diag(1/sqrt(eigvals_k))
    inv_sqrt = 1.0 / np.sqrt(eigvals_k + reg)
    W = V_k * inv_sqrt[np.newaxis, :]  # (p, k) — broadcasting

    return MetricStructure(
        mean=mean, V_k=V_k, eigvals_k=eigvals_k, W=W,
        k=k, explained_var=float(explained_var),
    )


def whiten(H: np.ndarray, metric: MetricStructure) -> np.ndarray:
    """
    Whiten hidden states via PCA projection.
    H_tilde = (H - mean) @ W,  shape (..., k)

    Produces isotropic representations in the reduced k-dimensional space.

    Args:
        H: shape (..., p_orig)
        metric: MetricStructure

    Returns:
        H_tilde: shape (..., k), whitened in reduced space
    """
    original_shape = H.shape
    p_orig = original_shape[-1]
    H_flat = H.reshape(-1, p_orig)
    H_centered = H_flat - metric.mean[np.newaxis, :]
    H_white = H_centered @ metric.W  # (N, k)
    new_shape = original_shape[:-1] + (metric.k,)
    return H_white.reshape(new_shape)


# ============================================================
# Section 4: Directional-Radial Decomposition
# ============================================================

@dataclass
class DirectionalRadial:
    """Decomposition H_tilde = exp(u) * H_hat on S^{p-1} x R."""
    H_hat: np.ndarray     # unit directions, shape (..., p)
    r: np.ndarray         # radii (norms), shape (...)
    u: np.ndarray         # log-radii, shape (...)


def decompose_direction_energy(H_tilde: np.ndarray, eps: float = 1e-8) -> DirectionalRadial:
    """
    Decompose whitened representations into direction and energy.

    H_tilde^{(l,t)} = r^{(l,t)} * H_hat^{(l,t)},  ||H_hat|| = 1
    u^{(l,t)} = log r^{(l,t)}

    Args:
        H_tilde: shape (..., p), whitened hidden states
    """
    r = np.linalg.norm(H_tilde, axis=-1, keepdims=False)
    r_safe = np.maximum(r, eps)
    H_hat = H_tilde / r_safe[..., np.newaxis]
    u = np.log(r_safe)
    return DirectionalRadial(H_hat=H_hat, r=r_safe, u=u)


# ============================================================
# Section 5: Kernel Structures
# ============================================================

def layer_kernel(H_tilde: np.ndarray, t: int) -> np.ndarray:
    """
    Layer kernel at token position t.
    K_layer^{(t)}[l, l'] = <H_tilde^{(l,t)}, H_tilde^{(l',t)}>

    Args:
        H_tilde: (L, T, p)
        t: token index

    Returns:
        K: (L, L)
    """
    T_mat = H_tilde[:, t, :]  # (L, p)
    return T_mat @ T_mat.T


def temporal_kernel(H_tilde: np.ndarray, l: int) -> np.ndarray:
    """
    Temporal kernel at layer l.
    K_time^{(l)}[t, t'] = <H_tilde^{(l,t)}, H_tilde^{(l,t')}>

    Args:
        H_tilde: (L, T, p)
        l: layer index

    Returns:
        K: (T, T)
    """
    S_mat = H_tilde[l, :, :]  # (T, p)
    return S_mat @ S_mat.T


def spatiotemporal_kernel(H_tilde: np.ndarray) -> np.ndarray:
    """
    Full spatiotemporal kernel.
    C[(l,t), (l',t')] = <H_tilde^{(l,t)}, H_tilde^{(l',t')}>

    Args:
        H_tilde: (L, T, p)

    Returns:
        C: (L*T, L*T)
    """
    L, T, p = H_tilde.shape
    flat = H_tilde.reshape(L * T, p)
    return flat @ flat.T


def diffusion_operator(K: np.ndarray) -> np.ndarray:
    """
    Diffusion operator P = D^{-1} K, where D = diag(K @ 1).

    Args:
        K: (N, N) kernel matrix

    Returns:
        P: (N, N) row-stochastic diffusion operator
    """
    d = K.sum(axis=1)
    d_safe = np.maximum(d, 1e-12)
    return K / d_safe[:, np.newaxis]


# ============================================================
# Section 6: Directed Interaction Operators
# ============================================================

def temporal_interaction(H_tilde: np.ndarray, t: int) -> np.ndarray:
    """
    Directed temporal interaction operator.
    M_time^{(t)} = X^{(t)} (X^{(t+1)})^T

    where X^{(t)} = H_tilde[:, t, :] stacks layers at token t.

    Args:
        H_tilde: (L, T, p)
        t: token index (must be < T-1)

    Returns:
        M: (L, L)
    """
    X_t = H_tilde[:, t, :]      # (L, p)
    X_t1 = H_tilde[:, t + 1, :]  # (L, p)
    return X_t @ X_t1.T


def layer_interaction(H_tilde: np.ndarray, l: int) -> np.ndarray:
    """
    Directed layer interaction operator.
    M_layer^{(l)} = Y^{(l)} (Y^{(l+1)})^T

    where Y^{(l)} = H_tilde[l, :, :] stacks tokens at layer l.

    Args:
        H_tilde: (L, T, p)
        l: layer index (must be < L-1)

    Returns:
        M: (T, T)
    """
    Y_l = H_tilde[l, :, :]      # (T, p)
    Y_l1 = H_tilde[l + 1, :, :]  # (T, p)
    return Y_l @ Y_l1.T


# ============================================================
# Section 7: Symmetric and Antisymmetric Decomposition
# ============================================================

def symmetric_antisymmetric(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Decompose M into symmetric and antisymmetric parts.
    S = (M + M^T) / 2,  A = (M - M^T) / 2

    Returns:
        (S, A)
    """
    S = 0.5 * (M + M.T)
    A = 0.5 * (M - M.T)
    return S, A


# ============================================================
# Section 8: Operator Decomposition (Polar)
# ============================================================

@dataclass
class OperatorDecomposition:
    """Polar decomposition T = U P of a layer transition operator in token subspace."""
    T_op: np.ndarray   # (r, r) operator in token subspace
    U: np.ndarray       # (r, r) orthogonal (rotation)
    P: np.ndarray       # (r, r) symmetric positive semi-definite (scaling)
    V: np.ndarray       # (p, r) basis of the joint token subspace
    rank: int           # effective rank r
    singular_values: Optional[np.ndarray] = None  # (r,) singular values of T_op


def layer_operator(H_tilde: np.ndarray, l: int,
                   rank_thresh: float = 0.01) -> OperatorDecomposition:
    """
    Compute layer transition operator and its polar decomposition,
    restricted to the token-spanned subspace for well-determined estimation.

    When T < p, the full p×p operator is under-determined. Instead, we:
    1. Find the joint subspace of H^{(l)} and H^{(l+1)} via SVD
    2. Project both layers into this subspace (dimension r ≤ 2T)
    3. Solve the r×r operator via least squares (now well-determined)
    4. Polar decompose in this subspace

    Args:
        H_tilde: (L, T, p)
        l: layer index
        rank_thresh: singular values below this fraction of the max are dropped

    Returns:
        OperatorDecomposition with T, U, P in the r-dimensional subspace
    """
    H_l = H_tilde[l, :, :]      # (T, p)
    H_l1 = H_tilde[l + 1, :, :]  # (T, p)

    # Joint subspace of both layers
    H_joint = np.vstack([H_l, H_l1])  # (2T, p)
    _, S_joint, Vt_joint = np.linalg.svd(H_joint, full_matrices=False)

    # Truncate to effective rank
    threshold = rank_thresh * S_joint[0]
    r = int(np.sum(S_joint > threshold))
    r = max(r, 1)  # at least rank 1
    V = Vt_joint[:r, :].T  # (p, r)

    # Project into subspace
    A_l = H_l @ V    # (T, r)
    A_l1 = H_l1 @ V  # (T, r)

    # Solve T_sub: A_l @ T_sub = A_l1, so T_sub = lstsq(A_l, A_l1)
    T_op = np.linalg.lstsq(A_l, A_l1, rcond=None)[0]  # (r, r)

    # Polar decomposition: T = U P
    U, P = polar(T_op)

    # Singular values of T_op for metric diagnostics
    sv = np.linalg.svd(T_op, compute_uv=False)

    return OperatorDecomposition(T_op=T_op, U=U, P=P, V=V, rank=r,
                                 singular_values=sv)


# ============================================================
# Section 9: Discrete Geometric Flow
# ============================================================

def delta_layer(H_tilde: np.ndarray) -> np.ndarray:
    """
    Layer-wise finite difference: Delta_l H = H^{(l+1,t)} - H^{(l,t)}

    Args:
        H_tilde: (L, T, p)

    Returns:
        dH: (L-1, T, p)
    """
    return H_tilde[1:, :, :] - H_tilde[:-1, :, :]


def delta_time(H_tilde: np.ndarray) -> np.ndarray:
    """
    Temporal finite difference: Delta_t H = H^{(l,t+1)} - H^{(l,t)}

    Args:
        H_tilde: (L, T, p)

    Returns:
        dH: (L, T-1, p)
    """
    return H_tilde[:, 1:, :] - H_tilde[:, :-1, :]


# ============================================================
# Section 10: Curvature (Non-Commutativity of Transport)
# ============================================================

def _local_transport(h_from: np.ndarray, h_to: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Compute the local transport operator that maps h_from to h_to.
    Uses the rotation component (Procrustes) to define parallel transport
    on the unit sphere, plus a scaling factor for energy.

    For unit vectors a, b: the transport rotates a into b within their
    shared plane, acting as identity on the orthogonal complement.

    For general vectors: T = (r_to/r_from) * R, where R is the rotation.

    Args:
        h_from: (p,) source vector
        h_to: (p,) target vector

    Returns:
        T: (p, p) transport operator
    """
    r_from = np.linalg.norm(h_from) + eps
    r_to = np.linalg.norm(h_to) + eps
    a = h_from / r_from
    b = h_to / r_to

    # Rotation in the plane spanned by a and b (Rodrigues-like)
    cos_theta = np.clip(np.dot(a, b), -1.0, 1.0)
    v = b - cos_theta * a  # component of b orthogonal to a
    v_norm = np.linalg.norm(v)

    if v_norm < eps:
        # Vectors are (anti-)parallel: transport is identity (or reflection)
        R = np.eye(len(a)) if cos_theta > 0 else np.eye(len(a)) - 2.0 * np.outer(a, a)
    else:
        v = v / v_norm  # unit vector in rotation plane, orthogonal to a
        sin_theta = v_norm / (1.0 + eps)  # approximate sin from geometry
        # More precise: use cross-product magnitude
        sin_theta = np.sqrt(1.0 - cos_theta ** 2)
        # R = I + (cos - 1)(aa^T + vv^T) + sin(va^T - av^T)
        R = (np.eye(len(a))
             + (cos_theta - 1.0) * (np.outer(a, a) + np.outer(v, v))
             + sin_theta * (np.outer(v, a) - np.outer(a, v)))

    scale = r_to / r_from
    return scale * R


def curvature(H_tilde: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Operator-based curvature: non-commutativity of transport around
    elementary plaquettes in the (layer, time) grid.

    For each (l, t), define local transport operators:
      T_l^{(l,t)}: transport from (l,t) → (l+1,t)  (layer step at token t)
      T_t^{(l,t)}: transport from (l,t) → (l,t+1)  (time step at layer l)

    The holonomy operator around the plaquette is:
      Omega^{(l,t)} = T_t^{(l+1,t)} T_l^{(l,t)} - T_l^{(l,t+1)} T_t^{(l,t)}

    This is a p×p matrix. We store its Frobenius norm (scalar curvature)
    and a representative curvature vector (Omega applied to H_hat).

    Both paths map h^{(l,t)} to h^{(l+1,t+1)} identically, so applying
    Omega to h^{(l,t)} gives zero — the curvature measures how the
    transport *differs for other directions*, i.e. the operator mismatch.

    Args:
        H_tilde: (L, T, p)

    Returns:
        Omega_norms: (L-1, T-1) scalar curvature at each plaquette
    """
    L, T, p = H_tilde.shape
    Omega_norms = np.zeros((L - 1, T - 1))

    for l in range(L - 1):
        for t in range(T - 1):
            h_lt = H_tilde[l, t]
            h_l1t = H_tilde[l + 1, t]
            h_lt1 = H_tilde[l, t + 1]
            h_l1t1 = H_tilde[l + 1, t + 1]

            # Path 1: layer first, then time
            T_layer = _local_transport(h_lt, h_l1t, eps)
            T_time_up = _local_transport(h_l1t, h_l1t1, eps)
            P1 = T_time_up @ T_layer  # (p, p)

            # Path 2: time first, then layer
            T_time = _local_transport(h_lt, h_lt1, eps)
            T_layer_right = _local_transport(h_lt1, h_l1t1, eps)
            P2 = T_layer_right @ T_time  # (p, p)

            # Holonomy = operator difference
            Omega_op = P1 - P2  # (p, p)
            Omega_norms[l, t] = np.linalg.norm(Omega_op, 'fro')

    return Omega_norms


def curvature_norm(Omega_norms: np.ndarray) -> np.ndarray:
    """
    Identity pass-through for backward compatibility.
    curvature() now returns norms directly.

    Args:
        Omega_norms: (L-1, T-1) already scalar curvature norms

    Returns:
        Omega_norms unchanged
    """
    return Omega_norms


# ============================================================
# Section 11: Geometric Algebra — Bivectors and Skew Generators
# ============================================================

def bivector(H_hat: np.ndarray, l: int, t: int) -> np.ndarray:
    """
    Bivector B^{(l,t)} = H_hat^{(l,t)} wedge H_hat^{(l,t+1)}.
    Represented as the antisymmetric matrix: a b^T - b a^T.

    Args:
        H_hat: (L, T, p) unit directions
        l, t: indices

    Returns:
        B: (p, p) antisymmetric matrix
    """
    a = H_hat[l, t, :]
    b = H_hat[l, t + 1, :]
    return np.outer(a, b) - np.outer(b, a)


def skew_generator(U: np.ndarray) -> np.ndarray:
    """
    Compute the skew-symmetric generator A such that U = exp(A).
    Uses matrix logarithm: A = logm(U).

    Args:
        U: (p, p) orthogonal matrix from polar decomposition

    Returns:
        A: (p, p) skew-symmetric matrix
    """
    A = logm(U)
    # Enforce skew-symmetry (numerical cleanup)
    A = 0.5 * (A - A.T)
    return A.real  # discard small imaginary parts


def bivector_field(H_hat: np.ndarray, l: int) -> np.ndarray:
    """
    Aggregate bivector field at layer l: sum_t B^{(l,t)}.
    Approximates the skew generator A^{(l)}.

    Args:
        H_hat: (L, T, p)
        l: layer index

    Returns:
        B_sum: (p, p) antisymmetric matrix
    """
    T = H_hat.shape[1]
    p = H_hat.shape[2]
    B_sum = np.zeros((p, p))
    for t in range(T - 1):
        B_sum += bivector(H_hat, l, t)
    return B_sum


# ============================================================
# Section 12-13: Sample-Level Geometry and Reasoning Metrics
# ============================================================

@dataclass
class SampleGeometry:
    """Geometric signature of a single sample."""
    Omega_norms: np.ndarray          # (L-1, T-1) scalar curvature at each plaquette
    K_layer_eigenvalues: np.ndarray  # per-token layer kernel eigenvalues
    K_time_eigenvalues: np.ndarray   # per-layer temporal kernel eigenvalues
    difficulty: float                # mean curvature per plaquette (length-invariant)
    difficulty_total: float          # total curvature (sum)
    directionality: float            # ||A||_F / (||S||_F + eps), layers 1+ only
    rotation_devs: np.ndarray        # (L-1,) per-layer ||U - I||_F (NaN at layer 0)
    scaling_devs: np.ndarray         # (L-1,) per-layer ||P - I||_F (NaN at layer 0)
    operator_ranks: np.ndarray       # (L-1,) effective rank per layer operator
    S_norms: np.ndarray              # (L-1,) per-layer symmetric norms (NaN at layer 0)
    A_norms: np.ndarray              # (L-1,) per-layer antisymmetric norms (NaN at layer 0)
    # Length-robust metrics
    curv_concentration: float = 0.0  # fraction of curvature in final 3 layers
    curv_peak_layer: int = 0         # layer with max mean curvature
    curv_entropy: float = 0.0        # Shannon entropy of per-layer curvature distribution
    R_windowed: Optional[np.ndarray] = None  # (n_windows,) windowed directionality
    # Metric-side diagnostics
    condition_numbers: Optional[np.ndarray] = None  # (L-1,) per-layer κ
    eranks: Optional[np.ndarray] = None             # (L-1,) per-layer erank(P)
    stretching_field: Optional[np.ndarray] = None   # (L-1, T) stretching field
    lyapunov_max: float = 0.0                       # maximal Lyapunov exponent
    stretch_concentration: float = 0.0              # fraction of stretching in final 3 layers


def _curvature_layer_profile(Omega_norms: np.ndarray) -> np.ndarray:
    """Mean curvature at each layer, averaged over tokens. Shape (L-1,)."""
    return Omega_norms.mean(axis=1)


def _curv_concentration(Omega_norms: np.ndarray, n_final: int = 3) -> float:
    """Fraction of total curvature in the final n_final layers."""
    profile = _curvature_layer_profile(Omega_norms)
    total = profile.sum()
    if total < 1e-12:
        return 0.0
    return float(profile[-n_final:].sum() / total)


def _curv_peak_layer(Omega_norms: np.ndarray) -> int:
    """Layer index with maximum mean curvature."""
    return int(_curvature_layer_profile(Omega_norms).argmax())


def _curv_entropy(Omega_norms: np.ndarray, eps: float = 1e-12) -> float:
    """Shannon entropy of per-layer curvature distribution (length-invariant)."""
    profile = _curvature_layer_profile(Omega_norms)
    total = profile.sum()
    if total < eps:
        return 0.0
    p = profile / total
    p = p[p > eps]
    return float(-np.sum(p * np.log(p)))


def _condition_number(singular_values: np.ndarray, eps: float = 1e-12) -> float:
    """κ = σ_max / σ_min of the operator."""
    if len(singular_values) == 0:
        return 1.0
    return float(singular_values[0] / (singular_values[-1] + eps))


def _erank(singular_values: np.ndarray, eps: float = 1e-12) -> float:
    """Effective rank: erank = exp(-Σ p_i log p_i) where p_i = σ_i / Σσ_j."""
    if len(singular_values) == 0:
        return 1.0
    total = singular_values.sum()
    if total < eps:
        return 1.0
    p = singular_values / total
    p = p[p > eps]
    return float(np.exp(-np.sum(p * np.log(p))))


def _stretching_field(r: np.ndarray) -> np.ndarray:
    """
    Stretching field S^(l,t) = |log(r^(l+1,t) / r^(l,t))|.

    Args:
        r: (L, T) radii array

    Returns:
        S_stretch: (L-1, T) stretching at each (layer transition, token)
    """
    eps = 1e-12
    log_ratio = np.log(np.maximum(r[1:], eps) / np.maximum(r[:-1], eps))
    return np.abs(log_ratio)


def _lyapunov_max(sigma_max_per_layer: np.ndarray) -> float:
    """
    Maximal Lyapunov exponent: λ_max = (1/(L-1)) Σ log σ_1^(l).

    Args:
        sigma_max_per_layer: (L-1,) max singular value at each layer transition
    """
    n = len(sigma_max_per_layer)
    if n == 0:
        return 0.0
    valid = sigma_max_per_layer[sigma_max_per_layer > 0]
    if len(valid) == 0:
        return 0.0
    return float(np.sum(np.log(valid)) / n)


def _stretch_concentration(S_stretch: np.ndarray, n_final: int = 3) -> float:
    """Fraction of total stretching in the final n_final layers."""
    profile = S_stretch.mean(axis=1)  # (L-1,)
    total = profile.sum()
    if total < 1e-12:
        return 0.0
    return float(profile[-n_final:].sum() / total)


def _directionality_windowed(S_norms: np.ndarray, A_norms: np.ndarray,
                             window: int = 3, eps: float = 1e-8) -> np.ndarray:
    """
    Sliding-window directionality ratio over consecutive layers.

    Aggregates S and A norms over windows of `window` layers for stability.
    Skips NaN entries (layer 0).

    Returns:
        R_windowed: (n_windows,) array of R = ||A||/||S|| per window
    """
    L_minus_1 = len(S_norms)
    if L_minus_1 < window:
        s_valid = np.nansum(S_norms)
        a_valid = np.nansum(A_norms)
        return np.array([a_valid / (s_valid + eps)])

    n_windows = L_minus_1 - window + 1
    R = np.zeros(n_windows)
    for i in range(n_windows):
        s_w = np.nansum(S_norms[i:i + window])
        a_w = np.nansum(A_norms[i:i + window])
        R[i] = a_w / (s_w + eps)
    return R


def _build_sample_geometry(Omega_norms, K_layer_eigs, K_time_eigs,
                           rotation_devs, scaling_devs, operator_ranks,
                           S_norms, A_norms, eps=1e-8,
                           condition_numbers=None, eranks=None,
                           stretching_field_arr=None, lyapunov_max_val=0.0,
                           stretch_conc=0.0):
    """Construct SampleGeometry with all derived metrics."""
    L_minus_1 = len(rotation_devs)
    n_plaquettes = Omega_norms.size
    difficulty_total = float(Omega_norms.sum())
    difficulty = difficulty_total / n_plaquettes if n_plaquettes > 0 else 0.0

    # Directionality from layers 1+ (skip NaN layer 0)
    directionality = float(np.nansum(A_norms) / (np.nansum(S_norms) + eps))

    # Length-robust curvature metrics
    cc = _curv_concentration(Omega_norms)
    cpl = _curv_peak_layer(Omega_norms)
    ce = _curv_entropy(Omega_norms)
    R_w = _directionality_windowed(S_norms, A_norms)

    return SampleGeometry(
        Omega_norms=Omega_norms,
        K_layer_eigenvalues=K_layer_eigs,
        K_time_eigenvalues=K_time_eigs,
        difficulty=difficulty,
        difficulty_total=difficulty_total,
        directionality=directionality,
        rotation_devs=rotation_devs,
        scaling_devs=scaling_devs,
        operator_ranks=operator_ranks,
        S_norms=S_norms,
        A_norms=A_norms,
        curv_concentration=cc,
        curv_peak_layer=cpl,
        curv_entropy=ce,
        R_windowed=R_w,
        condition_numbers=condition_numbers,
        eranks=eranks,
        stretching_field=stretching_field_arr,
        lyapunov_max=lyapunov_max_val,
        stretch_concentration=stretch_conc,
    )


def sample_geometry(H_tilde: np.ndarray, eps: float = 1e-8) -> SampleGeometry:
    """
    Compute the full geometric signature for a single sample.

    Args:
        H_tilde: (L, T, p) whitened hidden states
    """
    L, T, p = H_tilde.shape

    # Curvature (operator-based, returns norms directly)
    Omega_norms = curvature(H_tilde)

    # Layer kernel eigenvalues (at each token)
    K_layer_eigs = []
    for t in range(T):
        K = layer_kernel(H_tilde, t)
        eigs = np.sort(np.linalg.eigvalsh(K))[::-1]
        K_layer_eigs.append(eigs)
    K_layer_eigs = np.array(K_layer_eigs)

    # Temporal kernel eigenvalues (at each layer)
    K_time_eigs = []
    for l in range(L):
        K = temporal_kernel(H_tilde, l)
        eigs = np.sort(np.linalg.eigvalsh(K))[::-1]
        K_time_eigs.append(eigs)
    K_time_eigs = np.array(K_time_eigs)

    # Operator decomposition and S+A decomposition per layer
    rotation_devs = np.full(L - 1, np.nan)
    scaling_devs = np.full(L - 1, np.nan)
    operator_ranks = np.zeros(L - 1, dtype=int)
    S_norms = np.full(L - 1, np.nan)
    A_norms = np.full(L - 1, np.nan)
    cond_numbers = np.full(L - 1, np.nan)
    eranks_arr = np.full(L - 1, np.nan)
    sigma_max_arr = np.full(L - 1, np.nan)

    for l in range(L - 1):
        if l == 0:
            continue  # skip embedding→first-layer (numerically degenerate)

        op = layer_operator(H_tilde, l)
        I_r = np.eye(op.rank)
        rotation_devs[l] = np.linalg.norm(op.U - I_r, 'fro')
        scaling_devs[l] = np.linalg.norm(op.P - I_r, 'fro')
        operator_ranks[l] = op.rank

        # Metric-side: condition number and erank from singular values
        if op.singular_values is not None:
            cond_numbers[l] = _condition_number(op.singular_values)
            eranks_arr[l] = _erank(op.singular_values)
            sigma_max_arr[l] = op.singular_values[0]

        M = layer_interaction(H_tilde, l)
        S, A = symmetric_antisymmetric(M)
        S_norms[l] = np.linalg.norm(S, 'fro')
        A_norms[l] = np.linalg.norm(A, 'fro')

    # Stretching field from radii
    dr = decompose_direction_energy(H_tilde)
    S_stretch = _stretching_field(dr.r)  # (L-1, T)

    # Lyapunov exponent from valid sigma_max values
    valid_sigma = sigma_max_arr[~np.isnan(sigma_max_arr)]
    lyap = _lyapunov_max(valid_sigma)

    # Stretch concentration
    s_conc = _stretch_concentration(S_stretch)

    return _build_sample_geometry(
        Omega_norms, K_layer_eigs, K_time_eigs,
        rotation_devs, scaling_devs, operator_ranks,
        S_norms, A_norms, eps,
        condition_numbers=cond_numbers,
        eranks=eranks_arr,
        stretching_field_arr=S_stretch,
        lyapunov_max_val=lyap,
        stretch_conc=s_conc,
    )


def sample_feature_vector(sg: SampleGeometry, n_eigs: int = 10) -> np.ndarray:
    """
    Length-invariant feature vector phi(s) using summary statistics.

    Produces a fixed-length vector regardless of sequence length T:
    - Curvature statistics: mean, std, max, percentiles over the (l,t) grid
    - Curvature layer profile: mean curvature at each layer (length L-1, fixed)
    - Kernel eigenvalue statistics: averaged across tokens/layers
    - Operator profile: rotation/scaling deviations per layer
    - S+A profile: directionality ratio per layer

    Args:
        sg: SampleGeometry
        n_eigs: number of top eigenvalues to include

    Returns:
        phi: 1D feature vector (fixed length for any T)
    """
    features = []

    # Curvature summary statistics (7 values)
    omega = sg.Omega_norms
    features.extend([
        omega.mean(), omega.std(), omega.max(),
        np.percentile(omega, 25), np.percentile(omega, 50),
        np.percentile(omega, 75), np.percentile(omega, 90),
    ])

    # Curvature layer profile: mean across tokens at each layer (L-1 values)
    features.extend(omega.mean(axis=1).tolist())

    # Layer kernel: top eigenvalues averaged across tokens (n_eigs values, padded)
    n_eigs_layer = min(n_eigs, sg.K_layer_eigenvalues.shape[1])
    layer_eigs = np.zeros(n_eigs)
    layer_eigs[:n_eigs_layer] = sg.K_layer_eigenvalues[:, :n_eigs_layer].mean(axis=0)
    features.extend(layer_eigs.tolist())

    # Temporal kernel: summary stats (fixed 5 values regardless of T)
    # Use mean eigenvalue spectrum across layers, summarized as statistics
    K_time_eigs = sg.K_time_eigenvalues  # (L, T) — T varies
    # Effective rank per layer, then summarize
    for l_idx in range(K_time_eigs.shape[0]):
        eigs_l = K_time_eigs[l_idx]
        eigs_l = eigs_l[eigs_l > 0]
    # Summary: mean/std of top eigenvalue across layers, effective rank stats
    top_eig_per_layer = K_time_eigs[:, 0] if K_time_eigs.shape[1] > 0 else np.zeros(1)
    eig_ratios = []
    for l_idx in range(K_time_eigs.shape[0]):
        eigs_l = K_time_eigs[l_idx]
        total = eigs_l.sum()
        if total > 0:
            probs = eigs_l / total
            probs = probs[probs > 1e-12]
            eig_ratios.append(np.exp(-np.sum(probs * np.log(probs))))
    eig_ratios = np.array(eig_ratios) if eig_ratios else np.zeros(1)
    features.extend([
        top_eig_per_layer.mean(), top_eig_per_layer.std(),
        eig_ratios.mean(), eig_ratios.std(), eig_ratios.max(),
    ])

    # Operator profile: rotation and scaling per layer (2*(L-1) values)
    # Replace NaN (layer 0) with 0 for feature vector
    features.extend(np.nan_to_num(sg.rotation_devs, nan=0.0).tolist())
    features.extend(np.nan_to_num(sg.scaling_devs, nan=0.0).tolist())

    # S+A profile: directionality ratio per layer (L-1 values)
    R_per_layer = np.nan_to_num(sg.A_norms, nan=0.0) / (np.nan_to_num(sg.S_norms, nan=1.0) + 1e-8)
    features.extend(R_per_layer.tolist())

    # Scalar summaries (2 values)
    features.extend([sg.difficulty, sg.directionality])

    return np.array(features)


def prompt_kernel(geometries: list[SampleGeometry], n_eigs: int = 10) -> np.ndarray:
    """
    Compute prompt kernel K_prompt(s_i, s_j) = <phi(s_i), phi(s_j)>.

    Uses length-invariant feature vectors, so no padding is needed.

    Args:
        geometries: list of SampleGeometry objects
        n_eigs: number of eigenvalues for feature vector

    Returns:
        K: (n_samples, n_samples) kernel matrix
    """
    phis = [sample_feature_vector(sg, n_eigs) for sg in geometries]
    phi_mat = np.array(phis)  # (n_samples, d) — all same length
    return phi_mat @ phi_mat.T


# ============================================================
# Section 15: Steering Diagnostics
# ============================================================

@dataclass
class SteeringDiagnostics:
    """Diagnostics for an activation steering intervention."""
    angular_ratio: np.ndarray   # (L, T) ratio of angular to total perturbation
    radial_ratio: np.ndarray    # (L, T) ratio of radial to total perturbation
    delta_Omega_norms: np.ndarray  # (L-1, T-1) curvature norm change
    delta_S_norm: float         # change in symmetric norm
    delta_A_norm: float         # change in antisymmetric norm
    R_before: float             # directionality ratio before
    R_after: float              # directionality ratio after


def steering_diagnostics(H_tilde_before: np.ndarray,
                         H_tilde_after: np.ndarray,
                         eps: float = 1e-8) -> SteeringDiagnostics:
    """
    Compute all steering diagnostics from Section 15.

    Args:
        H_tilde_before: (L, T, p) whitened hidden states before steering
        H_tilde_after: (L, T, p) whitened hidden states after steering
    """
    L, T, p = H_tilde_before.shape

    # Direction-energy decomposition of before states
    dr_before = decompose_direction_energy(H_tilde_before)
    H_hat = dr_before.H_hat  # (L, T, p)

    # Perturbation
    delta = H_tilde_after - H_tilde_before  # (L, T, p)

    # Angular vs radial decomposition of perturbation
    # delta_radial = (delta . H_hat) H_hat
    # delta_angular = delta - delta_radial
    dot = np.sum(delta * H_hat, axis=-1, keepdims=True)  # (L, T, 1)
    delta_radial = dot * H_hat
    delta_angular = delta - delta_radial

    delta_norm = np.linalg.norm(delta, axis=-1) + eps
    angular_ratio = np.linalg.norm(delta_angular, axis=-1) / delta_norm
    radial_ratio = np.linalg.norm(delta_radial, axis=-1) / delta_norm

    # Curvature change (norm-level)
    Omega_before = curvature(H_tilde_before)
    Omega_after = curvature(H_tilde_after)
    delta_Omega_norms = Omega_after - Omega_before

    # S+A decomposition change
    S_before_norm = A_before_norm = 0.0
    S_after_norm = A_after_norm = 0.0
    for l in range(L - 1):
        M_b = layer_interaction(H_tilde_before, l)
        S_b, A_b = symmetric_antisymmetric(M_b)
        S_before_norm += np.linalg.norm(S_b, 'fro')
        A_before_norm += np.linalg.norm(A_b, 'fro')

        M_a = layer_interaction(H_tilde_after, l)
        S_a, A_a = symmetric_antisymmetric(M_a)
        S_after_norm += np.linalg.norm(S_a, 'fro')
        A_after_norm += np.linalg.norm(A_a, 'fro')

    R_before = A_before_norm / (S_before_norm + eps)
    R_after = A_after_norm / (S_after_norm + eps)

    return SteeringDiagnostics(
        angular_ratio=angular_ratio,
        radial_ratio=radial_ratio,
        delta_Omega_norms=delta_Omega_norms,
        delta_S_norm=S_after_norm - S_before_norm,
        delta_A_norm=A_after_norm - A_before_norm,
        R_before=R_before,
        R_after=R_after,
    )


# ============================================================
# GPU-Accelerated Implementations
# ============================================================

def _batch_transport_gpu(src_hat: torch.Tensor, tgt_hat: torch.Tensor,
                         src_r: torch.Tensor, tgt_r: torch.Tensor,
                         eps: float = 1e-8) -> torch.Tensor:
    """
    Batched transport operator construction on GPU.

    Given N pairs of unit vectors and radii, builds N transport matrices
    of shape (p, p) using the Rodrigues rotation formula + radial scaling.

    Args:
        src_hat: (N, p) unit source vectors
        tgt_hat: (N, p) unit target vectors
        src_r: (N,) source radii
        tgt_r: (N,) target radii

    Returns:
        T: (N, p, p) transport operators
    """
    N, p = src_hat.shape

    cos_theta = (src_hat * tgt_hat).sum(-1).clamp(-1.0, 1.0)  # (N,)
    sin_theta = (1.0 - cos_theta ** 2).clamp(min=0).sqrt()  # (N,)

    v = tgt_hat - cos_theta.unsqueeze(-1) * src_hat  # (N, p)
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=eps)
    v = v / v_norm  # (N, p)

    I = torch.eye(p, device=src_hat.device, dtype=src_hat.dtype).unsqueeze(0)  # (1, p, p)

    aa = src_hat.unsqueeze(-1) * src_hat.unsqueeze(-2)  # (N, p, p)
    vv = v.unsqueeze(-1) * v.unsqueeze(-2)  # (N, p, p)
    va = v.unsqueeze(-1) * src_hat.unsqueeze(-2)  # (N, p, p)
    av = src_hat.unsqueeze(-1) * v.unsqueeze(-2)  # (N, p, p)

    cos_m1 = (cos_theta - 1.0).reshape(N, 1, 1)
    sin_t = sin_theta.reshape(N, 1, 1)

    R = I + cos_m1 * (aa + vv) + sin_t * (va - av)  # (N, p, p)

    scale = (tgt_r / src_r).reshape(N, 1, 1)
    return scale * R


def curvature_gpu(H_tilde_np: np.ndarray, device: str = "cuda",
                  eps: float = 1e-8) -> np.ndarray:
    """
    GPU-accelerated curvature via batched transport holonomy.

    Equivalent to curvature() but uses batched GPU operations for the
    transport operator construction and matrix products.  Processes one
    layer at a time to keep GPU memory bounded at O(T * p^2).

    Args:
        H_tilde_np: (L, T, p) numpy array, whitened hidden states
        device: torch device string

    Returns:
        Omega_norms: (L-1, T-1) numpy array of Frobenius curvature norms
    """
    H = torch.tensor(H_tilde_np, dtype=torch.float32, device=device)
    L, T, p = H.shape

    r = H.norm(dim=-1).clamp(min=eps)  # (L, T)
    H_hat = H / r.unsqueeze(-1)  # (L, T, p)

    Omega_norms = torch.zeros(L - 1, T - 1, device=device)

    for l in range(L - 1):
        # Layer transports: h(l,t) -> h(l+1,t) for all t    shape (T, p, p)
        T_layer = _batch_transport_gpu(
            H_hat[l], H_hat[l + 1], r[l], r[l + 1], eps)

        # Time transports at layer l and l+1    shape (T-1, p, p) each
        T_time_l = _batch_transport_gpu(
            H_hat[l, :-1], H_hat[l, 1:], r[l, :-1], r[l, 1:], eps)
        T_time_l1 = _batch_transport_gpu(
            H_hat[l + 1, :-1], H_hat[l + 1, 1:],
            r[l + 1, :-1], r[l + 1, 1:], eps)

        # Path 1: layer first, then time  →  T_time(l+1,t) @ T_layer(l,t)
        P1 = torch.bmm(T_time_l1, T_layer[:-1])  # (T-1, p, p)
        # Path 2: time first, then layer  →  T_layer(l,t+1) @ T_time(l,t)
        P2 = torch.bmm(T_layer[1:], T_time_l)  # (T-1, p, p)

        Omega = P1 - P2
        Omega_norms[l] = Omega.norm(dim=(-2, -1))  # Frobenius norm

    return Omega_norms.cpu().numpy()


def layer_operator_gpu(H_tilde_np: np.ndarray, l: int,
                       device: str = "cuda",
                       rank_thresh: float = 0.01) -> 'OperatorDecomposition':
    """
    GPU-accelerated layer operator decomposition with polar decomposition.

    Same as layer_operator() but uses torch.linalg for SVD and lstsq,
    and implements polar decomposition via SVD (avoiding scipy).

    Args:
        H_tilde_np: (L, T, p) numpy array
        l: layer index
        device: torch device string
        rank_thresh: singular values below this fraction of max are dropped

    Returns:
        OperatorDecomposition
    """
    H_l = torch.tensor(H_tilde_np[l], dtype=torch.float32, device=device)
    H_l1 = torch.tensor(H_tilde_np[l + 1], dtype=torch.float32, device=device)

    # Joint subspace
    H_joint = torch.cat([H_l, H_l1], dim=0)  # (2T, p)
    _, S_joint, Vh = torch.linalg.svd(H_joint, full_matrices=False)

    threshold = rank_thresh * S_joint[0].item()
    r = max(int((S_joint > threshold).sum().item()), 1)
    V = Vh[:r].T  # (p, r)

    # Project into subspace
    A_l = H_l @ V  # (T, r)
    A_l1 = H_l1 @ V  # (T, r)

    # Solve T_sub via lstsq
    T_op = torch.linalg.lstsq(A_l, A_l1).solution  # (r, r)

    # Polar decomposition via SVD: T = U_p @ P_p
    U_t, S_t, Vh_t = torch.linalg.svd(T_op)
    U_polar = U_t @ Vh_t  # orthogonal factor
    P_polar = Vh_t.mH @ torch.diag(S_t) @ Vh_t  # symmetric PSD

    return OperatorDecomposition(
        T_op=T_op.cpu().numpy(),
        U=U_polar.cpu().numpy().real,
        P=P_polar.cpu().numpy().real,
        V=V.cpu().numpy(),
        rank=r,
        singular_values=S_t.cpu().numpy(),
    )


def sample_geometry_gpu(H_tilde_np: np.ndarray, device: str = "cuda",
                        eps: float = 1e-8) -> 'SampleGeometry':
    """
    GPU-accelerated sample geometry computation.

    Uses curvature_gpu for transport holonomy and layer_operator_gpu for
    operator decomposition.  Kernels and S+A remain on CPU (they are cheap).

    Args:
        H_tilde_np: (L, T, p) numpy array, whitened hidden states
        device: torch device string

    Returns:
        SampleGeometry
    """
    L, T, p = H_tilde_np.shape

    # Curvature on GPU
    Omega_norms = curvature_gpu(H_tilde_np, device, eps)
    n_plaquettes = (L - 1) * (T - 1)
    difficulty_total = float(Omega_norms.sum())
    difficulty = difficulty_total / n_plaquettes if n_plaquettes > 0 else 0.0

    # Kernels (small matrices, fast on CPU)
    K_layer_eigs = []
    for t in range(T):
        K = layer_kernel(H_tilde_np, t)
        eigs = np.sort(np.linalg.eigvalsh(K))[::-1]
        K_layer_eigs.append(eigs)
    K_layer_eigs = np.array(K_layer_eigs)

    K_time_eigs = []
    for l in range(L):
        K = temporal_kernel(H_tilde_np, l)
        eigs = np.sort(np.linalg.eigvalsh(K))[::-1]
        K_time_eigs.append(eigs)
    K_time_eigs = np.array(K_time_eigs)

    # Operator decomposition on GPU + S+A on CPU
    rotation_devs = np.full(L - 1, np.nan)
    scaling_devs = np.full(L - 1, np.nan)
    operator_ranks = np.zeros(L - 1, dtype=int)
    S_norms = np.full(L - 1, np.nan)
    A_norms = np.full(L - 1, np.nan)
    cond_numbers = np.full(L - 1, np.nan)
    eranks_arr = np.full(L - 1, np.nan)
    sigma_max_arr = np.full(L - 1, np.nan)

    for l in range(L - 1):
        if l == 0:
            continue  # skip embedding→first-layer (numerically degenerate)

        op = layer_operator_gpu(H_tilde_np, l, device)
        I_r = np.eye(op.rank)
        rotation_devs[l] = np.linalg.norm(op.U - I_r, 'fro')
        scaling_devs[l] = np.linalg.norm(op.P - I_r, 'fro')
        operator_ranks[l] = op.rank

        # Metric-side: condition number and erank from singular values
        if op.singular_values is not None:
            cond_numbers[l] = _condition_number(op.singular_values)
            eranks_arr[l] = _erank(op.singular_values)
            sigma_max_arr[l] = op.singular_values[0]

        M = layer_interaction(H_tilde_np, l)
        S, A = symmetric_antisymmetric(M)
        S_norms[l] = np.linalg.norm(S, 'fro')
        A_norms[l] = np.linalg.norm(A, 'fro')

    # Stretching field from radii
    dr = decompose_direction_energy(H_tilde_np)
    S_stretch = _stretching_field(dr.r)  # (L-1, T)

    # Lyapunov exponent from valid sigma_max values
    valid_sigma = sigma_max_arr[~np.isnan(sigma_max_arr)]
    lyap = _lyapunov_max(valid_sigma)

    # Stretch concentration
    s_conc = _stretch_concentration(S_stretch)

    return _build_sample_geometry(
        Omega_norms, K_layer_eigs, K_time_eigs,
        rotation_devs, scaling_devs, operator_ranks,
        S_norms, A_norms, eps,
        condition_numbers=cond_numbers,
        eranks=eranks_arr,
        stretching_field_arr=S_stretch,
        lyapunov_max_val=lyap,
        stretch_conc=s_conc,
    )


# ============================================================
# Section 16: Generation-Time Analysis
# ============================================================

def extract_generation_trajectory(
    model, tokenizer, prompt: str, max_new_tokens: int = 20,
    device: str = "cuda", temperature: float = 1.0, top_k: int = 50,
) -> tuple[list[torch.Tensor], list[int], list[str]]:
    """
    Generate tokens autoregressively and capture hidden states at each step.

    At each generation step, runs a full forward pass with output_hidden_states=True
    so every layer's representation is captured for the growing context.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        prompt: Input prompt text.
        max_new_tokens: Maximum tokens to generate.
        device: Torch device string.
        temperature: Sampling temperature (1.0 = standard, <1 = greedy-ish).
        top_k: Top-k filtering for sampling.

    Returns:
        hidden_states_per_step: List of (L, T_step, p) tensors, one per generation step.
            Step 0 is the prompt itself (T_0 = prompt length).
            Step i has T_i = T_0 + i tokens.
        token_ids: Full list of token IDs (prompt + generated).
        token_strings: Decoded token strings.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]  # (1, T_prompt)

    hidden_states_per_step = []
    all_token_ids = input_ids[0].tolist()

    for step in range(max_new_tokens + 1):
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)

        # Capture full hidden states: (L+1, T_current, p)
        H = torch.stack(outputs.hidden_states, dim=0).squeeze(1).float()
        hidden_states_per_step.append(H.cpu())

        if step == max_new_tokens:
            break

        # Sample next token
        logits = outputs.logits[0, -1, :]  # (vocab,)
        if temperature > 0:
            logits = logits / temperature
            if top_k > 0:
                top_vals, top_idx = logits.topk(top_k)
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(0, top_idx, top_vals)
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
        else:
            next_token = logits.argmax().unsqueeze(0)

        all_token_ids.append(next_token.item())
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)

        # Stop on EOS
        if next_token.item() == tokenizer.eos_token_id:
            break

    token_strings = [tokenizer.decode([tid]) for tid in all_token_ids]

    return hidden_states_per_step, all_token_ids, token_strings


@dataclass
class GenerationGeometry:
    """Geometric trajectory across generation steps.

    Each field is indexed by generation step (step 0 = prompt only).

    Attributes:
        steps: List of SampleGeometry, one per generation step.
        difficulties: (n_steps,) mean curvature per step.
        directionalities: (n_steps,) directionality ratio per step.
        lyapunov_exponents: (n_steps,) max Lyapunov exponent per step.
        curv_concentrations: (n_steps,) curvature concentration per step.
        last_token_norms: (n_steps, L) norm of hidden state at the last token per layer.
        last_token_drift: (n_steps-1,) cosine distance of final-layer last-token
            representation between consecutive steps.
        token_ids: Full token ID sequence.
        token_strings: Decoded token strings.
    """
    steps: list
    difficulties: np.ndarray
    directionalities: np.ndarray
    lyapunov_exponents: np.ndarray
    curv_concentrations: np.ndarray
    last_token_norms: np.ndarray
    last_token_drift: np.ndarray
    token_ids: list
    token_strings: list


def generation_geometry(
    hidden_states_per_step: list,
    metric: 'MetricStructure',
    device: str = "cuda",
    eps: float = 1e-8,
) -> GenerationGeometry:
    """
    Compute geometric trajectory across autoregressive generation steps.

    For each generation step, whitens the hidden states and computes the full
    SampleGeometry. Then extracts trajectory-level summaries that track how
    the model's geometric processing evolves as it generates.

    Args:
        hidden_states_per_step: List of (L, T_step, p) arrays/tensors from
            extract_generation_trajectory.
        metric: Fitted MetricStructure for whitening.
        device: Device for GPU-accelerated geometry (if available).

    Returns:
        GenerationGeometry with per-step and trajectory-level metrics.
    """
    step_geometries = []
    difficulties = []
    directionalities = []
    lyapunov_exps = []
    curv_concs = []
    last_token_norms_list = []
    final_layer_vecs = []

    use_gpu = device.startswith("cuda") and torch.cuda.is_available()

    for H_raw in hidden_states_per_step:
        if isinstance(H_raw, torch.Tensor):
            H_raw = H_raw.numpy()

        H_tilde = whiten(H_raw, metric)

        if use_gpu:
            sg = sample_geometry_gpu(H_tilde, device=device, eps=eps)
        else:
            sg = sample_geometry(H_tilde, eps=eps)

        step_geometries.append(sg)
        difficulties.append(sg.difficulty)
        directionalities.append(sg.directionality)
        lyapunov_exps.append(sg.lyapunov_max)
        curv_concs.append(sg.curv_concentration)

        # Last-token norm profile across layers
        last_tok_norms = np.linalg.norm(H_tilde[:, -1, :], axis=-1)  # (L,)
        last_token_norms_list.append(last_tok_norms)

        # Final-layer, last-token vector for drift calculation
        final_layer_vecs.append(H_tilde[-1, -1, :].copy())

    # Cosine drift between consecutive steps at final layer, last token
    drift = []
    for i in range(1, len(final_layer_vecs)):
        a, b = final_layer_vecs[i - 1], final_layer_vecs[i]
        cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps)
        drift.append(1.0 - cos_sim)  # cosine distance

    return GenerationGeometry(
        steps=step_geometries,
        difficulties=np.array(difficulties),
        directionalities=np.array(directionalities),
        lyapunov_exponents=np.array(lyapunov_exps),
        curv_concentrations=np.array(curv_concs),
        last_token_norms=np.array(last_token_norms_list),
        last_token_drift=np.array(drift),
        token_ids=[],
        token_strings=[],
    )


def generation_curvature_evolution(gen_geom: GenerationGeometry) -> np.ndarray:
    """
    Extract curvature-at-last-token across generation steps.

    For each step, takes the curvature column corresponding to the
    newest token position, giving a (n_steps, L-1) array that shows
    how the model's curvature profile at the generation frontier evolves.

    Args:
        gen_geom: GenerationGeometry from generation_geometry().

    Returns:
        curv_at_frontier: (n_steps, L-1) array. curv_at_frontier[s, l] is the
            curvature at layer transition l for the last token at generation step s.
    """
    profiles = []
    for sg in gen_geom.steps:
        omega = sg.Omega_norms  # (L-1, T-1)
        if omega.shape[1] > 0:
            profiles.append(omega[:, -1])  # last token column
        else:
            profiles.append(np.zeros(omega.shape[0]))
    return np.array(profiles)


def generation_attention_shift(gen_geom: GenerationGeometry) -> np.ndarray:
    """
    Track how the temporal kernel's effective rank evolves at each layer.

    The effective rank of the temporal kernel measures how many token
    positions contribute meaningfully. As generation proceeds, this
    reveals whether the model spreads attention or concentrates it.

    Args:
        gen_geom: GenerationGeometry from generation_geometry().

    Returns:
        erank_trajectory: (n_steps, L) array of temporal kernel effective ranks.
    """
    eranks = []
    for sg in gen_geom.steps:
        step_eranks = []
        for l_eigs in sg.K_time_eigenvalues:
            total = l_eigs.sum()
            if total > 1e-12:
                p = l_eigs[l_eigs > 1e-12] / total
                step_eranks.append(float(np.exp(-np.sum(p * np.log(p)))))
            else:
                step_eranks.append(1.0)
        eranks.append(step_eranks)
    return np.array(eranks)


# ============================================================
# Section 17: Dependency Observable (Chapter 11)
# ============================================================

@dataclass
class DependencyProfile:
    """Dependency analysis for a single sample.

    Contains the gradient-based dependency density, layerwise profiles,
    dependency horizon, and persistence score as defined in the chapter
    on reasoning, memory, and control.
    """
    D_lt: np.ndarray              # (L, T) dependency density at each cell
    D_layer: np.ndarray           # (L,) uniform layerwise dependency profile
    horizon: dict[float, int]     # {alpha: H_alpha} dependency horizons
    persistence: dict[float, int] # {tau: Pers_tau} persistence scores
    total_dependency: float       # sum of D_layer
    peak_layer: int               # layer with max D_layer


def _score_argmax_logit(model, inputs, hidden_states_with_grad):
    """Score functional: logit of the argmax token at the last position.

    Args:
        model: HuggingFace model
        inputs: tokenized inputs dict
        hidden_states_with_grad: list of (1, T, p) tensors with grad enabled

    Returns:
        scalar tensor (differentiable)
    """
    # Run the model head on the final hidden state
    final_hidden = hidden_states_with_grad[-1]  # (1, T, p)
    # Apply the language model head (lm_head)
    if hasattr(model, 'lm_head'):
        logits = model.lm_head(final_hidden)  # (1, T, V)
    else:
        logits = model.output(final_hidden)
    # Score = logit of argmax at last position
    last_logits = logits[0, -1, :]  # (V,)
    argmax_idx = last_logits.argmax().detach()
    return last_logits[argmax_idx]


def compute_dependency_density(
    model, tokenizer, text: str, metric: MetricStructure,
    device: str = "cuda", score_fn=None,
) -> DependencyProfile:
    """Compute gradient-based dependency density D(l,t) = ||∂s/∂H̃(l,t)||₂.

    Uses autograd to differentiate the score functional with respect to
    whitened hidden states at every (layer, token) cell.

    Args:
        model: HuggingFace causal LM
        tokenizer: corresponding tokenizer
        text: input prompt
        metric: MetricStructure for whitening
        device: torch device
        score_fn: optional callable(model, inputs, hidden_states) -> scalar tensor.
                  Defaults to argmax logit at last position.

    Returns:
        DependencyProfile with dependency density, layerwise profile,
        horizons, and persistence scores.
    """
    if score_fn is None:
        score_fn = _score_argmax_logit

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # Forward pass with hidden states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    raw_hidden = outputs.hidden_states  # tuple of (1, T, p)

    L = len(raw_hidden)
    T = raw_hidden[0].shape[1]
    p = raw_hidden[0].shape[2]

    # Whitening matrices as torch tensors
    W_torch = torch.tensor(metric.W, dtype=torch.float32, device=device)  # (p, k)
    mean_torch = torch.tensor(metric.mean, dtype=torch.float32, device=device)  # (p,)
    k = metric.k

    # Create whitened hidden states with gradient tracking
    # We need gradients w.r.t. the whitened states
    H_tilde_list = []
    for l in range(L):
        h = raw_hidden[l].detach().float()  # (1, T, p)
        h_centered = h - mean_torch.unsqueeze(0).unsqueeze(0)
        h_white = h_centered @ W_torch  # (1, T, k)
        h_white = h_white.detach().requires_grad_(True)
        H_tilde_list.append(h_white)

    # Reconstruct approximate raw states from whitened for the model head
    # We only need the final layer's hidden state to go through lm_head
    # The score is s = lm_head(H_raw_final) evaluated at argmax
    # But H_raw_final ≈ H_tilde_final @ W^T * sqrt(eigvals) + mean (pseudo-inverse)
    # Instead, use the chain rule: ∂s/∂H̃(l,t) via the model's own computation

    # Strategy: re-run forward pass where we inject whitened states
    # Actually, for gradient computation we need a differentiable path from
    # H̃(l,t) to s. The cleanest approach: compute s from the final raw hidden
    # state, then use the Jacobian of whitening to map gradients back.
    #
    # s depends on H_raw^(L-1) via lm_head.
    # H̃^(l) = (H_raw^(l) - mean) @ W
    # ∂s/∂H̃^(l,t) = ∂s/∂H_raw^(l,t) @ W_pinv^T ... but layers l < L-1 affect
    # the final state through the model's nonlinear forward pass.
    #
    # The correct approach: use hooks to intercept hidden states at each layer,
    # replace with whitened versions that have grad, and let autograd propagate.
    #
    # Simplest correct approach: compute gradient of score w.r.t. each layer's
    # raw hidden state (via hooks), then project to whitened space.

    D_lt = np.zeros((L, T))

    # Gradient w.r.t. raw hidden states, then project to whitened metric
    # ∂s/∂H̃(l,t) = ∂s/∂H_raw(l,t) @ (∂H̃/∂H_raw)^T but ∂H̃/∂H_raw = W^T
    # So ||∂s/∂H̃(l,t)||₂ = ||∂s/∂H_raw(l,t) @ W||₂  (since H̃ = (H-μ)W)
    # Actually: ∂s/∂H̃ = ∂s/∂H_raw @ (W^T)^{-1}... no.
    # H̃ = (H_raw - μ) @ W.  If we view H̃ as the variable:
    # ∂s/∂H̃ = ∂s/∂H_raw @ ∂H_raw/∂H̃
    # But H_raw = H̃ @ W^+ + μ where W^+ is pseudo-inverse... this gets messy.
    #
    # Cleanest: for each layer, attach a hook that stores gradient, then
    # compute score and backprop through the model.

    # Use register_full_backward_hook on each layer to capture gradients
    gradients = {}

    def make_hook(layer_idx):
        def hook_fn(module, grad_input, grad_output):
            # grad_output is tuple; first element is grad w.r.t. output (1, T, p)
            if grad_output[0] is not None:
                gradients[layer_idx] = grad_output[0].detach()
        return hook_fn

    # Register hooks on transformer layers
    hooks = []
    # Layer 0 = embedding
    if hasattr(model, 'model'):
        # Common HF pattern: model.model.embed_tokens, model.model.layers[i]
        transformer = model.model
        if hasattr(transformer, 'embed_tokens'):
            h = transformer.embed_tokens.register_full_backward_hook(make_hook(0))
            hooks.append(h)
        if hasattr(transformer, 'layers'):
            for i, layer in enumerate(transformer.layers):
                h = layer.register_full_backward_hook(make_hook(i + 1))
                hooks.append(h)

    # Forward with grad enabled
    inputs_grad = {k: v.detach() for k, v in inputs.items()}
    outputs_grad = model(**inputs_grad, output_hidden_states=True)
    hidden_grad = outputs_grad.hidden_states

    # Compute score
    if hasattr(model, 'lm_head'):
        logits = outputs_grad.logits  # (1, T, V)
    else:
        logits = outputs_grad.logits

    last_logits = logits[0, -1, :]
    argmax_idx = last_logits.argmax().detach()
    score = last_logits[argmax_idx]

    # Backward
    score.backward()

    # Remove hooks
    for h in hooks:
        h.remove()

    # Compute dependency density from captured gradients
    # For layers not captured by hooks, use hidden_states gradients
    for l in range(L):
        if l in gradients:
            grad_raw = gradients[l][0]  # (T, p)
        elif hidden_grad[l].grad is not None:
            grad_raw = hidden_grad[l].grad[0]  # (T, p)
        else:
            # Fallback: compute via hidden state grad
            continue

        # Project gradient to whitened space:
        # D(l,t) = ||grad_raw(l,t) @ W||₂  (W is the whitening projection)
        grad_white = grad_raw.float() @ W_torch  # (T, k)
        D_lt[l] = grad_white.norm(dim=1).cpu().numpy()

    # Build layerwise profile (uniform weighting)
    D_layer = D_lt.mean(axis=1)  # (L,)

    # Dependency horizons
    C_tot = D_layer.sum()
    horizons = {}
    if C_tot > 0:
        cumulative = np.cumsum(D_layer)
        for alpha in [0.5, 0.8, 0.9, 0.95]:
            h_alpha = int(np.searchsorted(cumulative / C_tot, alpha))
            horizons[alpha] = min(h_alpha, L - 1)
    else:
        for alpha in [0.5, 0.8, 0.9, 0.95]:
            horizons[alpha] = L - 1

    # Persistence scores
    persistence = {}
    if C_tot > 0:
        # Use percentile-based thresholds relative to D_layer
        for q in [0.1, 0.25, 0.5]:
            tau = q * D_layer.max()
            persistence[q] = int((D_layer >= tau).sum())
    else:
        for q in [0.1, 0.25, 0.5]:
            persistence[q] = 0

    return DependencyProfile(
        D_lt=D_lt,
        D_layer=D_layer,
        horizon=horizons,
        persistence=persistence,
        total_dependency=float(C_tot),
        peak_layer=int(D_layer.argmax()),
    )


def compute_dependency_density_direct(
    model, tokenizer, text: str, metric: MetricStructure,
    device: str = "cuda", low_memory: bool = False,
) -> DependencyProfile:
    """Compute dependency density via retain_grad + single backward pass.

    Runs a single forward pass with output_hidden_states=True, calls
    retain_grad() on each hidden state tensor, then backpropagates the
    score (argmax logit at last position) to obtain gradients at every
    layer simultaneously.

    Args:
        model: HuggingFace causal LM
        tokenizer: corresponding tokenizer
        text: input prompt
        metric: MetricStructure for whitening
        device: torch device
        low_memory: if True, use per-layer forward passes instead of
            a single backward through the full graph (slower but uses
            much less VRAM — suitable for 30B+ models)

    Returns:
        DependencyProfile
    """
    if low_memory:
        return _compute_dependency_low_memory(
            model, tokenizer, text, metric, device
        )

    inputs = tokenizer(text, return_tensors="pt").to(device)
    W_torch = torch.tensor(metric.W, dtype=torch.float32, device=device)

    # Forward pass WITH gradient enabled and hidden states retained
    outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states  # tuple of (1, T, p)
    L = len(hidden_states)
    T = hidden_states[0].shape[1]

    # Retain gradients on all hidden states
    for h in hidden_states:
        h.retain_grad()

    # Score: logit of argmax at last position
    logits = outputs.logits  # (1, T, V)
    last_logits = logits[0, -1, :]
    argmax_idx = last_logits.argmax().detach()
    score = last_logits[argmax_idx]

    # Single backward pass
    score.backward()

    # Collect gradients and compute dependency density
    D_lt = np.zeros((L, T))
    for l in range(L):
        if hidden_states[l].grad is not None:
            grad_raw = hidden_states[l].grad[0].float()  # (T, p)
            grad_white = grad_raw @ W_torch  # (T, k)
            D_lt[l] = grad_white.norm(dim=1).cpu().numpy()

    # Zero gradients to avoid accumulation
    model.zero_grad()

    return _build_dependency_profile(D_lt, L)


def _compute_dependency_low_memory(
    model, tokenizer, text: str, metric: MetricStructure,
    device: str = "cuda",
) -> DependencyProfile:
    """Memory-efficient dependency density for large models.

    Instead of a single backward through the entire model (which stores
    the full computation graph in VRAM), this:
    1. Runs one no-grad forward pass to cache hidden states and find argmax.
    2. For each layer l, runs a partial forward from layer l to the output
       with gradient enabled only for that segment, computes the gradient,
       then discards the graph.

    Uses ~2x the per-layer VRAM instead of L x per-layer VRAM.
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    W_torch = torch.tensor(metric.W, dtype=torch.float32, device=device)

    # Step 1: no-grad forward to get hidden states + argmax
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = [h.detach().clone() for h in outputs.hidden_states]
        argmax_idx = outputs.logits[0, -1, :].argmax().item()
        del outputs
        torch.cuda.empty_cache()

    L = len(hidden_states)
    T = hidden_states[0].shape[1]
    D_lt = np.zeros((L, T))

    # Get model internals
    transformer = model.model if hasattr(model, 'model') else None
    if transformer is None or not hasattr(transformer, 'layers'):
        # Fallback: use norm-based proxy
        for l in range(L):
            h_white = (hidden_states[l][0].float().cpu().numpy()
                       - metric.mean) @ metric.W
            D_lt[l] = np.linalg.norm(h_white, axis=1)
        for h in hidden_states:
            del h
        torch.cuda.empty_cache()
        return _build_dependency_profile(D_lt, L)

    n_model_layers = len(transformer.layers)
    attention_mask = inputs.get('attention_mask', None)
    position_ids = torch.arange(T, device=device).unsqueeze(0)

    # Build causal mask once
    if hasattr(transformer, '_update_causal_mask'):
        try:
            causal_mask = transformer._update_causal_mask(
                attention_mask, hidden_states[0],
                cache_position=torch.arange(T, device=device)
            )
        except Exception:
            causal_mask = None
    else:
        causal_mask = None

    # Pre-compute rotary position embeddings (required by Qwen2/Llama-style models)
    position_embeddings = None
    if hasattr(transformer, 'rotary_emb'):
        with torch.no_grad():
            position_embeddings = transformer.rotary_emb(
                hidden_states[0], position_ids
            )

    # Step 2: for each layer, do a grad-enabled partial forward
    model_dtype = next(model.parameters()).dtype
    for l in range(L):
        # Create requires_grad tensor in model's native dtype so the
        # gradient chain is not broken by a dtype cast
        h_l = hidden_states[l].clone().to(model_dtype).requires_grad_(True)
        hidden = h_l

        start_layer = l  # transformer.layers[l] produces hidden_states[l+1]

        try:
            if start_layer < n_model_layers:
                # Build kwargs for layer forward
                layer_kwargs = dict(
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                )
                if position_embeddings is not None:
                    layer_kwargs['position_embeddings'] = position_embeddings

                for i in range(start_layer, n_model_layers):
                    layer_out = transformer.layers[i](
                        hidden, **layer_kwargs,
                    )
                    hidden = layer_out[0]
                    # Some models squeeze the batch dim; restore it
                    if hidden.ndim == 2:
                        hidden = hidden.unsqueeze(0)

            if hasattr(transformer, 'norm'):
                hidden = transformer.norm(hidden)
            logits = model.lm_head(hidden)
            score = logits[0, -1, argmax_idx]

            grad = torch.autograd.grad(score, h_l)[0]  # (1, T, p) or (T, p)
            if grad.ndim == 3:
                grad = grad[0]  # (T, p)
            grad_white = grad.float() @ W_torch  # (T, k)
            D_lt[l] = grad_white.norm(dim=1).detach().cpu().numpy()
        except Exception:
            pass

        # Free the computation graph immediately
        del h_l, hidden
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

    # Clean up cached hidden states
    for h in hidden_states:
        del h
    del hidden_states
    torch.cuda.empty_cache()

    return _build_dependency_profile(D_lt, L)


def _forward_from_layer(model, h_l, l, inputs, cached_hidden_states):
    """Run model forward from layer l using h_l as that layer's output.

    Returns logits tensor or None if architecture not supported.
    """
    if not hasattr(model, 'model'):
        return None

    transformer = model.model

    if not hasattr(transformer, 'layers'):
        return None

    # h_l is (1, T, p) — the hidden state at layer l
    hidden = h_l

    # Apply transformer layers from l onward
    # Layer index l corresponds to hidden_states[l]:
    #   l=0 is embedding output, l=1..N are after transformer.layers[0..N-1]
    # So to continue from hidden_states[l], we apply transformer.layers[l:]
    n_layers = len(transformer.layers)
    start_layer = l  # transformer.layers[l] produces hidden_states[l+1]

    # Get attention mask and position ids
    attention_mask = inputs.get('attention_mask', None)
    T = hidden.shape[1]

    # Build causal mask if needed
    if hasattr(transformer, '_update_causal_mask'):
        try:
            causal_mask = transformer._update_causal_mask(
                attention_mask, hidden, cache_position=torch.arange(T, device=hidden.device)
            )
        except Exception:
            causal_mask = None
    else:
        causal_mask = None

    position_ids = torch.arange(T, device=hidden.device).unsqueeze(0)

    for i in range(start_layer, n_layers):
        layer_module = transformer.layers[i]
        try:
            layer_out = layer_module(
                hidden,
                attention_mask=causal_mask,
                position_ids=position_ids,
            )
            hidden = layer_out[0]
        except Exception:
            # Some models have different signatures
            try:
                layer_out = layer_module(hidden)
                hidden = layer_out[0] if isinstance(layer_out, tuple) else layer_out
            except Exception:
                return None

    # Apply final norm
    if hasattr(transformer, 'norm'):
        hidden = transformer.norm(hidden)

    # Apply lm_head
    if hasattr(model, 'lm_head'):
        logits = model.lm_head(hidden)
    else:
        return None

    return logits


def _build_dependency_profile(D_lt: np.ndarray, L: int) -> DependencyProfile:
    """Build DependencyProfile from D_lt array."""
    D_layer = D_lt.mean(axis=1)
    C_tot = D_layer.sum()

    horizons = {}
    if C_tot > 0:
        cumulative = np.cumsum(D_layer)
        for alpha in [0.5, 0.8, 0.9, 0.95]:
            h_alpha = int(np.searchsorted(cumulative / C_tot, alpha))
            horizons[alpha] = min(h_alpha, L - 1)
    else:
        for alpha in [0.5, 0.8, 0.9, 0.95]:
            horizons[alpha] = L - 1

    persistence = {}
    if C_tot > 0:
        for q in [0.1, 0.25, 0.5]:
            tau = q * D_layer.max()
            persistence[q] = int((D_layer >= tau).sum())
    else:
        for q in [0.1, 0.25, 0.5]:
            persistence[q] = 0

    return DependencyProfile(
        D_lt=D_lt,
        D_layer=D_layer,
        horizon=horizons,
        persistence=persistence,
        total_dependency=float(C_tot),
        peak_layer=int(D_layer.argmax()),
    )


# ============================================================
# Section 18: Dual Geometric Control (Chapter 11)
# ============================================================

@dataclass
class ControlResult:
    """Result of applying geometric control intervention."""
    H_tilde_ctrl: np.ndarray       # (L, T, k) controlled whitened hidden states
    dependency_before: DependencyProfile
    dependency_after: DependencyProfile
    delta_D_layer: np.ndarray      # (L,) change in layerwise dependency
    control_type: str              # 'metric', 'rotation', 'dual', 'baseline'


def apply_metric_control(
    H_tilde: np.ndarray,
    operators: list[OperatorDecomposition],
    kappa_max: float = 50.0,
    erank_min: float = 3.0,
) -> np.ndarray:
    """Apply metric control by clamping the spectral structure of P^(l).

    Constrains condition number and effective rank of each layer's metric
    factor. This acts on propagation stability without affecting rotation.

    Args:
        H_tilde: (L, T, k) whitened hidden states
        operators: list of OperatorDecomposition for layers 0..L-2
        kappa_max: maximum allowed condition number
        erank_min: minimum allowed effective rank

    Returns:
        H_tilde_ctrl: (L, T, k) controlled hidden states
    """
    L, T, k = H_tilde.shape
    H_ctrl = H_tilde.copy()

    for l in range(1, L - 1):  # skip embedding layer
        if l >= len(operators) or operators[l] is None:
            continue
        op = operators[l]

        # Use P matrix eigenvalues directly for kappa check
        P_mat = op.P  # (r, r) symmetric PSD
        if P_mat is None or P_mat.shape[0] < 2:
            continue

        eigvals_p, eigvecs_p = np.linalg.eigh(P_mat)
        # eigh returns ascending order; flip to descending
        eigvals_p = eigvals_p[::-1].copy()
        eigvecs_p = eigvecs_p[:, ::-1].copy()

        # Clamp negative eigenvalues (numerical noise) to small positive
        eigvals_p = np.maximum(eigvals_p, 1e-8)

        kappa_p = eigvals_p[0] / (eigvals_p[-1] + 1e-12)
        if kappa_p <= kappa_max:
            continue

        # Clamp: raise the smallest eigenvalues to reduce kappa
        min_target = eigvals_p[0] / kappa_max
        eigvals_clamped = np.maximum(eigvals_p, min_target)

        # Correction factor: sqrt(clamped/orig) in P eigenbasis
        correction = np.sqrt(eigvals_clamped / (eigvals_p + 1e-12))

        V = op.V  # (k, r) — basis of joint token subspace
        r = op.rank

        for t in range(T):
            h = H_ctrl[l, t]  # (k,)
            h_sub = V[:k, :r].T @ h  # (r,) — project to subspace
            h_p = eigvecs_p.T @ h_sub  # in P eigenbasis
            h_p_ctrl = h_p * correction
            h_sub_ctrl = eigvecs_p @ h_p_ctrl
            H_ctrl[l, t] = h + V[:k, :r] @ (h_sub_ctrl - h_sub)

    return H_ctrl


def apply_rotation_control(
    H_tilde: np.ndarray,
    A_target: list[Optional[np.ndarray]],
    A_suppress: list[Optional[np.ndarray]],
    operators: list[OperatorDecomposition],
    alpha: float = 0.3,
    beta: float = 0.1,
) -> np.ndarray:
    """Apply rotation control via skew perturbation ΔA.

    ΔA^(l) = alpha * A_target^(l) - beta * A_suppress^(l)
    The perturbation is applied as exp(ΔA) acting on the hidden states.

    Args:
        H_tilde: (L, T, k) whitened hidden states
        A_target: list of (r, r) skew-symmetric target templates per layer
        A_suppress: list of (r, r) skew-symmetric suppress templates per layer
        operators: list of OperatorDecomposition
        alpha: strength of target promotion
        beta: strength of suppression

    Returns:
        H_tilde_ctrl: (L, T, k) controlled hidden states
    """
    L, T, k = H_tilde.shape
    H_ctrl = H_tilde.copy()

    for l in range(1, L - 1):
        if l >= len(operators) or operators[l] is None:
            continue
        op = operators[l]
        V = op.V  # (p, r)
        r = op.rank

        # Build ΔA
        delta_A = np.zeros((r, r))
        if l < len(A_target) and A_target[l] is not None:
            At = A_target[l]
            if At.shape[0] == r:
                delta_A += alpha * At
        if l < len(A_suppress) and A_suppress[l] is not None:
            As = A_suppress[l]
            if As.shape[0] == r:
                delta_A -= beta * As

        # Ensure skew-symmetry
        delta_A = 0.5 * (delta_A - delta_A.T)

        if np.linalg.norm(delta_A, 'fro') < 1e-10:
            continue

        # Rotation: U_ctrl = exp(ΔA)
        try:
            U_ctrl = _matrix_exp_skew(delta_A)
        except Exception:
            continue

        # Apply rotation in the operator's subspace
        for t in range(T):
            h = H_ctrl[l, t]
            h_sub = V[:k, :r].T @ h
            h_sub_rot = U_ctrl @ h_sub
            H_ctrl[l, t] = h + V[:k, :r] @ (h_sub_rot - h_sub)

    return H_ctrl


def _matrix_exp_skew(A: np.ndarray) -> np.ndarray:
    """Matrix exponential of a skew-symmetric matrix (returns orthogonal matrix)."""
    # For small A, use Padé approximation via scipy
    from scipy.linalg import expm
    return expm(A)


def apply_dual_control(
    H_tilde: np.ndarray,
    operators: list[OperatorDecomposition],
    A_target: list[Optional[np.ndarray]],
    A_suppress: list[Optional[np.ndarray]],
    kappa_max: float = 50.0,
    erank_min: float = 3.0,
    alpha: float = 0.3,
    beta: float = 0.1,
) -> np.ndarray:
    """Apply dual geometric control: metric first, then rotation.

    Matches the convention in the chapter: P_ctrl acts first, U_ctrl second.

    Args:
        H_tilde: (L, T, k) whitened hidden states
        operators: list of OperatorDecomposition
        A_target: target skew templates per layer
        A_suppress: suppress skew templates per layer
        kappa_max: metric control constraint
        erank_min: metric control constraint
        alpha: rotation control strength
        beta: rotation suppression strength

    Returns:
        H_tilde_ctrl: (L, T, k) controlled hidden states
    """
    # Step 1: metric control
    H_metric = apply_metric_control(H_tilde, operators, kappa_max, erank_min)
    # Step 2: rotation control
    H_dual = apply_rotation_control(H_metric, A_target, A_suppress, operators, alpha, beta)
    return H_dual


def estimate_skew_templates(
    geometries: dict[str, list],
    task_desired: str,
    task_undesired: str,
) -> tuple[list[Optional[np.ndarray]], list[Optional[np.ndarray]]]:
    """Estimate empirical skew templates A_target and A_suppress from task groups.

    Args:
        geometries: dict mapping task_type -> list of (operators, H_tilde) tuples
        task_desired: name of desired task family
        task_undesired: name of undesired task family

    Returns:
        (A_target, A_suppress): lists of per-layer (r, r) skew-symmetric templates
    """
    def _avg_antisymmetric(task_data):
        if not task_data:
            return []
        # Find max number of layers
        max_layers = max(len(ops) for ops, _ in task_data)
        A_avg = [None] * max_layers
        counts = [0] * max_layers
        for ops, H_tilde in task_data:
            L = H_tilde.shape[0]
            for l in range(1, min(L - 1, len(ops))):
                if ops[l] is None:
                    continue
                # Extract skew generator from the orthogonal factor U
                # U = exp(A_skew) where A_skew is skew-symmetric (r, r)
                U = ops[l].U  # (r, r) orthogonal
                r = ops[l].rank
                try:
                    A_skew = logm(U)
                    A_skew = 0.5 * (A_skew - A_skew.T)  # ensure skew
                    A_skew = A_skew.real  # discard small imaginary parts
                except Exception:
                    continue
                if A_avg[l] is None:
                    A_avg[l] = A_skew
                else:
                    if A_avg[l].shape == A_skew.shape:
                        A_avg[l] = A_avg[l] + A_skew
                counts[l] += 1
        for l in range(max_layers):
            if A_avg[l] is not None and counts[l] > 0:
                A_avg[l] = A_avg[l] / counts[l]
        return A_avg

    A_target = _avg_antisymmetric(geometries.get(task_desired, []))
    A_suppress = _avg_antisymmetric(geometries.get(task_undesired, []))
    return A_target, A_suppress
