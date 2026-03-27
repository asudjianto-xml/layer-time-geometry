"""
Geometric Algebra primitives for layer-time geometry.

Represents rotors, bivectors, and their operations using standard NumPy / PyTorch
linear algebra.  The key insight is that in a *real* Clifford algebra Cl(k,0):

    - A **bivector** B is represented by a (k, k) skew-symmetric matrix.
      Its independent components are the k(k-1)/2 basis bivectors e_i ^ e_j.
    - A **rotor** R = exp(-B/2) is represented by a (k, k) orthogonal matrix
      with det = +1.  The matrix acts by the sandwich product R v R^{-1},
      which is ordinary matrix-vector multiplication for the orthogonal rep.
    - The **geometric product** of two vectors a, b decomposes as
      ab = a . b + a ^ b  (scalar + bivector).

All GA quantities wrap the underlying matrices with semantic names and
provide GA-native operations (composition, inversion, grade projection).

Implementation note
-------------------
We do *not* build a full multivector engine.  The objects live in the
vector representation of Cl(k,0), which is isomorphic to the space of
k x k real matrices for even-grade elements.  This keeps everything
compatible with the existing PyTorch / NumPy backend while exposing
GA semantics.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional
from scipy.linalg import logm, expm


# ── Bivector ──────────────────────────────────────────────────────

@dataclass
class Bivector:
    """A grade-2 element of Cl(k,0), stored as a skew-symmetric matrix.

    In the standard orthonormal basis {e_1, ..., e_k}, the bivector
        B = sum_{i<j} B_{ij} (e_i ^ e_j)
    is represented by the skew-symmetric matrix whose (i,j) entry is B_{ij}.

    Attributes:
        matrix: (k, k) skew-symmetric numpy array.
        dim: k, the dimension of the underlying vector space.
    """
    matrix: np.ndarray
    dim: int

    def __post_init__(self):
        self.dim = self.matrix.shape[0]

    @property
    def norm(self) -> float:
        """Frobenius norm ||B||_F = sqrt(2 * sum_{i<j} B_{ij}^2)."""
        return float(np.linalg.norm(self.matrix, "fro"))

    @property
    def angle(self) -> float:
        """Rotation angle theta = ||B||_F / sqrt(2).

        For a simple (rank-2) bivector, this is the rotation angle.
        For a general bivector (sum of simple bivectors), this is an
        aggregate measure.
        """
        return self.norm / np.sqrt(2.0)

    @property
    def n_components(self) -> int:
        """Number of independent bivector components: k(k-1)/2."""
        return self.dim * (self.dim - 1) // 2

    def components(self) -> np.ndarray:
        """Extract the k(k-1)/2 independent components (upper triangle)."""
        idx = np.triu_indices(self.dim, k=1)
        return self.matrix[idx]

    def principal_planes(self, n_planes: int = 3) -> list[dict]:
        """Decompose into principal simple bivectors via SVD of the skew matrix.

        A skew-symmetric matrix has purely imaginary eigenvalues ±iσ_j.
        Each pair defines a simple bivector (a plane of rotation) with
        rotation magnitude σ_j.

        Returns:
            List of dicts with keys 'angle', 'plane_vectors', 'weight',
            sorted by descending weight.
        """
        # Eigenvalues of skew-symmetric matrix are purely imaginary
        eigvals, eigvecs = np.linalg.eig(self.matrix)
        # Sort by magnitude of imaginary part (descending)
        imag_parts = np.abs(eigvals.imag)
        order = np.argsort(-imag_parts)

        planes = []
        seen = set()
        for idx in order:
            lam = eigvals[idx]
            if abs(lam.imag) < 1e-10:
                continue
            # Eigenvalues come in conjugate pairs; skip duplicates
            pair_key = round(abs(lam.imag), 8)
            if pair_key in seen:
                continue
            seen.add(pair_key)

            # The plane is spanned by real and imaginary parts of the eigenvector
            v = eigvecs[:, idx]
            plane_v1 = v.real / (np.linalg.norm(v.real) + 1e-12)
            plane_v2 = v.imag / (np.linalg.norm(v.imag) + 1e-12)

            planes.append({
                "angle": float(abs(lam.imag)),
                "plane_vectors": (plane_v1.astype(float), plane_v2.astype(float)),
                "weight": float(abs(lam.imag)),
            })
            if len(planes) >= n_planes:
                break

        return planes

    def __neg__(self) -> Bivector:
        return Bivector(matrix=-self.matrix, dim=self.dim)

    def __add__(self, other: Bivector) -> Bivector:
        return Bivector(matrix=self.matrix + other.matrix, dim=self.dim)

    def __sub__(self, other: Bivector) -> Bivector:
        return Bivector(matrix=self.matrix - other.matrix, dim=self.dim)

    def __mul__(self, scalar: float) -> Bivector:
        return Bivector(matrix=scalar * self.matrix, dim=self.dim)

    def __rmul__(self, scalar: float) -> Bivector:
        return self.__mul__(scalar)


# ── Rotor ─────────────────────────────────────────────────────────

@dataclass
class Rotor:
    """An even-grade element of Cl(k,0) representing a rotation.

    Stored as a (k, k) orthogonal matrix with det = +1.
    The rotor R = exp(-B/2) acts on vectors by conjugation:
        v' = R v R^{-1}
    which in the matrix representation is simply v' = M v.

    Attributes:
        matrix: (k, k) orthogonal matrix (SO(k)).
        bivector: the generator B such that matrix = expm(-B/2), if computed.
        dim: k.
    """
    matrix: np.ndarray
    bivector: Optional[Bivector] = None
    dim: int = 0

    def __post_init__(self):
        self.dim = self.matrix.shape[0]

    @property
    def angle(self) -> float:
        """Total rotation angle, computed from the bivector generator."""
        if self.bivector is not None:
            return self.bivector.angle
        # Fall back: compute from eigenvalues of the orthogonal matrix
        eigvals = np.linalg.eigvals(self.matrix)
        angles = np.abs(np.angle(eigvals))
        return float(np.sqrt(np.sum(angles**2) / 2.0))

    @property
    def is_identity(self) -> bool:
        """True if the rotor is close to the identity (no rotation)."""
        return float(np.linalg.norm(self.matrix - np.eye(self.dim), "fro")) < 1e-6

    def inverse(self) -> Rotor:
        """R^{-1} = R^T for orthogonal matrices (= R~ the reversion in GA)."""
        biv = Bivector(matrix=-self.bivector.matrix, dim=self.dim) if self.bivector else None
        return Rotor(matrix=self.matrix.T, bivector=biv, dim=self.dim)

    def compose(self, other: Rotor) -> Rotor:
        """Rotor composition: R_self * R_other (apply other first, then self)."""
        return Rotor(matrix=self.matrix @ other.matrix, dim=self.dim)

    def apply(self, v: np.ndarray) -> np.ndarray:
        """Apply the rotor to a vector: v' = R v R^{-1} = M v."""
        return self.matrix @ v

    def deviation_from_identity(self) -> float:
        """||R - I||_F, measures how far from no-rotation."""
        return float(np.linalg.norm(self.matrix - np.eye(self.dim), "fro"))


# ── Conversion functions ──────────────────────────────────────────

def bivector_from_skew(A: np.ndarray) -> Bivector:
    """Convert a skew-symmetric matrix to a Bivector.

    Args:
        A: (k, k) skew-symmetric matrix.

    Returns:
        Bivector wrapping the matrix.
    """
    # Enforce exact skew-symmetry
    A_skew = 0.5 * (A - A.T)
    return Bivector(matrix=A_skew.real if np.iscomplexobj(A_skew) else A_skew,
                    dim=A_skew.shape[0])


def skew_from_bivector(B: Bivector) -> np.ndarray:
    """Convert a Bivector to its skew-symmetric matrix representation.

    Args:
        B: Bivector object.

    Returns:
        (k, k) skew-symmetric numpy array.
    """
    return B.matrix


def rotor_from_orthogonal(U: np.ndarray, compute_bivector: bool = True) -> Rotor:
    """Convert an orthogonal matrix (from polar decomposition) to a Rotor.

    The bivector generator B is computed via the matrix logarithm:
        U = exp(A) where A is skew-symmetric
        B = 2A (so that U = exp(-B/2) ... wait, convention)

    Convention: We define B such that U = expm(A) where A = -B/2,
    i.e. B = -2A.  This matches the GA convention R = exp(-B theta/2).

    But for simplicity and consistency with the existing code, we store
    the skew generator A directly as the bivector (since the factor of 2
    is a convention choice).  The bivector B.matrix IS the skew generator A.

    Args:
        U: (k, k) orthogonal matrix with det ≈ +1.
        compute_bivector: if True, compute the bivector generator via logm.

    Returns:
        Rotor object.
    """
    # Ensure proper rotation (det = +1)
    if np.linalg.det(U) < 0:
        U = -U

    biv = None
    if compute_bivector:
        A = logm(U)
        A_skew = 0.5 * (A - A.T).real  # enforce skew-symmetry, discard imaginary
        biv = Bivector(matrix=A_skew, dim=U.shape[0])

    return Rotor(matrix=U.real if np.iscomplexobj(U) else U,
                 bivector=biv, dim=U.shape[0])


def rotor_angle(R: Rotor) -> float:
    """Total rotation angle of a rotor."""
    return R.angle


def rotor_plane(R: Rotor, n_planes: int = 1) -> list[dict]:
    """Extract the principal plane(s) of rotation from a rotor.

    Args:
        R: Rotor object (must have bivector computed).
        n_planes: number of principal planes to return.

    Returns:
        List of dicts with 'angle', 'plane_vectors', 'weight'.
    """
    if R.bivector is None:
        # Compute on the fly
        R = rotor_from_orthogonal(R.matrix, compute_bivector=True)
    return R.bivector.principal_planes(n_planes=n_planes)


def rotor_compose(R1: Rotor, R2: Rotor) -> Rotor:
    """Compose two rotors: R1 * R2 (apply R2 first, then R1)."""
    return R1.compose(R2)


def rotor_inverse(R: Rotor) -> Rotor:
    """Inverse (reversion) of a rotor: R~ = R^T."""
    return R.inverse()


# ── Commutator (Lie bracket of bivectors) ─────────────────────────

def commutator_bivector(B1: Bivector, B2: Bivector) -> Bivector:
    """Lie bracket [B1, B2] = B1 B2 - B2 B1.

    This is the commutator of the skew-symmetric generators, which is
    itself skew-symmetric and represents the infinitesimal obstruction
    to commutativity of the two rotations.

    In GA terms, this is the grade-2 part of the geometric product of
    the two bivectors.

    Args:
        B1, B2: Bivector objects.

    Returns:
        Bivector [B1, B2].
    """
    comm = B1.matrix @ B2.matrix - B2.matrix @ B1.matrix
    # Commutator of skew-symmetric matrices is skew-symmetric
    return bivector_from_skew(comm)


# ── Grade decomposition ──────────────────────────────────────────

def grade_decomposition(M: np.ndarray) -> dict:
    """Decompose a square matrix into grade-0 (symmetric) and grade-2 (skew) parts.

    For a matrix M, the decomposition is:
        M = S + A
    where S = (M + M^T)/2 is symmetric (grade-0 in the metric sense)
    and A = (M - M^T)/2 is skew-symmetric (grade-2, a bivector).

    This corresponds to the polar decomposition T = UP split into
    the stretching (grade-0 eigenvalues) and rotation (grade-2 bivector).

    Args:
        M: (k, k) square matrix.

    Returns:
        dict with keys:
            'grade_0': symmetric part (np.ndarray)
            'grade_2': Bivector (skew-symmetric part)
            'grade_0_norm': Frobenius norm of symmetric part
            'grade_2_norm': Frobenius norm of skew part
    """
    S = 0.5 * (M + M.T)
    A = 0.5 * (M - M.T)
    return {
        "grade_0": S,
        "grade_2": bivector_from_skew(A),
        "grade_0_norm": float(np.linalg.norm(S, "fro")),
        "grade_2_norm": float(np.linalg.norm(A, "fro")),
    }


# ── Geometric product of vectors ──────────────────────────────────

def geometric_product_vectors(a: np.ndarray, b: np.ndarray) -> dict:
    """Geometric product of two vectors: ab = a.b + a^b.

    The scalar part (grade 0) is the inner product.
    The bivector part (grade 2) is the outer (wedge) product.

    Args:
        a, b: (k,) vectors.

    Returns:
        dict with 'scalar' (float) and 'bivector' (Bivector).
    """
    scalar = float(np.dot(a, b))
    wedge = np.outer(a, b) - np.outer(b, a)
    return {
        "scalar": scalar,
        "bivector": bivector_from_skew(wedge),
    }


# ── Rodrigues rotation ──────────────────────────────────────────────

def rodrigues_rotation(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rodrigues rotation matrix that maps unit vector u to unit vector v.

    Uses the exact closed-form Rodrigues formula:
        R = I + sin(theta) * B_hat + (1 - cos(theta)) * B_hat^2
    where B_hat is the unit bivector (skew-symmetric) in the u^v plane.

    Exact for any angle, including large rotations (>60 degrees).

    Args:
        u: (k,) unit vector (source).
        v: (k,) unit vector (target).

    Returns:
        (k, k) rotation matrix R such that R @ u = v.
    """
    k = len(u)
    cos_theta = float(np.clip(np.dot(u, v), -1.0, 1.0))
    if cos_theta > 1.0 - 1e-12:
        return np.eye(k)
    sin_theta = np.sqrt(1.0 - cos_theta ** 2)
    # Skew-symmetric matrix in the u^v plane
    B = np.outer(v, u) - np.outer(u, v)
    if sin_theta < 1e-12:
        # 180-degree rotation: B is zero, use fallback
        return 2.0 * np.outer(v, v) - np.eye(k)
    B_hat = B / sin_theta  # unit bivector
    R = np.eye(k) + sin_theta * B_hat + (1.0 - cos_theta) * (B_hat @ B_hat)
    return R


def rodrigues_rotor(u: np.ndarray, v: np.ndarray) -> Rotor:
    """Rotor that maps unit vector u to unit vector v via Rodrigues formula.

    Args:
        u: (k,) unit vector (source).
        v: (k,) unit vector (target).

    Returns:
        Rotor with associated bivector generator.
    """
    R = rodrigues_rotation(u, v)
    return rotor_from_orthogonal(R, compute_bivector=True)


# ── Cayley bivector ─────────────────────────────────────────────────

def cayley_bivector(u: np.ndarray, v: np.ndarray) -> tuple[Bivector, float]:
    """Cayley bivector encoding the rotation from u to v.

    A(u, v) = (v u^T - u v^T) / (1 + v^T u)

    The steering magnitude is tau = ||A||_F / sqrt(2) = tan(theta/2)
    where theta = arccos(u^T v).  The sqrt(2) arises because the
    Frobenius norm of a rank-2 skew-symmetric matrix has a factor of
    sqrt(2) relative to the rotation-angle parameterisation.

    Args:
        u: (k,) unit vector (prior / source).
        v: (k,) unit vector (posterior / target).

    Returns:
        (Bivector, tau): the bivector and its steering magnitude.
    """
    denom = 1.0 + np.dot(v, u)
    if denom < 1e-12:
        raise ValueError("Vectors are antiparallel; Cayley map is undefined.")
    A = (np.outer(v, u) - np.outer(u, v)) / denom
    biv = bivector_from_skew(A)
    tau = biv.norm / np.sqrt(2.0)  # tan(theta/2)
    return biv, tau


# ── Binet-Cauchy cosine ────────────────────────────────────────────

def binet_cauchy_cosine(
    u_i: np.ndarray,
    u_next: np.ndarray,
    c_q: np.ndarray,
    c_ctx: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Binet-Cauchy cosine: signed bivector alignment between two planes.

    Measures whether the reasoning trajectory (u_i -> u_next) flows in
    the same oriented direction as the expected information flow
    (c_q -> c_ctx).

    BCcos = det(S) / (||u_i ^ u_next|| * ||c_q ^ c_ctx|| + eps)

    where S is the 2x2 matrix of inner products:
        S = [[u_i . c_q,   u_i . c_ctx],
             [u_next . c_q, u_next . c_ctx]]

    Returns:
        float in [-1, 1].  +1 = aligned, -1 = causal inversion.
    """
    S = np.array([
        [np.dot(u_i, c_q), np.dot(u_i, c_ctx)],
        [np.dot(u_next, c_q), np.dot(u_next, c_ctx)],
    ])
    det_S = S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0]

    # Norms of wedge products (= sin of angle between each pair)
    norm_u = np.sqrt(max(0.0, 1.0 - np.dot(u_i, u_next) ** 2))
    norm_c = np.sqrt(max(0.0, 1.0 - np.dot(c_q, c_ctx) ** 2))

    return float(det_S / (norm_u * norm_c + eps))


# ── Directional flow ratio ─────────────────────────────────────────

def directional_flow_ratio(M: np.ndarray, eps: float = 1e-12) -> float:
    """Grade-2 to grade-0 ratio of a matrix (directional flow ratio).

    R = ||A||_F / (||S||_F + eps)

    where S = (M + M^T)/2 (symmetric / grade-0) and
          A = (M - M^T)/2 (antisymmetric / grade-2 / bivector).

    R >> 1: rotation-dominated.  R << 1: metric-dominated.

    Args:
        M: (k, k) matrix (e.g. layer transition operator).

    Returns:
        float, the directional flow ratio.
    """
    S = 0.5 * (M + M.T)
    A = 0.5 * (M - M.T)
    return float(np.linalg.norm(A, "fro") / (np.linalg.norm(S, "fro") + eps))
