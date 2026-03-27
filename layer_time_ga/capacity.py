"""
Compositional capacity via bivector non-commutativity.

GA-native version of the capacity metrics from the scaling experiment.
The quantities are:

    C_acc  = Σ_{i<j} ||[B_i, B_j]||_F        (accumulated interaction)
    C_eff  = Σ_{i<j} ||[B_i, B_j]||_F √(D_i D_j)  (effective capacity)
    cconc  = (final-layer commutator mass) / C_acc   (concentration)

where B_i is the bivector generator at layer i and D_i is the dependency.

This module wraps the existing capacity.py computation but exposes
results through GA objects (bivectors, commutator bivectors).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

from .algebra import Bivector, bivector_from_skew, commutator_bivector
from .curvature import commutator_field, commutator_plane_decomposition
from .decomposition import extract_rotor_field, LayerRotorField


@dataclass
class GACapacityProfile:
    """GA-native capacity profile for a single sample.

    Attributes:
        bivectors: Per-layer bivector generators.
        commutator_norms: (n, n) pairwise commutator norm matrix.
        C_acc: Accumulated non-commutativity.
        C_eff: Dependency-weighted effective capacity.
        cconc: Commutator concentration in final layers.
        layer_contributions: (n,) per-layer contribution to C_acc.
        principal_planes: Dominant planes of non-commutativity.
        D_layer: Dependency profile used for weighting (if provided).
        rotor_field: The full rotor field (if retained).
    """
    bivectors: list[Bivector]
    commutator_norms: np.ndarray
    C_acc: float
    C_eff: float
    cconc: float
    layer_contributions: np.ndarray
    principal_planes: list[dict]
    D_layer: Optional[np.ndarray]
    rotor_field: Optional[LayerRotorField]


def ga_capacity_profile(
    H_tilde: np.ndarray,
    D_layer: Optional[np.ndarray] = None,
    skip_first: bool = True,
    n_final: int = 3,
    n_planes: int = 3,
    rank_thresh: float = 0.01,
) -> GACapacityProfile:
    """Compute GA-native capacity metrics from whitened hidden states.

    Args:
        H_tilde: (L, T, k) whitened hidden states.
        D_layer: (L,) dependency profile. If None, C_eff is set to 0.
        skip_first: Skip the first layer transition.
        n_final: Number of final layers for concentration.
        n_planes: Number of principal planes to extract.
        rank_thresh: Singular value threshold for rank truncation.

    Returns:
        GACapacityProfile with all capacity statistics.
    """
    # Extract rotor field and bivectors
    rf = extract_rotor_field(H_tilde, skip_first=skip_first,
                             rank_thresh=rank_thresh)
    bivectors = rf.bivectors
    n = len(bivectors)

    if n < 2:
        k = H_tilde.shape[2]
        return GACapacityProfile(
            bivectors=bivectors,
            commutator_norms=np.zeros((n, n)),
            C_acc=0.0,
            C_eff=0.0,
            cconc=0.0,
            layer_contributions=np.zeros(n),
            principal_planes=[],
            D_layer=D_layer,
            rotor_field=rf,
        )

    # Pairwise commutator norms
    comm_norms = commutator_field(bivectors)

    # C_acc: sum of upper triangle
    C_acc = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            C_acc += comm_norms[i, j]

    # C_eff: dependency-weighted
    C_eff = 0.0
    if D_layer is not None:
        offset = 2 if skip_first else 1
        for i in range(n):
            d_i = D_layer[min(i + offset, len(D_layer) - 1)]
            for j in range(i + 1, n):
                d_j = D_layer[min(j + offset, len(D_layer) - 1)]
                C_eff += comm_norms[i, j] * np.sqrt(max(d_i * d_j, 0.0))

    # Concentration in final layers
    total = 0.0
    final_total = 0.0
    final_start = max(0, n - n_final)
    for i in range(n):
        for j in range(i + 1, n):
            val = comm_norms[i, j]
            total += val
            if i >= final_start or j >= final_start:
                final_total += val
    cconc = final_total / (total + 1e-12)

    # Per-layer contributions
    layer_contrib = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                layer_contrib[i] += comm_norms[i, j]

    # Principal planes of non-commutativity
    plane_info = commutator_plane_decomposition(bivectors, n_planes=n_planes)

    return GACapacityProfile(
        bivectors=bivectors,
        commutator_norms=comm_norms,
        C_acc=C_acc,
        C_eff=C_eff,
        cconc=cconc,
        layer_contributions=layer_contrib,
        principal_planes=plane_info["principal_planes"],
        D_layer=D_layer,
        rotor_field=rf,
    )
