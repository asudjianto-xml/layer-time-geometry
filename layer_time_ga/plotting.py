"""
GA-specific visualization functions for the layer-time geometric algebra framework.

Provides matplotlib-based plots for rotor fields, holonomy maps,
commutator structure, grade profiles, and capacity summaries.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch

from .algebra import Bivector, Rotor
from .decomposition import LayerRotorField, extract_rotor_field
from .curvature import (
    holonomy_scalar_map,
    holonomy_field,
    commutator_field,
    commutator_plane_decomposition,
)
from .capacity import GACapacityProfile

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
NAVY = "#1F3864"
BLUE = "#2E6DAD"
_PALETTE = [NAVY, BLUE, "#5B9BD5", "#A3C4E0", "#D6E4F0"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numpy(x):
    """Convert tensor / array / scalar to numpy."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _finalise(fig, ax, save_path):
    """Apply tight_layout and optionally save the figure."""
    if fig is not None:
        fig.tight_layout()
    if save_path is not None:
        (fig or ax.get_figure()).savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig or ax.get_figure())


def _get_fig_ax(ax, figsize=(8, 4)):
    """Return (fig, ax).  If *ax* is provided, fig is None."""
    if ax is not None:
        return None, ax
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


# ===================================================================
# 1. Rotor angle profile
# ===================================================================

def plot_rotor_angle_profile(rotor_field, save_path=None, ax=None):
    """Plot rotation angle theta(l) vs layer index.

    Parameters
    ----------
    rotor_field : LayerRotorField
        Contains per-layer Rotor objects.
    save_path : str or Path, optional
        If given, save figure to this path.
    ax : matplotlib.axes.Axes, optional
        If given, plot on this axes instead of creating a new figure.
    """
    fig, ax = _get_fig_ax(ax, figsize=(8, 4))

    angles = []
    for rotor in rotor_field.rotors:
        # Rotation angle from the bivector norm: ||B|| = theta/2
        biv_norm = torch.linalg.norm(rotor.bivector.components).item()
        angles.append(2.0 * biv_norm)

    layers = np.arange(len(angles))
    ax.plot(layers, angles, color=NAVY, linewidth=2, marker="o", markersize=4)
    ax.set_xlabel("Layer index $\\ell$", fontsize=12)
    ax.set_ylabel("Rotation angle $\\theta(\\ell)$  [rad]", fontsize=12)
    ax.set_title("Rotor Angle Profile", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    _finalise(fig, ax, save_path)
    return ax


# ===================================================================
# 2. Bivector plane evolution
# ===================================================================

def plot_bivector_plane_evolution(rotor_field, n_planes=3, save_path=None, ax=None):
    """Stacked area chart of principal rotation-plane weights across layers.

    Parameters
    ----------
    rotor_field : LayerRotorField
    n_planes : int
        Number of dominant planes to show.
    save_path : str or Path, optional
    ax : matplotlib.axes.Axes, optional
    """
    fig, ax = _get_fig_ax(ax, figsize=(10, 5))

    L = len(rotor_field.rotors)
    # Collect singular values of each layer's bivector matrix
    plane_weights = []
    for rotor in rotor_field.rotors:
        B = rotor.bivector.components
        if B.dim() == 1:
            # Reshape antisymmetric vector into matrix for SVD
            d = int((1 + (1 + 8 * B.shape[0]) ** 0.5) / 2)
            mat = torch.zeros(d, d, dtype=B.dtype, device=B.device)
            idx = 0
            for i in range(d):
                for j in range(i + 1, d):
                    mat[i, j] = B[idx]
                    mat[j, i] = -B[idx]
                    idx += 1
        else:
            mat = B
        sv = torch.linalg.svdvals(mat.float())
        # SVD of antisymmetric matrix gives paired singular values
        vals = _to_numpy(sv[:n_planes])
        # Pad if fewer than n_planes
        if len(vals) < n_planes:
            vals = np.pad(vals, (0, n_planes - len(vals)))
        plane_weights.append(vals)

    plane_weights = np.array(plane_weights)  # (L, n_planes)
    layers = np.arange(L)

    colours = _PALETTE[:n_planes]
    ax.stackplot(
        layers,
        *[plane_weights[:, k] for k in range(n_planes)],
        labels=[f"Plane {k+1}" for k in range(n_planes)],
        colors=colours,
        alpha=0.85,
    )
    ax.set_xlabel("Layer index $\\ell$", fontsize=12)
    ax.set_ylabel("Singular value weight", fontsize=12)
    ax.set_title("Bivector Plane Evolution", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    _finalise(fig, ax, save_path)
    return ax


# ===================================================================
# 3. Holonomy map (scalar curvature heatmap)
# ===================================================================

def plot_holonomy_map(H_tilde, save_path=None):
    """2-D heatmap of scalar curvature from the holonomy field.

    Parameters
    ----------
    H_tilde : tensor-like, shape (L-1, T-1)
        Precomputed holonomy scalar map (e.g. from holonomy_scalar_map()).
    save_path : str or Path, optional
    """
    data = _to_numpy(H_tilde)

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "navy_blue", [NAVY, BLUE, "#A3C4E0", "#FFFFFF"]
    )
    im = ax.imshow(data, aspect="auto", cmap=cmap, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Scalar curvature", fontsize=11)

    ax.set_xlabel("Token position $t$", fontsize=12)
    ax.set_ylabel("Layer index $\\ell$", fontsize=12)
    ax.set_title("Holonomy Scalar Map", fontsize=13, fontweight="bold")

    _finalise(fig, ax, save_path)
    return ax


# ===================================================================
# 4. Holonomy planes
# ===================================================================

def plot_holonomy_planes(holonomy_results, save_path=None):
    """Plot dominant curvature-plane angle per layer (averaged over tokens).

    Parameters
    ----------
    holonomy_results : list of list
        Nested list from holonomy_field(); outer index = layer,
        inner index = token position.  Each element should expose
        a bivector / matrix from which we extract the dominant plane angle.
    save_path : str or Path, optional
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    layer_angles = []
    for layer_results in holonomy_results:
        token_angles = []
        for h in layer_results:
            # Extract bivector matrix from holonomy result
            if hasattr(h, "bivector"):
                mat = h.bivector.components
            elif isinstance(h, torch.Tensor):
                mat = h
            else:
                mat = torch.tensor(h)

            if mat.dim() == 1:
                d = int((1 + (1 + 8 * mat.shape[0]) ** 0.5) / 2)
                full = torch.zeros(d, d, dtype=mat.dtype)
                idx = 0
                for i in range(d):
                    for j in range(i + 1, d):
                        full[i, j] = mat[idx]
                        full[j, i] = -mat[idx]
                        idx += 1
                mat = full

            sv = torch.linalg.svdvals(mat.float())
            # Dominant plane angle ~ arctan(sv[0] / (sv[1] + 1e-12))
            if sv.numel() >= 2:
                angle = torch.atan2(sv[0], sv[1] + 1e-12).item()
            else:
                angle = 0.0
            token_angles.append(angle)
        layer_angles.append(np.mean(token_angles) if token_angles else 0.0)

    layers = np.arange(len(layer_angles))
    ax.bar(layers, layer_angles, color=BLUE, edgecolor=NAVY, linewidth=0.5)
    ax.set_xlabel("Layer index $\\ell$", fontsize=12)
    ax.set_ylabel("Dominant plane angle [rad]", fontsize=12)
    ax.set_title("Holonomy Dominant Curvature Planes", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    _finalise(fig, ax, save_path)
    return ax


# ===================================================================
# 5. Commutator heatmap
# ===================================================================

def plot_commutator_heatmap(bivectors, save_path=None, ax=None):
    """Heatmap of pairwise commutator norms ||[B_i, B_j]||_F.

    Parameters
    ----------
    bivectors : list of Bivector or tensors
        One bivector per layer.
    save_path : str or Path, optional
    ax : matplotlib.axes.Axes, optional
    """
    comm = commutator_field(bivectors)
    data = _to_numpy(comm)

    fig, ax = _get_fig_ax(ax, figsize=(7, 6))

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "navy_blue_hot", ["#FFFFFF", BLUE, NAVY]
    )
    im = ax.imshow(data, cmap=cmap, interpolation="nearest")
    cbar = (fig or ax.get_figure()).colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("$\\|[B_i, B_j]\\|_F$", fontsize=11)

    ax.set_xlabel("Layer $j$", fontsize=12)
    ax.set_ylabel("Layer $i$", fontsize=12)
    ax.set_title("Commutator Norm Heatmap", fontsize=13, fontweight="bold")

    _finalise(fig, ax, save_path)
    return ax


# ===================================================================
# 6. Commutator planes bar chart
# ===================================================================

def plot_commutator_planes(bivectors, n_planes=5, save_path=None, ax=None):
    """Bar chart of principal commutator planes and their weights.

    Parameters
    ----------
    bivectors : list of Bivector or tensors
    n_planes : int
        Number of principal planes to display.
    save_path : str or Path, optional
    ax : matplotlib.axes.Axes, optional
    """
    planes, weights = commutator_plane_decomposition(bivectors, n_planes=n_planes)
    weights = _to_numpy(weights)

    fig, ax = _get_fig_ax(ax, figsize=(8, 4))

    indices = np.arange(len(weights))
    colours = [NAVY if k % 2 == 0 else BLUE for k in indices]
    ax.bar(indices, weights, color=colours, edgecolor="white", linewidth=0.5)
    ax.set_xticks(indices)
    ax.set_xticklabels([f"Plane {k+1}" for k in indices], fontsize=10)
    ax.set_ylabel("Weight (singular value)", fontsize=12)
    ax.set_title("Principal Commutator Planes", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    _finalise(fig, ax, save_path)
    return ax


# ===================================================================
# 7. Grade profile
# ===================================================================

def plot_grade_profile(rotor_field, save_path=None, ax=None):
    """Side-by-side grade-0 and grade-2 content across layers.

    Grade-0 deviation: ||P - I||_F   (scalar part proximity to identity).
    Grade-2 content:   ||B||_F       (bivector norm).

    Parameters
    ----------
    rotor_field : LayerRotorField
    save_path : str or Path, optional
    ax : matplotlib.axes.Axes, optional
    """
    fig, ax = _get_fig_ax(ax, figsize=(9, 4))

    grade0 = []
    grade2 = []
    for rotor in rotor_field.rotors:
        # Grade-0: deviation of the projection part from identity
        P = rotor.projection
        if isinstance(P, torch.Tensor):
            d = P.shape[0]
            dev = torch.linalg.norm(P.float() - torch.eye(d, dtype=P.dtype, device=P.device).float()).item()
        else:
            dev = 0.0
        grade0.append(dev)

        # Grade-2: bivector norm
        biv = rotor.bivector.components
        grade2.append(torch.linalg.norm(biv.float()).item())

    layers = np.arange(len(grade0))
    width = 0.35

    ax.bar(layers - width / 2, grade0, width, label="Grade-0  $\\|P - I\\|_F$",
           color=NAVY, edgecolor="white", linewidth=0.5)
    ax.bar(layers + width / 2, grade2, width, label="Grade-2  $\\|B\\|_F$",
           color=BLUE, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Layer index $\\ell$", fontsize=12)
    ax.set_ylabel("Frobenius norm", fontsize=12)
    ax.set_title("Grade Profile (Grade-0 vs Grade-2)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    _finalise(fig, ax, save_path)
    return ax


# ===================================================================
# 8. Capacity summary (multi-panel)
# ===================================================================

def plot_capacity_summary(capacity_profile, save_path=None):
    """Multi-panel capacity figure.

    Panels:
      (a) Per-layer contributions to C_acc.
      (b) Commutator heatmap.
      (c) Principal planes bar chart.

    Parameters
    ----------
    capacity_profile : GACapacityProfile
        Must expose .per_layer_contributions, .bivectors.
    save_path : str or Path, optional
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Per-layer capacity contributions
    contribs = _to_numpy(capacity_profile.per_layer_contributions)
    layers = np.arange(len(contribs))
    axes[0].bar(layers, contribs, color=NAVY, edgecolor="white", linewidth=0.5)
    axes[0].set_xlabel("Layer index $\\ell$", fontsize=11)
    axes[0].set_ylabel("$\\Delta C(\\ell)$", fontsize=11)
    axes[0].set_title("(a) Per-layer capacity", fontsize=12, fontweight="bold")
    axes[0].grid(True, axis="y", alpha=0.3)

    # (b) Commutator heatmap
    plot_commutator_heatmap(capacity_profile.bivectors, ax=axes[1])
    axes[1].set_title("(b) Commutator heatmap", fontsize=12, fontweight="bold")

    # (c) Principal planes
    plot_commutator_planes(capacity_profile.bivectors, n_planes=5, ax=axes[2])
    axes[2].set_title("(c) Principal planes", fontsize=12, fontweight="bold")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return axes


# ===================================================================
# 9. GA summary (2x2)
# ===================================================================

def plot_ga_summary(rotor_field, H_tilde, save_path=None):
    """2x2 summary figure.

    Panels:
      (top-left)     Rotor angle profile.
      (top-right)    Grade profile.
      (bottom-left)  Holonomy scalar map.
      (bottom-right) Commutator heatmap.

    Parameters
    ----------
    rotor_field : LayerRotorField
    H_tilde : tensor-like, shape (L-1, T-1)
    save_path : str or Path, optional
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: rotor angle profile
    plot_rotor_angle_profile(rotor_field, ax=axes[0, 0])
    axes[0, 0].set_title("Rotor Angle Profile", fontsize=12, fontweight="bold")

    # Top-right: grade profile
    plot_grade_profile(rotor_field, ax=axes[0, 1])
    axes[0, 1].set_title("Grade Profile", fontsize=12, fontweight="bold")

    # Bottom-left: holonomy scalar map
    data = _to_numpy(H_tilde)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "navy_blue", [NAVY, BLUE, "#A3C4E0", "#FFFFFF"]
    )
    im = axes[1, 0].imshow(data, aspect="auto", cmap=cmap, interpolation="nearest")
    fig.colorbar(im, ax=axes[1, 0], shrink=0.8)
    axes[1, 0].set_xlabel("Token position $t$", fontsize=11)
    axes[1, 0].set_ylabel("Layer index $\\ell$", fontsize=11)
    axes[1, 0].set_title("Holonomy Scalar Map", fontsize=12, fontweight="bold")

    # Bottom-right: commutator heatmap
    bivectors = [r.bivector for r in rotor_field.rotors]
    plot_commutator_heatmap(bivectors, ax=axes[1, 1])
    axes[1, 1].set_title("Commutator Heatmap", fontsize=12, fontweight="bold")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return axes
