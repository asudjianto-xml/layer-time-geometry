"""
ltg_ga — Layer-Time Geometry: Geometric Algebra Student API
============================================================

A high-level wrapper around layer_time_geometry.py and layer_time_ga/
designed for undergraduate data science students.  Every function takes
simple inputs and returns interpretable outputs, with GA-native
diagnostics: rotors, bivectors, holonomy, and commutator structure.

Quick start:
    >>> import ltg_ga
    >>> model = ltg_ga.load_model("Qwen/Qwen2.5-7B")
    >>> result = ltg_ga.analyse("The capital of France is", model=model)
    >>> result.summary()
    >>> result.plot_ga_summary()
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional
import warnings

import layer_time_geometry as core
from layer_time_ga.decomposition import extract_rotor_field, LayerRotorField
from layer_time_ga.curvature import holonomy_scalar_map, commutator_field
from layer_time_ga.capacity import ga_capacity_profile, GACapacityProfile
from layer_time_ga.algebra import Bivector


# ════════════════════════════════════════════════════════════════
# Model loading  (reused from ltg.py pattern)
# ════════════════════════════════════════════════════════════════

@dataclass
class Model:
    """Wrapper holding a model, tokenizer, and device info."""
    hf_model: object
    tokenizer: object
    name: str
    device: str
    n_layers: int
    hidden_dim: int


def load_model(name: str = "Qwen/Qwen2.5-7B",
               device: str = "auto") -> Model:
    """
    Load a Hugging Face causal language model.

    Args:
        name: Model name on Hugging Face (e.g., "Qwen/Qwen2.5-7B")
        device: "cuda", "cpu", or "auto" (picks GPU if available)

    Returns:
        Model object ready for analysis.

    Example:
        >>> model = ltg_ga.load_model("Qwen/Qwen2.5-7B")
        >>> print(f"{model.name}: {model.n_layers} layers, dim={model.hidden_dim}")
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        name, trust_remote_code=True,
        dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device).eval()

    config = hf_model.config
    n_layers = getattr(config, 'num_hidden_layers', None)
    hidden_dim = getattr(config, 'hidden_size', None)

    return Model(
        hf_model=hf_model,
        tokenizer=tokenizer,
        name=name,
        device=device,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
    )


# ════════════════════════════════════════════════════════════════
# GA analysis result
# ════════════════════════════════════════════════════════════════

@dataclass
class GAResult:
    """
    Complete geometric-algebra analysis of a single prompt.

    Attributes:
        text: The input prompt.
        tokens: List of token strings.
        n_layers: Number of transformer layers (after skipping layer 0).
        n_tokens: Number of tokens.
        k: Whitening dimension.
        H_whitened: Whitened hidden states, shape (L, T, k).
        rotor_field: LayerRotorField with per-layer rotors and bivectors.
        bivector_field: List of Bivector generators, one per transition.
        holonomy_map: (L-1, T-1) scalar curvature from holonomy rotors.
        curvature_by_layer: Mean holonomy curvature per layer, shape (L-1,).
        condition_numbers: Condition number per transition, shape (n_transitions,).
        effective_ranks: Effective rank per transition, shape (n_transitions,).
        angles: Rotor angle per transition, shape (n_transitions,).
        dependency_profile: Dependency D_l per layer (optional).
        dep_entropy: Dependency entropy (optional).
        dep_total: Total dependency (optional).
        dep_horizon_90: Layer at which 90% dependency is reached (optional).
    """
    text: str
    tokens: list
    n_layers: int
    n_tokens: int
    k: int

    # Core GA arrays
    H_whitened: np.ndarray
    rotor_field: LayerRotorField
    bivector_field: list  # list[Bivector]
    holonomy_map: np.ndarray
    curvature_by_layer: np.ndarray
    condition_numbers: np.ndarray
    effective_ranks: np.ndarray
    angles: np.ndarray

    # Dependency (optional)
    dependency_profile: Optional[np.ndarray] = None
    dep_entropy: Optional[float] = None
    dep_total: Optional[float] = None
    dep_horizon_90: Optional[int] = None

    # ── Summary ──────────────────────────────────────────────────

    def summary(self):
        """Print a human-readable summary of the GA analysis."""
        print(f"=== Layer-Time Geometry: GA Analysis ===")
        print(f"Prompt: \"{self.text[:60]}{'...' if len(self.text) > 60 else ''}\"")
        print(f"Tokens: {self.n_tokens}  |  Layers: {self.n_layers}  |  Whitened dim: {self.k}")
        print()

        print(f"-- Rotor Field --")
        print(f"  Mean rotation angle:   {self.angles.mean():.4f} rad")
        print(f"  Peak angle layer:      {self.angles.argmax()}")
        print(f"  Angle range:           [{self.angles.min():.4f}, {self.angles.max():.4f}]")
        print()

        print(f"-- Holonomy Curvature --")
        if self.holonomy_map.size > 0:
            print(f"  Mean curvature:        {self.holonomy_map.mean():.4f}")
            print(f"  Peak layer:            {self.curvature_by_layer.argmax()}"
                  f" (of {len(self.curvature_by_layer)})")
            total_curv = self.curvature_by_layer.sum()
            if total_curv > 0:
                final3 = self.curvature_by_layer[-3:].sum() / total_curv
                print(f"  Final-3-layer share:   {final3:.1%}")
        else:
            print(f"  (no plaquettes — need >= 2 layers and >= 2 tokens)")
        print()

        print(f"-- Metric Deformation --")
        print(f"  Mean condition number: {self.condition_numbers.mean():.2f}")
        print(f"  Peak condition layer:  {self.condition_numbers.argmax()}")
        print(f"  Mean effective rank:   {self.effective_ranks.mean():.1f}")
        print()

        if self.dependency_profile is not None:
            print(f"-- Dependency --")
            print(f"  Total dependency:      {self.dep_total:.4f}")
            print(f"  Dependency entropy:    {self.dep_entropy:.3f}")
            print(f"  Horizon-90:            layer {self.dep_horizon_90}")

    # ── Plotting methods ─────────────────────────────────────────

    def plot_rotor_angles(self, save_path: Optional[str] = None):
        """Plot rotor rotation angles across layers."""
        fig, ax = plt.subplots(figsize=(8, 4))
        layers = np.arange(len(self.angles))
        ax.plot(layers, self.angles, color='#4477AA', linewidth=2,
                marker='o', markersize=3)
        ax.fill_between(layers, self.angles, alpha=0.15, color='#4477AA')
        ax.set_xlabel('Layer transition')
        ax.set_ylabel('Rotation angle (rad)')
        ax.set_title('Rotor Angle Profile')
        ax.axvline(self.angles.argmax(), color='grey', linestyle='--',
                    alpha=0.5, label=f'Peak = {self.angles.argmax()}')
        ax.legend()
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Saved: {save_path}")
        plt.close(fig)
        return fig

    def plot_holonomy(self, save_path: Optional[str] = None):
        """Plot holonomy curvature heatmap and layer profile."""
        if self.holonomy_map.size == 0:
            print("No holonomy data (need >= 2 layers and >= 2 tokens).")
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Heatmap
        im = ax1.imshow(self.holonomy_map, aspect='auto', cmap='YlOrRd',
                         origin='lower')
        ax1.set_xlabel('Token position')
        ax1.set_ylabel('Layer')
        ax1.set_title('Holonomy Scalar Curvature $\\|R_{\\mathrm{loop}} - I\\|_F$')
        plt.colorbar(im, ax=ax1)

        # Layer profile
        ax2.plot(self.curvature_by_layer, color='#EE6677', linewidth=2)
        ax2.fill_between(range(len(self.curvature_by_layer)),
                         self.curvature_by_layer, alpha=0.2, color='#EE6677')
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Mean holonomy curvature')
        ax2.set_title('Curvature by Layer (GA holonomy)')
        ax2.axvline(self.curvature_by_layer.argmax(), color='grey',
                     linestyle='--', alpha=0.5, label='Peak')
        ax2.legend()

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Saved: {save_path}")
        plt.close(fig)
        return fig

    def plot_grade_profile(self, save_path: Optional[str] = None):
        """Plot condition number and effective rank (metric deformation)."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        layers = np.arange(len(self.condition_numbers))

        ax1.plot(layers, self.condition_numbers, color='#AA3377', linewidth=2)
        ax1.set_xlabel('Layer transition')
        ax1.set_ylabel('Condition number $\\kappa$')
        ax1.set_title('Metric Selectivity (Grade-0)')

        ax2.plot(layers, self.effective_ranks, color='#66CCEE', linewidth=2)
        ax2.set_xlabel('Layer transition')
        ax2.set_ylabel('Effective rank')
        ax2.set_title('Metric Dimensionality (Grade-0)')

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Saved: {save_path}")
        plt.close(fig)
        return fig

    def plot_commutator(self, save_path: Optional[str] = None):
        """Plot bivector commutator heatmap [B_i, B_j]."""
        if len(self.bivector_field) < 2:
            print("Need at least 2 bivectors for commutator plot.")
            return None

        comm_norms = commutator_field(self.bivector_field)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(comm_norms, cmap='inferno', origin='lower')
        ax.set_xlabel('Layer transition $j$')
        ax.set_ylabel('Layer transition $i$')
        ax.set_title('Bivector Commutator $\\|[B_i, B_j]\\|_F$')
        plt.colorbar(im, ax=ax, label='Commutator norm')
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Saved: {save_path}")
        plt.close(fig)
        return fig

    def plot_ga_summary(self, save_path: Optional[str] = None):
        """Four-panel GA summary: angles, holonomy, grade profile, commutator."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Top-left: rotor angles
        ax = axes[0, 0]
        layers = np.arange(len(self.angles))
        ax.plot(layers, self.angles, color='#4477AA', linewidth=2,
                marker='o', markersize=3)
        ax.fill_between(layers, self.angles, alpha=0.15, color='#4477AA')
        ax.set_xlabel('Layer transition')
        ax.set_ylabel('Angle (rad)')
        ax.set_title('Rotor Angles')

        # Top-right: holonomy curvature by layer
        ax = axes[0, 1]
        if self.holonomy_map.size > 0:
            ax.plot(self.curvature_by_layer, color='#EE6677', linewidth=2)
            ax.fill_between(range(len(self.curvature_by_layer)),
                            self.curvature_by_layer, alpha=0.2, color='#EE6677')
            ax.set_title('Holonomy Curvature by Layer')
        else:
            ax.text(0.5, 0.5, 'No plaquettes', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title('Holonomy Curvature')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Mean curvature')

        # Bottom-left: condition number + effective rank
        ax = axes[1, 0]
        trans_layers = np.arange(len(self.condition_numbers))
        ax.plot(trans_layers, self.condition_numbers, color='#AA3377',
                linewidth=2, label='$\\kappa$')
        ax_twin = ax.twinx()
        ax_twin.plot(trans_layers, self.effective_ranks, color='#66CCEE',
                     linewidth=2, linestyle='--', label='erank')
        ax.set_xlabel('Layer transition')
        ax.set_ylabel('Condition number', color='#AA3377')
        ax_twin.set_ylabel('Effective rank', color='#66CCEE')
        ax.set_title('Metric Deformation (Grade-0)')

        # Bottom-right: commutator heatmap
        ax = axes[1, 1]
        if len(self.bivector_field) >= 2:
            comm_norms = commutator_field(self.bivector_field)
            im = ax.imshow(comm_norms, cmap='inferno', origin='lower')
            ax.set_xlabel('Layer $j$')
            ax.set_ylabel('Layer $i$')
            ax.set_title('Commutator $\\|[B_i, B_j]\\|_F$')
            plt.colorbar(im, ax=ax, label='Norm')
        else:
            ax.text(0.5, 0.5, 'Need >= 2 bivectors', ha='center',
                    va='center', transform=ax.transAxes)
            ax.set_title('Commutator')

        prompt_short = self.text[:50] + ('...' if len(self.text) > 50 else '')
        fig.suptitle(f'GA Summary: "{prompt_short}"', fontsize=12, y=1.01)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Saved: {save_path}")
        plt.close(fig)
        return fig

    def plot_all(self, prefix: str = "ga_analysis"):
        """Generate all plots with auto-naming."""
        self.plot_rotor_angles(f"{prefix}_rotor_angles.png")
        self.plot_holonomy(f"{prefix}_holonomy.png")
        self.plot_grade_profile(f"{prefix}_grade_profile.png")
        self.plot_commutator(f"{prefix}_commutator.png")
        self.plot_ga_summary(f"{prefix}_summary.png")
        print(f"All GA plots saved with prefix: {prefix}")


# ════════════════════════════════════════════════════════════════
# Main analysis function
# ════════════════════════════════════════════════════════════════

def analyse(text: str,
            model: Model,
            compute_dependency: bool = True,
            whiten_components: int = 256) -> GAResult:
    """
    Run complete GA layer-time analysis on a prompt.

    Extracts hidden states, whitens them, decomposes each layer
    transition into a rotor + metric deformation, computes holonomy
    curvature over the layer-time grid, and optionally computes
    gradient-based dependency.

    Args:
        text: Input prompt string.
        model: Model object from load_model().
        compute_dependency: Whether to compute gradient-based dependency
                           (slower but very informative). Default True.
        whiten_components: Whitening dimension (default 256).

    Returns:
        GAResult with all GA quantities and plotting methods.

    Example:
        >>> model = ltg_ga.load_model("Qwen/Qwen2.5-7B")
        >>> result = ltg_ga.analyse("What is 2 + 3?", model=model)
        >>> result.summary()
        >>> result.plot_ga_summary(save_path="ga_summary.png")
    """
    # ── 1. Extract hidden states ──
    H_raw = core.extract_hidden_states(
        model.hf_model, model.tokenizer, text, model.device
    )
    H_np = H_raw[1:].cpu().numpy()  # skip layer 0, shape (L, T, p)
    L, T, p = H_np.shape

    # Get token strings
    token_ids = model.tokenizer.encode(text)
    tokens = [model.tokenizer.decode([tid]) for tid in token_ids]

    # ── 2. Whiten ──
    k = whiten_components
    H_flat = H_np.reshape(L * T, p)
    metric = core.estimate_metric(
        H_flat, n_components=min(k, min(L * T, p) - 1)
    )
    H_w = core.whiten(H_np, metric)  # (L, T, k_actual)
    k_actual = H_w.shape[2]

    # ── 3. Extract rotor field (GA decomposition) ──
    rotor_field_obj = extract_rotor_field(H_w, skip_first=False)
    bivectors = rotor_field_obj.bivectors
    angles = rotor_field_obj.angles
    cond_numbers = rotor_field_obj.condition_numbers
    eranks = rotor_field_obj.effective_ranks

    # ── 4. Holonomy scalar curvature map ──
    if L >= 2 and T >= 2:
        holo_map = holonomy_scalar_map(H_w)
        curv_by_layer = holo_map.mean(axis=1) if holo_map.size > 0 else np.zeros(max(L - 1, 0))
    else:
        holo_map = np.array([])
        curv_by_layer = np.array([])

    # ── 5. Dependency (optional) ──
    dep_profile = None
    dep_total = dep_entropy = dep_h90 = None

    if compute_dependency:
        try:
            dep_result = core.compute_dependency_density(
                model.hf_model, model.tokenizer, text, metric,
                device=model.device
            )
            dep_profile = dep_result.D_layer  # (L,)
            dep_total = float(dep_profile.sum())

            D_norm = dep_profile / (dep_total + 1e-12)
            dep_entropy = float(-np.sum(D_norm * np.log(D_norm + 1e-12)))

            D_cum = np.cumsum(D_norm)
            h90_indices = np.where(D_cum >= 0.9)[0]
            dep_h90 = int(h90_indices[0]) if len(h90_indices) > 0 else L - 1
        except Exception as e:
            warnings.warn(f"Dependency computation failed: {e}")

    return GAResult(
        text=text,
        tokens=tokens,
        n_layers=L,
        n_tokens=T,
        k=k_actual,
        H_whitened=H_w,
        rotor_field=rotor_field_obj,
        bivector_field=bivectors,
        holonomy_map=holo_map,
        curvature_by_layer=curv_by_layer,
        condition_numbers=cond_numbers,
        effective_ranks=eranks,
        angles=angles,
        dependency_profile=dep_profile,
        dep_entropy=dep_entropy,
        dep_total=dep_total,
        dep_horizon_90=dep_h90,
    )


# ════════════════════════════════════════════════════════════════
# Comparison
# ════════════════════════════════════════════════════════════════

def compare(results: list, save_path: Optional[str] = None):
    """
    Compare GA profiles across multiple prompts.

    Args:
        results: List of GAResult objects.
        save_path: Optional path to save the comparison plot.

    Example:
        >>> r1 = ltg_ga.analyse("2 + 3 = ?", model=model)
        >>> r2 = ltg_ga.analyse("What causes rain?", model=model)
        >>> ltg_ga.compare([r1, r2], save_path="ga_compare.png")
    """
    n = len(results)
    if n == 0:
        print("No results to compare.")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = plt.cm.Set2(np.linspace(0, 1, max(n, 3)))

    # ── Rotor angles ──
    ax = axes[0, 0]
    for i, r in enumerate(results):
        label = r.text[:30] + ('...' if len(r.text) > 30 else '')
        ax.plot(r.angles, color=colors[i], linewidth=1.5, label=label)
    ax.set_xlabel('Layer transition')
    ax.set_ylabel('Angle (rad)')
    ax.set_title('Rotor Angle Profiles')
    ax.legend(fontsize=7)

    # ── Holonomy curvature by layer ──
    ax = axes[0, 1]
    for i, r in enumerate(results):
        if r.curvature_by_layer.size > 0:
            ax.plot(r.curvature_by_layer, color=colors[i], linewidth=1.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean holonomy curvature')
    ax.set_title('Holonomy Curvature Profiles')

    # ── Condition number ──
    ax = axes[1, 0]
    for i, r in enumerate(results):
        ax.plot(r.condition_numbers, color=colors[i], linewidth=1.5)
    ax.set_xlabel('Layer transition')
    ax.set_ylabel('$\\kappa$')
    ax.set_title('Condition Number')

    # ── Summary table ──
    ax = axes[1, 1]
    ax.axis('off')
    cell_text = []
    for r in results:
        peak_curv = (int(r.curvature_by_layer.argmax())
                     if r.curvature_by_layer.size > 0 else "-")
        row = [
            r.text[:25] + '...' if len(r.text) > 25 else r.text,
            f"{r.angles.mean():.3f}",
            str(peak_curv),
            f"{r.condition_numbers.mean():.1f}",
        ]
        if r.dep_entropy is not None:
            row.append(f"{r.dep_entropy:.2f}")
        else:
            row.append("-")
        cell_text.append(row)

    table = ax.table(
        cellText=cell_text,
        colLabels=['Prompt', 'Mean\nangle', 'Peak curv\nlayer',
                   'Mean kappa', 'Dep.\nentropy'],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.8)
    ax.set_title('Summary', pad=20)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved: {save_path}")
    plt.close(fig)
    return fig


# ════════════════════════════════════════════════════════════════
# GA capacity
# ════════════════════════════════════════════════════════════════

def capacity(text: str,
             model: Model,
             whiten_components: int = 256) -> GACapacityProfile:
    """
    Compute the GA capacity profile for a prompt.

    Measures compositional capacity via bivector non-commutativity:
    how much successive layer rotations fail to commute, weighted
    by gradient-based dependency.

    Args:
        text: Input prompt string.
        model: Model object from load_model().
        whiten_components: Whitening dimension (default 256).

    Returns:
        GACapacityProfile with C_acc, C_eff, concentration, and
        per-layer contributions.

    Example:
        >>> model = ltg_ga.load_model("Qwen/Qwen2.5-7B")
        >>> cap = ltg_ga.capacity("Explain quantum entanglement", model=model)
        >>> print(f"C_acc = {cap.C_acc:.3f}, C_eff = {cap.C_eff:.3f}")
    """
    # ── 1. Extract and whiten ──
    H_raw = core.extract_hidden_states(
        model.hf_model, model.tokenizer, text, model.device
    )
    H_np = H_raw[1:].cpu().numpy()
    L, T, p = H_np.shape

    k = whiten_components
    H_flat = H_np.reshape(L * T, p)
    metric = core.estimate_metric(
        H_flat, n_components=min(k, min(L * T, p) - 1)
    )
    H_w = core.whiten(H_np, metric)

    # ── 2. Dependency (best-effort) ──
    D_layer = None
    try:
        dep_result = core.compute_dependency_density(
            model.hf_model, model.tokenizer, text, metric,
            device=model.device
        )
        D_layer = dep_result.D_layer
    except Exception as e:
        warnings.warn(f"Dependency computation failed (capacity will use "
                      f"C_acc only): {e}")

    # ── 3. GA capacity profile ──
    cap = ga_capacity_profile(H_w, D_layer=D_layer)
    return cap
