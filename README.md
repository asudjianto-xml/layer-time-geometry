# Learning Geometric Algebra Through Transformer Geometry

This is a book about **Geometric Algebra**. It teaches you the Clifford algebra Cl(*k*, 0) through a concrete, fascinating application: understanding how large language models process information internally.

You learn GA one concept at a time. Each chapter introduces a single algebraic idea, develops it with small examples you can work by hand, and then immediately applies it to real data extracted from a transformer language model. The transformer is your *laboratory*: a place where abstract algebraic concepts become visible, measurable, and surprising.

> *Rotors, bivectors, and holonomy in the hidden-state field of transformers*

**Authors:** Agus Sudjianto, Sandi Setiawan, Aijun Zhang

## Why Transformers?

Geometric Algebra was born in physics and has thrived in computer graphics, robotics, and signal processing. But transformers offer something these fields do not: a high-dimensional space (*k* = 256 after whitening) where *every* GA concept — bivectors, rotors, holonomy, commutators — appears naturally in the data.

In 3D physics, a bivector is just a dressed-up cross product. In R^256, bivectors have 32,640 independent components, multiple principal planes, and rich internal structure. This is where GA earns its keep.

By learning GA through this lens, you gain two things at once: a deep understanding of a beautiful algebra, and a new way to see inside the models that are reshaping technology.

## What You Need

- **Linear algebra**: matrix multiplication, eigenvalues, orthogonal matrices, SVD. A first undergraduate course is sufficient.
- **Python**: NumPy-level comfort. The code is simple and self-contained.
- **Calculus**: partial derivatives for the dependency chapter. You can skip that chapter if needed.
- **No prior GA knowledge** — the book starts from vectors.
- **No prior ML knowledge** — Appendix A covers everything you need about transformers.

## How to Read This Book

Each chapter follows the same pattern:

1. **The GA concept** — introduced abstractly with small (R^3 or R^4) examples.
2. **In the transformer** — the same concept applied to real hidden-state data from a language model.
3. **Code** — working Python using the `layer_time_ga` library. Every code block runs in the companion Jupyter notebooks.
4. **Exercises** — both pure GA problems and transformer-data explorations.

The chapters build sequentially. Read them in order the first time.

## Three Kinds of Statement

This book mixes mathematics, computation, and empirical observation. To keep the distinction clear, we use three categories throughout:

- **GA identity** (exact). A mathematical fact that holds in any Clifford algebra. Example: the geometric product decomposes as *ab* = *a* · *b* + *a* ∧ *b*. These do not depend on any model or dataset.
- **Computational construction** (defined). A procedure we define and apply to transformer data: whitening, the layer transition operator, versor decomposition, dependency density, etc. These are methodological choices, not theorems.
- **Empirical observation** (observed). A pattern we find in a specific model under a specific setup. These observations motivate the narrative, but they are not guaranteed to hold for all models, prompts, or whitening dimensions.

## Structure

### Part I: Vectors and Products (Chapters 1–4)
| Chapter | GA Concept | Transformer Insight |
|---------|-----------|-------------------|
| 1. Vectors Live Somewhere | Inner product, grade-0 | Hidden states encode alignment |
| 2. The Product That Does Everything | Geometric product | Both alignment and plane in one operation |
| 3. Planes, Not Axes | Bivectors, principal planes | Layer rotations have multiple planes |
| 4. When Your Coordinates Lie | Orthonormal frames, Cl(k,0) | Whitening establishes valid GA |

### Part II: Rotations and Rotors (Chapters 5–8)
| Chapter | GA Concept | Transformer Insight |
|---------|-----------|-------------------|
| 5. Rotations Without Matrices | Rotors, Rodrigues, Cayley map | Three ways to build rotations |
| 6. What a Layer Actually Does | Versor decomposition, three-phase structure, directional flow ratio | Grade-0 (stretch) vs grade-2 (rotation) |
| 7. Reading the Planes | Plane evolution, similarity | Rotation planes evolve across layers |
| 8. The Eigenvalue Story | Grade-0 dominance | Stretching controls gradients, not rotation |

### Part III: Curvature and Commutators (Chapters 9–11)
| Chapter | GA Concept | Transformer Insight |
|---------|-----------|-------------------|
| 9. When Order Matters | Commutators, Lie algebra so(k), Baker–Campbell–Hausdorff | Non-commutativity lives in specific planes |
| 10. Walking in Circles | Holonomy, nonseparability index, curvature regimes | Curvature has direction, not just magnitude |
| 11. How Much Computation? | Capacity, Jacobi identity | Decompose complexity by plane |

### Part IV: The Full Picture (Chapters 12–14)
| Chapter | GA Concept | Transformer Insight |
|---------|-----------|-------------------|
| 12. Which Planes Matter? | Dependency + bivectors | Identify functionally relevant planes |
| 13. Diagnosing and Steering | Cayley bivector, bivector coherence, Binet–Cauchy cosine, directional–radial decomposition | Context detection, hallucination diagnosis, plane-specific interventions |
| 14. What GA Gave Us | Retrospective | Directions invisible to scalar summaries |

### Appendices
| Appendix | Contents |
|----------|----------|
| A. Transformers: A Primer | What hidden states are, the layer–time grid, gradients and dependency |
| B. GA from Scratch | Formal Clifford algebra, blades, spin group, Lie algebra, pseudoscalar |
| C. The Matrix–Multivector Bridge | Systematic mapping between matrix operations and GA equivalents |
| D. Code Reference | Full API documentation for `ltg_ga` and `layer_time_ga` |

## Repository Contents

| Component | Path | Description |
|-----------|------|-------------|
| **Textbook** | `monograph_ga_learning.pdf` | 98-page book (PDF) |
| **LaTeX source** | `monograph_ga_learning.tex` | Book source |
| **Tutorials** | `tutorials_ga_learning/` | 14 companion Jupyter notebooks |
| **GA library** | `layer_time_ga/` | Bivectors, rotors, holonomy, commutators, capacity |
| **Student API** | `ltg_ga.py` | High-level 3-line interface |
| **Backend** | `layer_time_geometry.py` | PyTorch/NumPy numerical engine |
| **Figures** | `figures_ga_learning/` | All book figures (regenerable via `run_book_figures.py`) |
| **Tests** | `tests/` | Unit tests |

## Installation

```bash
pip install ga-transformer-geometry
```

Or install from source:

```bash
git clone https://github.com/asudjianto-xml/GA-Transformer-Geometry.git
cd GA-Transformer-Geometry
pip install -e .
```

With tutorial dependencies:

```bash
pip install -e ".[tutorials]"
```

## Quick Start

```python
import ltg_ga

# Load a model
model = ltg_ga.load_model("Qwen/Qwen2.5-7B")

# Run GA analysis
result = ltg_ga.analyse("The capital of France is", model=model)

# See the summary
result.summary()

# Generate the 4-panel GA summary plot
result.plot_ga_summary(save_path="ga_summary.png")
```

## The `layer_time_ga` Package

The GA library provides 34 public functions and classes across five modules:

### Core Algebra (`layer_time_ga.algebra`)

```python
from layer_time_ga.algebra import (
    Bivector, Rotor,
    geometric_product_vectors,     # ab = a.b + a^b
    bivector_from_skew,            # skew-symmetric matrix → Bivector
    rotor_from_orthogonal,         # orthogonal matrix → Rotor (via log)
    rodrigues_rotation,            # two vectors → rotation matrix (exact)
    rodrigues_rotor,               # two vectors → Rotor (via Rodrigues)
    cayley_bivector,               # two vectors → (Bivector, tau)
    rotor_compose, rotor_inverse,  # rotor arithmetic
    commutator_bivector,           # Lie bracket [B1, B2]
    grade_decomposition,           # M → grade-0 + grade-2
    binet_cauchy_cosine,           # oriented bivector alignment
    directional_flow_ratio,        # ||A||_F / ||S||_F
)
```

### Decomposition (`layer_time_ga.decomposition`)

```python
from layer_time_ga.decomposition import extract_rotor_field

rf = extract_rotor_field(H_whitened)
for vd in rf.decompositions:
    print(f"Layer {vd.layer_index}: angle={vd.rotor.angle:.4f}, "
          f"kappa={vd.condition_number:.1f}")
    planes = vd.bivector.principal_planes(n_planes=3)
```

### Curvature (`layer_time_ga.curvature`)

```python
from layer_time_ga.curvature import (
    holonomy_rotor,           # curvature at a single plaquette
    holonomy_scalar_map,      # (L-1, T-1) scalar curvature map
    nonseparability_index,    # D(s) = total curvature + regime classification
    commutator_field,         # pairwise ||[B_i, B_j]||_F
)

# Holonomy: curvature with direction
hr = holonomy_rotor(H_whitened, l=20, t=2)
print(f"Scalar curvature: {hr.scalar_curvature}")
print(f"Curvature plane: {hr.principal_plane}")

# Nonseparability: total interactive computation
ns = nonseparability_index(H_whitened)
print(f"D(s) = {ns['D_total']:.2f}, regime = {ns['regime']}")
```

### Capacity (`layer_time_ga.capacity`)

```python
from layer_time_ga.capacity import ga_capacity_profile

cap = ga_capacity_profile(H_whitened, D_layer=dep.D_layer)
print(f"Accumulated capacity: {cap.C_acc:.2f}")
print(f"Effective capacity: {cap.C_eff:.2f}")
```

## The Key Mapping

| Matrix Operation | GA Equivalent | What GA Adds |
|-----------------|--------------|-------------|
| Dot product a^T b | Inner product a · b | Same |
| Skew-symmetric A | Bivector B | Principal plane decomposition |
| Orthogonal U ∈ SO(k) | Rotor R = exp(−B/2) | Explicit plane + angle |
| Polar decomp T = UP | Versor decomp T = RP | Grade-2 × grade-0 separation |
| ‖U − I‖_F | Rotor angle θ | Plus: which plane |
| ‖[A_i, A_j]‖_F | ‖[B_i, B_j]‖ | Plus: decompose into planes |
| ‖Ω − I‖_F | Holonomy rotor | Plus: curvature direction |
| — | Cayley bivector | Magnitude + direction of context influence |
| — | Binet–Cauchy cosine | Oriented alignment (detects causal inversions) |
| — | Nonseparability index | Total interactive computation (single scalar) |

## Running the Tutorials

```bash
pip install -e ".[tutorials]"
cd tutorials_ga_learning
jupyter lab
```

Start with `ch01_vectors_live_somewhere.ipynb` and work through sequentially.

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Citation

If you use this work, please cite:

```
Sudjianto, A., Setiawan, S., and Zhang, A. (2026).
Learning Geometric Algebra Through Transformer Geometry.
https://github.com/asudjianto-xml/GA-Transformer-Geometry
```

## License

Apache 2.0. See [LICENSE](LICENSE).
