
"""Torch-free implementation of the a-energy estimator (Theorem A of the manuscript)."""

from .estimator import (
    AEnergyResult,
    a_energy_estimator,
    estimator_from_vectors,
    true_error_a_quadrature,
)

from .fem import (
    assemble_stiffness,
    cell_average,
    residual_vector,
    triangle_rule,
)

from .mesh import (
    TriMesh,
    build_mesh,
    l_shaped_mesh,
    unit_square_mesh,
)

from .references import (
    boundary_norms,
    lshape_grad_u,
    lshape_u,
    lshape_true_error_boundary_integral,
    lshape_true_error_graded,
)

from .evaluate import evaluate_from_callables

__all__ = [
    "AEnergyResult",
    "TriMesh",
    "a_energy_estimator",
    "assemble_stiffness",
    "boundary_norms",
    "build_mesh",
    "cell_average",
    "estimator_from_vectors",
    "evaluate_from_callables",
    "l_shaped_mesh",
    "lshape_grad_u",
    "lshape_u",
    "lshape_true_error_boundary_integral",
    "lshape_true_error_graded",
    "residual_vector",
    "triangle_rule",
    "true_error_a_quadrature",
    "unit_square_mesh",
]
