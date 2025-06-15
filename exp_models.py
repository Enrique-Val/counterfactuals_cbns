from scipy.stats import laplace, norm, lognorm

from causal_ale.build_model import ModelBuilder
from causal_ale.model import Model

import numpyro.distributions as dist

from causal_ale.pyro_utils import layer, layerm, invert_layer, inv_softplus
from jax import nn


def get_large_backdoor_scm() -> Model:
    binary_treatment = False
    vars = ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"]
    exogenous_vars = [f"U_{v}" for v in vars]  # No noise for x8 (deterministic/derived)

    model_builder = ModelBuilder(vars + exogenous_vars, "large_backdoor_scm")

    # Add exogenous noise variables
    for u in exogenous_vars[:-2]:
        model_builder.add_equation(u, noise=norm(0, 1))

    # x0 (special case)
    def inv_x0(u, x0):
        return (inv_softplus(x0 + 1) / 1.8)
    model_builder.add_equation("x0", ["U_x0"], lambda u,u_x0: nn.softplus(1.8 * u_x0) - 1)
    model_builder.add_inverse_equation("U_x0", ["x0"], inv_x0)

    # x1 (special case)
    model_builder.add_equation("x1", ["x0", "U_x1"], lambda u,x0,u_x1: layer(x0, 0)*1.5 + 0.25*u_x1)
    model_builder.add_inverse_equation("U_x1", ["x1", "x0"], lambda u,x1,x0: (x1 - layer(x0, 0)*1.5) / 0.25)

    # x2-x6 (loopable structure)
    for i in range(2, 7):
        parent = f"x{i - 2}"  # x2 depends on x0, others chain
        model_builder.add_equation(
            f"x{i}",
            [parent, f"U_x{i}"],
            lambda u,parent_val, u_assoc: layer(parent_val, u_assoc)
        )
        model_builder.add_inverse_equation(
            f"U_x{i}",
            [f"x{i}", parent],
            lambda u,x_i, parent: invert_layer(x_i, parent)
        )

    # Treatment (x7)
    if binary_treatment:
        model_builder.add_equation("U_x7", noise=lognorm(0, 0.15))
        model_builder.add_equation("x7", ["x5", "U_x7"], lambda u,x5,u_x7: 1 if nn.softplus(x5 + 1) / 1.5 - 1 + u_x7 > 0.5 else 0)
    else:
        model_builder.add_equation("U_x7", noise=norm(0, 0.3))
        model_builder.add_equation("x7", ["x5","U_x7"], lambda u,x5,u_x7: nn.softplus(x5 + 1) * 1.5 - 1 + u_x7)
    model_builder.add_inverse_equation("U_x7", ["x7", "x5"], lambda u,x7,x5: x7 - nn.softplus(x5 + 1) * 1.5 + 1)
    # Outcome (x8)
    model_builder.add_equation("U_x8", noise=laplace(0, 0.6))
    model_builder.add_equation("x8", ["x6", "x7","U_x8"],
                               lambda u,x6, x7,u_x8: -nn.softplus((-x6 * 1.3 + x7) / 3 + 1) + 2+ u_x8)
    model_builder.add_inverse_equation("U_x8", ["x8", "x6", "x7"],
                                        lambda u,x8,x6,x7: nn.softplus((-x6 * 1.3 + x7) / 3 + 1) - 2 + x8)

    return model_builder.build()

def get_large_backdoor_cbn() -> Model:
    binary_treatment = False
    vars = ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"]

    model_builder = ModelBuilder(vars, "large_backdoor_cbn")

    # x0 (special case)
    model_builder.add_equation("x0", [], lambda u: nn.softplus(1.8 * u) - 1, noise=norm(0, 1))

    # x1 (special case)
    model_builder.add_equation("x1", ["x0"], lambda u,x0: layer(x0, 0)*1.5 + 0.25*u, noise=norm(0, 1))

    # x2-x6 (loopable structure)
    for i in range(2, 7):
        parent = f"x{i - 2}"  # x2 depends on x0, others chain
        model_builder.add_equation(
            f"x{i}",
            [parent],
            lambda u,parent_val: layer(parent_val, u),
            noise=norm(0, 1)
        )

    # Treatment (x7)
    if binary_treatment:
        model_builder.add_equation("x7", ["x5"], lambda u, x5: 1 if nn.softplus(x5 + 1) / 1.5 - 1 + u > 0.5 else 0, noise = lognorm(0, 0.15))
    else:
        model_builder.add_equation("x7", ["x5"], lambda u,x5: nn.softplus(x5 + 1) * 1.5 - 1 + u, noise = norm(0, 0.3))

    # Outcome (x8)
    model_builder.add_equation("x8", ["x6", "x7"],
                               lambda u,x6, x7: -nn.softplus((-x6 * 1.3 + x7) / 3 + 1) + 2 + u,
                               noise = laplace(0, 0.6))

    return model_builder.build()

def get_weak_arrow_scm() -> Model:
    binary_treatment = False  # or True, depending on your needs
    vars = ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"]
    exogenous_vars = [f"U_{v}" for v in vars]  # Noise for all variables

    model_builder = ModelBuilder(vars + exogenous_vars, "weak_arrow_scm")

    # Add exogenous noise variables
    for u in exogenous_vars:
        model_builder.add_equation(u, noise=norm(0, 1))

    # x0 (special case)
    model_builder.add_equation("x0", ["U_x0"], lambda u, u_x0: nn.softplus(1.8 * u_x0) - 1)

    # x1 (special case - different noise scale)
    model_builder.add_equation("x1", ["x0", "U_x1"], lambda u, x0, u_x1: layer(x0, 0) * 0.75 + 0.75 * u_x1)

    # x2-x6 (loopable structure)
    parents = ["x0", "x1", "x2", "x3", "x4"]  # Parents for x2-x6 respectively
    for i in range(2, 7):
        layer_fn = layerm if i in [2, 3] else layer  # Use layerm for x2, x3
        model_builder.add_equation(
            f"x{i}",
            [parents[i-2], f"U_x{i}"],
            lambda u, parent_val, u_xi: layer_fn(parent_val, u_xi)
        )

    # Treatment (x7)
    if binary_treatment:
        model_builder.add_equation("x7_cts", ["x5"], lambda u, x5: dist.LogNormal(nn.softplus(x5 + 1)/2 - 1, 0.12))
        model_builder.add_equation("x7", ["x7_cts"], lambda u, x7_cts: 1 if x7_cts > 0.6 else 0)
    else:
        model_builder.add_equation("U_x7", noise=norm(0, 0.3))
        model_builder.add_equation("x7", ["x5", "U_x7"], lambda u, x5, u_x7: nn.softplus(x5 + 1)*1.5 - 1 + u_x7)

    # Outcome (x8)
    model_builder.add_equation("U_x8", noise=laplace(0, 0.5))
    model_builder.add_equation("x8",
                             ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "U_x8"],
                             lambda u, x0, x1, x2, x3, x4, x5, x6, x7, u_x8:
                             nn.softplus(0.1*x0 + 0.1*x1 + 0.1*x2 + 0.1*x3 + 0.1*x4 + 0.1*x5 + 0.5*x6 + 0.7*x7 + 1) - 2 + u_x8)

    return model_builder.build()

