import numpy as np

NORMAL = 0
STRIKE_SLIP = 1
INVERSE = 2

def get_nomogram(pp, s_v, fric_coef):
    """
    Returns the two points that describes a nomogram.

    The nomogram is useful to compute and visualize the possible values of Shmin and SHmax in induced seismicity.

    Parameters
        ----------
        pp : double
            The value of the pore pressure
        s_v : double
            The value of the vertical effort

        Returns
        ------
        tuple
            A tuple that contains another two tuples with that represent the points that describe the nomogram
    """
    K = np.power(np.sqrt(fric_coef ** 2 + 1) + fric_coef, 2)

    a = (s_v - pp) / K + pp
    b = K * (a - pp) + pp
    d = K * (s_v - pp) + pp
    c = (d - pp) / K + pp
    return (a, b), (c, d)


def get_point_from_triangle(a, b, c, r1, r2):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    return (1 - np.sqrt(r1)) * a + (np.sqrt(r1) * (1 - r2)) * b + (np.sqrt(r1) * r2) * c


# Compute pore pressure as estimated using hydrostatic pp and s_v
def f_pp(u, hpp, s_v, r_pp):
    # return uniform(loc=hpp, scale=s_v - hpp).ppf(r_pp)
    return (hpp + s_v) / 2 + r_pp


def f_shmin(u, pp, s_v, stress_regime, fric_coef, r1, r2):
    # 1. Compute nomogram points for ALL samples simultaneously
    (a, b), (c, d) = get_nomogram(pp, s_v, fric_coef)

    # 2. Define masks for the regimes
    conds = [
        stress_regime == NORMAL,
        stress_regime == STRIKE_SLIP,
        stress_regime == INVERSE
    ]

    # 3. Vectorize the routing logic using np.select
    p1 = np.select(conds, [a, a, c])
    p2 = np.select(conds, [a, c, c])
    p3 = np.select(conds, [b, b, d])

    # 4. Compute all triangle points in one array operation
    return get_point_from_triangle(p1, p2, p3, r1, r2)


def f_shmax(u, pp, s_v, stress_regime, fric_coef, r1, r2):
    # 1. Compute nomogram points for ALL samples simultaneously
    (a, b), (c, d) = get_nomogram(pp, s_v, fric_coef)

    # 2. Define masks for the regimes
    conds = [
        stress_regime == NORMAL,
        stress_regime == STRIKE_SLIP,
        stress_regime == INVERSE
    ]

    # 3. Vectorize the routing logic using np.select
    p1 = np.select(conds, [b, b, d])
    p2 = np.select(conds, [a, d, c])
    p3 = np.select(conds, [b, b, d])

    # 4. Compute all triangle points in one array operation
    return get_point_from_triangle(p1, p2, p3, r1, r2)


def f_shmin_normal(u, pp, s_v, fric_coef, u_min):
    """
    SCM node for Shmin (Normal Regime).
    Shmin acts as the baseline horizontal stress.
    """
    # 1. Get nomogram bounds (a = frictional lower limit, b = Sv)
    (a, b), (c, d) = get_nomogram(pp, s_v, fric_coef)

    # 2. Deterministic function: Expected Shmin is between frictional limit and Sv
    # Often Shmin sits closer to the lower limit 'a' in active normal faulting
    expected_shmin = (a + b) / 2.0

    # 3. Additive noise
    return expected_shmin + u_min


def f_shmax_normal(u, pp, s_v, fric_coef, shmin, u_max):
    """
    SCM node for Shmax (Normal Regime).
    Takes 'shmin' as a causal parent to enforce Shmin <= Shmax <= Sv.
    """
    # 1. Get nomogram bounds
    (a, b), (c, d) = get_nomogram(pp, s_v, fric_coef)

    # 2. Deterministic function: Shmax must sit between Shmin and Sv (b)
    # This guarantees the Andersonian definition for normal faulting is maintained.
    expected_shmax = (shmin + b) / 2.0

    # 3. Additive noise
    return expected_shmax + u_max


## CBN variants
def f_pp_cbn(u, hpp, s_v):
    return (hpp + s_v) / 2 + u

def f_shmin_normal_cbn(u,pp, s_v, fric_coef):
    (a, b), (c, d) = get_nomogram(pp, s_v, fric_coef)
    expected_shmin = (a + b) / 2.0
    return expected_shmin + u

def f_shmax_normal_cbn(u,pp, s_v, fric_coef, shmin):
    (a, b), (c, d) = get_nomogram(pp, s_v, fric_coef)
    expected_shmax = (shmin + b) / 2.0
    return expected_shmax + u

import numpy as np

def f_seismic_failure(u, s_v, s_hmax, s_hmin, pp, mu):
    """
    Vectorized SCM node for Seismic Failure (Deterministic).
    Returns an array of 1s (slip/seism) and 0s (stable).
    """
    # Pre-existing fault assumption
    cohesion = 0.0

    # 1. Identify the absolute max and min principal stresses across the arrays
    # axis=0 ensures it compares element-by-element across the three arrays
    sigma_1 = np.max([s_v, s_hmax, s_hmin], axis=0)
    sigma_3 = np.min([s_v, s_hmax, s_hmin], axis=0)

    # 2. Calculate Mohr Circle geometric properties (Vectorized by default)
    radius = (sigma_1 - sigma_3) / 2.0
    center_eff = ((sigma_1 + sigma_3) / 2.0) - pp

    # 3. Convert friction coefficient to friction angle (Vectorized by default)
    phi = np.arctan(mu)

    # 4. Calculate the failure threshold
    failure_threshold = (center_eff * np.sin(phi)) + (cohesion * np.cos(phi))

    # 5. Evaluate Failure Conditions using bitwise OR (|)
    # Condition A: Effective stress is negative (tension break)
    # Condition B: Shear drive (radius) exceeds frictional resistance
    is_failure = np.where((center_eff < 0) | (radius >= failure_threshold), 1, 0)

    return is_failure

