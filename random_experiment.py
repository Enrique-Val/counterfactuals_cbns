import numpy as np
import pandas as pd
from scipy.stats import norm

import counterfactual_cbn
from counterfactual_cbn import utils
from counterfactual_cbn.build_model import ModelBuilder


def random_dag(d, p=0.5):
    """
    Create a random DAG with nodes X1..Xd.
    Edge Xi->Xj allowed only if i<j.
    Edge probability = p.
    """
    parents = {i: [] for i in range(d)}
    for i in range(d):
        for j in range(i+1, d):
            if np.random.rand() < p:
                parents[j].append(i)
    return parents


def scm_from_dag(parents):
    """
    Convert DAG to SCM dictionary with:
      X_i = sum_{j in parents[i]} X_j + U_i
      U_i ~ N(0,1)
    Returns:
      {
        "nodes": ["X1", ..., "Xd"],
        "parents": {i: [...]},
        "weights": {i: np.ones(len(parents[i]))},
        "sigmas": {i: 1.0}
      }
    """
    d = len(parents)
    nodes = [f"X{i}" for i in range(d)]
    weights = {i: np.ones(len(parents[i])) for i in parents}
    sigmas = {i: 1.0 for i in parents}

    return {
        "nodes": nodes,
        "parents": parents,
        "weights": weights,
        "sigmas": sigmas
    }

def build_model_from_random_gaussian_scm(scm, name="random_scm") -> causal_ale.model.Model:
    """
    scm: dict produced by your random generator:
        - scm["nodes"]
        - scm["parents"]
        - scm["weights"]
        - scm["sigmas"]
    """

    # -------------------------------------------------------------
    # 1. Construct variable list: U_i first, then X_i
    # -------------------------------------------------------------
    U_vars = [f"U_{x}" for x in scm["nodes"]]
    X_vars = scm["nodes"]
    all_vars = U_vars + X_vars

    builder = ModelBuilder(all_vars, name=name)

    # -------------------------------------------------------------
    # 2. Add noise equations U_i := N(0, sigma_i)
    # -------------------------------------------------------------
    for i, x in enumerate(X_vars):
        sigma = scm["sigmas"][i]
        builder.add_equation(
            variable=f"U_{x}",
            parents=[],
            function=utils.identity,
            noise=norm(0, sigma)
        )

    # -------------------------------------------------------------
    # 3. Add structural equations: X_i := sum_j w_ij X_j + U_i
    # -------------------------------------------------------------
    for i, x in enumerate(X_vars):
        pa = scm["parents"][i]
        w = scm["weights"][i].copy()

        # build function f(X_pa..., U_i)
        def make_func(*args, w=w):
            if len(w) == 0:
                return args[-1]  # just return noise U_i
            u, *parents_values = args
            noise = parents_values[-1]
            parents_values = parents_values[:-1]
            to_ret = np.sum(np.multiply(w.reshape(-1,1),parents_values), axis=0) + noise
            return to_ret

        parent_names = [X_vars[j] for j in pa] + [f"U_{x}"]

        builder.add_equation(
            variable=x,
            parents=parent_names,
            function=make_func,
            noise=norm(0, 0)
        )

    # -------------------------------------------------------------
    # 5. Build and return Model
    # -------------------------------------------------------------
    return builder.build()


def scm_to_mvn(scm):
    """
    Returns the full MVN of (U, X):
    mean = 0
    covariance = [[I, A^T],
                  [A, A A^T]]
    """
    d = len(scm["nodes"])
    parents = scm["parents"]

    B = np.zeros((d, d))
    for i in range(d):
        for j in parents[i]:
            B[i,j] = 1.0

    print("B:\n", B)

    A = np.linalg.inv(np.eye(d) - B)

    Sig_UU = np.eye(d)
    Sig_UX = A.T
    Sig_XU = A
    Sig_XX = A @ A.T

    Sigma = np.block([
        [Sig_UU, Sig_UX],
        [Sig_XU, Sig_XX]
    ])

    mu = np.zeros(2 * d)
    return mu, Sigma, A

def prel_test() :
    d = 3
    p = 0.8
    parents = random_dag(d, p)
    scm = scm_from_dag(parents)
    print("DAG Parents:", parents)
    print("SCM:", scm)

    model = build_model_from_random_gaussian_scm(scm, name="random_gaussian_scm")
    data = model.generate_data(100000)
    summ = data.describe()
    print("Data Summary:\n", summ.to_string())

    mu, Sigma = scm_to_mvn(scm)
    mvn = np.random.multivariate_normal(mu, Sigma, size=100000)
    mvn_df = pd.DataFrame(mvn, columns=[f"U_X{i}" for i in range(d)] + [f"X{i}" for i in range(d)])
    print("MVN Sample Summary:\n", mvn_df.describe().to_string())

    diff = np.abs((data.describe() - mvn_df.describe())).mean(axis=1)
    print("Difference Summary:\n", diff.to_string())


def experiment_conditional_U(d=6, p=0.5, N=200000, k=5000, v=0.0, seed = None):
    if seed is not None and isinstance(seed, int) and seed >= 0:
        np.random.seed(seed)

    parents = random_dag(d, p)
    scm = scm_from_dag(parents)
    mu, Sigma, A = scm_to_mvn(scm)

    Sig_UU = Sigma[:d, :d]
    Sig_UX = Sigma[:d, d:]
    Sig_XX = Sigma[d:, d:]

    i = np.random.randint(0, d)  # variable to condition on

    cov_U_Xi = Sig_UX[:, i]
    var_Xi = Sig_XX[i, i]

    ############################################################
    # A. MVN conditional distribution of U | X_i=v
    ############################################################

    # mean
    meanU_mvn = cov_U_Xi * (v / var_Xi)

    # covariance
    covU_mvn = Sig_UU - np.outer(cov_U_Xi, cov_U_Xi) / var_Xi

    var_sumU_mvn = np.sum(covU_mvn)

    ############################################################
    # B. SCM closed-form conditional distribution
    #
    # Using:
    #   X_i = a_i^T U
    #   E[U | X_i=v] = Σ_U a_i * v / (a_i^T Σ_U a_i)
    #   Cov(U|X_i)    = Σ_U - Σ_U a_i a_i^T Σ_U / (a_i^T Σ_U a_i)
    #
    ############################################################

    Sigma_U = np.eye(d)
    a_i = A[i, :]

    denom = a_i @ a_i

    meanU_scm = (a_i / denom) * v
    covU_scm = Sigma_U - np.outer(a_i, a_i) / denom

    var_sumU_scm = np.sum(covU_scm)

    ############################################################
    # C. Monte-Carlo estimation of U | X_i ≈ v
    ############################################################
    rng = np.random.default_rng(3)
    samples = rng.multivariate_normal(mu, Sigma, size=N)

    U_samples = samples[:, :d]
    X_samples = samples[:, d:]
    Xi_samples = X_samples[:, i]

    idx = np.argsort(np.abs(Xi_samples - v))[:k]
    U_sel = U_samples[idx]

    meanU_mc = U_sel.mean(axis=0)
    covU_mc = np.cov(U_sel.T)
    var_sumU_mc = np.var(U_sel.sum(axis=1), ddof=1)

    ############################################################
    return {
        "parents": parents,
        "condition on X_i": i,
        "MVN_meanU": meanU_mvn,
        "SCM_meanU": meanU_scm,
        "MC_meanU": meanU_mc,
        "MVN_covU": covU_mvn,
        "SCM_covU": covU_scm,
        "MC_covU": covU_mc,
        "MVN_var_sumU": float(var_sumU_mvn),
        "SCM_var_sumU": float(var_sumU_scm),
        "MC_var_sumU": float(var_sumU_mc)
    }


############################################################
# 3. Run and print results
############################################################

if __name__ == "__main__":
    res = experiment_conditional_U(d = 4, p=0.8)
    print("Parents:", res["parents"])
    print("Conditioning on variable index:", res["condition on X_i"])
    print("\n=== Conditional Mean of U ===")
    print("MVN:", res["MVN_meanU"])
    print("SCM:", res["SCM_meanU"])
    print("MC :", res["MC_meanU"])
    print("\n=== Conditional Cov(U) ===")
    print("MVN:\n", res["MVN_covU"])
    print("\nSCM:\n", res["SCM_covU"])
    print("\nMC:\n", res["MC_covU"])
    print("\n=== Conditional Var(sum U) ===")
    print("MVN:", res["MVN_var_sumU"])
    print("SCM:", res["SCM_var_sumU"])
    print("MC :", res["MC_var_sumU"])