import argparse
import time

import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
from scipy.stats import norm

from causal_ale import utils
from causal_ale.build_model import ModelBuilder
from random_experiment import random_dag, scm_to_mvn, scm_from_dag


def experiment_counterfactual(scm, i,j, v=0.0):
    mu, Sigma, A = scm_to_mvn(scm)
    d = A.shape[0]

    ############################################################
    # STEP 1: ABDUCTION (Update U | X_i=v)
    ############################################################
    Sigma_U = np.eye(d)
    a_i = A[i, :]
    denom = a_i @ a_i

    meanU_post = (a_i / denom) * v
    covU_post = Sigma_U - np.outer(a_i, a_i) / denom

    ############################################################
    # STEP 2: ACTION / INTERVENTION (Mutilate the SCM)
    ############################################################
    # The matrix A is the reduced form matrix: X = A*U.
    # In a linear SCM, X = B*X + U, meaning A = (I - B)^-1.
    # We can recover the structural weight matrix B from A:
    B = np.eye(d) - np.linalg.inv(A)

    # Mutilate the graph for do(X_j = 0)
    # Remove the parents of X_j by setting the j-th row of B to 0
    B_cf = B.copy()
    B_cf[j, :] = 0.0

    # Since the intervention fixes X_j = 0 deterministically,
    # we replace the j-th exogenous variable's distribution with a Dirac delta at 0.
    meanU_cf = meanU_post.copy()
    meanU_cf[j] = 0.0  # Set to the intervention value

    covU_cf = covU_post.copy()
    covU_cf[j, :] = 0.0
    covU_cf[:, j] = 0.0

    ############################################################
    # STEP 3: PREDICTION (Compute Counterfactual Distribution)
    ############################################################
    # Calculate the new structural matrix A_cf = (I - B_cf)^-1
    A_cf = np.linalg.inv(np.eye(d) - B_cf)

    # The counterfactual distribution of X given X_i=v and do(X_j=0)
    meanX_cf = A_cf @ meanU_cf
    covX_cf = A_cf @ covU_cf @ A_cf.T

    return meanX_cf, covX_cf, i, j

def experiment_counterfactual_mvn(scm,i,j, v=0.0):
    mu, Sigma, A = scm_to_mvn(scm)
    d = A.shape[0]

    ############################################################
    # 1. Structural Mutilation (Action)
    ############################################################
    # Recover B from A, mutilate it, and compute A_cf
    B = np.eye(d) - np.linalg.inv(A)
    B_cf = B.copy()
    B_cf[j, :] = 0.0
    A_cf = np.linalg.inv(np.eye(d) - B_cf)

    # Matrix to zero out the j-th exogenous noise
    J = np.eye(d)
    J[j, j] = 0.0

    ############################################################
    # 2. The Cross-World Joint MVN Distribution
    ############################################################
    # Factual covariance (same as Sig_XX from your original code)
    Sig_XX = A @ A.T

    # Counterfactual covariance
    Sig_Xcf_Xcf = A_cf @ J @ A_cf.T

    # Cross-world covariance between Counterfactual X and Factual X
    Sig_Xcf_X = A_cf @ J @ A.T

    ############################################################
    # 3. MVN Conditional Distribution (Prediction via Abduction)
    # Target: X_cf | X_i = v
    ############################################################
    # We extract the cross-covariance specifically for the observed variable X_i
    cov_Xcf_Xi = Sig_Xcf_X[:, i]
    var_Xi = Sig_XX[i, i]

    # Standard MVN Conditional Formulas:
    # mu_a|b = mu_a + Cov(a,b) * Var(b)^-1 * (b - mu_b)
    # Cov_a|b = Var(a) - Cov(a,b) * Var(b)^-1 * Cov(b,a)

    meanX_cf_mvn = cov_Xcf_Xi * (v / var_Xi)
    covX_cf_mvn = Sig_Xcf_Xcf - np.outer(cov_Xcf_Xi, cov_Xcf_Xi) / var_Xi

    return meanX_cf_mvn, covX_cf_mvn, i, j


def experiment_counterfactual_mixed_mvn(scm, i,j, v=0.0):
    mu, Sigma, A = scm_to_mvn(scm)
    d = A.shape[0]

    ############################################################
    # 1. Structural Mutilation (Action)
    ############################################################
    B = np.eye(d) - np.linalg.inv(A)
    B_cf = B.copy()
    B_cf[j, :] = 0.0
    A_cf = np.linalg.inv(np.eye(d) - B_cf)

    J = np.eye(d)
    J[j, j] = 0.0

    ############################################################
    # 2. Identify Descendants & Create Mask Matrix M
    ############################################################
    # In the structural matrix A, A[k, j] != 0 means U_j affects X_k.
    # Therefore, k is a descendant of j if A[k, j] is non-zero.
    is_descendant = np.abs(A[:, j]) > 1e-9
    is_descendant[j] = False  # The intervened variable is handled by J

    # M_kk = 1 if non-descendant (keep covariance), 0 if descendant (break covariance)
    M_diag = np.where(is_descendant, 0.0, 1.0)
    M = np.diag(M_diag)

    ############################################################
    # 3. The Modified Cross-World Joint MVN
    ############################################################
    Sig_XX = A @ A.T

    # Unchanged: The marginal variance of U_cf is still I
    Sig_Xcf_Xcf = A_cf @ J @ A_cf.T

    # CHANGED: We insert M to break the covariance for descendants
    Sig_Xcf_X = A_cf @ J @ M @ A.T

    ############################################################
    # 4. MVN Conditional Distribution (Prediction)
    ############################################################
    cov_Xcf_Xi = Sig_Xcf_X[:, i]
    var_Xi = Sig_XX[i, i]

    meanX_cf_mvn = cov_Xcf_Xi * (v / var_Xi)
    covX_cf_mvn = Sig_Xcf_Xcf - np.outer(cov_Xcf_Xi, cov_Xcf_Xi) / var_Xi

    return meanX_cf_mvn, covX_cf_mvn, i, j


def kl_same_mean(cov_p, cov_q):
    """
    Computes the standard KL divergence KL(P || Q) between two Gaussians
    with the same mean, where P is the true distribution and Q is the approximation.
    """
    d = cov_p.shape[0]

    # Use slogdet for better numerical stability than np.linalg.det
    _, logdet_p = np.linalg.slogdet(cov_p)
    _, logdet_q = np.linalg.slogdet(cov_q)

    # We only need the inverse of the approximate distribution's covariance
    inv_cov_q = np.linalg.inv(cov_q)

    # Calculate KL(P || Q)
    kl_divergence = 0.5 * (np.trace(inv_cov_q @ cov_p) - d + logdet_q - logdet_p)

    return kl_divergence


def wasserstein_distance_same_mean(cov1, cov2):
    """
    Computes the 2-Wasserstein distance between two Gaussians
    with the same mean, given their covariance matrices.
    """
    # 1. Compute the matrix square root of cov1
    sqrt_cov1 = sqrtm(cov1)

    # 2. Compute the inner term: sqrt(cov1) * cov2 * sqrt(cov1)
    inner_term = sqrt_cov1 @ cov2 @ sqrt_cov1

    # 3. Compute the matrix square root of the inner term
    sqrt_inner_term = sqrtm(inner_term)

    # 4. Calculate the squared Wasserstein distance
    w2_squared = np.trace(cov1) + np.trace(cov2) - 2 * np.trace(sqrt_inner_term)

    # 5. Handle numerical precision issues (discard negligible imaginary parts)
    w2_squared = np.real(w2_squared)

    # 6. Guard against tiny negative values due to floating-point errors
    w2_squared = max(0.0, w2_squared)

    # Return the actual distance (not squared)
    return np.sqrt(w2_squared)

if __name__ == "__main__":
    # Add command line args
    parser = argparse.ArgumentParser(description="Run causal experiments with SCM and CBN models.")
    parser.add_argument('--p', type=float, default=0.5, help="Probability of edge creation in the random DAG")
    parser.add_argument('--parallel', action='store_true', help="Run experiments in parallel")

    args = parser.parse_args()
    t0 = time.time()
    prob_arcs = 0.8
    d=8
    dag = random_dag(d=d, p=prob_arcs)
    print("Generated DAG (Parents List):", dag)
    scm = scm_from_dag(dag)
    i = np.random.randint(0, d)  # Variable to condition on (Observation)
    j = np.random.randint(0, d)  # Variable to intervene on (Action)
    print("Observing variable index (i):", i)
    print("Intervening variable index (j):", j)
    res_mvn = experiment_counterfactual_mvn(scm,i,j)
    res_mixed_mvn = experiment_counterfactual_mixed_mvn(scm,i,j)

    # Compute JSD between the two counterfactual distributions (assume they are Gaussian and the means are the same for simplicity)
    mean_cf_mvn, cov_cf_mvn, _, _ = res_mvn
    mean_cf_mixed, cov_cf_mixed, _, _ = res_mixed_mvn

    print("Cov of MVN Counterfactual:\n", cov_cf_mvn)
    print("Cov of Mixed-MVN Counterfactual:\n", cov_cf_mixed)

    wass = wasserstein_distance_same_mean(cov_cf_mvn, cov_cf_mixed)
    print(f"Wasserstein Distance between MVN and Mixed-MVN counterfactuals: {wass:.4f}")
    print(f"Time taken: {time.time() - t0:.2f} seconds")


