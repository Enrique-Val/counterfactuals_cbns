import os
import time

import numpy as np
import pandas as pd
from scipy.stats import sem, skew, kurtosis

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


import numpy as np
import time
import argparse
from scipy.linalg import sqrtm


# Assuming scm_to_mvn, scm_from_dag, random_dag are defined elsewhere in your code

def experiment_counterfactual_mvn(scm, obs_indices, int_indices, v=None):
    mu, Sigma, A = scm_to_mvn(scm)
    d = A.shape[0]

    # Default observation vector to zeros if not provided
    if v is None:
        v = np.zeros(len(obs_indices))

    ############################################################
    # 1. Structural Mutilation (Action)
    ############################################################
    B = np.eye(d) - np.linalg.inv(A)
    B_cf = B.copy()
    B_cf[int_indices, :] = 0.0  # Zero out all intervened rows
    A_cf = np.linalg.inv(np.eye(d) - B_cf)

    # Matrix to zero out the exogenous noise for all intervened vars
    J = np.eye(d)
    J[int_indices, int_indices] = 0.0

    ############################################################
    # 2. The Cross-World Joint MVN Distribution
    ############################################################
    Sig_XX = A @ A.T
    Sig_Xcf_Xcf = A_cf @ J @ A_cf.T
    Sig_Xcf_X = A_cf @ J @ A.T

    ############################################################
    # 3. MVN Conditional Distribution (Prediction via Abduction)
    ############################################################
    # Extract submatrices for the observed variables
    cov_Xcf_Xobs = Sig_Xcf_X[:, obs_indices]

    # np.ix_ allows us to extract the square submatrix for the observed vars
    var_Xobs = Sig_XX[np.ix_(obs_indices, obs_indices)]

    # Standard MVN Conditional Formulas using matrix inversion
    inv_var_Xobs = np.linalg.inv(var_Xobs)

    meanX_cf_mvn = cov_Xcf_Xobs @ inv_var_Xobs @ v
    covX_cf_mvn = Sig_Xcf_Xcf - cov_Xcf_Xobs @ inv_var_Xobs @ cov_Xcf_Xobs.T

    return meanX_cf_mvn, covX_cf_mvn, obs_indices, int_indices


def experiment_counterfactual_mixed_mvn(scm, obs_indices, int_indices, v=None):
    mu, Sigma, A = scm_to_mvn(scm)
    d = A.shape[0]

    if v is None:
        v = np.zeros(len(obs_indices))

    ############################################################
    # 1. Structural Mutilation (Action)
    ############################################################
    B = np.eye(d) - np.linalg.inv(A)
    B_cf = B.copy()
    B_cf[int_indices, :] = 0.0
    A_cf = np.linalg.inv(np.eye(d) - B_cf)

    J = np.eye(d)
    J[int_indices, int_indices] = 0.0

    ############################################################
    # 2. Identify Descendants & Create Mask Matrix M
    ############################################################
    # Node k is a descendant if A[k, j] is non-zero for ANY j in int_indices
    is_descendant = np.any(np.abs(A[:, int_indices]) > 1e-9, axis=1)
    is_descendant[int_indices] = False  # Intervened variables handled by J

    M_diag = np.where(is_descendant, 0.0, 1.0)
    M = np.diag(M_diag)

    ############################################################
    # 3. The Modified Cross-World Joint MVN
    ############################################################
    Sig_XX = A @ A.T
    Sig_Xcf_Xcf = A_cf @ J @ A_cf.T
    Sig_Xcf_X = A_cf @ J @ M @ A.T

    ############################################################
    # 4. MVN Conditional Distribution (Prediction)
    ############################################################
    cov_Xcf_Xobs = Sig_Xcf_X[:, obs_indices]
    var_Xobs = Sig_XX[np.ix_(obs_indices, obs_indices)]

    inv_var_Xobs = np.linalg.inv(var_Xobs)

    meanX_cf_mvn = cov_Xcf_Xobs @ inv_var_Xobs @ v
    covX_cf_mvn = Sig_Xcf_Xcf - cov_Xcf_Xobs @ inv_var_Xobs @ cov_Xcf_Xobs.T

    return meanX_cf_mvn, covX_cf_mvn, obs_indices, int_indices

def wasserstein_distance_same_mean(cov1, cov2):
    """
    Computes the 2-Wasserstein distance between two Gaussians
    with the same mean, given their covariance matrices.
    """
    # 1. Compute the matrix square root of cov1
    # 1. Decompose the matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov1)

    # 2. Force any tiny negative floating point errors to be exactly 0
    eigenvalues[eigenvalues < 0] = 0

    # 3. Reconstruct the square root matrix manually
    sqrt_cov1 = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T

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

def build_graph_plus_counterfactual_experiment(d=4, p=0.5, num_obs=2, num_ints=2):
    # Generate a random DAG and corresponding SCM
    dag = random_dag(d=d, p=p)

    scm = scm_from_dag(dag)

    # Run the counterfactual experiment with multiple observations and interventions
    wass = counterfactual_experiment(scm, num_obs=num_obs, num_ints=num_ints)
    return wass
    print(f"Final Wasserstein Distance: {wass:.4f}")

def counterfactual_experiment(scm, num_obs, num_ints):
    d = len(scm["nodes"])
    # Randomly select multiple unique variables for observation and intervention
    obs_indices = np.random.choice(d, num_obs, replace=False)
    int_indices = np.random.choice(d, num_ints, replace=False)


    res_mvn = experiment_counterfactual_mvn(scm, obs_indices, int_indices)
    res_mixed_mvn = experiment_counterfactual_mixed_mvn(scm, obs_indices, int_indices)

    mean_cf_mvn, cov_cf_mvn, _, _ = res_mvn
    mean_cf_mixed, cov_cf_mixed, _, _ = res_mixed_mvn

    #print("Cov of MVN Counterfactual:\n", np.round(cov_cf_mvn, 4))
    #print("Cov of Mixed-MVN Counterfactual:\n", np.round(cov_cf_mixed, 4))

    wass = wasserstein_distance_same_mean(cov_cf_mvn, cov_cf_mixed)
    #print(f"Wasserstein Distance between MVN and Mixed-MVN counterfactuals: {wass:.4f}")

    return wass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run causal experiments with SCM and CBN models.")
    parser.add_argument('--p', type=float, default=0.5, help="Probability of edge creation")
    parser.add_argument('--nodes', type=int, default=8, help="Number of nodes")
    parser.add_argument('--dir', type=str, default="./results/synthetic", help="Directory to save results to")
    args = parser.parse_args()

    res_dir = os.path.join(args.dir, "nodes_"+str(args.nodes))
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    N = 10000


    results_list = []

    for num_obs in range(1, args.nodes+1) :
        for num_ints in range(1, args.nodes) :
            print(f"\nRunning experiment with {num_obs} observations and {num_ints} interventions...")

            t0 = time.time()
            wass_list = []
            for _ in range(N):  # Run multiple trials to get an average Wasserstein
                wass = build_graph_plus_counterfactual_experiment(d=args.nodes, p=args.p, num_obs=num_obs, num_ints=num_ints)
                wass_list.append(wass)
            tn = time.time()

            print(f"Total time for {N} trials: {tn - t0:.2f} seconds")
            print()
            results_list.append({
                'nodes': args.nodes,
                'p': args.p,
                'num_obs': num_obs,
                'num_ints': num_ints,
                'mean_wass': np.mean(wass_list),
                'std_wass': np.std(wass_list),
                'median_wass' : np.median(wass_list),
                'iqr_wass' : np.percentile(wass_list, 75) - np.percentile(wass_list, 25),
                'sem_wass': sem(wass_list),

                # Extremes & Tails
                'min_wass': np.min(wass_list),
                'max_wass': np.max(wass_list),
                'p95_wass': np.percentile(wass_list, 95),
                'p99_wass': np.percentile(wass_list, 99),

                # Shape
                'skew_wass': skew(wass_list),
                'kurtosis_wass': kurtosis(wass_list),
                'time_seconds': tn - t0
            })

            results_df = pd.DataFrame(results_list)
            file_name = os.path.join(res_dir, "p_"+str(args.p) + ".csv")
            results_df.to_csv(file_name, index = False)

