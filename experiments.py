import itertools
import time
from functools import partial

import networkx as nx
import numpy as np
import pandas as pd

from causal_ale.model import Model
import matplotlib.pyplot as plt

import multiprocess as mp

from exp_models import get_large_backdoor_scm, get_weak_arrow_scm, get_large_backdoor_cbn

def compute_marginal_means(model : Model):
    """Returns dict of marginal means for X1-X7"""
    # Run inference/sampling on base model to get marginals
    samples = model.generate_data(10000)  # Your inference method here
    return {
        'x0': np.mean(samples['x0']),
        'x1': np.mean(samples['x1']),
        'x2': np.mean(samples['x2']),
        'x3': np.mean(samples['x3']),
        'x4': np.mean(samples['x4']),
        'x5': np.mean(samples['x5']),
        'x6': np.mean(samples['x6']),
        'x7': np.mean(samples['x7']),
        'x8': np.mean(samples['x8'])
    }


def process_observation_vars(observation_vars, model, model_cbn, n_samples, eps, marginal_means, intervention_combs,
                             all_vars, joint_data):
    local_results = []
    local_result_mat = np.zeros((9, 2))
    local_count_mat = np.zeros((9, 2))

    k = len(observation_vars)
    obs_index = k - 1
    observation_set = {v: marginal_means[v] for v in observation_vars}
    observational_data_scm = model.get_observational_data(n_samples, observation_list=observation_set, eps=eps)
    observational_data_cbn = observational_data_scm[all_vars]  # CBN uses the same observational data

    # Ancestor set
    ancestor_set = set()
    for v in observation_vars:
        ancestor_set.update(nx.ancestors(model.get_graph(), v))
    ancestral_u_set = set(model.get_root_nodes()).intersection(ancestor_set)

    for intervention_vars in intervention_combs:
        int_index = len(intervention_vars) - 1
        intervention_set = {v: marginal_means[v] for v in intervention_vars}

        # Check d-separation
        if nx.is_d_separator(model.get_graph(), set(["x8"]), ancestral_u_set, set(intervention_vars)):
            print("Skipped")
            print(
                f"Skipping intervention {intervention_vars} with observations {observation_vars} due to d-separation condition.")
            x8_samples = model.intervene_data(joint_data, intervention_list=intervention_set,
                                              method='non_descendants')
            x8_samples = x8_samples['x8']
            local_results.append({
                'model': 'SCM',
                'intervention_vars': intervention_vars,
                'observation_vars': observation_vars,
                'x8_mean': np.mean(x8_samples),
                'x8_var': np.var(x8_samples)
            })
            local_results.append({
                'model': 'CBN',
                'intervention_vars': intervention_vars,
                'observation_vars': observation_vars,
                'x8_mean': np.mean(x8_samples),
                'x8_var': np.var(x8_samples)
            })
        else:
            print(f"Running SCM experiment with interventions: {intervention_vars}, observations: {observation_vars}")

            x8_samples_scm = model.intervene_data(observational_data_scm, intervention_list=intervention_set,
                                                  method='root_nodes')
            x8_samples_scm = x8_samples_scm['x8']

            local_results.append({
                'model' : 'SCM',
                'intervention_vars': intervention_vars,
                'observation_vars': observation_vars,
                'x8_mean': np.mean(x8_samples_scm),
                'x8_var': np.var(x8_samples_scm)
            })

            print(f"Running CBN experiment with interventions: {intervention_vars}, observations: {observation_vars}")

            x8_samples_cbn = model_cbn.intervene_data(observational_data_cbn, intervention_list=intervention_set,
                                                      method='non_descendants')
            x8_samples_cbn = x8_samples_cbn['x8']

            local_results.append({
                'model': 'CBN',
                'intervention_vars': intervention_vars,
                'observation_vars': observation_vars,
                'x8_mean': np.mean(x8_samples_cbn),
                'x8_var': np.var(x8_samples_cbn)
            })
            local_result_mat[obs_index, int_index] += np.var(x8_samples_cbn) - np.var(x8_samples_scm)
        local_count_mat[obs_index, int_index] += 1

    return local_results, local_result_mat, local_count_mat


def full_experiment_suite(model: Model, model_cbn: Model, n_samples, eps, parallel=True):
    marginal_means = compute_marginal_means(model)
    joint_data = model.generate_data(1000)

    # Generate all non-empty subsets of X1-X7
    total_iters = 0
    skipped_iters = 0
    intervention_combs = [("x" + str(i),) for i in range(0, 8)]  # X1 to X7
    intervention_combs += [("x0", "x" + str(i)) for i in range(1, 8)]  # X0 with X1 to X7
    intervention_combs += list(itertools.product(["x1", "x3", "x5", "x7"], ["x2", "x4", "x6"]))
    all_vars = ['x' + str(i) for i in range(0, 9)]  # X1 to X7

    # Prepare arguments for parallel processing
    observation_combinations = []
    for k in range(1, len(all_vars) + 1):
        observation_combinations.extend(list(itertools.combinations(all_vars, k)))


    if not parallel:
        results = []
        for observation_combination in observation_combinations:
            results.append(process_observation_vars(observation_combination, model, model_cbn, n_samples, eps,
                                     marginal_means, intervention_combs, all_vars, joint_data))
    else :
        pool = mp.Pool(processes=mp.cpu_count()-1)
        results = pool.starmap(
            process_observation_vars,
            [(obs_comb, model, model_cbn, n_samples, eps, marginal_means, intervention_combs, all_vars, joint_data)
             for obs_comb in observation_combinations]
        )
        pool.close()
        pool.join()


    # Combine results
    combined_results = []
    combined_result_mat = np.zeros((9, 2))
    combined_count_mat = np.zeros((9, 2))

    for result in results:
        res, res_mat, count_mat = result
        combined_results.extend(res)
        combined_result_mat += res_mat
        combined_count_mat += count_mat

    # Calculate total iterations
    total_iters = np.sum(combined_count_mat)
    skipped_iters = total_iters - np.count_nonzero(combined_result_mat)

    print(f"Total iterations: {total_iters}, Skipped iterations: {skipped_iters}")
    average_var_mat = np.divide(combined_result_mat, combined_count_mat)

    return pd.DataFrame(combined_results), average_var_mat

if __name__ == "__main__" :
    #mp.set_start_method('spawn')
    N = 10000000
    eps = 0.05
    scm = get_large_backdoor_scm()
    cbn = get_large_backdoor_cbn()
    '''end_vars = ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"]
    data = scm.generate_data(10000)
    #data = scm.get_observational_data_kde(N, [("x"+str(i), data[end_vars[i]].mean()) for i in range(1)], eps,
                                          #functional=True)
    # Create matrix of plots. Diagonal is marginal distribution, else it is scatter plot of variables
    fig, axs = plt.subplots(len(end_vars), len(end_vars), figsize=(30,30))
    for i in range(len(end_vars)):
        for j in range(len(end_vars)):
            if i == j:
                # Plot density with seaborn

                sns.histplot(data[end_vars[i]], ax=axs[i, j], stat="density", bins=200)
            else:
                # Plot scatter plot with seaborn with grid
                sns.scatterplot(x=data[end_vars[j]], y=data[end_vars[i]], ax=axs[i, j], alpha=0.1)
                axs[i, j].set_xlabel(end_vars[j])
                axs[i, j].set_ylabel(end_vars[i])
    plt.tight_layout()
    plt.show()
    print(len(data.index))
    print(data.describe())'''

    # Plot graphs of scm and cbn

    scm_graph : nx.DiGraph = scm.get_graph()

    cbn_graph : nx.DiGraph = cbn.get_graph()

    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    nx.draw(scm_graph, with_labels=True, node_size=700, node_color='lightblue', font_size=10, font_color='black', arrows=True)
    plt.title("SCM Graph")
    plt.subplot(1, 2, 2)
    nx.draw(cbn_graph, with_labels=True, node_size=700, node_color='lightgreen', font_size=10, font_color='black', arrows=True)
    plt.title("CBN Graph")
    plt.tight_layout()
    plt.show()




    # Run the full experiment suite
    t0 = time.time()
    results, mat = full_experiment_suite(scm, cbn, N, eps, parallel=True)
    t1 = time.time()
    print(f"Experiment completed in {t1 - t0:.2f} seconds.")
    # Save result to json
    results.to_json("results.json", orient="records", lines=True)
    # Save matrix to csv
    mat = pd.DataFrame(mat, columns=["Intervention 1", "Intervention 2"], index=[f"Observation {i+1}" for i in range(9)])
    mat.to_csv("result_matrix.csv", index=True)
    print(mat)
