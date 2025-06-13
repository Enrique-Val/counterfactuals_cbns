import itertools

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import norm, laplace

from causal_ale.build_model import ModelBuilder
from causal_ale.model import Model
import matplotlib.pyplot as plt

from causal_ale.pyro_utils import layer
from causal_ale.utils import make_equation

import seaborn as sns

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

def full_experiment_suite(model : Model, model_cbn : Model, n_samples, eps):
    marginal_means = compute_marginal_means(model)
    results = []
    result_mat = np.zeros((9,2))
    count_mat = np.zeros((9,2))

    joint_data = model.generate_data(1000)

    '''x8_marginal = model.generate_data(n_samples)
    x8_marginal = x8_marginal['x8']
    results.append({
        'intervention_vars': [],
        'observation_vars': [],
        'x8_mean': np.mean(x8_marginal),
        'x8_std': np.std(x8_marginal)
    })'''

    # Generate all non-empty subsets of X1-X7
    total_iters = 0
    skipped_iters = 0
    intervention_combs = [("x" + str(i),) for i in range(0, 8)]  # X1 to X7
    # Consider combination of x0 with the rest of the variables
    intervention_combs += [("x0", "x" + str(i)) for i in range(1, 8)][-2:]  # X0 with X1 to X7
    # Consider the product of ["x1","x3","x5","x7"] with the ["x2","x4","x6"] variables
    intervention_combs += list(itertools.product(["x1", "x3", "x5", "x7"], ["x2", "x4", "x6"]))[-2:]
    all_vars = ['x' + str(i) for i in range(0, 9)]  # X1 to X7
    for k in range(1, len(all_vars) + 1-7):
        obs_index = k - 1
        for observation_vars in list(itertools.combinations(all_vars, k))[:2]:
            observation_set = {v: marginal_means[v] for v in observation_vars}
            observational_data_scm = model.get_observational_data(n_samples, observation_list=observation_set, eps=eps)
            observational_data_cbn = model_cbn.get_observational_data(n_samples, observation_list=observation_set, eps=eps, is_scm=False)
            # Ancestor set
            ancestor_set = set()
            for v in observation_vars:
                ancestor_set.update(nx.ancestors(model.get_graph(), v))

            ancestral_u_set = set(model.get_root_nodes()).intersection(ancestor_set)

            for intervention_vars in intervention_combs:
                int_index = len(intervention_vars) - 1
                # Create intervention/observation dicts
                intervention_set = {v: marginal_means[v] for v in intervention_vars}

                # Check if ancestors of observations that are also root nodes are d-separated
                # from x_8 given the intervened nodes

                # Check d-separation
                if nx.is_d_separator(model.get_graph(), set(["x8"]), ancestral_u_set, set(intervention_vars)):
                    print("Skipped")
                    print(f"Skipping intervention {intervention_vars} with observations {observation_vars} due to d-separation condition.")
                    x8_samples = model.intervene_data(joint_data, intervention_list=intervention_set,
                                                      method='non_descendants')
                    x8_samples = x8_samples['x8']
                    results.append({
                        'intervention_vars': intervention_vars,
                        'observation_vars': observation_vars,
                        'x8_mean': np.mean(x8_samples),
                        'x8_var': np.var(x8_samples)
                    })
                    results.append({
                        'intervention_vars': intervention_vars,
                        'observation_vars': observation_vars,
                        'x8_mean': np.mean(x8_samples),
                        'x8_var': np.var(x8_samples)
                    })
                    skipped_iters += 1


                else :
                    print(f"Running SCM experiment with interventions: {intervention_vars}, observations: {observation_vars}")

                    x8_samples_scm = model.intervene_data(observational_data_scm, intervention_list=intervention_set,
                                                      method = 'root_nodes')

                    x8_samples_scm = x8_samples_scm['x8']

                    results.append({
                        'intervention_vars': intervention_vars,
                        'observation_vars': observation_vars,
                        'x8_mean': np.mean(x8_samples_scm),
                        'x8_var': np.var(x8_samples_scm)
                    })

                    print(f"Running CBN experiment with interventions: {intervention_vars}, observations: {observation_vars}")

                    # Run experiment
                    x8_samples_cbn = model_cbn.intervene_data(observational_data_cbn, intervention_list=intervention_set,
                                                          method='non_descendants')

                    x8_samples_cbn = x8_samples_cbn['x8']

                    results.append({
                        'intervention_vars': intervention_vars,
                        'observation_vars': observation_vars,
                        'x8_mean': np.mean(x8_samples_cbn),
                        'x8_var': np.var(x8_samples_cbn)
                    })
                    result_mat[obs_index, int_index] += np.var(x8_samples_cbn) - np.var(x8_samples_scm)
                total_iters += 1
                count_mat[obs_index, int_index] += 1

    print(f"Total iterations: {total_iters}, Skipped iterations: {skipped_iters}")
    # Calculate average variance for each observation set and intervention set using elementwise division
    average_var_mat = np.divide(result_mat, count_mat)


    return pd.DataFrame(results), average_var_mat

if __name__ == "__main__" :
    N = 1000000
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
    results, mat = full_experiment_suite(scm, cbn, N, eps)
    # Save result to json
    results.to_json("results.json", orient="records", lines=True)
    print(mat)
