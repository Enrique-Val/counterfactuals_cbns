import json
import os
import time
from itertools import product, combinations, permutations
from statistics import LinearRegression

import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm, uniform, truncnorm, rv_discrete
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR

from causal_ale.build_model import ModelBuilder
import seaborn as sns
import numpy as np
import pybnesian as pb

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from experiments.geophysics_equations import f_pp, f_shmin_normal, f_shmax_normal, f_shmin_normal_cbn, f_shmax_normal_cbn, f_pp_cbn, f_seismic_failure


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

# VARIABLE NAMES
WD = "Water\n density"
RD = "Rock\n density"
HPP = "Hydrostatic\n pore pressure"
SV = "Vertical\n stress"
FC = "Friction\n coefficient"
PP = "Pore pressure"
SHMIN = "Minimum\n horizontal stress"
SHMAX = "Maximum\n horizontal stress"
FAILURE = "Slip"

variables = [WD,
             RD,
             HPP,
             SV,
             FC,
             PP,
             SHMIN,
             SHMAX,
             FAILURE]

if __name__ == '__main__':

    exogenous_vars = ["r_pp","r1","r2"]

    json_file = "model_inputs_original.json"
    inputs = json.load(open(json_file))

    # CONSTANTES
    g = 9.81  # aceleración de la gravedad
    Z_f = 0  # Phreatic water depth

    # Build the priors
    H_1 = inputs["H_1"]
    Z = inputs["Z"]
    Z_f = 0
    H_2 = Z - H_1  # Diferencia entre profundidad del agua y del pozo

    ## A PRIORIS / Inputs a introducir
    ro_s = norm(inputs["rd_mean"], inputs["rd_sd"] / 3)  # Entre 2300 y 2800   2550 unc range 250
    ro_w = norm(inputs["wd_mean"], inputs["wd_sd"] / 3)  # Entre 1000 y 1100    1050 unc range 50
    # Codificación de variables discretas
    NORMAL = 1
    STRIKE_SLIP = 2
    INVERSE = 3
    stress_regime = rv_discrete(
        values=([NORMAL, STRIKE_SLIP, INVERSE], [inputs["normal"], inputs["strike_slip"], inputs["inverse"]]))
    fric_coef = uniform(loc=inputs["fc_mean"] - inputs["fc_sd"], scale=2 * inputs["fc_sd"])



    
    model_builder = ModelBuilder(variables+exogenous_vars, name="SlopeStability")
    model_builder_cbn = ModelBuilder(variables, name="SlopeStabilityCBN")

    model_builder.add_equation(WD, noise = ro_w)
    model_builder.add_equation(RD, noise = ro_s)
    model_builder.add_equation(FC, noise=fric_coef)

    model_builder_cbn.add_equation(WD, noise=ro_w)
    model_builder_cbn.add_equation(RD, noise=ro_s)
    model_builder_cbn.add_equation(FC, noise=fric_coef)


    ## Equations
    # Estimation of the vertical stress if we don't have density logs
    def f_sv_estimation(u, ro_s, ro_w):
        sv_Pa = (H_1 * ro_w + H_2 * ro_s) * g
        sv_MPa = sv_Pa / (10 ** 6)
        return sv_MPa


    model_builder.add_equation(variable=SV,
                               parents=[RD, WD],
                               function=f_sv_estimation)
    model_builder_cbn.add_equation(variable=SV,
                                   parents=[RD, WD],
                               function=f_sv_estimation)


    # Compute hydrostatic pore pressure
    def f_hydrostatic_pp(u, ro_w):
        hydrostatic_pp_Pa = (Z - Z_f) * ro_w * g
        hydrostatic_pp_MPa = hydrostatic_pp_Pa / (10 ** 6)
        return hydrostatic_pp_MPa


    model_builder.add_equation(variable=HPP,
                               parents=[WD],
                               function=f_hydrostatic_pp)

    model_builder_cbn.add_equation(variable=HPP,
                               parents=[WD],
                               function=f_hydrostatic_pp)

    # Variable that defines were the pp will be placed between the hydrostatic pp and the s_v
    r_pp_noise = norm(0, 3)
    model_builder.add_equation("r_pp", noise=r_pp_noise)

    '''
    utilizamos el valor de 0.9 para tener un margen a la hora de restar dpp y dT
    '''
    # return np.random.uniform(np.minimum(hpp,s_v),s_v)
    model_builder.add_equation(PP, [HPP, SV, "r_pp"], f_pp)
    model_builder_cbn.add_equation(variable=PP, parents=[HPP, SV], function=f_pp_cbn, noise=r_pp_noise)


    # Obtaining values for Shmin and SHmax given pp and s_v. We will use two exogenous uniform distributions r1 and r2
    # to sample uniformly distributed points from the nomogram

    model_builder.add_equation("r1", noise=uniform())
    model_builder.add_equation("r2", noise=uniform())


    model_builder.add_equation(SHMIN, [PP, SV, FC, "r2"],
                               function=f_shmin_normal)


    model_builder.add_equation(SHMAX, [PP, SV, FC, SHMIN, "r1"],
                               function=f_shmax_normal)

    model_builder_cbn.add_equation(variable=SHMAX, parents=[PP, SV, FC, SHMIN], function=f_shmax_normal_cbn, noise=uniform())
    model_builder_cbn.add_equation(variable=SHMIN, parents=[PP, SV, FC], function=f_shmin_normal_cbn, noise=uniform())

    # Finally, we will add a node (that will not be plotted) that computes whether the slope will fail or not
    model_builder.add_equation(variable=FAILURE, parents=[SV, SHMAX, SHMIN, PP, FC], function=f_seismic_failure)
    model_builder_cbn.add_equation(variable=FAILURE, parents=[SV, SHMAX, SHMIN, PP, FC], function=f_seismic_failure)


    model = model_builder.build()
    model_cbn = model_builder_cbn.build()

    N = 10000000
    t0 = time.time()
    data = model.generate_data(N)
    print(data.describe().to_string())
    print("Data generation took", time.time() - t0, "seconds")

    def experiment_final(N = 10000000) -> tuple[pd.DataFrame,pd.DataFrame]:
        range_int = np.arange(36,40,0.25)
        ret_df = pd.DataFrame(index = range_int, columns = variables[:8])
        ret_df_cbn = pd.DataFrame(index=range_int, columns=variables[:8])
        data_raw = model.generate_data(N)
        # Extract mean and std of the variables
        means = data_raw.mean()
        stds = data_raw.std()

        # We observe in an incremental manner. First, the densities, next we add fric_coef, next the pore pressure
        for i, var in enumerate(variables[:8]):
            for int_value in range_int :
                observation_list = {v: means[v] for v in variables[:i+1]}
                data_obs = model.get_observational_data(N, observation_list = observation_list, eps=0.05)
                data_count_scm = model.intervene_data(data_obs, intervention_list={PP : int_value}, method="root_nodes")
                data_count_cbn = model_cbn.intervene_data(data_obs, intervention_list={PP : int_value}, method="non_descendants")
                print(f"Observing {var} - Description of the data:")
                print(f"Intervening on {PP} with value {int_value}")
                print(data_count_scm[variables].describe().to_string())
                print("\n\n")
                ret_df.loc[int_value, var] = data_count_scm[FAILURE].mean()
                ret_df_cbn.loc[int_value, var] = data_count_cbn[FAILURE].mean()
                # Store the samples as well
                # First, create a 2D array with both samples for the FAILURE var:
                mat = np.hstack((data_count_scm[FAILURE].values.reshape(-1,1), data_count_cbn[FAILURE].values.reshape(-1,1)))
                # Then, create a dataframe with the mat and columns "SCM" and "CBN"
                df_samples = pd.DataFrame(mat, columns=["SCM", "CBN"])
                dir_store = os.path.join(".","results", "applied", "samples")
                file_name = os.path.join(dir_store, f"{i}_{int_value}.csv")
                df_samples.to_csv(file_name, index=False)
        return ret_df, ret_df_cbn

    def experiment_final_child(N = 10000000) -> tuple[pd.DataFrame,pd.DataFrame]:
        range_int = np.arange(36,40,0.25)
        children = [PP,SHMIN, SHMAX]
        # Get combinatory of children list (ignore empty set)
        children_combinations = list(combinations(children, 1)) + list(combinations(children, 2)) + list(combinations(children, 3))
        print(children_combinations)
        ret_df = pd.DataFrame(index = range_int, columns = range(len(children_combinations)))
        ret_df_cbn = pd.DataFrame(index=range_int, columns= range(len(children_combinations)))
        data_raw = model.generate_data(N)
        # Extract mean and std of the variables
        means = data_raw.mean()
        stds = data_raw.std()

        # We observe in an incremental manner. First, the densities, next we add fric_coef, next the pore pressure
        for i, var in enumerate(children_combinations):
            for int_value in range_int :
                observation_list = {v: means[v] for v in var}
                data_obs = model.get_observational_data(N, observation_list = observation_list, eps=0.05)
                data_count_scm = model.intervene_data(data_obs, intervention_list={PP : int_value}, method="root_nodes")
                data_count_cbn = model_cbn.intervene_data(data_obs, intervention_list={PP : int_value}, method="non_descendants")
                print(f"Observing {var} - Description of the data:")
                print(f"Intervening on {PP} with value {int_value}")
                print(data_count_scm[variables].describe().to_string())
                print("\n\n")
                ret_df.loc[int_value, var] = data_count_scm[FAILURE].mean()
                ret_df_cbn.loc[int_value, var] = data_count_cbn[FAILURE].mean()
                # Store the samples as well
                # First, create a 2D array with both samples for the FAILURE var:
                mat = np.hstack((data_count_scm[FAILURE].values.reshape(-1,1), data_count_cbn[FAILURE].values.reshape(-1,1)))
                # Then, create a dataframe with the mat and columns "SCM" and "CBN"
                df_samples = pd.DataFrame(mat, columns=["SCM", "CBN"])
                dir_store = os.path.join(".","results", "applied", "samples_children")
                file_name = os.path.join(dir_store, f"{i}_{int_value}.csv")
                df_samples.to_csv(file_name, index=False)
        return ret_df, ret_df_cbn

    final_df, final_df_cbn = experiment_final_child(N)
    print(final_df.to_string())
    print(final_df_cbn.to_string())
    # Export results
    #final_df.to_csv(os.path.join(".","results", "applied", "final_df.csv"))
    #final_df_cbn.to_csv(os.path.join(".","results", "applied", "final_df_cbn.csv"))
    raise Exception("Stop here")


    # Assume SHMIN == 34
    obs = 38
    data_obs = model.get_observational_data(N, observation_list={SHMIN : obs}, eps=0.05)
    print(data_obs[[SHMAX, SV, SHMIN, FAILURE, PP]].describe().to_string())

    inter = 40
    data_count = model.get_counterfactual_data(N, observation_list={SHMIN : obs}, intervention_list={PP : inter}, eps=0.05)
    print(data_count[[SHMAX, SV, SHMIN, FAILURE, PP]].describe().to_string())



    raise Exception("Stop here")
    # Plot model using nx
    import networkx as nx
    G = nx.DiGraph()
    for node in model.var_dict.keys():
        G.add_node(node, label = model.var_dict[node])
    for eq in model.model.structural_equations:
        for parent in eq.indices_rhs:
            G.add_edge(parent, eq.index_lhs)
    MULT_SIZE = 0.65
    plt.figure(figsize=(10*MULT_SIZE, 6*MULT_SIZE))
    # Remove exogenous variables from the graph
    G.remove_nodes_from([model.var_dict.inverse[var_name] for var_name in exogenous_vars + [FAILURE]])

    # dot layout
    pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
    labels = nx.get_node_attributes(G, 'label')
    nx.draw(G, pos, labels=labels, with_labels=True, node_size=2000, node_color='lightblue', font_size=10,
            font_weight='bold', arrowsize=20, edge_color='gray')
    plt.title("Causal Graph of the Slope Stability Model")
    plt.tight_layout()
    file_path = os.path.join(".","results", "applied", "graph.pdf")
    plt.savefig(file_path)
    plt.show()

    # Sample from the CBN
    t0 = time.time()
    data_cbn = model_cbn.generate_data(1000000)
    print(data_cbn[[HPP,SV]].describe())
    print("CBN Data generation took", time.time() - t0, "seconds")







