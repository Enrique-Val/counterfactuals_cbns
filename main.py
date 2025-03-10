from itertools import product
from statistics import LinearRegression

import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm, uniform
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR

from causal_ale.build_model import ModelBuilder
import seaborn as sns
import numpy as np
import pybnesian as pb

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


########################
# ALE BN approximation #
########################
def ale_bn_approximation(model, variable, target, data=None, intervals=20):
    if data is None:
        data = model.sample(100000).to_pandas()
    else:
        data = data.copy()

    # First of all, create the intervals
    if isinstance(intervals, int) :
        cuts = np.linspace(data[variable].quantile(0.01), data[variable].quantile(0.99), intervals + 1)
        intervals = [(cuts[i], cuts[i + 1]) for i in range(intervals)]
    else :
        intervals = intervals
    # Compute the local effect for each interval
    local_effects = []
    for interval in intervals:
        # Get the local data (data within the interval)
        data = data[data[variable] > interval[0]]
        local_data = data[data[variable] < interval[1]]
        local_data = local_data.reset_index(drop=True)
        # Approximate ALE with data
        # Substitute the variable with the interval extremes
        local_data[variable] = interval[0]
        # Get the likelihood
        logls = model.logl(local_data)
        # Get the value of the target. Essentially, a weighted mean of the values
        mean_target_val_left = np.average(local_data[target], weights=np.e ** logls)

        # Compute for the right interval
        local_data[variable] = interval[1]
        logls = model.logl(local_data)
        mean_target_val_right = np.average(local_data[target], weights=np.e ** logls)

        # Compute the local effect
        le = mean_target_val_right - mean_target_val_left
        local_effects.append(le)
    # Accumulate the local effects
    accumulated_local_effects = np.cumsum(local_effects)
    return np.array([np.mean(i) for i in intervals]), accumulated_local_effects - np.mean(accumulated_local_effects)


def ale_sklearn_approximation(model, variable, data, intervals=20):
    # First of all, create the intervals
    if isinstance(intervals, int):
        cuts = np.linspace(data[variable].quantile(0.01), data[variable].quantile(0.99), intervals + 1)
        intervals = [(cuts[i], cuts[i + 1]) for i in range(intervals)]
    else:
        intervals = intervals

    # Compute the local effect for each interval
    local_effects = []
    for interval in intervals:
        # Get the local data (data within the interval)
        data = data[data[variable] > interval[0]]
        local_data = data[data[variable] < interval[1]]
        local_data = local_data.reset_index(drop=True)
        # Approximate ALE with data
        # Substitute the variable with the interval extremes
        local_data[variable] = interval[0]
        # Predict with perturbed data
        y_pred_left = model.predict(local_data)

        # Compute for the right interval
        local_data[variable] = interval[1]
        y_pred_right = model.predict(local_data)

        y_pred_diff = y_pred_right - y_pred_left
        le = np.mean(y_pred_diff)

        local_effects.append(le)
    # Accumulate the local effects
    accumulated_local_effects = np.cumsum(local_effects)
    return np.array([np.mean(i) for i in intervals]), accumulated_local_effects - np.mean(accumulated_local_effects)


def get_data_based_models(data, bn_metric = "bic", sklearn_model = LinearRegression()):
    # Learn BN using pybnesian
    bn = pb.hc(data, bn_type=pb.GaussianNetworkType(), score="bge", operators=["arcs"])
    bn.fit(data)

    '''    data_bn = bn.sample(100000).to_pandas()

    # Compute ALE using the BN
    ale_bns = []
    for var in data.columns:
        if var == "Y":
            continue
        ale_bn = ale_bn_approximation(bn, var, "Y", data=data_bn, intervals=intervals)
        ale_bns.append(ale_bn)'''


    predictors = [i for i in data.columns if i != "Y"]
    X = data[predictors]
    y = data["Y"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = sklearn_model
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(mean_squared_error(y_test, y_pred))
    # Print the learned equation
    if isinstance(model, LinearRegression):
        print(model.coef_, model.intercept_)

    '''    # Compute ALE using the model
    ale_sklearns = []
    for var in data.columns:
        if var == "Y":
            continue
        ale_sklearn = ale_sklearn_approximation(model, var, X, intervals=intervals)
        ale_sklearns.append(ale_sklearn)'''

    return bn, model

def get_confounder_structures(n_og_samples, noise_x, noise_y, real_asoc_coef = 0, noise_c = 1):
    #######
    # SCM #
    #######
    # X <- C -> Y
    vars = ["X", "C", "Y"]
    exogenous_vars = ["U_X", "U_Y"]

    model_builder = ModelBuilder(vars + exogenous_vars, "model")

    # Confounder prior
    model_builder.add_equation("C", noise=norm(0, noise_c))

    # Noises
    model_builder.add_equation("U_X", noise=norm(0, noise_x))
    model_builder.add_equation("U_Y", noise=norm(0, noise_y))

    # Structural equations
    def f_x(u, X_U, C):
        return X_U + C

    def f_y(u, Y_U, C, X):
        return Y_U - C + real_asoc_coef*X

    model_builder.add_equation("X", ["U_X", "C"], f_x)
    model_builder.add_equation("Y", ["U_Y", "C", "X"], f_y)

    model_scm = model_builder.build()

    # Generate data
    data = model_scm.generate_data(n_og_samples)

    #############
    # Causal BN #
    #############

    model_builder = ModelBuilder(vars, "model")

    # Confounder prior
    model_builder.add_equation("C", noise=norm(0, noise_c))

    # Structural equations
    def f_x(u, C):
        return u + C

    def f_y(u, C, X):
        return u - C + real_asoc_coef*X

    model_builder.add_equation("X", ["C"], f_x, noise=norm(0, noise_x))
    model_builder.add_equation("Y", ["C", "X"], f_y, noise=norm(0, noise_y))

    model_cbn = model_builder.build()

    ###############
    # Learned CBN #
    ###############
    cbn = pb.GaussianNetwork(data[vars].columns)
    arc_list = [("C", "X"), ("C", "Y"), ("X", "Y")]
    # Random shuffle arc list
    # TODO
    for arc in arc_list:
        cbn.add_arc(arc[0], arc[1])
    cbn.fit(data[vars])

    # Convert to SCM library
    model_builder = ModelBuilder(vars, "model")

    model_builder.add_equation("C", noise=norm(cbn.cpd("C").beta[0], np.sqrt(cbn.cpd("C").variance)))
    model_builder.add_equation("X", ["C"], lambda u, C: cbn.cpd("X").beta[1] * C + u + cbn.cpd("X").beta[0],
                               noise=norm(0, np.sqrt(cbn.cpd("X").variance)))
    model_builder.add_equation("Y", cbn.cpd("Y").evidence(),
                               lambda u, b1, b2: cbn.cpd("Y").beta[0] + b1 * cbn.cpd("Y").beta[1] +
                                                 cbn.cpd("Y").beta[2] * b2 + u,
                               noise=norm(0, np.sqrt(cbn.cpd("Y").variance)))

    model_cbn_learned = model_builder.build()

    return (model_scm, model_cbn, model_cbn_learned), data, data[vars]

def get_mediator_structures(n_og_samples, noise_m, noise_y, real_asoc_coef, noise_x = 1):
    #######
    # SCM #
    #######
    # X -> M -> Y
    vars = ["X", "M", "Y"]
    exogenous_vars = ["U_M", "U_Y"]

    model_builder = ModelBuilder(vars + exogenous_vars, "model")

    # Confounder prior
    model_builder.add_equation("X", noise=norm(0, noise_x))

    # Noises
    model_builder.add_equation("U_M", noise=norm(0, noise_m))
    model_builder.add_equation("U_Y", noise=norm(0, noise_y))

    # Structural equations
    def f_m(u, U_M, X):
        return U_M + X

    def f_y(u, U_Y, M, X):
        return U_Y - M + real_asoc_coef*X

    model_builder.add_equation("M", ["U_M", "X"], f_m)
    model_builder.add_equation("Y", ["U_Y", "M", "X"], f_y)

    model_scm = model_builder.build()

    # Generate data
    data = model_scm.generate_data(n_og_samples)

    scm_ales = []
    for var in vars :
        if var == "Y" :
            continue
        ale_scm = model_scm.ale_plot(var, "Y", data=data, intervals=intervals)
        scm_ales.append(ale_scm)

    #############
    # Causal BN #
    #############

    model_builder = ModelBuilder(vars, "model")

    # X prior
    model_builder.add_equation("X", noise=norm(0, noise_x))

    # Structural equations
    def f_m(u, X):
        return u + X

    def f_y(u, M, X):
        return u - M + real_asoc_coef*X

    model_builder.add_equation("M", ["X"], f_m, noise=norm(0, noise_m))
    model_builder.add_equation("Y", ["M", "X"], f_y, noise=norm(0, noise_y))

    model_cbn = model_builder.build()

    # Learn a CBN
    cbn = pb.GaussianNetwork(data[vars].columns)
    arc_list = [("X", "M"), ("X", "Y"), ("M", "Y")]
    # Random shuffle arc list
    # TODO
    for arc in arc_list:
        cbn.add_arc(arc[0], arc[1])
    cbn.fit(data[vars])

    # Convert to SCM library
    model_builder = ModelBuilder(vars, "model")



    model_builder.add_equation("X", noise=norm(cbn.cpd("X").beta[0], np.sqrt(cbn.cpd("X").variance)))
    model_builder.add_equation("M", ["X"], lambda u, X: cbn.cpd("M").beta[1]*X+u+cbn.cpd("M").beta[0],
                               noise=norm(0, np.sqrt(cbn.cpd("M").variance)))
    model_builder.add_equation("Y", cbn.cpd("Y").evidence(), lambda u, b1, b2: cbn.cpd("Y").beta[0]+b1*cbn.cpd("Y").beta[1]+ cbn.cpd("Y").beta[2]*b2+u,
                                 noise=norm(0, np.sqrt(cbn.cpd("Y").variance)))

    model_cbn_learned = model_builder.build()

    return (model_scm, model_cbn, model_cbn_learned), data, data[vars]



def get_collider_structures(n_og_samples, noise_c, real_asoc_coef, noise_x = 1, noise_y = 1):
    #######
    # SCM #
    #######
    # X -> C <- Y
    vars = ["X", "C",  "Y"]
    exogenous_vars = ["U_C", "U_Y"]

    model_builder = ModelBuilder(vars + exogenous_vars, "model")

    # Priors
    model_builder.add_equation("X", noise=norm(0, noise_x))
    model_builder.add_equation("U_Y", noise=norm(0, noise_y))

    # Noise
    model_builder.add_equation("U_C", noise=norm(0, noise_c))

    # Structural equations
    def f_c(u, U_C, X, Y):
        return U_C + X - Y

    def f_y(u, U_Y, X):
        return U_Y + real_asoc_coef*X

    model_builder.add_equation("C", ["U_C", "X", "Y"], f_c)
    model_builder.add_equation("Y", ["U_Y", "X"], f_y)

    model_scm = model_builder.build()

    # Generate data
    data = model_scm.generate_data(n_og_samples)

    #############
    # Causal BN #
    #############

    model_builder = ModelBuilder(vars, "model")

    # Priors
    model_builder.add_equation("X", noise=norm(0, noise_x))

    # Structural equations
    def f_c(u, X, Y):
        return u + X - Y

    def f_y(u, X):
        return u + real_asoc_coef*X

    model_builder.add_equation("C", ["X", "Y"], f_c, noise=norm(0, noise_c))
    model_builder.add_equation("Y", ["X"], f_y, noise=norm(0, noise_y))

    model_cbn = model_builder.build()

    # Learn a CBN
    cbn = pb.GaussianNetwork(data[vars].columns)
    arc_list = [("X", "C"), ("Y", "C"), ("X", "Y")]
    # Random shuffle arc list
    # TODO
    for arc in arc_list:
        cbn.add_arc(arc[0], arc[1])
    cbn.fit(data[vars])

    # Convert to SCM library
    model_builder = ModelBuilder(vars, "model")

    model_builder.add_equation("X", noise=norm(cbn.cpd("X").beta[0], np.sqrt(cbn.cpd("X").variance)))
    model_builder.add_equation("Y", ["X"], lambda u, X: cbn.cpd("Y").beta[1] * X + u + cbn.cpd("Y").beta[0],
                               noise=norm(0, np.sqrt(cbn.cpd("Y").variance)))
    model_builder.add_equation("C", cbn.cpd("C").evidence(),
                               lambda u, b1, b2: cbn.cpd("C").beta[0] + b1 * cbn.cpd("C").beta[1] +
                                                 cbn.cpd("C").beta[2] * b2 + u,
                               noise=norm(0, np.sqrt(cbn.cpd("C").variance)))

    model_cbn_learned = model_builder.build()

    '''data_cbn = model_cbn_learned.generate_data(n_og_samples * 100)

    learned_cbn_ales = []
    for var in vars:
        if var == "Y":
            continue
        ale_cbn = model_cbn_learned.ale_plot(var, "Y", data=data_cbn, intervals=intervals, model_type="cbn")
        learned_cbn_ales.append(ale_cbn)'''

    return (model_scm, model_cbn, model_cbn_learned), data, data[vars]


if __name__ == '__main__':
    noise_y = .5
    real_asoc_coef = 0
    n_intervals = 30
    dfs = {}
    for noise_y,real_asoc_coef in product([.001, .25, .5, .75, 1, 2],[0, .25, .5, .75, 1, 2]) :
        noise_var = noise_y
        fig, ax = plt.subplots(figsize=(10, 8), nrows=3, ncols=2)
        for i,struct in enumerate(["Confounder", "Mediator", "Collider"]) :
            if struct == "Confounder" :
                models = get_confounder_structures(1000,noise_var, noise_y, real_asoc_coef=real_asoc_coef)
            elif struct == "Mediator" :
                models = get_mediator_structures(1000,noise_var, noise_y, real_asoc_coef=real_asoc_coef)
            elif struct == "Collider" :
                models = get_collider_structures(1000, noise_var, real_asoc_coef=real_asoc_coef)
            else :
                raise ValueError("Invalid structure")
            (scm, cbn, learned_cbn), data_total, data = models
            print(data)
            bn, sklearn_model = get_data_based_models(data, sklearn_model=LinearRegression())

            for j,var in enumerate(data.columns) :
                if var == "Y" :
                    continue
                cuts = np.linspace(data[var].quantile(0.05), data[var].quantile(0.95), n_intervals + 1)
                intervals = [(cuts[i], cuts[i + 1]) for i in range(n_intervals)]
                scm_ale = scm.ale_plot(var, "Y", data=data_total, intervals=intervals)
                cbn_ale = cbn.ale_plot(var, "Y", intervals=intervals, model_type="cbn")
                learned_cbn_ale = learned_cbn.ale_plot(var, "Y", intervals=intervals, model_type="cbn")
                bn_ale = ale_bn_approximation(bn, var, "Y", intervals=intervals)
                sklearn_ale = ale_sklearn_approximation(sklearn_model, var, data.drop("Y", axis=1), intervals=intervals)
                dfs[(noise_y, real_asoc_coef, struct, var)] = [np.sum((cbn_ale[1]-scm_ale[1])**2),
                                                               np.sum((learned_cbn_ale[1]-scm_ale[1])**2),
                                                               np.sum((bn_ale[1]-scm_ale[1])**2),
                                                               np.sum((sklearn_ale[1]-scm_ale[1])**2)]

                ax_i = ax[i, j]
                ax_i.plot(scm_ale[0], scm_ale[1], label="SCM")
                ax_i.plot(cbn_ale[0], cbn_ale[1], label="CBN")
                ax_i.plot(learned_cbn_ale[0], learned_cbn_ale[1], label="Learned CBN")
                ax_i.plot(bn_ale[0], bn_ale[1], label="BN")
                ax_i.plot(sklearn_ale[0], sklearn_ale[1], label="Sklearn")
                # Set x label
                ax_i.set_xlabel(var)
                # Set y label
                ax_i.set_ylabel("ALE")
                # Legend in axis
                ax_i.legend()
                ax_i.set_title(f"{struct}: Variable {var} ALE")
        fig.suptitle(f"Var noise {noise_y}, Asociation coef. {real_asoc_coef} ALE")
        fig.show()
        plt.close()

    df = pd.DataFrame(dfs).T
    # Set columns
    df.columns = ["CBN", "Learned CBN", "BN", "Sklearn"]
    df.to_csv("ale_comparison.csv")




