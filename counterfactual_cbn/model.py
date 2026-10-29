import networkx as nx
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from sklearn.model_selection import GridSearchCV

from causal_ale import utils

from StructuralCausalModels import structural_causal_model, structural_equation
from scipy.ndimage import gaussian_filter1d

DEFAULT_NB_SAMPLES = 100000
SF_THRESH = 2





class Model:
    def __init__(self, model: structural_causal_model.StructuralCausalModel, var_dict, inverse_functions = None):
        self.model : structural_causal_model.StructuralCausalModel = model
        self.var_dict = var_dict
        self.original_equations = set()
        self.inverse_functions = [] if None else inverse_functions

    #############
    ## GETTERS ##
    #############

    def get_model(self, copy = True):
        """
        Get the underlying model used for inference from the library StructuralCausalModels.

        Parameters
        ----------
        copy : bool
            Whether to return a copy of the model or the model itself. Optional. Default is True.

        Returns
        -------
        structural_causal_model.StructuralCausalModel
            The model used for inference
        """
        if copy:
            return utils.copy_model(self.model)
        else :
            return self.model

    def get_variables(self) -> list:
        return list(self.var_dict.values())

    def get_structural_equations(self) -> set[structural_equation.StructuralEquation]:
        """
        Get the structural equations of all the nodes of the model. The whole set of structural equations defines the
        behaviour and inference of the model.

        Returns
        -------
        set[structural_equation.StructuralEquation]
            The set of structural equations of the model.
        """
        to_ret = set()
        for i in self.original_equations:
            to_ret.add(utils.copy_structural_equation(i))
        return to_ret

    def get_structural_equation(self, node:str) -> structural_equation.StructuralEquation:
        """
        Get the structural equation of a given node in the model.

        Parameters
        ----------
        node : str
            The node to get the structural equation from.

        Returns
        -------
        structural_equation.StructuralEquation
            The structural equation of the given node.
        """
        for i in self.model.structural_equations:
            if i.index_lhs == self.var_dict.inverse[node]:
                return utils.copy_structural_equation(i)
        raise Exception('Node ', node, " not present in the model")

    #################
    ## COPY METHOD ##
    #################

    def copy(self):
        """
        Copy method for the class.

        Returns
        -------
        Model
            A copy of the model.
        """
        copied_model = Model(utils.copy_model(self.model), self.var_dict.copy())
        for i in self.original_equations:
            copied_model.original_equations.add(utils.copy_structural_equation(i))
        return copied_model

    ###################
    ## FUNCTIONALITY ##
    ###################

    def generate_data(self, nb_samples: int, random_state: int = 42) -> pd.DataFrame:
        """
        Generate data from the model.

        Parameters
        ----------
        nb_samples : int
            The number of samples to generate.
        random_state : int
            The random state to be used in the generation of the data. Optional.

        Returns
        -------
        pd.DataFrame
            The generated data.
        """
        if random_state is not None:
            np.random.seed(random_state)
        data = self.model.generate_data(nb_samples)
        data = data.rename(self.var_dict, axis='columns')
        return data

    def intervene_data(self, data: pd.DataFrame, intervention_list: dict, method = "root_nodes", random_state: int = 42) -> pd.DataFrame:
        all_columns = data.columns
        if method == "root_nodes":
            data = data[self.get_root_nodes()].copy()
        elif method == "non_descendants":
            total_non_descendants = set(self.get_variables())
            for i in intervention_list.keys():
                total_non_descendants = total_non_descendants.intersection(set(self.get_non_descendants(i)))
            data = data[list(total_non_descendants)].copy()
        else:
            raise Exception("Method not recognized. Please use 'root_nodes' or 'non_descendants'")
        for i in intervention_list.keys():
            if i not in all_columns:
                raise Exception('Node ', i[0], " not present in the model")
            data[i] = intervention_list[i]
        data = self.causal_inference(data, random_state=random_state)
        return data

    def get_observational_data(self, nb_samples: int, observation_list, eps=1e-8, random_state: int = 42, is_scm = False) -> pd.DataFrame:
        """
        Generate data from the model with the given observations.

        Parameters
        ----------
        observation_list : dict
            The list of observations to be used in the generation of the data.
        nb_samples : int
            The number of samples to generate.
        random_state : int
            The random state to be used in the generation of the data. Optional.

        Returns
        -------
        pd.DataFrame
            The generated data.
        """

        data = self.generate_data(nb_samples, random_state)
        graph = self.get_graph()
        if is_scm :
            # Create a subgraph without the root nodes
            end_vars = self.get_variables()
            end_vars = [i for i in end_vars if i not in self.get_root_nodes()]
            graph = nx.subgraph(graph, end_vars)
        pi_nodes = {}
        # Iterate over an ancestral ordering of the subgraph
        for i in observation_list.keys():
            # If i has no parents or all parents are in the observation list, we can just sample it, add to pi_nodes
            if len(list(graph.predecessors(i))) == 0 or \
                    all([j in observation_list.keys() for j in graph.predecessors(i)]):
                pi_nodes[i] = observation_list[i]
        if len(pi_nodes) > 0:
            data = self.intervene_data(data, pi_nodes, method="root_nodes", random_state=random_state)

        if not is_scm:
            unobserved_vars = set(self.get_variables()).difference(set(observation_list.keys()))
            joint_markov_blanket = self.joint_markov_blanket(list(unobserved_vars))
            # If the whole Markov blanket is observed, we can just filter it and ignore the rest
            observed_in_mb = list(set(observation_list.keys()).intersection(set(joint_markov_blanket)))
            non_mb_vars = list(set(list(observation_list.keys())).difference(set(joint_markov_blanket)))
        else:
            # Same as before, but remove the root nodes from the joint markov blanket
            unobserved_vars = set(self.get_variables()).difference(set(observation_list.keys()))
            unobserved_vars = unobserved_vars.difference(set(self.get_root_nodes()))
            joint_markov_blanket = self.joint_markov_blanket(list(unobserved_vars))
            observed_in_mb = list(set(observation_list.keys()).intersection(set(joint_markov_blanket)))
            non_mb_vars = list(set(list(observation_list.keys())).difference(set(joint_markov_blanket)))
        print("Total MB", observed_in_mb)
        for i in observed_in_mb:
            if i not in data.columns:
                raise Exception('Node ', i, " not present in the model")
            data = data[data[i] < observation_list[i] + eps]
            data = data[data[i] > observation_list[i] - eps]
        data = data.reset_index(drop=True)
        # Now, we just pass the values of the non-markov blanket variables
        for i in non_mb_vars:
            if i not in data.columns:
                raise Exception('Node ', i, " not present in the model")
            data[i] = observation_list[i]
        # Now, if it is a SCM, we still need to compute the values of the exogenous variables
        if is_scm:
            print("Entra ex")
            data = self.fill_exogenous_data(data)
        # Let's do that in an a
        if data.shape[0] == 0:
            raise Exception('No data generated with the given observations ' + str(observation_list))
        return data

    def get_observational_data_kde(self, nb_samples: int, observation_list, eps=1e-8, random_state: int = 42,
                                   functional = True, cv=2) -> pd.DataFrame:
        """
        Generate data from the model with the given observations.

        Parameters
        ----------
        observation_list : dict
            The list of observations to be used in the generation of the data.
        nb_samples : int
            The number of samples to generate.
        random_state : int
            The random state to be used in the generation of the data. Optional.

        Returns
        -------
        pd.DataFrame
            The generated data.
        """
        data = self.generate_data(nb_samples, random_state)
        # First filter data
        for i in observation_list.keys():
            if i not in data.columns:
                raise Exception('Node ', i, " not present in the model")
            data = data[data[i] < observation_list[i] + eps*10]
            data = data[data[i] > observation_list[i] - eps*10]
        weights = np.zeros(data.shape[0])
        for i in observation_list.keys():
            weighting_norm = norm(observation_list[i], eps)
            if i not in data.columns:
                raise Exception('Node ', i, " not present in the model")
            # Weight the data according to the observation
            weights += weighting_norm.pdf(data[i])
        # Normalize the weights
        # Substitute NaN values with 0
        weights = np.exp(weights)
        weights[np.isnan(weights)] = 0
        weights /= np.sum(weights)
        # Sample the data according to the weights. First, learn join distribution of the data via KDE
        if functional:
            root_nodes = self.get_root_nodes()
            data_exog = data[root_nodes]
            # Learn KDE using sklearn
            from sklearn.neighbors import KernelDensity
            #kde_learned = KernelDensity(kernel='gaussian', bandwidth=eps)
            #kde_learned.fit(data_exog, sample_weight=weights)
            # Repeat the learning, but use grid-search and cross validate
            rang = np.logspace(-3, 0, 10)  # Bandwidth range for grid search
            gs_results = GridSearchCV(KernelDensity(kernel='gaussian'), {'bandwidth': rang}, cv=cv, n_jobs=-1)
            gs_results.fit(data_exog, sample_weight=weights)
            kde_learned = gs_results.best_estimator_
            # Sample from the learned KDE
            sampled_data_exog = kde_learned.sample(nb_samples, random_state=random_state)
            sampled_data_exog = pd.DataFrame(sampled_data_exog, columns=root_nodes)
            # Now, we need to compute the values of the endogenous variables
            resample = self.causal_inference(sampled_data_exog, random_state=random_state)
            # Concatenate the exogenous and endogenous variables
            return resample
        else :
            from sklearn.neighbors import KernelDensity
            # Sample the data according to the weights
            kde_learned = KernelDensity(kernel='gaussian')
            kde_learned.fit(data, sample_weight=weights)
            sampled_data = kde_learned.sample(nb_samples, random_state=random_state)
            sampled_data = pd.DataFrame(sampled_data, columns=data.columns)
            return sampled_data

        if data.shape[0] == 0:
            raise Exception('No data generated with the given observations')
        return data

    def get_interventional_data(self, nb_samples: int, intervention_list : dict, random_state: int = 42) -> pd.DataFrame:
        """
        Generate data from the model with the given interventions.
        Parameters
        ----------
        intervention_list : dict
            The list of interventions to be used in the generation of the data.
        nb_samples : int
            The number of samples to generate.
        random_state : int
            The random state to be used in the generation of the data. Optional.
        Returns
        -------
        pd.DataFrame
            The generated data.

        """

        toy_data = self.generate_data(1)
        for i in intervention_list:
            if i not in toy_data.columns:
                raise Exception('Node ', i, " not present in the model")
            self.perform_intervention(utils.make_equation(i, noise=norm(intervention_list[i],0), var_dict=self.var_dict))
        data = self.generate_data(nb_samples, random_state)
        self.delete_all_interventions()
        return data

    def get_counterfactual_data(self, nb_samples: int, observation_list, intervention_list,
                                eps = 1e-8, random_state: int = 42, is_scm = True) -> pd.DataFrame:
        """
        Generate data from the model with the given observations and interventions.
        Parameters
        ----------
        observation_list : dict
            The list of observations to be used in the generation of the data.
        intervention_list : dict
            The list of interventions to be used in the generation of the data.
        method : str
            The method to be used for the counterfactual generation, specifically regarding the abduction phase.
            The options are "root_nodes" (recommended for SCMs where the noise is modelled as a variable) and
            "non_descendants" (recommended for CBNs where the noise is not modelled as a variable).
        nb_samples : int
            The number of samples to generate.
        random_state : int
            The random state to be used in the generation of the data. Optional.
        Returns
        -------
        pd.DataFrame
            The generated data.

        """


        data = self.get_observational_data(nb_samples, observation_list=observation_list, eps=eps,
                                           random_state=random_state, is_scm=is_scm)
        return self.intervene_data(data, intervention_list, method="root_nodes" if is_scm else "non_descendants",
                                   random_state=random_state)

    def fill_exogenous_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Given data of the endogenous variables (not root nodes in an SCM), get the values of the exogenous variables
        """

        root_nodes = self.get_root_nodes()
        endogenous_vars = set(self.get_variables()).difference(set(root_nodes))
        # Check that the dataframe contains all the endogenous variables and there are no nans over them
        if not endogenous_vars.issubset(set(data.columns)):
            raise Exception('Data does not contain all the endogenous variables. Missing: ',
                            list(endogenous_vars.difference(set(data.columns))))
        if data[list(endogenous_vars)].isnull().values.any():
            raise Exception('Data contains NaN values over the endogenous variables. Please provide a complete dataset.')

        data = data.copy()

        # First, we have to rename columns using the var_dict inverse
        data = data.rename(self.var_dict.inverse, axis='columns')

        for structural_equation in self.inverse_functions:
            data = structural_equation.generate_data(data)

        # Now, restore the original names
        data = data.rename(self.var_dict, axis='columns')

        return data

    ###################
    ## INTERVENTIONS ##
    ###################

    def get_intervened_nodes(self):
        """
        Get the intervened nodes in the model.

        Returns
        -------
        list
            The list of intervened nodes in the model.
        """
        to_ret = []
        for i in self.original_equations:
            to_ret.append(self.var_dict[i.index_lhs])
        return to_ret

    def perform_intervention(self, new_structural_equation):
        """
        Perform an intervention on the model. The intervention is performed by replacing the structural equation of the
        variable with the new structural equation. This can break the edges with parents of the intervened variable.

        Parameters
        ----------
        new_structural_equation : structural_equation.StructuralEquation
            The new structural equation to replace the existing one. We recommend using the make_equation method to
            simplify the creation of the new structural equation.

        Returns
        -------
        None
        """
        self.model, eq = self.__perform_intervention__(new_structural_equation)
        self.original_equations.add(eq)

    def delete_intervention(self, node):
        """
        Delete an intervention performed on the model.

        Parameters
        ----------
        node : str
            The node to delete the intervention from.

        Returns
        -------
        None

        Raises
        ------
        Exception
            If the node is not present in the model or if the node was not intervened.
        """
        to_del = None
        deleted_intervention = None
        for i, og_equation in enumerate(self.original_equations):
            if og_equation.index_lhs == self.var_dict.inverse[node]:
                deleted_intervention = self.get_structural_equation(node)
                self.model, _ = self.__perform_intervention__(og_equation)
                self.original_equations.remove(og_equation)
                return
        err = 'Node ' + node + " not present in the model or not intervened"
        raise Exception(err)

    def delete_all_interventions(self):
        """
        Delete all interventions performed on the model. This will restore the model to its original state.

        Returns
        -------
        None
        """
        for i, og_equation in enumerate(self.original_equations):
            self.model, _ = self.__perform_intervention__(og_equation)
        self.original_equations = set()

    def __perform_intervention__(self, new_structural_equation):
        new_eqs = self.model.structural_equations
        for i, eq in enumerate(self.model.structural_equations):
            if eq.index_lhs == new_structural_equation.index_lhs:
                new_eqs[i] = new_structural_equation
                return structural_causal_model.StructuralCausalModel(name="uset", nb_var=len(new_eqs),
                                                                     structural_equations=new_eqs), eq
        error_msg = 'Node ' + self.var_dict[new_structural_equation.index_lhs] + " not present in the model"
        raise Exception(error_msg)

    def get_graph(self):
        """
        Get the graph of the model. The graph is a directed graph where the nodes are the variables and the edges
        represent the causal relationships between the variables.

        Returns
        -------
        networkx.DiGraph
            The graph of the model
        """

        to_print_matrix = self.model.adjacency_matrix()
        graph = None
        if int(nx.__version__[0]) < 3:
            graph = nx.from_numpy_matrix(to_print_matrix, create_using=nx.DiGraph)
        else:
            graph = nx.from_numpy_array(to_print_matrix, create_using=nx.DiGraph)
        graph = nx.relabel_nodes(graph, self.var_dict)
        return graph

    def get_root_nodes(self):
        """Get the root nodes of the network (i.e. variables with no parents in the graph).
        This method is useful to quickly locate the source of uncertainty, as the only variables with uncertainty
        are root nodes in the graph

        Returns
        -------
        list
            The list of root nodes in the graph
        """
        graph = self.get_graph()
        return [node for node, degree in graph.in_degree() if degree == 0]

    def get_descendants(self, node):
        """
        Get the descendants of a given node in the model.

        Parameters
        ----------
        node : str
            The node to get the descendants from.

        Returns
        -------
        list
            The list of descendants of the given node.
        """
        graph = self.get_graph()
        return list(nx.descendants(graph, node))

    def get_non_descendants(self, node):
        """
        Get the non-descendants of a given node in the model.

        Parameters
        ----------
        node : str
            The node to get the non-descendants from.

        Returns
        -------
        list
            The list of non-descendants of the given node.
        """
        graph = self.get_graph()
        return [i for i in self.get_variables() if i not in nx.descendants(graph, node) and i != node]

    def get_markov_blanket(self, node):
        """
            Find the Markov blanket of a node in a directed graph.

            The Markov blanket of a node consists of:
            1. Its parents
            2. Its children
            3. The other parents of its children (spouses)

            Parameters:
            -----------
            node : str
                The target node whose Markov blanket we want to find

            Returns:
            --------
            set
                A set of nodes comprising the Markov blanket (excluding the node itself)
            """
        graph = self.get_graph()

        if node not in graph:
            raise ValueError(f"Node {node} not found in the graph")

        # Get parents of the node
        parents = set(graph.predecessors(node))

        # Get children of the node
        children = set(graph.successors(node))

        # Get spouses (other parents of children)
        spouses = set()
        for child in children:
            spouses.update(graph.predecessors(child))
        spouses.discard(node)  # Remove the node itself if it was included

        # Combine all components of the Markov blanket
        markov_blanket = parents.union(children).union(spouses)

        return markov_blanket

    def joint_markov_blanket(self, nodes : list):
        """
        Find the joint Markov blanket of a set of nodes in a directed graph.

        The joint Markov blanket of a set of nodes consists of:
        1. Their parents
        2. Their children
        3. The other parents of their children (spouses)

        Parameters
        ----------
        nodes : list
            The list of nodes whose joint Markov blanket we want to find

        Returns
        -------
        set
            A set of nodes comprising the joint Markov blanket (excluding the nodes themselves)
        """
        markov_blanket = set()
        for node in nodes:
            markov_blanket.update(set(self.get_markov_blanket(node)))
        # Remove the nodes themselves
        markov_blanket = markov_blanket.difference(set(nodes))
        return list(markov_blanket.difference(set(nodes)))

    def causal_inference(self, input_data: pd.DataFrame = None, random_state=None):
        """
        Performs inference in the network. Every variable in input data is considered as intervened, NOT observed.
        However, if all variables in input_data are root nodes, interventional and observational inference becomes
        equivalent.

        Parameters
        ----------
        input_data : pandas.DataFrame
            A dataframe containing the values to be propagated. Since
            samples are not univariate distributions are requested, this allows for much more interesting interventions (
            or observations if only root nodes are provided

        random_state : int
            The random state to be used in the generation of the data.

        Returns
        -------
        pandas.DataFrame
            The samples.
        """
        if input_data is None:
            return self.generate_data(DEFAULT_NB_SAMPLES, random_state)
        if random_state is not None:
            np.random.seed(random_state)
        nb_samples = input_data.shape[0]

        index = range(nb_samples)
        data = pd.DataFrame(
            np.empty((nb_samples, self.model.nb_var,)),
            index=index,
        )
        data.values[:] = np.nan


        for i in input_data.columns:
            data[self.var_dict.inverse[i]] = input_data[i]

        ordered_structural_equations = self.model.order_structural_equations()
        for structural_equation in ordered_structural_equations:
            if not self.var_dict[structural_equation.index_lhs] in input_data.columns:
                data = structural_equation.generate_data(data)

        data = data.rename(self.var_dict, axis='columns')
        return data

    ############################
    ## EXPLAINABILITY METHODS ##
    ############################
    def m_plot(self, variable, target_var, data=None, n_bins=20) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the M-plot for a given variable. The M-plot is a way to visualize the effect of a variable on the
        safety factor. It is computed by discretizing the variable into bins and computing the expected value of the
        safety factor given the variable bin values. This is done by filtering the data values to the ones that fall
        within the bin, and then computing the average safety factor for that bin.

        References:

         - Apley, Daniel W., and Jingyu Zhu. “Visualizing the effects of predictor variables in black box
        supervised learning models.” Journal of the Royal Statistical Society: Series B (Statistical Methodology)
        82.4 (2020): 1059-1086

        - Molnar, Christoph. Interpretable machine learning.
        https://christophm.github.io/interpretable-ml-book/ale.html, 2019.

        Parameters
        ----------
        variable : str
            Variable to compute M-plot over
        data : pd.DataFrame
            Optional parameter to provide data for the M-plot computation. If no data is provided, the model will
            generate it
        n_bins : int
            Number of bins to discretize the variable of study

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple with the x and y values of the M-plot. The x value are the means of the bins, and the y values are
        """
        SMOOTH_KERNEL = 1
        if data is None:
            data = self.generate_data(DEFAULT_NB_SAMPLES)

        data = data.copy()

        conditional_expectations = []
        cuts = pd.qcut(data[variable], q=n_bins, retbins=True)[1]
        intervals = [(cuts[i], cuts[i + 1]) for i in range(n_bins)]

        # Compute the local effect for each interval
        local_effects = []
        for interval in intervals:
            # Get the local data (data within the interval)
            data = data[data[variable] > interval[0]]
            local_data = data[data[variable] < interval[1]]
            local_data = local_data.reset_index(drop=True)
            local_data_risk = len(local_data[local_data[target_var] < 1]) / len(local_data) * 100
            conditional_expectations.append(local_data_risk)
        conditional_expectations = np.array(conditional_expectations)
        conditional_expectations = gaussian_filter1d(conditional_expectations, sigma=SMOOTH_KERNEL)
        return np.array([np.mean(i) for i in intervals]), conditional_expectations

    # Interventional coherent ALE
    def ale_plot(self, variable, target_var, data=None, intervals=20, model_type="scm") -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the accumulated local effect (ALE) for a given variable. The ALE is a way to visualize the effect of a
        variable on the safety factor.

        It is computed by discretizing the variable into bins and computing the local
        effect for each bin. The local effect is computed by first filtering the data values to the ones that fall
        within the bin, then intervening on the variable with the left and right extreme of the interval, and computing
        the difference in the safety factor. The accumulated local effect is the cumulative sum of the local effects.

        References:

         - Apley, Daniel W., and Jingyu Zhu. “Visualizing the effects of predictor variables in black box
        supervised learning models.” Journal of the Royal Statistical Society: Series B (Statistical Methodology)
        82.4 (2020): 1059-1086

        - Molnar, Christoph. Interpretable machine learning.
        https://christophm.github.io/interpretable-ml-book/ale.html, 2019.

        Parameters
        ----------
        variable : str
            Variable to compute ALE over
        data : pd.DataFrame
            Optional parameter to provide data for the ALE computation. If no data is provided, the model will
            generate it
        intervals : int
            Number of intervals to discretize the variable of study

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple with the x and y values of the ALE. The x value are the means of the bins, and the y values are
        """

        def compute_local_effect(variable, target_var, local_data, left_extreme, right_extreme, model_type = "scm"):
            # Get the distribution of the root nodes
            if model_type == "scm":
                vars = self.get_root_nodes()
            elif model_type == "cbn":
                # All variables except descendants of the variable
                vars = self.get_non_descendants(variable)
            else :
                raise Exception("Model type not recognized. Please use 'scm' or 'cbn'")
            stochastic_data = local_data[vars]

            # Intervene with left extreme of interval
            stochastic_data_0 = stochastic_data.copy()
            substitute = np.repeat(left_extreme, len(stochastic_data_0))
            stochastic_data_0[variable] = substitute
            data_0 = self.causal_inference(stochastic_data_0)
            sf0 = data_0[target_var]

            # Intervene with right extreme of interval
            stochastic_data_1 = stochastic_data.copy()
            substitute = np.repeat(right_extreme, len(stochastic_data_1))
            stochastic_data_1[variable] = substitute
            data_1 = self.causal_inference(stochastic_data_1)
            sf1 = data_1[target_var]

            # Get the local effect
            return sf1.mean()-sf0.mean(), len(stochastic_data) / len(data)

        if data is None:
            data = self.generate_data(DEFAULT_NB_SAMPLES)

        if isinstance(intervals, int):
            # Compute the intervals in which to compute the local effect
            cuts = np.linspace(data[variable].quantile(0.01), data[variable].quantile(0.99), intervals+1)
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
            # Compute the local effect for the interval
            le, prob = compute_local_effect(variable, target_var, local_data, left_extreme=interval[0],
                                            right_extreme=interval[1], model_type=model_type)
            local_effects.append(le)
        # Accumulate the local effects
        accumulated_local_effects = np.cumsum(local_effects)
        return np.array([np.mean(i) for i in intervals]), accumulated_local_effects - np.mean(accumulated_local_effects)
