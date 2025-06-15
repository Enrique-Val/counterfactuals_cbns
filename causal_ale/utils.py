import numpy as np
import pandas as pd
from bidict import bidict

from StructuralCausalModels import structural_equation, structural_causal_model
from scipy.stats import norm

def copy_model(model):
    new_eq_list = []
    for i in model.structural_equations:
        new_eq_list.append(copy_structural_equation(i))
    return structural_causal_model.StructuralCausalModel(name=model.name, nb_var=len(new_eq_list),
                                                         structural_equations=new_eq_list)


def copy_structural_equation(eq):
    return structural_equation.StructuralEquation(eq.index_lhs, eq.indices_rhs,
                                                  eq.exogenous_variable, eq.function)


def identity(u):
    return u


def pass_values(u, parent):
    return parent


def make_equation(variable, parents=None, function=identity, noise=norm(0, 0), var_dict=None):
    if parents is None:
        parents = []
    if var_dict is None:
        raise ValueError("var_dict must be provided")
        var_dict = bidict({i: var for i, var in enumerate([variable] + parents)})
    index_lhs = var_dict.inverse[variable]
    indices_rhs = [var_dict.inverse[i] for i in parents]
    return structural_equation.StructuralEquation(index_lhs, indices_rhs, noise, function)

def get_pdf(a:np.array, density:bool=True, start:float=0, stop:float=100, step:float=1e-2):
    '''
    calcula la funcion de distribucion acumulada de un array sobre unos bins definidos
    con density = True calcula la densidad de probabilidad normalizando
    pero condicionado al ancho de paso de los intervalos
    If True, the result is the value of the probability density function at the bin,
    normalized such that the integral over the range is 1.
    Note that the sum of the histogram values will not be equal to 1 unless bins of unity width are chosen;
    it is not a probability mass function.
    '''

    bins = np.arange(start, stop, step)
    p_, bin_edges = np.histogram(a, density=density, bins=bins)
    p = p_*step

    return p, bin_edges[:-1]
