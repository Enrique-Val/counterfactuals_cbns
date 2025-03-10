from bidict import bidict

from StructuralCausalModels import structural_causal_model
from scipy.stats import norm

from causal_ale import utils
from causal_ale.model import Model


class ModelBuilder:
    def __init__(self, variables, name = ""):
        self.nodes = variables
        self.equations = list()
        self.name = name
        self.var_dict = bidict({i: var for i, var in enumerate(self.nodes)})

    def get_var_dict(self) -> bidict:
        return self.var_dict

    def add_equation(self, variable, parents=None, function=utils.identity, noise=norm(0, 0)):
        self.equations.append(utils.make_equation(variable, parents, function, noise, self.var_dict))

    def build(self):
        #print(self.equations)
        children = [i.index_lhs for i in self.equations]
        no_equation = [i for i in range(len(self.nodes)) if i not in children]
        if len(no_equation) > 0:
            raise Exception('No equation provided for nodes: ', [self.var_dict[i] for i in no_equation])

        raw_model = structural_causal_model.StructuralCausalModel(name=self.name,
                                                             nb_var=len(self.equations),
                                                             structural_equations=self.equations)
        return Model(model=raw_model, var_dict=self.var_dict)#, var_type=self.var_type)
