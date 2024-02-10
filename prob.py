from typing import Any
import pyomo.environ as pyo
from pyomo.core.util import quicksum
import numpy as np
import numpy.typing as npt
from collections import defaultdict



def generate_scenarios(product:dict[int,Any], dist:dict[int,Any], ns:int) -> dict[int,npt.NDArray]:
    # scenarios is numpy array where the element ij represents the demand variation of product i in scenario j
    scenarios = {}
    for i, product_data in product.items():
        scenarios[i] = np.clip(dist[product_data['variance_id']].rvs(size=ns), a_min = 0., a_max=None) # ensure that the scenario is nonnegative

    return scenarios


def build_model(product:dict[int,Any], dist:dict[int,Any], args:dict[str,Any]) -> pyo.ConcreteModel:
    
    ######################### Preprocessing #########################
    scenarios = generate_scenarios(product, dist, args['ns'])
    G = set()
    Gi = defaultdict(list)
    d = {} # demand vector
    a = {} # COGS
    b = {} # selling price of product i, that is, margin + COGS
    c = {} # capacity limit of product i

    for i, data in product.items():
        G.add(data['substitutability_id'])
        Gi[data['substitutability_id']].append(i)
        d[i] = data['demand']
        a[i] = data['cogs']
        b[i] = data['margin'] + data['cogs']
        if not np.isnan(data['capacity']):
            c[i] = data['capacity']
    
    d_tilda = {} # demand realization vector
    for i in d.keys():
        for s in range(args['ns']):
            d_tilda[(i,s)] = d[i] * scenarios[i][s]
    

    # print('check substitutability id', G)
    # dsum = 0.
    # for _, val in d.items():
    #     dsum += val
    # print('sum d', dsum)
    # for s in range(args['ns']):
    #     dsum = 0.
    #     for i in d.keys():
    #         dsum += d_tilda[(i,s)]
    #     print('scenario',s,dsum)

    # print('check scenarios')
    # for i in range(6):
    #     print(scenarios[i][:2])


    ######################### Optimization Modeling #########################
    model = pyo.ConcreteModel()
    
    # # ====================
    # # I.    Sets
    # # ====================
    model.P = pyo.Set(initialize = list(product.keys())) # product IDs
    model.S = pyo.Set(initialize = range(args['ns'])) # scenario IDs
    model.G = pyo.Set(initialize = list(G)) # the set of the product ID sets
    model.Gi = pyo.Set(model.G, initialize = Gi) # product IDs belonging to the substitutability group i where i in {0,...,197}
    model.Pc = pyo.Set(initialize = list(c.keys())) # product IDs that capacity limits are defined

    # # ====================
    # # II.   Parameters
    # # ====================
    model.d = pyo.Param(model.P, initialize=d) # demand of product i
    model.d_tilda = pyo.Param(model.P, model.S, initialize=d_tilda) # demand realization of product i in scenario s
    model.ns = pyo.Param(initialize=args['ns']) # the number of scenarios, scalar
    model.a = pyo.Param(model.P, initialize=a) # COGS of product i
    model.b = pyo.Param(model.P, initialize=b) # selling price of product i, that is, margin + COGS
    model.c = pyo.Param(model.Pc, initialize=c) # capacity limit of product i
    model.mu = pyo.Param(initialize=args['mu']) # the ratio of aggregate surplus quantity across all products

    # # ====================
    # # III.  Variables
    # # ====================
    model.x = pyo.Var(model.P, within=pyo.NonNegativeReals) # surplus of product i
    model.y = pyo.Var(model.P, model.S, within=pyo.NonNegativeReals) # amount of demand of product i that cannot be met by the supplies in scenario s

    # # ====================
    # # IV.   Objective
    # # ====================
    model.obj = pyo.Objective( # maximize profit (expected revenue minus production cost)
        sense = pyo.maximize,
        rule = 1/model.ns * quicksum(model.b[i] * quicksum(model.d_tilda[i,s]-model.y[i,s] for s in model.S) for i in model.P) \
               - quicksum(model.a[i]*(model.x[i]+model.d[i]) for i in model.P)
    )

    # # ====================
    # # V.    Constraints
    # # ====================
    # 1. The surplus quantity added to each product's demand must not surpass its designated capacity limit.
    @model.Constraint(model.Pc)
    def cnst_capacity_limit(model, i): # upper bound of x
        return model.x[i] <= model.d[i]*model.c[i]
        
    # 2. The aggregate surplus quantity across all products should not exceed the total demand for all products, 
    #    adjusted by a macro target percentage. This macro target percentage is an adjustable input parameter, 
    #    ranging between 10% and 50%.
    @model.Constraint()
    def cnst_total_surplus_limit(model):
        return quicksum(model.x[i] for i in model.P) <= model.mu * quicksum(model.d[i] for i in model.P)

    # 3. For products classified within the same substitutability group, it's important to maintain adequate total 
    #    surplus quantities. This approach aims to mitigate the risk of lost sales by leveraging the substitutability 
    #    of products within these groups, ensuring that demand can be met even if specific products are over or undersupplied.
    @model.Constraint(model.G, model.S)
    def cnst_substitutability(model, g, s):
        return quicksum(model.x[i]+model.d[i] for i in model.Gi[g]) >= quicksum(model.d_tilda[i,s]-model.y[i,s] for i in model.Gi[g])

    return model


def solve_model(model:pyo.ConcreteModel, solver:str='highs') -> pyo.ConcreteModel:
    if solver.lower() == 'highs':
        opt = pyo.SolverFactory('appsi_highs')
    elif solver.lower() == 'glpk':
        opt = pyo.SolverFactory('glpk')
        
    result = opt.solve(model, tee=True)
    # for i in model.P:
    #     print(i, model.x[i].value)

    # xval_sum = 0.
    # for i in model.P:
    #     xval_sum += model.x[i].value
    return model