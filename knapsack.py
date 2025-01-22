import numpy as np
import gurobipy as gp
from gurobipy import GRB

def generate_knapsack(num_items):
    # Fix seed value
    rng = np.random.default_rng(seed=0)
    # Item values, weights
    values = rng.uniform(low=1, high=25, size=num_items)
    weights = rng.uniform(low=5, high=100, size=num_items)
    # Knapsack capacity
    capacity = 0.7 * weights.sum()

    return values, weights, capacity


def solve_knapsack_model(values, weights, capacity):
    num_items = len(values)
    # Turn values and weights numpy arrays to dict
    dict_values = {i: values[i] for i in range(num_items)}
    dict_weights = {i: weights[i] for i in range(num_items)}

    with gp.Env() as env:
        with gp.Model(name="knapsack", env=env) as model:
            # Define decision variables using the Model.addVars() method    
            x = model.addVars(num_items, vtype=GRB.BINARY, name="x")

            # Define objective function using the Model.setObjective() method
            # Build the LinExpr using the tupledict.prod() method
            LinExpr = x.prod(dict_values)
            model.setObjective(LinExpr, GRB.MAXIMIZE)


            # Define capacity constraint using the Model.addConstr() method
            model.addConstr(x.prod(dict_weights) <= capacity)

            model.optimize()

data = generate_knapsack(10000)
solve_knapsack_model(*data)