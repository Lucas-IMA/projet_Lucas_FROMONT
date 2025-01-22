import json
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB

with open("data/portfolio-example.json", "r") as f:
    data = json.load(f)

n = data["num_assets"]
sigma = np.array(data["covariance"])
mu = np.array(data["expected_return"])
mu_0 = data["target_return"]
k = data["portfolio_max_size"]


with gp.Model("portfolio") as model:
    # Name the modeling objects to retrieve them
    dict_sigma = {(i, j): sigma[i, j] for i in range(n) for j in range(n)}
    dict_mu = {i: mu[i] for i in range(n)}

    x = model.addVars(n, lb=0, ub=100, vtype=GRB.CONTINUOUS, name="x")
    y = model.addVars(n, vtype=GRB.BINARY, name="y")

    LinExpr = gp.quicksum(x[i] * x[j] * sigma[i, j] for i in range(n) for j in range(n))
    model.setObjective(LinExpr, GRB.MINIMIZE)

    LinExpr2 = x.prod(dict_mu)
    model.addConstr(LinExpr2 >= mu_0, "return")
    model.addConstr(y.sum() <= k, "portfolio_size")
    model.addConstr(x.sum() == 100, "relative investment")
    model.addConstrs(x[i] <= 100 * y[i] for i in range(n))

    model.optimize()

    # Write the solution into a DataFrame
    portfolio = [var.X for var in model.getVars() if "x" in var.VarName]
    risk = model.ObjVal
    expected_return = model.getRow(model.getConstrByName("return")).getValue()
    df = pd.DataFrame(
        data=portfolio + [risk, expected_return],
        index=[f"asset_{i}" for i in range(n)] + ["risk", "return"],
        columns=["Portfolio"],
    )
    print(df)