import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy

# 24 Hour Load Forecast (MW)
load_forecast = np.array([
    4, 4, 4, 4, 4, 4, 6, 6, 12, 12, 12, 12, 12, 4, 4, 4, 4, 16, 16, 16, 16, 6.5, 6.5, 6.5,
])

# Solar energy forecast (MW)
solar_forecast = np.array([
    0, 0, 0, 0, 0, 0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.5, 3.5, 2.5, 2.0, 1.5, 1.0, 0.5, 0, 0, 0, 0, 0, 0,
])

# Global number of time intervals
nTimeIntervals = len(load_forecast)

# Thermal units
thermal_units = ["gen1", "gen2", "gen3"]

# Thermal units' costs and limits
thermal_units_cost, a, b, c, sup_cost, sdn_cost = gp.multidict({
    "gen1": [5.0, 0.5, 1.0, 2, 1],
    "gen2": [5.0, 0.5, 0.5, 2, 1],
    "gen3": [5.0, 3.0, 2.0, 2, 1],
})
thermal_units_limits, pmin, pmax = gp.multidict({
    "gen1": [1.5, 5.0],
    "gen2": [2.5, 10.0],
    "gen3": [1.0, 3.0],
})
init_status = np.array([0, 0, 0])

# Convert dictionaries to numpy arrays for vectorized operations
a = np.array([a[g] for g in thermal_units])
b = np.array([b[g] for g in thermal_units])
c = np.array([c[g] for g in thermal_units])
sup_cost = np.array([sup_cost[g] for g in thermal_units])
sdn_cost = np.array([sdn_cost[g] for g in thermal_units])
pmin = np.array([pmin[g] for g in thermal_units])
pmax = np.array([pmax[g] for g in thermal_units])

with gp.Env() as env, gp.Model(env=env) as model:
    # Create decision variables using matrix API
    thermal_units_out_power = model.addMVar(
        (len(thermal_units), nTimeIntervals), lb=0, name="thermal_units_out_power"
    )
    thermal_units_startup_status = model.addMVar(
        (len(thermal_units), nTimeIntervals), vtype=GRB.BINARY, name="thermal_units_startup_status"
    )
    thermal_units_shutdown_status = model.addMVar(
        (len(thermal_units), nTimeIntervals), vtype=GRB.BINARY, name="thermal_units_shutdown_status"
    )
    thermal_units_comm_status = model.addMVar(
        (len(thermal_units), nTimeIntervals), vtype=GRB.BINARY, name="thermal_units_comm_status"
    )

    # Objective function (vectorized)
    obj_fun_expr = gp.quicksum(
        c @ thermal_units_out_power[:, t] ** 2 +
        b @ thermal_units_out_power[:, t] +
        a @ thermal_units_comm_status[:, t] +
        sup_cost @ thermal_units_startup_status[:, t] +
        sdn_cost @ thermal_units_shutdown_status[:, t]
        for t in range(nTimeIntervals)
    )
    model.setObjective(obj_fun_expr, GRB.MINIMIZE)

    # Power balance constraints (vectorized)
    model.addConstr(
        thermal_units_out_power.sum(axis=0) + solar_forecast == load_forecast,
        name="power_balance"
    )

    # Logical constraints for startup and shutdown
    model.addConstr(
        thermal_units_comm_status[:, 0] - init_status ==
        thermal_units_startup_status[:, 0] - thermal_units_shutdown_status[:, 0],
        name="logical_constraints_t0"
    )
    if nTimeIntervals > 1:
        model.addConstrs(
            thermal_units_comm_status[:, t] - thermal_units_comm_status[:, t - 1] ==
            thermal_units_startup_status[:, t] - thermal_units_shutdown_status[:, t]
            for t in range(1, nTimeIntervals)
        )

    model.addConstrs(
        thermal_units_startup_status[:, t] + thermal_units_shutdown_status[:, t] <= 1
        for t in range(nTimeIntervals)
    )

    # Physical constraints with indicator constraints (loop required)
    for g in range(len(thermal_units)):
        for t in range(nTimeIntervals):
            model.addGenConstrIndicator(
                thermal_units_comm_status[g, t], True,
                thermal_units_out_power[g, t] >= pmin[g],
                name=f"min_output_{g}_{t}",
            )
            model.addGenConstrIndicator(
                thermal_units_comm_status[g, t], True,
                thermal_units_out_power[g, t] <= pmax[g],
                name=f"max_output_{g}_{t}",
            )
            model.addGenConstrIndicator(
                thermal_units_comm_status[g, t], False,
                thermal_units_out_power[g, t] == 0,
                name=f"off_output_{g}_{t}",
            )

    model.optimize()

    # Display results
    if model.status == GRB.OPTIMAL:
        print(f"Overall Cost = {round(model.ObjVal, 2)}\n")
        for g_idx, g in enumerate(thermal_units):
            print(f"{g} Output:", thermal_units_out_power[g_idx, :].X)
        print("Solar Forecast:", solar_forecast)
        print("Load Forecast:", load_forecast)
    else:
        print("No optimal solution found.")
