from functools import partial

import gurobipy as gp
from gurobipy import GRB


class CallbackData:
    def __init__(self):
        self.last_gap_change_time = GRB.INFINITY
        self.last_gap = GRB.INFINITY


def callback(model, where, *, cbdata):
    if where != GRB.Callback.MIP:
        return
    if model.cbGet(GRB.Callback.MIP_SOLCNT) == 0:
        return
    
    # Use model.terminate() to end the search when required...
    gap = abs(model.cbGet(GRB.Callback.MIP_OBJBST) - model.cbGet(GRB.Callback.MIP_OBJBND))/model.cbGet(GRB.Callback.MIP_OBJBST)

    if abs(gap - cbdata.last_gap) > epsilon_to_compare_gap:
        cbdata.last_gap_change_time = model.cbGet(GRB.Callback.RUNTIME)
        cbdata.last_gap = gap
        return
    elif model.cbGet(GRB.Callback.RUNTIME) - cbdata.last_gap_change_time > time_from_best:
        model.terminate()


with gp.read("data/mkp.mps.bz2") as model:
    # Global variables used in the callback function
    time_from_best = 50
    epsilon_to_compare_gap = 1e-4

    # Initialize data passed to the callback function
    callback_data = CallbackData()
    callback_func = partial(callback, cbdata=callback_data)

    model.optimize(callback_func)