import gurobipy as gp

from base.results import ILPException


def solve_ilp_file(filepath: str) -> tuple[float, list[str], list[float]]:
    """Takes a filepath to an ILP file and returns the objective value, variable names, and corresponding variable values."""

    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()

        model = gp.read(filepath, env=env)
        model.optimize()

        if model.status == gp.GRB.INFEASIBLE:
            raise ILPException("Infeasible ILP problem")

        all_vars = model.getVars()

        obj = model.getObjective()
        names = model.getAttr("VarName", all_vars)
        values = model.getAttr("X", all_vars)

    return obj.getValue(), names, values
