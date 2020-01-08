from pyomo.environ import (
    ConcreteModel,
    Binary,
    Var,
    Constraint,
    ConstraintList,
    Objective,
    SolverFactory,
    TerminationCondition,
    SolverStatus,
    value,
)
import numpy as np


def solve_sudoku(constraint_matrix):
    # Initialise concrete model
    model = ConcreteModel()

    # Basic sudoku dimensions
    model.dim = list(range(1, 10))

    # Create model variables
    model.x = Var(model.dim, model.dim, model.dim, within=Binary)

    # Rule one instance of each number per row
    def sum_rows_rule(model, r, v):
        return sum(model.x[r, c, v] for c in model.dim) == 1

    model.sum_rows = Constraint(model.dim, model.dim, rule=sum_rows_rule)

    # Rule one instance of each number per col
    def sum_cols_rule(model, c, v):
        return sum(model.x[r, c, v] for r in model.dim) == 1

    model.sum_cols = Constraint(model.dim, model.dim, rule=sum_cols_rule)

    # One instance of each number per block
    model.sum_blocks_rules = ConstraintList()
    for rb in [1, 4, 7]:
        for cb in [1, 4, 7]:
            for v in model.dim:
                model.sum_blocks_rules.add(
                    sum(
                        sum(model.x[r, c, v] for c in [cb, cb + 1, cb + 2])
                        for r in [rb, rb + 1, rb + 2]
                    )
                    == 1
                )

    # One value per cell
    def sum_cells_rule(model, r, c):
        return sum(model.x[r, c, v] for v in model.dim) == 1

    model.sum_cells = Constraint(model.dim, model.dim, rule=sum_cells_rule)

    # Dummy objective
    def dummy_obj(model):
        return sum(sum(model.x[r, c, 1] for r in model.dim) for c in model.dim)

    model.obj = Objective(rule=dummy_obj)

    # Hard rules (given numbers)
    model.initial_vals_rules = ConstraintList()
    for r in model.dim:
        for c in model.dim:
            # Fetch value from given matrix
            hard_v = constraint_matrix[r - 1, c - 1]

            if hard_v != 0:
                # Force solution to be equal this value
                model.initial_vals_rules.add(model.x[r, c, hard_v] == 1)

    # Solve sudoku
    solver = SolverFactory("glpk")
    status = solver.solve(model)

    # Save solution to a matrix 9x9
    if (status.solver.termination_condition == TerminationCondition.optimal) and (
        status.solver.status == SolverStatus.ok
    ):
        solution_ = np.empty([9, 9], dtype=int)
        for r in model.dim:
            for c in model.dim:
                solution_[r - 1, c - 1] = sum(
                    value(model.x[r, c, v]) * v for v in model.dim
                )
    else:
        solution_ = np.tile("?", (9, 9))

    return solution_
