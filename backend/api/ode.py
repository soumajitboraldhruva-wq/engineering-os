from fastapi import APIRouter
from pydantic import BaseModel
import sympy as sp
import numpy as np

router = APIRouter()

class GeneralODERequest(BaseModel):
    equation: str  
    y0: float  # Initial position y(0)
    v0: float  # Initial velocity y'(0)
    time: float # Time to simulate
    variable: str = "t"
    function: str = "y"

@router.post("/solve")
def solve_general_ode(req: GeneralODERequest):
    try:
        t = sp.Symbol(req.variable)
        y_applied = sp.Function(req.function)(t)
        
        if "=" in req.equation:
            lhs_str, rhs_str = req.equation.split("=")
        else:
            lhs_str, rhs_str = req.equation, "0"

        def parse_derivatives(expr_str):
            f, v = req.function, req.variable
            expr_str = expr_str.replace(f"{f}'''", f"Derivative({f}, {v}, 3)")
            expr_str = expr_str.replace(f"{f}''", f"Derivative({f}, {v}, 2)")
            expr_str = expr_str.replace(f"{f}'", f"Derivative({f}, {v})")
            return expr_str

        parsed_lhs = parse_derivatives(lhs_str.strip())
        parsed_rhs = parse_derivatives(rhs_str.strip())
        
        local_dict = {req.function: y_applied, req.variable: t}
        lhs_math = sp.sympify(parsed_lhs, locals=local_dict)
        rhs_math = sp.sympify(parsed_rhs, locals=local_dict)
        diff_eq = sp.Eq(lhs_math, rhs_math)
        
        # Apply Initial Conditions! y(0) = y0, y'(0) = v0
        ics = {y_applied.subs(t, 0): req.y0, y_applied.diff(t).subs(t, 0): req.v0}
        
        # Solve Symbolically
        exact_solution = sp.dsolve(diff_eq, y_applied, ics=ics)
        
        # Convert exact math to numerical arrays for Plotly Graph
        func_numpy = sp.lambdify(t, exact_solution.rhs, modules=['numpy'])
        t_vals = np.linspace(0, req.time, 300)
        y_vals = func_numpy(t_vals)
        
        # If the solution is a constant, numpy might return a single number instead of an array
        if isinstance(y_vals, (int, float)):
            y_vals = np.full_like(t_vals, y_vals)
            
        return {
            "latex_eq": sp.latex(diff_eq),
            "latex_sol": f"{req.function}({req.variable}) = " + sp.latex(exact_solution.rhs),
            "t": t_vals.tolist(),
            "y": np.real(y_vals).tolist(), # Extract real part in case of tiny complex rounding
            "error": None
        }
    except Exception as e:
        return {"error": f"Failed to solve ODE: {str(e)}"}