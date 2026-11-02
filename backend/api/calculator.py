from fastapi import APIRouter
from pydantic import BaseModel
import sympy as sp

router = APIRouter()

class CalcRequest(BaseModel):
    expression: str

@router.post("/")
def evaluate_expression(req: CalcRequest):
    try:
        # Safely evaluate mathematical strings
        expr = sp.sympify(req.expression)
        result = float(expr.evalf())
        return {"result": result, "error": None}
    except Exception as e:
        return {"result": None, "error": f"Invalid expression: {str(e)}"}