from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import numpy as np
import sympy as sp
from fastapi import APIRouter
from pydantic import BaseModel, Field

try:
    import scipy.linalg as scipy_linalg
    import scipy.sparse as scipy_sparse
    import scipy.sparse.linalg as scipy_sparse_linalg
    SCIPY_AVAILABLE = True
except ImportError:
    scipy_linalg = None
    scipy_sparse = None
    scipy_sparse_linalg = None
    SCIPY_AVAILABLE = False

router = APIRouter()

# -----------------------------------------------------------------------------
# Safe symbolic environment
# -----------------------------------------------------------------------------

BASE_SYMBOLS = {
    name: sp.Symbol(name, real=True)
    for name in ("x", "y", "z", "t", "u", "v", "s")
}

SAFE_LOCALS: Dict[str, Any] = {
    **BASE_SYMBOLS,
    "pi": sp.pi,
    "e": sp.E,
    "E": sp.E,
    "I": sp.I,
    "oo": sp.oo,
    "sqrt": sp.sqrt,
    "cbrt": lambda q: sp.real_root(q, 3),
    "abs": sp.Abs,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "cot": sp.cot,
    "sec": sp.sec,
    "csc": sp.csc,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "asinh": sp.asinh,
    "acosh": sp.acosh,
    "atanh": sp.atanh,
    "exp": sp.exp,
    "log": sp.log,
    "ln": sp.log,
    "log10": lambda q: sp.log(q, 10),
    "floor": sp.floor,
    "ceiling": sp.ceiling,
    "factorial": sp.factorial,
    "gamma": sp.gamma,
    "erf": sp.erf,
    "Min": sp.Min,
    "Max": sp.Max,
    "diff": sp.diff,
    "differentiate": sp.diff,
    "integrate": sp.integrate,
    "int": sp.integrate,
    "limit": sp.limit,
    "factor": sp.factor,
    "expand": sp.expand,
    "simplify": sp.simplify,
}

# SymPy's sympify is intentionally given an explicit locals dictionary. The
# expression is still math-oriented, but callers should not treat this as a
# security sandbox for arbitrary untrusted Python.
_ALLOWED_EXPRESSION = re.compile(r"^[A-Za-z0-9_+\-*/().,\[\]{} ^%:=<>|!]+$")
IDENTIFIER = r"[A-Za-z_][A-Za-z0-9_]*"


def clean_expression(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return "0"
    text = (
        text.replace("−", "-")
        .replace("π", "pi")
        .replace("∞", "oo")
        .replace("²", "^2")
        .replace("³", "^3")
    )
    if not _ALLOWED_EXPRESSION.fullmatch(text):
        raise ValueError(f"Unsupported characters in expression: {text}")
    return text.replace("^", "**")


def parse_expression(value: Any) -> sp.Expr:
    if isinstance(value, bool):
        return sp.Integer(int(value))
    if isinstance(value, (int, float)):
        if not np.isfinite(value):
            raise ValueError("Expression contains NaN or Infinity.")
        return sp.sympify(value)
    text = clean_expression(value)
    try:
        return sp.sympify(text, locals=SAFE_LOCALS)
    except Exception as exc:
        raise ValueError(f"Could not parse expression '{value}': {exc}") from exc


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, complex):
        if abs(value.imag) < 1e-12:
            return float(value.real)
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, sp.Basic):
        if value.is_Integer:
            return int(value)
        if value.is_Rational:
            return str(value)
        if value.is_Float:
            return float(value)
        if value.is_real is True:
            try:
                return float(value)
            except Exception:
                return str(value)
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    return value


def split_top_level(text: str) -> List[str]:
    parts: List[str] = []
    start = 0
    depth = 0
    quote: Optional[str] = None
    for i, ch in enumerate(text):
        if quote:
            if ch == quote and (i == 0 or text[i - 1] != "\\"):
                quote = None
            continue
        if ch in "'\"":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth < 0:
                raise ValueError("Unbalanced brackets.")
        elif ch == "," and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def parse_matrix(values: List[List[Any]]) -> sp.Matrix:
    if not isinstance(values, list) or not values:
        raise ValueError("Matrix must be a non-empty 2D array.")
    if not isinstance(values[0], list) or not values[0]:
        raise ValueError("Matrix must have at least one column.")
    width = len(values[0])
    if any(not isinstance(row, list) or len(row) != width for row in values):
        raise ValueError("All matrix rows must have the same number of columns.")
    return sp.Matrix([[parse_expression(cell) for cell in row] for row in values])


def parse_vector(values: List[Any]) -> sp.Matrix:
    if not isinstance(values, list) or not values:
        raise ValueError("Vector cannot be empty.")
    return sp.Matrix([parse_expression(value) for value in values])


def parse_matrix_text(text: str) -> sp.Matrix:
    text = text.strip()
    if not (text.startswith("[[") and text.endswith("]]")):
        raise ValueError("Expected matrix syntax like [[1,2],[3,4]].")
    inner = text[1:-1].strip()
    rows: List[List[sp.Expr]] = []
    depth = 0
    start: Optional[int] = None
    chunks: List[str] = []
    for i, ch in enumerate(inner):
        if ch == "[":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth < 0:
                raise ValueError("Unbalanced brackets in matrix.")
            if depth == 0 and start is not None:
                chunks.append(inner[start + 1 : i])
                start = None
        elif depth == 0 and ch not in {",", " ", "\n", "\t", "\r"}:
            raise ValueError("Invalid matrix syntax.")
    if not chunks:
        raise ValueError("Matrix cannot be empty.")
    for chunk in chunks:
        rows.append([parse_expression(cell) for cell in split_top_level(chunk)])
    return parse_matrix([[str(cell) for cell in row] for row in rows])


def parse_vector_text(text: str) -> sp.Matrix:
    text = text.strip()
    if not (text.startswith("[") and text.endswith("]")):
        raise ValueError("Expected vector syntax like [1,2,3].")
    return parse_vector(split_top_level(text[1:-1]))


def numeric_matrix(matrix: sp.Matrix) -> np.ndarray:
    if matrix.free_symbols:
        names = ", ".join(sorted(str(s) for s in matrix.free_symbols))
        raise ValueError(
            f"Numeric operation requires numeric entries. Found symbols: {names}."
        )
    try:
        arr = np.array(matrix.evalf(), dtype=float)
    except Exception as exc:
        raise ValueError(f"Could not convert matrix to numeric values: {exc}") from exc
    if not np.all(np.isfinite(arr)):
        raise ValueError("Matrix contains NaN or Infinity.")
    return arr


def numeric_vector(vector: sp.Matrix) -> np.ndarray:
    if vector.free_symbols:
        names = ", ".join(sorted(str(s) for s in vector.free_symbols))
        raise ValueError(
            f"Numeric operation requires numeric values. Found symbols: {names}."
        )
    try:
        arr = np.array(vector.evalf(), dtype=float).reshape(-1)
    except Exception as exc:
        raise ValueError(f"Could not convert vector to numeric values: {exc}") from exc
    if not np.all(np.isfinite(arr)):
        raise ValueError("Vector contains NaN or Infinity.")
    return arr


def render_matrix(matrix: sp.Matrix) -> Dict[str, Any]:
    return {
        "result": [
            [str(sp.simplify(matrix[r, c])) for c in range(matrix.cols)]
            for r in range(matrix.rows)
        ],
        "latex": sp.latex(matrix),
        "shape": [matrix.rows, matrix.cols],
    }


def matrix_metadata(A: np.ndarray, tolerance: float) -> Dict[str, Any]:
    m, n = A.shape
    tol = max(float(tolerance), 1e-15)
    rank = int(np.linalg.matrix_rank(A, tol=tol))
    determinant = None
    trace = None
    condition_number = None
    if m == n:
        determinant = float(np.linalg.det(A))
        trace = float(np.trace(A))
        if rank == n:
            condition_number = float(np.linalg.cond(A))
    singular_values = np.linalg.svd(A, compute_uv=False)
    symmetric = bool(m == n and np.allclose(A, A.T, atol=tol, rtol=1e-8))
    positive_definite = False
    if symmetric:
        try:
            positive_definite = bool(np.all(np.linalg.eigvalsh(A) > tol))
        except np.linalg.LinAlgError:
            positive_definite = False
    return {
        "shape": [m, n],
        "rows": m,
        "columns": n,
        "rank": rank,
        "determinant": determinant,
        "trace": trace,
        "condition_number": condition_number,
        "singular_values": singular_values.tolist(),
        "sparsity": float(np.count_nonzero(A == 0) / A.size),
        "symmetry": symmetric,
        "positive_definite": positive_definite,
        "invertible": bool(m == n and rank == n),
        "frobenius_norm": float(np.linalg.norm(A, "fro")),
    }


def calculus_variable(name: str) -> sp.Symbol:
    name = str(name).strip()
    if name not in BASE_SYMBOLS:
        raise ValueError("Variable must be one of x, y, z, t, u, v, s.")
    return BASE_SYMBOLS[name]


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

class MatrixSolveRequest(BaseModel):
    matrix_a: List[List[Any]]
    vector_b: List[Any]
    tolerance: float = 1e-10


class MatrixOperationRequest(BaseModel):
    operation: str
    matrix_a: List[List[Any]]
    matrix_b: Optional[List[List[Any]]] = None
    vector_b: Optional[List[Any]] = None
    power: Optional[int] = None
    tolerance: float = 1e-10
    substitutions: Dict[str, Any] = Field(default_factory=dict)


class MatrixCalculusRequest(BaseModel):
    operation: str
    matrix: List[List[Any]]
    variable: str = "x"
    order: int = 1
    lower: Optional[Any] = None
    upper: Optional[Any] = None


class SymbolicRequest(BaseModel):
    expression: str
    operation: str = "simplify"
    variable: str = "x"
    order: int = 1
    lower: Optional[Any] = None
    upper: Optional[Any] = None


class SparseSolveRequest(BaseModel):
    matrix_a: List[List[Any]]
    vector_b: List[Any]
    tolerance: float = 1e-10


class EinsumRequest(BaseModel):
    equation: str
    operands: List[Any]


class CommandRequest(BaseModel):
    command: str
    variables: Dict[str, Any] = Field(default_factory=dict)
    tolerance: float = 1e-10


# -----------------------------------------------------------------------------
# Solvers and operations
# -----------------------------------------------------------------------------


def solve_system(req: MatrixSolveRequest) -> Dict[str, Any]:
    try:
        A_symbolic = parse_matrix(req.matrix_a)
        b_symbolic = parse_vector(req.vector_b)
        A = numeric_matrix(A_symbolic)
        b = numeric_vector(b_symbolic)
        if A.shape[0] != b.size:
            raise ValueError(f"Vector b must contain {A.shape[0]} values.")

        tol = max(float(req.tolerance), 1e-15)
        rank_A = int(np.linalg.matrix_rank(A, tol=tol))
        augmented = np.column_stack((A, b))
        rank_augmented = int(np.linalg.matrix_rank(augmented, tol=tol))
        m, n = A.shape

        if m == n and rank_A == n:
            x = np.linalg.solve(A, b)
            method = "direct"
            status = "unique_solution"
            explanation = "A is square and full rank, so Ax = b has a unique solution."
        else:
            x, _, _, _ = np.linalg.lstsq(A, b, rcond=tol)
            method = "least_squares"
            if rank_augmented > rank_A:
                status = "no_exact_solution"
                explanation = "The system is inconsistent; the returned vector is the least-squares fit."
            elif rank_A < n:
                status = "infinitely_many_solutions"
                explanation = "The system is rank-deficient; the returned vector is the minimum-norm least-squares solution."
            else:
                status = "unique_solution"
                explanation = "A has full column rank and the solution is unique."

        residual = A @ x - b
        inverse = None
        determinant = None
        condition_number = None
        if m == n:
            determinant = float(np.linalg.det(A))
            if rank_A == n:
                inverse = np.linalg.inv(A).tolist()
                condition_number = float(np.linalg.cond(A))

        return json_safe({
            "operation": "solve",
            "status": status,
            "method": method,
            "explanation": explanation,
            "shape": [m, n],
            "solution": x.tolist(),
            "rank": rank_A,
            "augmented_rank": rank_augmented,
            "determinant": determinant,
            "inverse": inverse,
            "residual_vector": residual.tolist(),
            "residual_norm": np.linalg.norm(residual),
            "condition_number": condition_number,
            "metadata": matrix_metadata(A, tol),
            "error": None,
        })
    except Exception as exc:
        return {"operation": "solve", "error": str(exc)}


def operation_result(req: MatrixOperationRequest) -> Dict[str, Any]:
    try:
        M = parse_matrix(req.matrix_a)
        if req.substitutions:
            subs = {calculus_variable(k): parse_expression(v) for k, v in req.substitutions.items()}
            M = M.subs(subs)
        operation = req.operation.lower().strip()
        tol = max(float(req.tolerance), 1e-15)

        symbolic_ops = {"rref", "nullspace"}
        if operation in symbolic_ops:
            if operation == "rref":
                rref, pivots = M.rref()
                return {"operation": operation, **render_matrix(rref), "pivots": list(pivots)}
            basis = M.nullspace()
            basis_matrix = sp.Matrix.hstack(*basis) if basis else sp.zeros(M.cols, 0)
            return {
                "operation": operation,
                "dimension": len(basis),
                "basis": [[str(v) for v in vec] for vec in basis],
                "result": [[str(basis_matrix[r, c]) for c in range(basis_matrix.cols)] for r in range(basis_matrix.rows)],
                "latex": sp.latex(basis_matrix),
                "shape": list(basis_matrix.shape),
            }

        if operation in {"transpose"}:
            return {"operation": operation, **render_matrix(M.T)}

        if operation in {"determinant", "trace", "inverse", "eigenvalues", "eigenvectors", "svd", "qr", "lu", "cholesky", "pseudoinverse", "pinv", "norm", "multiply", "power"}:
            A = numeric_matrix(M)

        if operation == "determinant":
            if A.shape[0] != A.shape[1]:
                raise ValueError("Determinant requires a square matrix.")
            return json_safe({"operation": operation, "value": np.linalg.det(A), "latex": r"\det(A)", "metadata": matrix_metadata(A, tol)})

        if operation == "trace":
            if A.shape[0] != A.shape[1]:
                raise ValueError("Trace requires a square matrix.")
            return json_safe({"operation": operation, "value": np.trace(A), "latex": r"\operatorname{tr}(A)"})

        if operation == "inverse":
            if A.shape[0] != A.shape[1]:
                raise ValueError("Inverse requires a square matrix.")
            if np.linalg.matrix_rank(A, tol) < A.shape[0]:
                raise ValueError("Matrix is singular and cannot be inverted.")
            return json_safe({"operation": operation, "result": np.linalg.inv(A), "metadata": matrix_metadata(A, tol), "latex": r"A^{-1}"})

        if operation == "rank":
            return {"operation": operation, "value": int(np.linalg.matrix_rank(A, tol=tol)), "metadata": matrix_metadata(A, tol)}

        if operation == "eigenvalues":
            if A.shape[0] != A.shape[1]:
                raise ValueError("Eigenvalues require a square matrix.")
            return json_safe({"operation": operation, "values": np.linalg.eigvals(A), "metadata": matrix_metadata(A, tol)})

        if operation == "eigenvectors":
            if A.shape[0] != A.shape[1]:
                raise ValueError("Eigenvectors require a square matrix.")
            values, vectors = np.linalg.eig(A)
            return json_safe({"operation": operation, "values": values, "vectors": vectors, "metadata": matrix_metadata(A, tol)})

        if operation == "svd":
            U, S, Vt = np.linalg.svd(A, full_matrices=False)
            return json_safe({"operation": operation, "U": U, "singular_values": S, "Vt": Vt, "reconstruction": U @ np.diag(S) @ Vt, "rank": int(np.linalg.matrix_rank(A, tol)), "metadata": matrix_metadata(A, tol), "latex": r"A=U\Sigma V^T"})

        if operation == "qr":
            Q, R = np.linalg.qr(A, mode="reduced")
            return json_safe({"operation": operation, "Q": Q, "R": R, "reconstruction": Q @ R, "orthogonality_error": np.linalg.norm(Q.T @ Q - np.eye(Q.shape[1])), "latex": r"A=QR"})

        if operation == "lu":
            if not SCIPY_AVAILABLE:
                raise RuntimeError("SciPy is required for LU decomposition.")
            P, L, U = scipy_linalg.lu(A)
            return json_safe({"operation": operation, "P": P, "L": L, "U": U, "reconstruction": P @ L @ U, "latex": r"A=PLU"})

        if operation == "cholesky":
            if A.shape[0] != A.shape[1]:
                raise ValueError("Cholesky requires a square matrix.")
            if not np.allclose(A, A.T, atol=tol, rtol=1e-8):
                raise ValueError("Matrix must be symmetric.")
            L = np.linalg.cholesky(A)
            return json_safe({"operation": operation, "L": L, "reconstruction": L @ L.T, "latex": r"A=LL^T"})

        if operation in {"pinv", "pseudoinverse"}:
            return json_safe({"operation": "pseudoinverse", "result": np.linalg.pinv(A, rcond=tol), "latex": r"A^+"})

        if operation == "norm":
            return json_safe({"operation": operation, "frobenius": np.linalg.norm(A, "fro"), "spectral": np.linalg.norm(A, 2), "one_norm": np.linalg.norm(A, 1), "infinity_norm": np.linalg.norm(A, np.inf)})

        if operation == "multiply":
            if req.matrix_b is None:
                raise ValueError("Matrix B is required for multiplication.")
            B = numeric_matrix(parse_matrix(req.matrix_b))
            if A.shape[1] != B.shape[0]:
                raise ValueError(f"Cannot multiply {A.shape[0]}×{A.shape[1]} by {B.shape[0]}×{B.shape[1]}.")
            C = A @ B
            return json_safe({"operation": operation, "result": C, "shape": list(C.shape), "latex": r"AB"})

        if operation == "power":
            if A.shape[0] != A.shape[1]:
                raise ValueError("Matrix power requires a square matrix.")
            if req.power is None:
                raise ValueError("Power n is required.")
            if abs(req.power) > 100:
                raise ValueError("Matrix power is limited to |n| ≤ 100.")
            return json_safe({"operation": operation, "power": req.power, "result": np.linalg.matrix_power(A, req.power), "latex": rf"A^{{{req.power}}}"})

        raise ValueError(f"Unsupported matrix operation '{req.operation}'.")
    except Exception as exc:
        return {"operation": req.operation, "error": str(exc)}


# -----------------------------------------------------------------------------
# API endpoints
# -----------------------------------------------------------------------------

@router.get("/health")
def health():
    return {
        "status": "ok",
        "service": "Engineering OS Linear Algebra",
        "numpy": np.__version__,
        "sympy": sp.__version__,
        "scipy": SCIPY_AVAILABLE,
    }


@router.post("/solve")
def solve_endpoint(req: MatrixSolveRequest):
    return solve_system(req)


# Same enhanced solver is also exposed at the exact old MatrixSolver contract.
@router.post("/legacy-solve")
def legacy_solve(req: MatrixSolveRequest):
    return solve_system(req)


@router.post("/operation")
def operation_endpoint(req: MatrixOperationRequest):
    return operation_result(req)


@router.post("/calculus")
def calculus_endpoint(req: MatrixCalculusRequest):
    try:
        M = parse_matrix(req.matrix)
        var = calculus_variable(req.variable)
        order = int(req.order)
        if order < 1 or order > 20:
            raise ValueError("Calculus order must be between 1 and 20.")

        op = req.operation.lower().strip()
        if op in {"diff", "differentiate", "derivative"}:
            result = M.applyfunc(lambda expr: sp.simplify(sp.diff(expr, var, order)))
        elif op in {"integrate", "integral", "int"}:
            if (req.lower is None) != (req.upper is None):
                raise ValueError("Provide both lower and upper bounds for definite integration.")
            if req.lower is None:
                result = M.applyfunc(lambda expr: sp.simplify(sp.integrate(expr, var, order)))
            else:
                low = parse_expression(req.lower)
                high = parse_expression(req.upper)
                result = M.applyfunc(lambda expr: sp.simplify(sp.integrate(expr, (var, low, high))))
        else:
            raise ValueError("Operation must be differentiate or integrate.")

        return {"operation": op, "variable": str(var), "order": order, "lower": req.lower, "upper": req.upper, **render_matrix(result)}
    except Exception as exc:
        return {"operation": req.operation, "error": str(exc)}


@router.post("/symbolic")
def symbolic_endpoint(req: SymbolicRequest):
    try:
        expr = parse_expression(req.expression)
        var = calculus_variable(req.variable)
        order = int(req.order)
        op = req.operation.lower().strip()
        if op in {"diff", "differentiate", "derivative"}:
            result = sp.diff(expr, var, order)
            op = "differentiate"
        elif op in {"integrate", "integral", "int"}:
            if (req.lower is None) != (req.upper is None):
                raise ValueError("Provide both lower and upper bounds for definite integration.")
            if req.lower is None:
                result = sp.integrate(expr, var, order)
            else:
                result = sp.integrate(expr, (var, parse_expression(req.lower), parse_expression(req.upper)))
            op = "integrate"
        else:
            result = sp.simplify(expr)
            op = "simplify"
        return {"operation": op, "input": req.expression, "result": str(sp.simplify(result)), "latex": sp.latex(result), "variable": str(var), "order": order}
    except Exception as exc:
        return {"operation": req.operation, "error": str(exc)}


@router.post("/sparse/solve")
def sparse_solve(req: SparseSolveRequest):
    try:
        if not SCIPY_AVAILABLE:
            raise RuntimeError("SciPy is required for sparse solving.")
        A = numeric_matrix(parse_matrix(req.matrix_a))
        b = numeric_vector(parse_vector(req.vector_b))
        if A.shape[0] != b.size or A.shape[0] != A.shape[1]:
            raise ValueError("Sparse direct solve requires a square matrix and matching vector.")
        csr = scipy_sparse.csr_matrix(A)
        x = scipy_sparse_linalg.spsolve(csr, b)
        residual = A @ x - b
        return json_safe({"operation": "sparse_solve", "solution": x, "residual": residual, "residual_norm": np.linalg.norm(residual), "shape": list(A.shape), "nonzero_entries": int(csr.nnz), "density": float(csr.nnz / A.size)})
    except Exception as exc:
        return {"operation": "sparse_solve", "error": str(exc)}


@router.post("/einsum")
def einsum_endpoint(req: EinsumRequest):
    try:
        operands = [np.asarray(item, dtype=float) for item in req.operands]
        result = np.einsum(req.equation, *operands)
        return json_safe({"operation": "einsum", "equation": req.equation, "result": result, "shape": list(result.shape) if hasattr(result, "shape") else []})
    except Exception as exc:
        return {"operation": "einsum", "error": str(exc)}


# -----------------------------------------------------------------------------
# Command console
# -----------------------------------------------------------------------------


def command_matrix(value: Any) -> sp.Matrix:
    return parse_matrix(value)


def execute_command(command: str, variables: Dict[str, Any], tolerance: float) -> Dict[str, Any]:
    command = command.strip()
    if not command:
        raise ValueError("Empty command.")

    if command.lower() == "help()":
        return {
            "type": "help",
            "commands": [
                "A = [[1,2],[sin(pi/4),log(10)]]",
                "b = [5,6]",
                "solve(A,b)", "det(A)", "rank(A)", "inverse(A)",
                "rref(A)", "nullspace(A)", "eigenvalues(A)", "eigenvectors(A)",
                "svd(A)", "qr(A)", "lu(A)", "cholesky(A)", "pinv(A)",
                "transpose(A)", "trace(A)", "norm(A)", "shape(A)",
                "multiply(A,B)", "power(A,2)",
                "diff(x^3*sin(x),x)", "integrate(sin(x),x)", "integrate(x^2,x,0,2)",
                "clear()", "symbols()",
            ],
        }

    if command.lower() == "clear()":
        variables.clear()
        return {"type": "clear"}

    if command.lower() == "symbols()":
        return {"type": "variables", "variables": variables}

    assign = re.match(rf"^({IDENTIFIER})\s*=\s*(.+)$", command)
    if assign:
        name, rhs = assign.groups()
        rhs = rhs.strip()
        if rhs.startswith("[["):
            value = parse_matrix_text(rhs)
            variables[name] = [[str(value[r, c]) for c in range(value.cols)] for r in range(value.rows)]
        elif rhs.startswith("["):
            value = parse_vector_text(rhs)
            variables[name] = [str(item) for item in value]
        else:
            variables[name] = str(sp.simplify(parse_expression(rhs)))
        return {"type": "assignment", "name": name, "value": variables[name]}

    shape_match = re.match(rf"^shape\(\s*({IDENTIFIER})\s*\)$", command, re.I)
    if shape_match:
        value = command_matrix(variables[shape_match.group(1)])
        return {"type": "shape", "value": [value.rows, value.cols]}

    calc_match = re.match(r"^(diff|differentiate|derivative|integrate|integral|int)\((.*)\)$", command, re.I)
    if calc_match:
        function_name, inside = calc_match.groups()
        parts = split_top_level(inside)
        if len(parts) < 2:
            raise ValueError("Use diff(expression,variable) or integrate(expression,variable[,lower,upper]).")
        expr = parse_expression(parts[0])
        var = calculus_variable(parts[1])
        if function_name.lower() in {"diff", "differentiate", "derivative"}:
            order = int(parts[2]) if len(parts) > 2 else 1
            result = sp.diff(expr, var, order)
        else:
            if len(parts) == 4:
                result = sp.integrate(expr, (var, parse_expression(parts[2]), parse_expression(parts[3])))
            else:
                result = sp.integrate(expr, var)
        return {"type": "calculus", "operation": function_name.lower(), "result": str(result), "latex": sp.latex(result)}

    unary = re.match(rf"^(det|determinant|rank|inverse|inv|rref|nullspace|eigenvalues|eigvals|eigenvectors|eigvecs|svd|qr|lu|cholesky|pinv|pseudoinverse|transpose|trace|norm)\(\s*({IDENTIFIER})\s*\)$", command, re.I)
    if unary:
        fn, name = unary.groups()
        op = {
            "det": "determinant", "determinant": "determinant",
            "inv": "inverse", "inverse": "inverse",
            "eigvals": "eigenvalues", "eigvecs": "eigenvectors",
            "pinv": "pseudoinverse", "pseudoinverse": "pseudoinverse",
        }.get(fn.lower(), fn.lower())
        return operation_result(MatrixOperationRequest(operation=op, matrix_a=variables[name], tolerance=tolerance))

    solve_match = re.match(rf"^solve\(\s*({IDENTIFIER})\s*,\s*({IDENTIFIER})\s*\)$", command, re.I)
    if solve_match:
        A_name, b_name = solve_match.groups()
        return solve_system(MatrixSolveRequest(matrix_a=variables[A_name], vector_b=variables[b_name], tolerance=tolerance))

    multiply_match = re.match(rf"^multiply\(\s*({IDENTIFIER})\s*,\s*({IDENTIFIER})\s*\)$", command, re.I)
    if multiply_match:
        A_name, B_name = multiply_match.groups()
        return operation_result(MatrixOperationRequest(operation="multiply", matrix_a=variables[A_name], matrix_b=variables[B_name], tolerance=tolerance))

    power_match = re.match(rf"^power\(\s*({IDENTIFIER})\s*,\s*(-?\d+)\s*\)$", command, re.I)
    if power_match:
        A_name, exponent = power_match.groups()
        return operation_result(MatrixOperationRequest(operation="power", matrix_a=variables[A_name], power=int(exponent), tolerance=tolerance))

    raise ValueError("Unknown command. Use help() for examples.")


@router.post("/command")
def command_endpoint(req: CommandRequest):
    try:
        variables = dict(req.variables)
        result = execute_command(req.command, variables, req.tolerance)
        return json_safe({"command": req.command, "result": result, "variables": variables, "error": None})
    except Exception as exc:
        return {"command": req.command, "result": None, "variables": req.variables, "error": str(exc)}