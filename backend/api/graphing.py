from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Tuple
import json
import re
import numpy as np
import sympy as sp
from functools import lru_cache

from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)

router = APIRouter()
TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)

# Keep the graph module self-contained while preventing arbitrary Python names from
# reaching SymPy's expression parser. These are mathematical identifiers only.
ALLOWED_IDENTIFIERS = {
    "x", "y", "z", "r", "theta", "rho", "phi", "t", "pi", "e", "E", "I", "i", "j",
    "sin", "cos", "tan", "sec", "csc", "cot", "asin", "acos", "atan",
    "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "log", "ln", "log10", "exp",
    "sqrt", "abs", "Abs", "arg", "re", "im", "conjugate", "floor", "ceil", "sign",
    "Heaviside", "Min", "Max", "factorial",
}

PARSER_LOCALS = {
    "x": sp.Symbol("x"), "y": sp.Symbol("y"), "z": sp.Symbol("z"),
    "r": sp.Symbol("r"), "theta": sp.Symbol("theta"), "rho": sp.Symbol("rho"),
    "phi": sp.Symbol("phi"), "t": sp.Symbol("t"), "pi": sp.pi, "e": sp.E, "E": sp.E,
    "I": sp.I, "i": sp.I, "j": sp.I, "ln": sp.log, "log10": lambda a: sp.log(a, 10),
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "sec": sp.sec, "csc": sp.csc, "cot": sp.cot,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan, "sinh": sp.sinh, "cosh": sp.cosh,
    "tanh": sp.tanh, "asinh": sp.asinh, "acosh": sp.acosh, "atanh": sp.atanh,
    "log": sp.log, "exp": sp.exp, "sqrt": sp.sqrt, "abs": sp.Abs, "Abs": sp.Abs,
    "arg": sp.arg, "re": sp.re, "im": sp.im, "conjugate": sp.conjugate,
    "floor": sp.floor, "ceil": sp.ceiling, "sign": sp.sign, "Heaviside": sp.Heaviside,
    "Min": sp.Min, "Max": sp.Max, "factorial": sp.factorial,
}
MAX_EXPRESSIONS = 20
MAX_EXPRESSION_LENGTH = 500
MAX_COMMANDS = 8
MAX_COMMAND_LENGTH = 300
MAX_3D_GRID_POINTS = 125000

_TRIG_FUNC_NAMES = (
    "asinh", "acosh", "atanh", "asin", "acos", "atan",
    "sinh", "cosh", "tanh", "sin", "cos", "tan", "sec", "csc", "cot",
)
_TRIG_POWER_RE = re.compile(
    r"\b(" + "|".join(_TRIG_FUNC_NAMES) + r")\*\*(\d+)\s*(?:\(([a-zA-Z_][a-zA-Z0-9_]*)\)|([a-zA-Z_][a-zA-Z0-9_]*))"
)
_LATEX_FRAC_RE = re.compile(r"\\frac\{([^{}]*)\}\{([^{}]*)\}")
_BRACE_GROUP_RE = re.compile(r"\{([^{}]*)\}")


def _normalize_math_text(text: str) -> str:
    """Accept common LaTeX-ish shorthand (as typed or pasted) and rewrite it into
    plain SymPy-parseable syntax *before* the strict validator runs. This keeps the
    validator itself strict (still blocks genuine injection characters) while no
    longer treating ordinary mathematical grouping/notation as an attack.

    Handles, e.g.:
      r = 4(cos^4theta + sin^4theta)^{-1/4}
        -> r = 4((cos(theta))**4 + (sin(theta))**4)**(-1/4)
    """
    s = text.strip()
    s = re.sub(r"\\left|\\right", "", s)
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    # \frac{a}{b} -> (a)/(b), repeated for nested fractions.
    prev = None
    while prev != s:
        prev = s
        s = _LATEX_FRAC_RE.sub(r"(\1)/(\2)", s)
    s = s.replace("\\", "")
    # Caret to Python power.
    s = s.replace("^", "**")
    # LaTeX-style curly-brace grouping (commonly left over in exponents, e.g.
    # `**{-1/4}`) is just parenthesization — convert innermost-first.
    prev = None
    while prev != s:
        prev = s
        s = _BRACE_GROUP_RE.sub(r"(\1)", s)
    # Trig-power shorthand: cos^4theta / cos**4theta / cos**4(theta) all mean
    # (cos(theta))**4, not cos * 4 * theta.
    def _trig_power_repl(m: "re.Match") -> str:
        func, power = m.group(1), m.group(2)
        arg = m.group(3) if m.group(3) is not None else m.group(4)
        return f"({func}({arg}))**{power}"

    prev = None
    while prev != s:
        prev = s
        s = _TRIG_POWER_RE.sub(_trig_power_repl, s)
    return s


def _validate_math_text(text: str) -> str:
    clean = _normalize_math_text(text)
    if len(clean) > MAX_EXPRESSION_LENGTH:
        raise ValueError(f"Expression is too long (maximum {MAX_EXPRESSION_LENGTH} characters).")
    # Braces/brackets should already be gone after normalization; if any remain,
    # the input had unbalanced/unsupported grouping rather than a hidden attack.
    if "__" in clean or any(ch in clean for ch in ["'", '"', ";", "[", "]", "{", "}"]):
        raise ValueError("Expression contains unsupported syntax.")
    names = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", clean)
    unknown = sorted({name for name in names if name not in ALLOWED_IDENTIFIERS})
    if unknown:
        raise ValueError("Unsupported identifier(s): " + ", ".join(unknown))
    return clean


QUALITY_PRESETS = {
    "low": {"line_1d": 420, "grid_2d": 150, "grid_3d": 24, "polar_theta": 360, "polar_pts": 900, "polar_radial": 160, "complex": 150},
    "medium": {"line_1d": 700, "grid_2d": 260, "grid_3d": 34, "polar_theta": 540, "polar_pts": 1400, "polar_radial": 220, "complex": 210},
    "high": {"line_1d": 1200, "grid_2d": 420, "grid_3d": 44, "polar_theta": 720, "polar_pts": 2400, "polar_radial": 280, "complex": 300},
}

DEFAULT_AXIS = {
    "x_min": -20.0, "x_max": 20.0,
    "y_min": -20.0, "y_max": 20.0,
    "z_min": -20.0, "z_max": 20.0,
    "theta_min": 0.0, "theta_max": 360.0,
    "r_max": 20.0,
    "phi_min": 0.0, "phi_max": 180.0,
    "re_min": -8.0, "re_max": 8.0,
    "im_min": -8.0, "im_max": 8.0,
}


class GraphRequest(BaseModel):
    expressions: List[str]
    expression_names: List[str] = Field(default_factory=list)
    t_val: float = 0.0
    commands: List[str] = Field(default_factory=list)
    include_plots: bool = True
    quality: str = "medium"
    coordinate_system: str = "cartesian2d"
    complex_view: str = "magnitude"
    angle_unit: str = "degrees"
    axis: Dict[str, Any] = Field(default_factory=dict)


@lru_cache(maxsize=256)
def _cached_parse(clean_expr: str):
    clean_expr = _validate_math_text(clean_expr)
    is_implicit = "=" in clean_expr
    if is_implicit:
        lhs_str, rhs_str = clean_expr.split("=", 1)
        if not lhs_str.strip() or not rhs_str.strip() or "=" in rhs_str:
            raise ValueError("Use a single equation sign, e.g. y=x^2 or x^2+y^2=25.")
        lhs_expr = parse_expr(lhs_str, local_dict=PARSER_LOCALS, transformations=TRANSFORMATIONS)
        rhs_expr = parse_expr(rhs_str, local_dict=PARSER_LOCALS, transformations=TRANSFORMATIONS)
        return lhs_expr - rhs_expr, True, lhs_str.strip(), rhs_expr
    base_eq = parse_expr(clean_expr, local_dict=PARSER_LOCALS, transformations=TRANSFORMATIONS)
    return base_eq, False, "", None


def _axis(req: GraphRequest, key: str, fallback: float) -> float:
    try:
        value = float(req.axis.get(key, fallback))
        return value if np.isfinite(value) else fallback
    except Exception:
        return fallback


def _range(req: GraphRequest, lo_key: str, hi_key: str, fallback=(0.0, 1.0)):
    lo = _axis(req, lo_key, fallback[0])
    hi = _axis(req, hi_key, fallback[1])
    if hi <= lo:
        lo, hi = fallback
    return lo, hi



def _json_safe(value: Any) -> Any:
    """Recursively convert NumPy/SymPy values and non-finite floats into JSON-safe data."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        out = float(value)
        return out if np.isfinite(out) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, complex):
        if abs(value.imag) < 1e-10 and np.isfinite(value.real):
            return float(value.real)
        return None
    return value


def _split_command(command: str) -> Tuple[str, List[str]]:
    """Accept both function syntax and legacy space-separated syntax."""
    text = command.strip()
    match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)$", text)
    if match:
        name = match.group(1).strip().lower()
        raw_args = match.group(2).strip()
        if not raw_args:
            return name, []
        return name, [part.strip().strip('\"\'`') for part in raw_args.split(',') if part.strip()]

    parts = text.split()
    if not parts:
        return '', []
    return parts[0].lower(), parts[1:]


def _resolve_expression_ref(ref: str, expression_names: List[str], count: int) -> Optional[int]:
    """Resolve 1-based numbers, EqN, or exact expression names (case-insensitive)."""
    token = re.sub(r"\s+", " ", ref.strip()).strip().casefold()
    if not token:
        return None

    if token.isdigit():
        index = int(token) - 1
        return index if 0 <= index < count else None

    eq_match = re.fullmatch(r"eq\s*(\d+)", token)
    if eq_match:
        index = int(eq_match.group(1)) - 1
        return index if 0 <= index < count else None

    for index, name in enumerate(expression_names[:count]):
        normalized = re.sub(r"\s+", " ", str(name).strip()).casefold()
        if normalized and normalized == token:
            return index

    # Be forgiving when a frontend only knows the default label but no name was sent.
    if token.startswith('equation '):
        suffix = token.removeprefix('equation ').strip()
        if suffix.isdigit():
            index = int(suffix) - 1
            return index if 0 <= index < count else None
    return None

def _safe_array(values: Any) -> np.ndarray:
    arr = np.asarray(values)
    if np.iscomplexobj(arr):
        arr = np.where(np.abs(arr.imag) < 1e-8, arr.real, np.nan)
    try:
        arr = arr.astype(float)
    except Exception:
        arr = np.vectorize(lambda v: _as_real_number(v))(arr)
    return np.where(np.isfinite(arr), arr, np.nan)


def _complex_array(values: Any) -> np.ndarray:
    arr = np.asarray(values)
    try:
        return arr.astype(complex)
    except Exception:
        return np.vectorize(lambda v: complex(v))(arr)


def _as_real_number(value: Any) -> Optional[float]:
    try:
        c = complex(sp.N(value))
        if abs(c.imag) > 1e-8:
            return None
        out = float(c.real)
        return out if np.isfinite(out) else None
    except Exception:
        return None


def _unique_points(points: List[Dict[str, Any]], digits: int = 7):
    # Numerical fallbacks can land on the same root several grid cells apart.
    # Use a small geometric tolerance in addition to rounded-coordinate matching.
    out = []
    tol2 = 1e-7
    for p in points:
        try:
            x = float(p.get("x", 0)); y = float(p.get("y", 0)); z = float(p.get("z", 0))
            if not all(np.isfinite(v) for v in (x, y, z)):
                continue
        except Exception:
            continue
        duplicate = False
        for q in out:
            dx=x-float(q.get("x",0)); dy=y-float(q.get("y",0)); dz=z-float(q.get("z",0))
            scale = max(1.0, abs(x), abs(y), abs(z), abs(float(q.get("x",0))), abs(float(q.get("y",0))), abs(float(q.get("z",0))))
            if dx*dx + dy*dy + dz*dz <= tol2 * scale * scale:
                duplicate = True
                break
        if not duplicate:
            out.append(p)
    return out


def _is_3d_symbols(syms: set) -> bool:
    return bool(syms & {"z", "phi", "rho"}) or ("r" in syms and "z" in syms)


def _curve_equation(base_eq: sp.Expr, is_implicit: bool) -> sp.Expr:
    if is_implicit:
        return base_eq
    x, y = sp.symbols("x y")
    return base_eq if y in base_eq.free_symbols else y - base_eq


def _numeric_roots_1d(expr: sp.Expr, var: sp.Symbol, lo: float = -500.0, hi: float = 500.0, samples: int = 4000):
    """Bounded, vectorized (NumPy) real-root scan for a single-variable expression.
    Used instead of sp.solve for transcendental/composite expressions, where symbolic
    solving is either very slow or simply gives up (ConditionSet/no closed form).
    This always terminates in well under a millisecond regardless of expression
    complexity, at the cost of only finding roots inside [lo, hi] — which comfortably
    covers any axis range someone would actually be looking at."""
    try:
        fn = sp.lambdify(var, expr, modules=["numpy"])
        grid = np.linspace(lo, hi, samples)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            vals = _safe_array(np.broadcast_to(fn(grid), grid.shape))
    except Exception:
        return []
    roots: List[float] = []
    finite = np.isfinite(vals)
    with np.errstate(over="ignore", invalid="ignore"):
        for j in range(samples - 1):
            if not (finite[j] and finite[j + 1]):
                continue
            a, b = vals[j], vals[j + 1]
            if abs(a) < 1e-9:
                roots.append(float(grid[j]))
            elif a * b < 0:
                roots.append(float(grid[j] - a * (grid[j + 1] - grid[j]) / (b - a)))
    return sorted({round(r, 8) for r in roots})


@lru_cache(maxsize=2048)
def _real_solutions(expr: sp.Expr, var: sp.Symbol, limit: int = 16):
    """Real roots of expr==0 in var. Polynomials get exact symbolic solving (fast
    and robust via SymPy's polynomial solvers); anything transcendental (sin/cos/
    exp/log/composite implicit forms) skips sp.solve entirely and goes straight to
    the bounded numeric scanner above — sp.solve on transcendental systems is the
    single biggest source of slowdown/hangs in this module, and a numeric scan is
    both faster and, for anything a person would actually plot, just as accurate.
    Cached: this is called on every visible curve on every plot request via
    _axis_points, so repeat requests for an unchanged curve are effectively free."""
    try:
        is_poly = expr.is_polynomial(var)
    except Exception:
        is_poly = False

    vals: List[float] = []
    if is_poly:
        try:
            sols = sp.solve(expr, var, dict=False)
        except Exception:
            sols = []
        if isinstance(sols, (sp.Expr, sp.Number)):
            sols = [sols]
        for root in sols or []:
            value = _as_real_number(root)
            if value is not None and abs(value) < 1e6:
                vals.append(value)
    else:
        vals = _numeric_roots_1d(expr, var)

    vals.sort()
    return list(dict.fromkeys(round(v, 10) for v in vals))[:limit]


@lru_cache(maxsize=1024)
def _axis_points(base_eq: sp.Expr, is_implicit: bool, syms: frozenset, is_3d: bool):
    pts: List[Dict[str, Any]] = []
    x, y, z, r, theta = sp.symbols("x y z r theta")
    try:
        if not is_3d and (syms & {"r", "theta"}):
            expr = base_eq
            if is_implicit and "r" in syms and "theta" in syms:
                fn = sp.lambdify(theta, expr.subs(r, sp.Symbol("r")), modules=["numpy"])
                del fn
            # For r=f(theta), evaluate the curve on the four cardinal rays.
            if not is_implicit:
                for ang, label in [(0, "x+"), (sp.pi, "x-"), (sp.pi/2, "y+"), (3*sp.pi/2, "y-")]:
                    try:
                        val = _as_real_number(sp.solve(sp.Eq(expr, r), r)[0].subs(theta, ang)) if r in expr.free_symbols else None
                    except Exception:
                        val = None
                    if val is None:
                        try:
                            val = _as_real_number(expr.subs(theta, ang))
                        except Exception:
                            val = None
                    if val is None:
                        continue
                    if label == "x+": point = (val, 0.0); axis_name = "x"
                    elif label == "x-": point = (-abs(val), 0.0); axis_name = "x"
                    elif label == "y+": point = (0.0, abs(val)); axis_name = "y"
                    else: point = (0.0, -abs(val)); axis_name = "y"
                    pts.append({"kind": "axis", "axis": axis_name, "x": point[0], "y": point[1], "z": 0.0, "label": f"{label}-axis ({point[0]:.4g}, {point[1]:.4g})"})
            else:
                # Convert any implicit polar equation to Cartesian and reuse the Cartesian intercept logic.
                converted = sp.simplify(expr.subs({r: sp.sqrt(x**2+y**2), theta: sp.atan2(y, x)}))
                for xv in _real_solutions(converted.subs(y, 0), x):
                    pts.append({"kind": "axis", "axis": "x", "x": xv, "y": 0.0, "z": 0.0, "label": f"x-int ({xv:.4g}, 0)"})
                for yv in _real_solutions(converted.subs(x, 0), y):
                    pts.append({"kind": "axis", "axis": "y", "x": 0.0, "y": yv, "z": 0.0, "label": f"y-int (0, {yv:.4g})"})
            return _unique_points(pts)

        if not is_3d and syms <= {"x", "y"}:
            eq = _curve_equation(base_eq, is_implicit)
            for xv in _real_solutions(sp.simplify(eq.subs(y, 0)), x):
                pts.append({"kind": "axis", "axis": "x", "x": xv, "y": 0.0, "z": 0.0, "label": f"x-int ({xv:.4g}, 0)"})
            if not is_implicit:
                yv = _as_real_number(base_eq.subs(x, 0))
                if yv is not None:
                    pts.append({"kind": "axis", "axis": "y", "x": 0.0, "y": yv, "z": 0.0, "label": f"y-int (0, {yv:.4g})"})
            else:
                for yv in _real_solutions(sp.simplify(eq.subs(x, 0)), y):
                    pts.append({"kind": "axis", "axis": "y", "x": 0.0, "y": yv, "z": 0.0, "label": f"y-int (0, {yv:.4g})"})
        elif is_3d and syms <= {"x", "y", "z"} and is_implicit:
            for xv in _real_solutions(base_eq.subs({y: 0, z: 0}), x, 8):
                pts.append({"kind": "axis", "axis": "x", "x": xv, "y": 0.0, "z": 0.0, "label": f"X ({xv:.4g}, 0, 0)"})
            for yv in _real_solutions(base_eq.subs({x: 0, z: 0}), y, 8):
                pts.append({"kind": "axis", "axis": "y", "x": 0.0, "y": yv, "z": 0.0, "label": f"Y (0, {yv:.4g}, 0)"})
            for zv in _real_solutions(base_eq.subs({x: 0, y: 0}), z, 8):
                pts.append({"kind": "axis", "axis": "z", "x": 0.0, "y": 0.0, "z": zv, "label": f"Z (0, 0, {zv:.4g})"})
    except Exception:
        pass
    return _unique_points(pts)

@lru_cache(maxsize=512)
def _intersection_points(eq1: sp.Expr, eq2: sp.Expr, limit: int = 32, x_bounds=(-20.0, 20.0), y_bounds=(-20.0, 20.0)):
    """Hybrid exact/numerical 2D intersections. Symbolic solutions are preferred;
    explicit y=f(x)/x=f(y) curves get a bounded numerical fallback."""
    x, y = sp.symbols("x y")
    out: List[Dict[str, Any]] = []
    try:
        sols = sp.solve([eq1, eq2], [x, y], dict=True)
    except Exception:
        sols = []
    for sol in sols or []:
        if x not in sol or y not in sol:
            continue
        xv, yv = _as_real_number(sol[x]), _as_real_number(sol[y])
        if xv is None or yv is None or abs(xv) > 1e6 or abs(yv) > 1e6:
            continue
        out.append({"kind": "intersection", "x": xv, "y": yv, "z": 0.0, "label": f"∩ ({xv:.4g}, {yv:.4g})"})
        if len(out) >= limit:
            return _unique_points(out)[:limit]

    # Numerical fallback for explicit curves. This intentionally stays NumPy-only
    # so the module does not acquire a SciPy dependency.
    v1, f1 = _explicit_curve_function(eq1)
    v2, f2 = _explicit_curve_function(eq2)
    try:
        if v1 == x and v2 == x:
            lo, hi = x_bounds; grid = np.linspace(lo, hi, 1800)
            a = np.broadcast_to(_safe_array(sp.lambdify(x, f1, modules=["numpy"])(grid)), grid.shape)
            b = np.broadcast_to(_safe_array(sp.lambdify(x, f2, modules=["numpy"])(grid)), grid.shape)
            d = a - b
            for j in range(len(grid) - 1):
                if not (np.isfinite(d[j]) and np.isfinite(d[j + 1])): continue
                if abs(d[j]) < 1e-8:
                    root = grid[j]
                elif d[j] * d[j + 1] < 0:
                    root = grid[j] - d[j] * (grid[j + 1] - grid[j]) / (d[j + 1] - d[j])
                else:
                    continue
                yv = _as_real_number(f1.subs(x, float(root)))
                if yv is not None:
                    out.append({"kind": "intersection", "x": float(root), "y": yv, "z": 0.0, "label": f"∩ ({root:.4g}, {yv:.4g})"})
        elif v1 == y and v2 == y:
            lo, hi = y_bounds; grid = np.linspace(lo, hi, 1800)
            a = np.broadcast_to(_safe_array(sp.lambdify(y, f1, modules=["numpy"])(grid)), grid.shape)
            b = np.broadcast_to(_safe_array(sp.lambdify(y, f2, modules=["numpy"])(grid)), grid.shape)
            d = a - b
            for j in range(len(grid) - 1):
                if not (np.isfinite(d[j]) and np.isfinite(d[j + 1])): continue
                if abs(d[j]) < 1e-8: root = grid[j]
                elif d[j] * d[j + 1] < 0: root = grid[j] - d[j] * (grid[j + 1] - grid[j]) / (d[j + 1] - d[j])
                else: continue
                xv = _as_real_number(f1.subs(y, float(root)))
                if xv is not None:
                    out.append({"kind": "intersection", "x": xv, "y": float(root), "z": 0.0, "label": f"∩ ({xv:.4g}, {root:.4g})"})
    except Exception:
        pass

    # Do not add sampled near-roots when exact/numerical curve solving already
    # supplied a complete pair of intersections.
    exact_unique = _unique_points(out)
    if len(exact_unique) >= 2:
        return exact_unique[:limit]

    # Final fallback: sample both implicit level sets on a bounded grid. A valid
    # intersection cell must be crossed by both zero-contours; this avoids false
    # positives from simply choosing two individually small residuals. Vectorized
    # with NumPy (no Python-level double loop) — the earlier per-cell Python loop
    # over a 220x220 grid was one of the slower fallback paths in this module.
    try:
        gx = np.linspace(float(x_bounds[0]), float(x_bounds[1]), 220)
        gy = np.linspace(float(y_bounds[0]), float(y_bounds[1]), 220)
        X, Y = np.meshgrid(gx, gy)
        f1 = sp.lambdify((x, y), eq1, modules=["numpy"])
        f2 = sp.lambdify((x, y), eq2, modules=["numpy"])
        A = _safe_array(f1(X, Y)); B = _safe_array(f2(X, Y))

        A00, A01, A10, A11 = A[:-1, :-1], A[:-1, 1:], A[1:, :-1], A[1:, 1:]
        B00, B01, B10, B11 = B[:-1, :-1], B[:-1, 1:], B[1:, :-1], B[1:, 1:]
        finite = (
            np.isfinite(A00) & np.isfinite(A01) & np.isfinite(A10) & np.isfinite(A11)
            & np.isfinite(B00) & np.isfinite(B01) & np.isfinite(B10) & np.isfinite(B11)
        )
        a_min = np.minimum(np.minimum(A00, A01), np.minimum(A10, A11))
        a_max = np.maximum(np.maximum(A00, A01), np.maximum(A10, A11))
        b_min = np.minimum(np.minimum(B00, B01), np.minimum(B10, B11))
        b_max = np.maximum(np.maximum(B00, B01), np.maximum(B10, B11))
        mask = finite & (a_min <= 0) & (0 <= a_max) & (b_min <= 0) & (0 <= b_max)

        cell_cx = 0.5 * (gx[:-1] + gx[1:])
        cell_cy = 0.5 * (gy[:-1] + gy[1:])
        rows, cols = np.where(mask)
        for i, j in zip(rows.tolist(), cols.tolist()):
            xv, yv = float(cell_cx[j]), float(cell_cy[i])
            if all((xv - p["x"]) ** 2 + (yv - p["y"]) ** 2 > 0.0025 for p in out):
                out.append({"kind": "intersection", "x": xv, "y": yv, "z": 0.0, "label": f"∩ ({xv:.4g}, {yv:.4g})"})
                if len(out) >= limit:
                    break
    except Exception:
        pass
    return _unique_points(out)[:limit]


def _format_point_latex(p: Dict[str, Any]) -> str:
    return f"({sp.latex(sp.N(p['x'], 7))},\\ {sp.latex(sp.N(p['y'], 7))})"


def _line_form(expr: sp.Expr):
    """Return A,B,C for a linear Cartesian equation Ax+By+C=0, else None."""
    x, y = sp.symbols("x y")
    try:
        poly = sp.Poly(sp.expand(expr), x, y)
        if poly.total_degree() != 1:
            return None
        return (float(poly.coeff_monomial(x)), float(poly.coeff_monomial(y)), float(poly.coeff_monomial(1)))
    except Exception:
        return None

def _explicit_curve_function(expr: sp.Expr):
    """Return y=f(x) or x=f(y) when a curve can be solved explicitly."""
    x, y = sp.symbols("x y")
    try:
        ys = sp.solve(expr, y, dict=False)
        ys = [v for v in (ys if isinstance(ys, list) else [ys]) if x in getattr(v, "free_symbols", set()) and y not in getattr(v, "free_symbols", set())]
        if len(ys) == 1:
            return x, sp.simplify(ys[0])
    except Exception:
        pass
    try:
        xs = sp.solve(expr, x, dict=False)
        xs = [v for v in (xs if isinstance(xs, list) else [xs]) if y in getattr(v, "free_symbols", set()) and x not in getattr(v, "free_symbols", set())]
        if len(xs) == 1:
            return y, sp.simplify(xs[0])
    except Exception:
        pass
    return None, None

def _circle_form(expr: sp.Expr):
    x, y = sp.symbols("x y")
    try:
        poly=sp.Poly(sp.expand(expr),x,y)
        if poly.total_degree()!=2:
            return None
        A=sp.Rational(poly.coeff_monomial(x**2)); B=sp.Rational(poly.coeff_monomial(x*y)); C=sp.Rational(poly.coeff_monomial(y**2))
        if A != 1 or B != 0 or C != 1:
            if A == C and A != 0 and B == 0:
                poly_expr=sp.expand(expr/A); poly=sp.Poly(poly_expr,x,y)
            else:
                return None
        D=float(poly.coeff_monomial(x)); E=float(poly.coeff_monomial(y)); F=float(poly.coeff_monomial(1))
        cx=-D/2.0; cy=-E/2.0; r2=cx*cx+cy*cy-F
        if r2 > 0 and np.isfinite(r2):
            return cx,cy,float(np.sqrt(r2))
    except Exception:
        pass
    return None


@lru_cache(maxsize=512)
def _curve_distance(eq1: sp.Expr, eq2: sp.Expr, x_bounds=(-20.0, 20.0), y_bounds=(-20.0, 20.0)):
    """Geometry-aware distance with exact lines/circles and bounded numerical fallback."""
    # If the boundaries intersect, the geometric distance is exactly zero.
    try:
        if _intersection_points(eq1, eq2, 2, x_bounds, y_bounds):
            return 0.0, "intersecting"
    except Exception:
        pass

    line1, line2 = _line_form(eq1), _line_form(eq2)
    if line1 and line2:
        A1, B1, C1 = line1; A2, B2, C2 = line2
        cross = A1 * B2 - A2 * B1
        if abs(cross) < 1e-10:
            norm1 = (A1 * A1 + B1 * B1) ** 0.5
            if norm1 < 1e-12: return None, "Invalid line"
            # Normalize second line to the first line orientation before comparing C.
            scale = (A1 * A2 + B1 * B2) / max(A1 * A1 + B1 * B1, 1e-30)
            C2n = C2 / scale if abs(scale) > 1e-12 else C2
            return abs(C2n - C1) / norm1, "exact"
        return 0.0, "intersecting"

    c1, c2 = _circle_form(eq1), _circle_form(eq2)
    if c1 and c2:
        cx1, cy1, r1 = c1; cx2, cy2, r2 = c2
        center_d = float(np.hypot(cx1-cx2, cy1-cy2))
        if center_d >= r1 + r2:
            return center_d - r1 - r2, "exact circles"
        if center_d <= abs(r1-r2):
            return abs(r1-r2) - center_d, "exact nested circles"
        return 0.0, "intersecting circles"

    x, y = sp.symbols("x y")
    try:
        v1, f1 = _explicit_curve_function(eq1)
        v2, f2 = _explicit_curve_function(eq2)
        if v1 == x and v2 == x:
            lo, hi = x_bounds; grid = np.linspace(lo, hi, 1200)
            a = np.broadcast_to(_safe_array(sp.lambdify(x, f1, modules=["numpy"])(grid)), grid.shape)
            b = np.broadcast_to(_safe_array(sp.lambdify(x, f2, modules=["numpy"])(grid)), grid.shape)
            gap = np.abs(a - b); finite = np.isfinite(gap)
            if finite.any(): return float(np.nanmin(gap[finite])), "sampled vertical"
    except Exception:
        pass

    try:
        # General implicit/parametric fallback. First check whether the sampled
        # boundaries intersect; otherwise estimate the closest pair of boundary points.
        pts = _intersection_points(eq1, eq2, 8, x_bounds, y_bounds)
        if pts:
            return 0.0, "intersecting"
        gx = np.linspace(float(x_bounds[0]), float(x_bounds[1]), 150)
        gy = np.linspace(float(y_bounds[0]), float(y_bounds[1]), 150)
        X, Y = np.meshgrid(gx, gy)
        f1 = sp.lambdify((x,y), eq1, modules=["numpy"]); f2 = sp.lambdify((x,y), eq2, modules=["numpy"])
        A = _safe_array(f1(X,Y)); B = _safe_array(f2(X,Y))
        def boundary_points(V):
            finite=np.isfinite(V)
            if not finite.any(): return np.empty((0,2))
            scale=max(float(np.nanpercentile(np.abs(V[finite]), 85)),1.0)
            score=np.abs(V)/scale; score[~finite]=np.inf
            flat=score.ravel(); k=min(240,flat.size); ids=np.argpartition(flat,k-1)[:k]
            rr,cc=np.unravel_index(ids,score.shape)
            keep = score[rr,cc] <= 0.025
            return np.column_stack((X[rr[keep],cc[keep]],Y[rr[keep],cc[keep]]))
        P,Q=boundary_points(A),boundary_points(B)
        if len(P) and len(Q):
            best=float("inf")
            for p0 in P:
                d=((Q-p0)**2).sum(axis=1)
                if d.size: best=min(best,float(np.min(d)))
            if np.isfinite(best): return float(np.sqrt(best)), "sampled boundary"
    except Exception:
        pass
    return None, "unsupported"

def _fit_locus(points):
    if len(points) < 2:
        return {"success": False, "message": "At least two intersection points are required for a fitted locus."}
    xs = np.array([p["x"] for p in points], dtype=float)
    ys = np.array([p["y"] for p in points], dtype=float)
    if np.ptp(xs) > 1e-9:
        degree = min(3, len(points) - 1, max(1, len(np.unique(np.round(xs, 9))) - 1))
        coeffs = np.polyfit(xs, ys, degree)
        grid = np.linspace(xs.min(), xs.max(), 260)
        y_fit = np.polyval(coeffs, grid)
        x = sp.Symbol("x")
        expr = sum(float(c) * x ** (degree - i) for i, c in enumerate(coeffs))
        return {"success": True, "orientation": "y=f(x)", "equation": sp.expand(expr), "x": grid.tolist(), "y": y_fit.tolist()}
    if np.ptp(ys) > 1e-9:
        degree = min(3, len(points) - 1, max(1, len(np.unique(np.round(ys, 9))) - 1))
        coeffs = np.polyfit(ys, xs, degree)
        grid = np.linspace(ys.min(), ys.max(), 260)
        x_fit = np.polyval(coeffs, grid)
        y = sp.Symbol("y")
        expr = sum(float(c) * y ** (degree - i) for i, c in enumerate(coeffs))
        return {"success": True, "orientation": "x=f(y)", "equation": sp.expand(expr), "x": x_fit.tolist(), "y": grid.tolist()}
    return {"success": False, "message": "The intersection points do not provide a usable 2D locus fit."}



def _angle_arrays(req: GraphRequest, count: int):
    theta_min = _axis(req, "theta_min", 0.0)
    theta_max = _axis(req, "theta_max", 360.0)
    theta_display = np.linspace(theta_min, theta_max, count)
    theta_rad = np.deg2rad(theta_display) if req.angle_unit == "degrees" else theta_display
    return theta_display, theta_rad


def _polar_substitution():
    x, y, r, theta = sp.symbols("x y r theta")
    return {x: r * sp.cos(theta), y: r * sp.sin(theta)}


def _cartesian_substitution():
    x, y, r, theta = sp.symbols("x y r theta")
    return {r: sp.sqrt(x**2 + y**2), theta: sp.atan2(y, x)}


def _cylindrical_to_cartesian_substitution():
    x, y, z, r, theta = sp.symbols("x y z r theta")
    return {r: sp.sqrt(x**2 + y**2), theta: sp.atan2(y, x)}


def _spherical_to_cartesian_substitution():
    x, y, z, rho, theta, phi = sp.symbols("x y z rho theta phi")
    return {
        rho: sp.sqrt(x**2 + y**2 + z**2),
        theta: sp.atan2(y, x),
        phi: sp.acos(sp.Max(-1, sp.Min(1, z / sp.sqrt(x**2 + y**2 + z**2)))),
    }


def _substitute_display_expression(expr: sp.Expr, target: str):
    """Convert a source expression to a target coordinate expression where practical."""
    x, y, z, r, rho, theta, phi = sp.symbols("x y z r rho theta phi")
    if target == "cartesian2d":
        return sp.simplify(expr.subs({r: sp.sqrt(x**2 + y**2), theta: sp.atan2(y, x)}))
    if target == "polar2d":
        return sp.simplify(expr.subs({x: r * sp.cos(theta), y: r * sp.sin(theta)}))
    if target == "cartesian3d":
        return sp.simplify(expr.subs({
            r: sp.sqrt(x**2 + y**2),
            rho: sp.sqrt(x**2 + y**2 + z**2),
            theta: sp.atan2(y, x),
            phi: sp.acos(sp.Max(-1, sp.Min(1, z / sp.sqrt(x**2 + y**2 + z**2)))),
        }))
    if target == "cylindrical3d":
        return sp.simplify(expr.subs({x: r * sp.cos(theta), y: r * sp.sin(theta)}))
    if target == "spherical3d":
        return sp.simplify(expr.subs({x: rho * sp.sin(phi) * sp.cos(theta), y: rho * sp.sin(phi) * sp.sin(theta), z: rho * sp.cos(phi)}))
    return expr


def _polar_explicit(rhs: sp.Expr, req: GraphRequest, preset, t_sym):
    theta_display, theta_rad = _angle_arrays(req, preset["polar_pts"])
    r_max = abs(_axis(req, "r_max", 20.0)) or 20.0
    fn = sp.lambdify("theta", rhs.subs(t_sym, req.t_val), modules=["numpy"])
    raw = _safe_array(fn(theta_rad))
    negative = np.isfinite(raw) & (raw < 0)
    r = np.abs(raw)
    theta_out = theta_display.copy()
    theta_period = 360.0 if req.angle_unit == "degrees" else 2 * np.pi
    theta_out[negative] = (theta_out[negative] + theta_period / 2) % theta_period
    r = np.where(r <= r_max * 4, r, np.nan)
    return {"theta": theta_out.tolist(), "r": r.tolist()}


def _polar_implicit(eq: sp.Expr, req: GraphRequest, preset):
    """Plot F(x,y)=0 in polar coordinates using direct radial roots first."""
    theta_display, theta_rad = _angle_arrays(req, preset["polar_theta"])
    r_max = abs(_axis(req, "r_max", 20.0)) or 20.0
    r_sym, th_sym = sp.symbols("r theta")
    x_sym, y_sym = sp.symbols("x y")
    transformed = sp.simplify(eq.subs({x_sym: r_sym * sp.cos(th_sym), y_sym: r_sym * sp.sin(th_sym)}))

    symbolic_roots = []
    try:
        raw_roots = sp.solve(sp.Eq(transformed, 0), r_sym, dict=False)
        if not isinstance(raw_roots, list): raw_roots = [raw_roots]
        symbolic_roots = [sp.simplify(root) for root in raw_roots if r_sym not in getattr(root, "free_symbols", set())]
    except Exception:
        symbolic_roots = []

    if symbolic_roots:
        root_fns = []
        for root in symbolic_roots:
            try: root_fns.append(sp.lambdify(th_sym, root, modules=["numpy"]))
            except Exception: pass
        theta_out, r_out = [], []
        for th_plot, th in zip(theta_display, theta_rad):
            vals=[]
            for fn in root_fns:
                try:
                    value=complex(fn(th))
                    if abs(value.imag)<1e-8 and np.isfinite(value.real):
                        rr=float(value.real)
                        if rr < -1e-9: rr=-rr
                        if 0 <= rr <= r_max*1.000001: vals.append(rr)
                except Exception: pass
            vals=sorted({round(v,10) for v in vals})
            for rr in vals:
                theta_out.append(float(th_plot)); r_out.append(float(rr))
            if len(vals)>1:
                theta_out.append(np.nan); r_out.append(np.nan)
        if r_out:
            return {"theta":theta_out,"r":r_out}

    # Numerical fallback for equations SymPy cannot solve radially.
    rgrid=np.linspace(0,r_max,preset["polar_radial"])
    fn=sp.lambdify((r_sym,th_sym),transformed,modules=["numpy"])
    theta_out, r_out=[],[]
    for th_plot, th in zip(theta_display,theta_rad):
        try: vals=_safe_array(fn(rgrid,th))
        except Exception: continue
        roots=[]
        for j in range(len(rgrid)-1):
            a,b=vals[j],vals[j+1]
            if not (np.isfinite(a) and np.isfinite(b)): continue
            if abs(a)<1e-7: roots.append(float(rgrid[j]))
            elif a*b<0:
                frac=abs(a)/max(abs(a)+abs(b),1e-15)
                roots.append(float(rgrid[j]+frac*(rgrid[j+1]-rgrid[j])))
        roots=sorted({round(v,8) for v in roots})
        for rr in roots:
            theta_out.append(float(th_plot)); r_out.append(float(rr))
        if len(roots)>1: theta_out.append(np.nan); r_out.append(np.nan)
    return {"theta":theta_out,"r":r_out}


def _polar_cartesian_parametric(rhs: sp.Expr, req: GraphRequest, preset, t_sym):
    vals = _polar_explicit(rhs, req, preset, t_sym)
    th_display = np.asarray(vals["theta"], dtype=float)
    rr = np.asarray(vals["r"], dtype=float)
    th_rad = np.deg2rad(th_display) if req.angle_unit == "degrees" else th_display
    return {
        "x": (rr * np.cos(th_rad)).tolist(),
        "y": (rr * np.sin(th_rad)).tolist(),
        "type": "line2d",
    }


def _cartesian_from_polar_contour(eq: sp.Expr, req: GraphRequest, preset):
    x0, x1 = _range(req, "x_min", "x_max", (-20, 20)); y0, y1 = _range(req, "y_min", "y_max", (-20, 20))
    n = preset["grid_2d"]
    X, Y = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))
    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    if req.angle_unit == "degrees":
        theta = np.rad2deg(theta)
    x_sym, y_sym, r_sym, th_sym = sp.symbols("x y r theta")
    transformed = sp.simplify(eq.subs({x_sym: r_sym * sp.cos(th_sym), y_sym: r_sym * sp.sin(th_sym)}))
    # If the source really contains polar symbols, eq already has them. If it contains x/y,
    # the substitution above turns it into r/theta, allowing one common evaluator.
    if (r_sym in transformed.free_symbols) or (th_sym in transformed.free_symbols):
        fn = sp.lambdify((r_sym, th_sym), transformed, modules=["numpy"])
        Z = _safe_array(fn(r, np.deg2rad(theta) if req.angle_unit == "degrees" else theta))
    else:
        cart_eq = _substitute_display_expression(eq, "cartesian2d")
        fn = sp.lambdify((x_sym, y_sym), cart_eq, modules=["numpy"])
        Z = _safe_array(fn(X, Y))
    return {"x": X[0, :].tolist(), "y": Y[:, 0].tolist(), "z": Z.tolist(), "type": "contour2d"}


def _plot_2d_cartesian(eq, is_implicit, lhs, rhs, syms, req, preset, t_sym):
    if lhs.casefold() == "r" and rhs is not None:
        return _polar_cartesian_parametric(rhs, req, preset, t_sym)
    if syms & {"r", "theta"}:
        return _cartesian_from_polar_contour(eq, req, preset)
    x0, x1 = _range(req, "x_min", "x_max", (-20, 20)); y0, y1 = _range(req, "y_min", "y_max", (-20, 20))
    if is_implicit:
        n = preset["grid_2d"]
        X, Y = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))
        fn = sp.lambdify(("x", "y"), eq.subs(t_sym, req.t_val), modules=["numpy"])
        return {"x": X[0, :].tolist(), "y": Y[:, 0].tolist(), "z": _safe_array(fn(X, Y)).tolist(), "type": "contour2d"}
    if lhs.casefold() == "x" and rhs is not None:
        n = preset["line_1d"]
        Y = np.linspace(y0, y1, n)
        fn = sp.lambdify("y", rhs.subs(t_sym, req.t_val), modules=["numpy"])
        X = _safe_array(fn(Y))
        return {"x": X.tolist(), "y": Y.tolist(), "type": "line2d"}
    if syms - {"x"}:
        raise ValueError("2D Cartesian explicit plots are interpreted as y=f(x) or x=f(y).")
    n = preset["line_1d"]
    X = np.linspace(x0, x1, n)
    fn = sp.lambdify("x", eq.subs(t_sym, req.t_val), modules=["numpy"]) if "x" in syms else None
    Y = _safe_array(fn(X)) if fn else np.full_like(X, float(eq.evalf()))
    return {"x": X.tolist(), "y": Y.tolist(), "type": "line2d"}


def _plot_2d_polar(eq, is_implicit, lhs, rhs, syms, req, preset, t_sym):
    if not (syms <= {"x", "y", "r", "theta", "t"}):
        raise ValueError("Polar mode expects a 2D x/y or r/theta expression.")
    if lhs.casefold() == "r" and rhs is not None:
        vals = _polar_explicit(rhs, req, preset, t_sym)
        return {**vals, "type": "polar2d", "latex_eq": f"r = {sp.latex(rhs)}"}
    # Any Cartesian x/y equation is genuinely converted to r/theta coordinates.
    transformed = sp.simplify(eq.subs({sp.Symbol("x"): sp.Symbol("r") * sp.cos(sp.Symbol("theta")), sp.Symbol("y"): sp.Symbol("r") * sp.sin(sp.Symbol("theta"))}))
    vals = _polar_implicit(transformed, req, preset)
    return {**vals, "type": "polar2d"}


def _pretty_simplify(expr: sp.Expr, budget_ops: int = 80) -> sp.Expr:
    """A simplification pipeline aimed at readable LaTeX output rather than a
    canonical form: collapse sin^2+cos^2, tidy nested radicals/fractions, and
    fold obvious algebraic redundancy. Plain sp.simplify() alone often leaves
    coordinate-substituted expressions in an ugly, unreduced state (e.g. raw
    atan2/sqrt compositions); this chain catches most of what's left.

    Guarded by op-count: substituting theta=atan2(y,x) etc. into an already
    transcendental original (e.g. sin(3x) -> sin(3 r cos(theta))) can produce a
    deeply nested trig-of-trig expression that SymPy's trigsimp heuristic search
    blows up on (multi-second, sometimes much worse). Above the threshold this
    skips straight to a cheap, bounded tidy-up instead."""
    try:
        ops = sp.count_ops(expr)
    except Exception:
        ops = 0
    if ops > budget_ops:
        try:
            return sp.powsimp(sp.expand(expr), force=True)
        except Exception:
            return expr
    try:
        e = sp.expand_trig(expr)
        e = sp.trigsimp(e)
        e = sp.radsimp(e)
        e = sp.powsimp(e, force=True)
        e = sp.simplify(e)
        return e
    except Exception:
        try:
            return sp.simplify(expr)
        except Exception:
            return expr


def _coordinate_forms(meta: Dict[str, Any], eq: sp.Expr) -> Dict[str, str]:
    """Public entry point: unpacks the (unhashable) meta dict and delegates to the
    cached implementation below, so repeat requests for an unchanged expression
    (e.g. the user tweaks an unrelated axis setting) cost nothing the second time."""
    implicit = bool(meta.get("is_implicit"))
    lhs = (meta.get("lhs_str") or "").strip().casefold()
    rhs = meta.get("rhs_expr")
    syms = frozenset(meta.get("syms", set()))
    return _coordinate_forms_cached(eq, implicit, lhs, rhs, syms)


@lru_cache(maxsize=512)
def _coordinate_forms_cached(eq: sp.Expr, implicit: bool, lhs: str, rhs: Optional[sp.Expr], syms: frozenset) -> Dict[str, str]:
    """Return equivalent forms for every coordinate system this graph module supports."""
    x, y, z = sp.symbols("x y z")
    r, theta, rho, phi = sp.symbols("r theta rho phi")
    u, v = sp.symbols("u v")
    forms: Dict[str, str] = {}

    def fmt(expr: sp.Expr, force_zero: bool = False) -> str:
        return sp.latex(_pretty_simplify(expr)) + (" = 0" if force_zero else "")

    base_simplified = _pretty_simplify(eq)

    def equation_in(target: str) -> Optional[sp.Expr]:
        source_symbols = syms
        expr = base_simplified
        if target in {"Cartesian 2D", "Polar 2D", "Complex plane"}:
            if source_symbols <= {"x", "y", "r", "theta", "t"}:
                if target == "Cartesian 2D":
                    if source_symbols & {"r", "theta"}:
                        expr = expr.subs({r: sp.sqrt(x*x+y*y), theta: sp.atan2(y, x)})
                elif target == "Polar 2D":
                    if source_symbols & {"x", "y"}:
                        expr = expr.subs({x: r*sp.cos(theta), y: r*sp.sin(theta)})
                else:
                    expr = expr.subs({x: u, y: v, r: sp.sqrt(u*u+v*v), theta: sp.atan2(v, u)})
                return _pretty_simplify(expr)
            return None

        if target == "Cartesian 3D":
            if source_symbols <= {"x", "y", "z", "r", "rho", "theta", "phi", "t"}:
                expr = expr.subs({
                    r: sp.sqrt(x*x+y*y), theta: sp.atan2(y,x),
                    rho: sp.sqrt(x*x+y*y+z*z),
                    phi: sp.acos(sp.Max(-1, sp.Min(1, z/sp.sqrt(x*x+y*y+z*z)))),
                })
                return _pretty_simplify(expr)
            return None

        if target == "Cylindrical 3D":
            expr = _pretty_simplify(expr.subs({
                x: r*sp.cos(theta), y: r*sp.sin(theta), z: z,
                rho: sp.sqrt(r*r+z*z), phi: sp.atan2(r, z),
            }))
            return expr if source_symbols & {"x","y","z","r","rho","theta","phi"} else None

        if target == "Spherical 3D":
            expr = _pretty_simplify(expr.subs({
                x: rho*sp.sin(phi)*sp.cos(theta),
                y: rho*sp.sin(phi)*sp.sin(theta),
                z: rho*sp.cos(phi),
                r: rho*sp.sin(phi),
            }))
            return expr if source_symbols & {"x","y","z","r","rho","theta","phi"} else None
        return None

    # Explicit functions get cleaner conversions than raw implicit rearrangement.
    if not implicit and rhs is not None and lhs == "y" and syms <= {"x", "t"}:
        forms["Cartesian 2D"] = f"y = {sp.latex(_pretty_simplify(rhs))}"
        polar_rhs = _pretty_simplify(rhs.subs(x, r*sp.cos(theta)))
        forms["Polar 2D"] = f"r\\sin(\\theta) = {sp.latex(polar_rhs)}"
        forms["Complex plane"] = f"v = {sp.latex(_pretty_simplify(rhs.subs(x, u)))}\\quad(u=\\operatorname{{Re}}z,\\ v=\\operatorname{{Im}}z)"
    elif not implicit and rhs is not None and lhs == "r" and syms <= {"r", "theta", "t"}:
        forms["Polar 2D"] = f"r = {sp.latex(_pretty_simplify(rhs))}"
        cart = _pretty_simplify((r*sp.cos(theta)).subs(r, rhs))
        cart_y = _pretty_simplify((r*sp.sin(theta)).subs(r, rhs))
        forms["Cartesian 2D"] = f"x = {sp.latex(cart)},\\quad y = {sp.latex(cart_y)}"
        forms["Complex plane"] = f"u = {sp.latex(cart)},\\quad v = {sp.latex(cart_y)}\\quad(u=\\operatorname{{Re}}z,\\ v=\\operatorname{{Im}}z)"
    else:
        mapping = [
            ("Cartesian 2D", "Cartesian 2D", True),
            ("Polar 2D", "Polar 2D", True),
            ("Cartesian 3D", "Cartesian 3D", True),
            ("Cylindrical 3D", "Cylindrical 3D", True),
            ("Spherical 3D", "Spherical 3D", True),
            ("Complex plane", "Complex plane", True),
        ]
        for label, target, active in mapping:
            if not active:
                continue
            converted = equation_in(target)
            if converted is None:
                continue
            if implicit or lhs in {"y", "x", "r", "rho", "z"}:
                forms[label] = fmt(converted, implicit)
            else:
                forms[label] = fmt(converted)

    # Add explicit 3D native forms for ordinary z=f(x,y), r=f(theta,z), rho=f(theta,phi).
    if not implicit and rhs is not None and lhs == "z" and syms <= {"x", "y", "z", "t"}:
        forms["Cartesian 3D"] = f"z = {sp.latex(_pretty_simplify(rhs))}"
        forms["Cylindrical 3D"] = f"z = {sp.latex(_pretty_simplify(rhs.subs({x:r*sp.cos(theta), y:r*sp.sin(theta)})))}"
        forms["Spherical 3D"] = f"\\rho\\cos(\\phi) = {sp.latex(_pretty_simplify(rhs.subs({x:rho*sp.sin(phi)*sp.cos(theta), y:rho*sp.sin(phi)*sp.sin(theta)})))}"

    # Keep target order stable and omit forms that are identical to the source when possible.
    ordered = {}
    for key in ("Cartesian 2D", "Polar 2D", "Cartesian 3D", "Cylindrical 3D", "Spherical 3D", "Complex plane"):
        if key in forms and forms[key]:
            ordered[key] = forms[key]
    return ordered



def _robust_range(values, fallback, low_q=2.0, high_q=98.0):
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size < 8:
        return list(fallback)
    lo = float(np.percentile(arr, low_q)); hi = float(np.percentile(arr, high_q))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return list(fallback)
    if hi <= lo or abs(hi - lo) < 1e-10:
        center = 0.0 if not np.isfinite(lo) else lo
        pad = max(abs(center) * 0.20, 1.0)
        return [center - pad, center + pad]
    span = hi - lo
    pad = max(span * 0.08, 1e-6)
    return [lo - pad, hi + pad]


def _suggested_view(plots, req: GraphRequest):
    """Prevent pathological spikes/asymptotes from dominating the initial viewport."""
    if not plots:
        return None
    xs=[]; ys=[]; zs=[]; rs=[]
    for plot in plots:
        if not isinstance(plot, dict):
            continue
        for key, bucket in (("x",xs),("y",ys),("z",zs),("r",rs)):
            vals = plot.get(key)
            if vals is None:
                continue
            try:
                bucket.extend(np.asarray(vals, dtype=float).ravel().tolist())
            except Exception:
                pass
    view={}
    if req.coordinate_system == "polar2d":
        rr=_robust_range(rs, [0.0, abs(_axis(req,"r_max",20.0)) or 20.0], 1.0, 99.0)
        view["r"]=[max(0.0, rr[0]), rr[1]]
    elif req.coordinate_system in {"cartesian2d","complex"}:
        view["x"]=_robust_range(xs, [_axis(req,"re_min",-8.0) if req.coordinate_system=="complex" else _axis(req,"x_min",-20.0), _axis(req,"re_max",8.0) if req.coordinate_system=="complex" else _axis(req,"x_max",20.0)])
        view["y"]=_robust_range(ys, [_axis(req,"im_min",-8.0) if req.coordinate_system=="complex" else _axis(req,"y_min",-20.0), _axis(req,"im_max",8.0) if req.coordinate_system=="complex" else _axis(req,"y_max",20.0)])
    else:
        if req.coordinate_system == "cartesian3d":
            view["x"]=_robust_range(xs, [_axis(req,"x_min",-20.0),_axis(req,"x_max",20.0)])
            view["y"]=_robust_range(ys, [_axis(req,"y_min",-20.0),_axis(req,"y_max",20.0)])
            view["z"]=_robust_range(zs, [_axis(req,"z_min",-20.0),_axis(req,"z_max",20.0)])
        elif req.coordinate_system == "cylindrical3d":
            view["x"]=[_axis(req,"theta_min",0.0),_axis(req,"theta_max",360.0)]
            view["y"]=_robust_range(ys,[0.0,abs(_axis(req,"r_max",20.0)) or 20.0],1.0,99.0)
            view["z"]=_robust_range(zs,[_axis(req,"z_min",-20.0),_axis(req,"z_max",20.0)])
        elif req.coordinate_system == "spherical3d":
            view["x"]=[_axis(req,"theta_min",0.0),_axis(req,"theta_max",360.0)]
            view["y"]=[_axis(req,"phi_min",0.0),_axis(req,"phi_max",180.0)]
            view["z"]=_robust_range(zs,[0.0,abs(_axis(req,"r_max",20.0)) or 20.0],1.0,99.0)
    return view or None

def _smart_suggestions(parsed_meta, parsed_eqs, expression_names, req):
    suggestions=[]
    valid=[i for i,e in enumerate(parsed_eqs) if e is not None and parsed_meta[i] and "error" not in parsed_meta[i]]
    for i in valid:
        m=parsed_meta[i]; syms=set(m.get("syms",set())); name=expression_names[i]; actions=[]
        if syms & {"x","y"} or m.get("is_implicit"): actions += [{"label":"Show intercepts","command":f"intercepts({name})"},{"label":"Convert coordinates","command":f"convert({name})"}]
        if (m.get("lhs_str") or "").strip().casefold()=="y" and not m.get("is_implicit"): actions.append({"label":"Area under graph","command":f"area_under({name}, {req.axis.get('x_min',-20)}, {req.axis.get('x_max',20)})"})
        if (m.get("lhs_str") or "").strip().casefold()=="r": actions += [{"label":"Polar area","command":f"polar_area({name})"},{"label":"Convert coordinates","command":f"convert({name})"}]
        if m.get("is_implicit") and {"x","y"}.issubset(syms):
            try:
                degree=sp.Poly(sp.expand(parsed_eqs[i]), x, y).total_degree()
            except Exception:
                degree=0
            if degree >= 2:
                actions.append({"label":"Area inside/closed","command":f"area_closed({name})"})
        if actions: suggestions.append({"title":f"Analyze {name}","expression":name,"actions":actions[:5]})
    if len(valid)>=2:
        for p,i1 in enumerate(valid):
            for i2 in valid[p+1:p+4]:
                if (set(parsed_meta[i1].get("syms",set()))|set(parsed_meta[i2].get("syms",set()))) <= {"x","y","r","theta","t"}:
                    a,b=expression_names[i1],expression_names[i2]
                    suggestions.append({"title":f"Compare {a} ↔ {b}","expression":f"{a}+{b}","actions":[{"label":"Find intersections","command":f"intersection({a}, {b})"},{"label":"Area of bounded region","command":f"area_region({a}, {b})"},{"label":"Minimum distance","command":f"distance({a}, {b})"}]})
    return suggestions[:10]


def _convert_3d_expression_to_source(eq: sp.Expr, source_mode: str):
    x, y, z, r, rho, theta, phi = sp.symbols("x y z r rho theta phi")
    if source_mode == "cylindrical3d":
        return sp.simplify(eq.subs({x: r * sp.cos(theta), y: r * sp.sin(theta)}))
    if source_mode == "spherical3d":
        return sp.simplify(eq.subs({x: rho * sp.sin(phi) * sp.cos(theta), y: rho * sp.sin(phi) * sp.sin(theta), z: rho * sp.cos(phi)}))
    return eq


def _display_3d_symbols(mode: str):
    if mode == "cylindrical3d": return {"r", "theta", "z"}
    if mode == "spherical3d": return {"rho", "theta", "phi"}
    return {"x", "y", "z"}


def _plot_3d_cartesian(eq, is_implicit, lhs, rhs, syms, req, preset, t_sym):
    x0, x1 = _range(req, "x_min", "x_max", (-20, 20)); y0, y1 = _range(req, "y_min", "y_max", (-20, 20)); z0, z1 = _range(req, "z_min", "z_max", (-20, 20))
    x_sym, y_sym, z_sym = sp.symbols("x y z")
    cart_eq = _substitute_display_expression(eq.subs(t_sym, req.t_val) if t_sym in eq.free_symbols else eq, "cartesian3d")
    if is_implicit and lhs.casefold() == "z" and z_sym not in rhs.free_symbols and len(rhs.free_symbols & {x_sym, y_sym, sp.Symbol("r"), sp.Symbol("theta")}) >= 0:
        n = preset["grid_2d"]
        X, Y = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))
        rhs_cart = _substitute_display_expression(rhs.subs(t_sym, req.t_val), "cartesian3d")
        try:
            fn = sp.lambdify((x_sym, y_sym), rhs_cart, modules=["numpy"])
            Z = _safe_array(fn(X, Y))
            if np.isfinite(Z).any():
                return {"x": X[0, :].tolist(), "y": Y[:, 0].tolist(), "z": Z.tolist(), "type": "surface3d"}
        except Exception:
            pass
    n = preset["grid_3d"]
    X, Y, Z = np.mgrid[x0:x1:complex(0, n), y0:y1:complex(0, n), z0:z1:complex(0, n)]
    fn = sp.lambdify((x_sym, y_sym, z_sym), cart_eq, modules=["numpy"])
    val = _safe_array(fn(X, Y, Z))
    return {"x": X.ravel().tolist(), "y": Y.ravel().tolist(), "z": Z.ravel().tolist(), "val": val.ravel().tolist(), "type": "isosurface3d"}


def _plot_3d_transformed(eq, mode, req, preset, t_sym, syms: set):
    n = preset["grid_3d"]
    if mode == "cylindrical3d":
        a0, a1 = _range(req, "theta_min", "theta_max", (0, 360))
        p0, p1 = _range(req, "z_min", "z_max", (-20, 20))
        a0r, a1r = (a0, a1) if req.angle_unit == "radians" else (np.deg2rad(a0), np.deg2rad(a1))
        r0, r1 = 0.0, abs(_axis(req, "r_max", 20.0)) or 20.0
        theta = np.linspace(a0r, a1r, n); r = np.linspace(r0, r1, n); z = np.linspace(p0, p1, n)
        TH, RR, ZZ = np.meshgrid(theta, r, z, indexing="ij")
        r_sym, th_sym, z_sym = sp.symbols("r theta z")
        cart_x = RR * np.cos(TH); cart_y = RR * np.sin(TH)
        display_eq = eq.subs(t_sym, req.t_val) if t_sym in eq.free_symbols else eq
        if syms & {"x", "y"}:
            display_eq = display_eq.subs({sp.Symbol("x"): r_sym * sp.cos(th_sym), sp.Symbol("y"): r_sym * sp.sin(th_sym)})
        display_eq = sp.simplify(display_eq)
        if syms <= {"x", "y", "z", "t"} and eq.free_symbols - {sp.Symbol("x"), sp.Symbol("y"), sp.Symbol("z"), t_sym}:
            raise ValueError("Unsupported 3D symbols.")
        fn_symbols = tuple(sym for sym in (r_sym, th_sym, z_sym) if sym in display_eq.free_symbols)
        fn = sp.lambdify(fn_symbols or (r_sym,), display_eq, modules=["numpy"])
        if fn_symbols == (r_sym, th_sym, z_sym): val = fn(RR, TH, ZZ)
        elif fn_symbols == (r_sym, th_sym): val = fn(RR, TH)
        elif fn_symbols == (r_sym, z_sym): val = fn(RR, ZZ)
        elif fn_symbols == (th_sym, z_sym): val = fn(TH, ZZ)
        elif fn_symbols == (r_sym,): val = fn(RR)
        elif fn_symbols == (th_sym,): val = fn(TH)
        elif fn_symbols == (z_sym,): val = fn(ZZ)
        else:
            val = np.broadcast_to(float(display_eq.evalf()), RR.shape)
        x_out = np.rad2deg(TH) if req.angle_unit == "degrees" else TH
        return {"x": x_out.ravel().tolist(), "y": RR.ravel().tolist(), "z": ZZ.ravel().tolist(), "val": _safe_array(val).ravel().tolist(), "type": "isosurface3d"}

    # spherical
    t0, t1 = _range(req, "theta_min", "theta_max", (0, 360)); p0, p1 = _range(req, "phi_min", "phi_max", (0, 180)); rho1 = abs(_axis(req, "r_max", 20.0)) or 20.0
    theta = np.linspace(np.deg2rad(t0), np.deg2rad(t1), n) if req.angle_unit == "degrees" else np.linspace(t0, t1, n)
    phi = np.linspace(np.deg2rad(p0), np.deg2rad(p1), n) if req.angle_unit == "degrees" else np.linspace(p0, p1, n)
    rho = np.linspace(0, rho1, n)
    TH, PH, RH = np.meshgrid(theta, phi, rho, indexing="ij")
    rho_sym, th_sym, ph_sym = sp.symbols("rho theta phi")
    display_eq = eq.subs(t_sym, req.t_val) if t_sym in eq.free_symbols else eq
    if syms & {"x", "y", "z"}:
        display_eq = display_eq.subs({
            sp.Symbol("x"): rho_sym * sp.sin(ph_sym) * sp.cos(th_sym),
            sp.Symbol("y"): rho_sym * sp.sin(ph_sym) * sp.sin(th_sym),
            sp.Symbol("z"): rho_sym * sp.cos(ph_sym),
        })
    fn_symbols = tuple(sym for sym in (rho_sym, th_sym, ph_sym) if sym in display_eq.free_symbols)
    fn = sp.lambdify(fn_symbols or (rho_sym,), display_eq, modules=["numpy"])
    if fn_symbols == (rho_sym, th_sym, ph_sym): val = fn(RH, TH, PH)
    elif fn_symbols == (rho_sym, th_sym): val = fn(RH, TH)
    elif fn_symbols == (rho_sym, ph_sym): val = fn(RH, PH)
    elif fn_symbols == (th_sym, ph_sym): val = fn(TH, PH)
    elif fn_symbols == (rho_sym,): val = fn(RH)
    elif fn_symbols == (th_sym,): val = fn(TH)
    elif fn_symbols == (ph_sym,): val = fn(PH)
    else: val = np.broadcast_to(float(display_eq.evalf()), RH.shape)
    x_out = np.rad2deg(TH) if req.angle_unit == "degrees" else TH
    y_out = np.rad2deg(PH) if req.angle_unit == "degrees" else PH
    return {"x": x_out.ravel().tolist(), "y": y_out.ravel().tolist(), "z": RH.ravel().tolist(), "val": _safe_array(val).ravel().tolist(), "type": "isosurface3d"}


def _plot_3d(req, eq, is_implicit, lhs, rhs, syms, preset, t_sym):
    allowed = {"x", "y", "z", "r", "rho", "theta", "phi", "t"}
    if syms - allowed:
        raise ValueError("3D mode supports x/y/z, cylindrical r/theta/z, or spherical rho/theta/phi variables.")
    if req.coordinate_system == "cartesian3d":
        return _plot_3d_cartesian(eq, is_implicit, lhs, rhs, syms, req, preset, t_sym)
    if req.coordinate_system == "cylindrical3d" and lhs.casefold() == "r" and rhs is not None and {"theta", "z"}.intersection(syms):
        a0, a1 = _range(req, "theta_min", "theta_max", (0, 360)); z0, z1 = _range(req, "z_min", "z_max", (-20, 20))
        theta = np.linspace(np.deg2rad(a0), np.deg2rad(a1), preset["polar_theta"] // 2) if req.angle_unit == "degrees" else np.linspace(a0, a1, preset["polar_theta"] // 2)
        zz = np.linspace(z0, z1, preset["grid_2d"])
        TH, Z = np.meshgrid(theta, zz)
        fn = sp.lambdify(tuple(sym for sym in (sp.Symbol("theta"), sp.Symbol("z")) if sym in rhs.free_symbols), rhs.subs(t_sym, req.t_val), modules=["numpy"])
        needed = [sym for sym in (sp.Symbol("theta"), sp.Symbol("z")) if sym in rhs.free_symbols]
        if needed == [sp.Symbol("theta"), sp.Symbol("z")]: R = _safe_array(fn(TH, Z))
        elif needed == [sp.Symbol("theta")]: R = _safe_array(fn(TH))
        elif needed == [sp.Symbol("z")]: R = _safe_array(fn(Z))
        else: R = np.full_like(TH, float(rhs.evalf()))
        return {"x": (np.rad2deg(TH) if req.angle_unit == "degrees" else TH).tolist(), "y": R.tolist(), "z": Z.tolist(), "type": "surface3d", "latex_eq": f"r = {sp.latex(rhs)}"}
    if req.coordinate_system == "spherical3d" and lhs.casefold() == "rho" and rhs is not None:
        a0, a1 = _range(req, "theta_min", "theta_max", (0, 360)); p0, p1 = _range(req, "phi_min", "phi_max", (0, 180))
        theta = np.linspace(np.deg2rad(a0), np.deg2rad(a1), preset["polar_theta"] // 2) if req.angle_unit == "degrees" else np.linspace(a0, a1, preset["polar_theta"] // 2)
        phi = np.linspace(np.deg2rad(p0), np.deg2rad(p1), preset["grid_2d"]) if req.angle_unit == "degrees" else np.linspace(p0, p1, preset["grid_2d"])
        TH, PH = np.meshgrid(theta, phi)
        needed = [sym for sym in (sp.Symbol("theta"), sp.Symbol("phi")) if sym in rhs.free_symbols]
        fn = sp.lambdify(tuple(needed), rhs.subs(t_sym, req.t_val), modules=["numpy"])
        if needed == [sp.Symbol("theta"), sp.Symbol("phi")]: RHO = _safe_array(fn(TH, PH))
        elif needed == [sp.Symbol("theta")]: RHO = _safe_array(fn(TH))
        elif needed == [sp.Symbol("phi")]: RHO = _safe_array(fn(PH))
        else: RHO = np.full_like(TH, float(rhs.evalf()))
        return {"x": (np.rad2deg(TH) if req.angle_unit == "degrees" else TH).tolist(), "y": (np.rad2deg(PH) if req.angle_unit == "degrees" else PH).tolist(), "z": RHO.tolist(), "type": "surface3d", "latex_eq": f"\\rho = {sp.latex(rhs)}"}
    return _plot_3d_transformed(eq, req.coordinate_system, req, preset, t_sym, syms)

def _complex_expression(text: str):
    clean = _normalize_math_text(text)
    if "=" in clean:
        clean = clean.split("=", 1)[1].strip()
    z = sp.Symbol("z")
    local = {
        "z": z, "I": sp.I, "i": sp.I, "j": sp.I, "pi": sp.pi,
        "re": sp.re, "im": sp.im, "arg": sp.arg, "Abs": sp.Abs, "abs": sp.Abs, "conjugate": sp.conjugate,
    }
    return parse_expr(clean, local_dict=local, transformations=TRANSFORMATIONS), z


def _plot_complex_geometry(eq: sp.Expr, req: GraphRequest, preset):
    """Plot a Cartesian 2D relation directly on Re(z)-Im(z) axes."""
    re0, re1 = _range(req, "re_min", "re_max", (-8, 8)); im0, im1 = _range(req, "im_min", "im_max", (-8, 8))
    n=preset["grid_2d"]
    U,V=np.meshgrid(np.linspace(re0,re1,n),np.linspace(im0,im1,n))
    x,y=sp.symbols("x y")
    fn=sp.lambdify((x,y),eq,modules=["numpy"])
    Z=_safe_array(fn(U,V))
    return {"x":U[0,:].tolist(),"y":V[:,0].tolist(),"z":Z.tolist(),"type":"complex_geometry2d","latex_eq":sp.latex(eq)+" = 0"}


def _plot_complex(expr_text: str, req: GraphRequest, preset):
    expr, z = _complex_expression(expr_text)
    re0, re1 = _range(req, "re_min", "re_max", (-8, 8)); im0, im1 = _range(req, "im_min", "im_max", (-8, 8))
    n = preset["complex"]
    u = np.linspace(re0, re1, n); v = np.linspace(im0, im1, n)
    U, V = np.meshgrid(u, v)
    ZZ = U + 1j * V
    fn = sp.lambdify(z, expr.subs(sp.Symbol("t"), req.t_val), modules=["numpy"])
    F = _complex_array(fn(ZZ))
    view = req.complex_view
    if view == "real": values = np.real(F)
    elif view == "imag": values = np.imag(F)
    elif view == "phase": values = np.angle(F)
    else: values = np.abs(F)
    if view == "surface":
        return {"x": u.tolist(), "y": v.tolist(), "z": np.abs(F).tolist(), "type": "surface3d", "latex_eq": sp.latex(expr)}
    return {"x": u.tolist(), "y": v.tolist(), "z": np.asarray(values, dtype=float).tolist(), "type": "complex_heatmap", "latex_eq": sp.latex(expr), "complex_expression": str(expr)}


def _analysis_eq(meta, eq, t_sym, req):
    equation = eq.subs(t_sym, req.t_val) if t_sym in eq.free_symbols else eq
    # Analysis is coordinate-independent: convert polar/cylindrical/spherical forms into physical Cartesian coordinates.
    if (meta and (meta.get("lhs_str") or "").strip().casefold() == "r" and meta.get("rhs_expr") is not None):
        # Polar r=f(theta) can often be converted to a clean Cartesian implicit
        # numerator. This is crucial for intersections, distance and region tools.
        x, y = sp.symbols("x y")
        r, theta = sp.symbols("r theta")
        try:
            polar_eq = r - meta["rhs_expr"].subs(t_sym, req.t_val) if t_sym in meta["rhs_expr"].free_symbols else r - meta["rhs_expr"]
            cart = sp.simplify(polar_eq.subs({r: sp.sqrt(x*x+y*y), theta: sp.atan2(y,x)}))
            numerator = sp.together(cart).as_numer_denom()[0]
            if numerator != 0:
                equation = sp.simplify(numerator)
        except Exception:
            equation = equation.subs({r: sp.sqrt(x*x+y*y), theta: sp.atan2(y,x)})
    elif req.coordinate_system in {"polar2d", "cylindrical3d"} or (meta and set(meta.get("syms", set())) & {"r", "theta"}):
        equation = equation.subs({
            sp.Symbol("r"): sp.sqrt(sp.Symbol("x")**2 + sp.Symbol("y")**2),
            sp.Symbol("theta"): sp.atan2(sp.Symbol("y"), sp.Symbol("x")),
        })
    if req.coordinate_system == "spherical3d" or (meta and set(meta.get("syms", set())) & {"rho", "phi"}):
        x, y, z = sp.symbols("x y z")
        equation = equation.subs({
            sp.Symbol("rho"): sp.sqrt(x**2 + y**2 + z**2),
            sp.Symbol("theta"): sp.atan2(y, x),
            sp.Symbol("phi"): sp.acos(sp.Max(-1, sp.Min(1, z/sp.sqrt(x**2+y**2+z**2)))),
        })
    return _curve_equation(equation, meta["is_implicit"])


def _function_data(meta, eq, req):
    """Return ('cartesian', variable, function) or ('polar', theta, r(theta)) where the expression is explicit."""
    x, y, r, theta = sp.symbols("x y r theta")
    equation = eq.subs(sp.Symbol("t"), req.t_val) if sp.Symbol("t") in eq.free_symbols else eq
    lhs = (meta.get("lhs_str") or "").strip().casefold()
    rhs = meta.get("rhs_expr")
    if rhs is not None and sp.Symbol("t") in rhs.free_symbols:
        rhs = rhs.subs(sp.Symbol("t"), req.t_val)
    source_syms = set(meta.get("syms", set()))
    if lhs == "r" and rhs is not None:
        return "polar", theta, rhs
    if lhs == "y" and rhs is not None and rhs.free_symbols <= {x, sp.Symbol("t")}:
        return "cartesian", x, rhs
    if lhs == "x" and rhs is not None and rhs.free_symbols <= {y, sp.Symbol("t")}:
        return "cartesian", y, rhs
    if not meta.get("is_implicit") and source_syms <= {"x", "t"}:
        return "cartesian", x, equation
    return None, None, None


def _numeric_integral(values: np.ndarray, coords: np.ndarray) -> float:
    finite = np.isfinite(values) & np.isfinite(coords)
    if finite.sum() < 2:
        return 0.0
    return float(np.trapezoid(values[finite], coords[finite]))


def _eval_explicit_function(fn_expr: sp.Expr, variable: sp.Symbol, lo: float, hi: float, count: int = 3000):
    grid = np.linspace(lo, hi, count)
    fn = sp.lambdify(variable, fn_expr, modules=["numpy"])
    values = _safe_array(fn(grid))
    values = np.broadcast_to(values, grid.shape)
    return grid, values


def _auto_intersection_bounds(eq: sp.Expr, other: Optional[sp.Expr] = None):
    x = sp.Symbol("x")
    target = eq if other is None else sp.simplify(eq - other)
    roots = _real_solutions(target, x, 20)
    if len(roots) >= 2:
        return min(roots), max(roots)
    return None


def _polar_limits(req: GraphRequest, args: List[str], start_index: int = 1):
    if len(args) >= start_index + 2:
        a = float(args[start_index]); b = float(args[start_index + 1])
        return (a, b)
    return (_axis(req, "theta_min", 0.0), _axis(req, "theta_max", 360.0))


def _area_under(meta, eq, req, lo: Optional[float], hi: Optional[float]):
    kind, variable, function = _function_data(meta, eq, req)
    if kind == "polar":
        if lo is None or hi is None: lo, hi = _axis(req, "theta_min", 0.0), _axis(req, "theta_max", 360.0)
        lo_r = np.deg2rad(lo) if req.angle_unit == "degrees" else lo
        hi_r = np.deg2rad(hi) if req.angle_unit == "degrees" else hi
        grid = np.linspace(lo_r, hi_r, 4000)
        fn = sp.lambdify(variable, function.subs(sp.Symbol("theta"), variable), modules=["numpy"])
        rvals = _safe_array(fn(grid))
        rvals = np.broadcast_to(rvals, grid.shape)
        return 0.5 * _numeric_integral(rvals**2, grid), f"\\frac12\\int r^2\\,d\\theta"
    if kind != "cartesian":
        raise ValueError("area_under requires an explicit y=f(x) function (or r=f(theta) in polar mode).")
    if variable != sp.Symbol("x"):
        raise ValueError("area_under currently uses y=f(x).")
    if lo is None or hi is None:
        raise ValueError("Provide limits: area_under(Name, a, b)")
    grid, values = _eval_explicit_function(function, variable, lo, hi)
    return _numeric_integral(values, grid), f"\\int_{{{lo}}}^{{{hi}}} f(x)\\,dx"


def _closed_implicit_area(eq: sp.Expr, lo: Optional[float], hi: Optional[float]):
    x, y = sp.symbols("x y")
    branches = []
    try:
        branches = [sp.simplify(v) for v in sp.solve(eq, y) if not getattr(v, "free_symbols", set()) <= {x} or x in v.free_symbols]
    except Exception:
        branches = []
    branches = [b for b in branches if x in b.free_symbols]
    if len(branches) < 2:
        return None
    if lo is None or hi is None:
        bounds = _auto_intersection_bounds(eq.subs(y, 0))
        if bounds:
            lo, hi = bounds
        else:
            return None
    grid = np.linspace(lo, hi, 3500)
    branch_values = []
    for branch in branches:
        fn = sp.lambdify(x, branch, modules=["numpy"])
        branch_values.append(_safe_array(fn(grid)))
    stack = np.vstack(branch_values)
    height = np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0)
    if not np.isfinite(height).any():
        return None
    return _numeric_integral(height, grid)


def _y_branches(meta, eq):
    """Return symbolic y=f(x) branches for an implicit/explicit Cartesian curve."""
    x, y = sp.symbols("x y")
    kind, variable, function = _function_data(meta, eq, GraphRequest(expressions=[]))
    if kind == "cartesian" and variable == x:
        return [function]
    if meta.get("is_implicit"):
        try:
            branches = [sp.simplify(v) for v in sp.solve(_analysis_eq(meta, eq, sp.Symbol("t"), GraphRequest(expressions=[])), y)]
            return [b for b in branches if x in b.free_symbols]
        except Exception:
            return []
    return []


def _numeric_branch_area(branch_a, branch_b, lo, hi, count=3200):
    x = sp.Symbol("x")
    grid = np.linspace(lo, hi, count)
    fa = sp.lambdify(x, branch_a, modules=["numpy"])
    fb = sp.lambdify(x, branch_b, modules=["numpy"])
    a = np.broadcast_to(_safe_array(fa(grid)), grid.shape)
    b = np.broadcast_to(_safe_array(fb(grid)), grid.shape)
    gap = np.abs(a-b)
    finite = np.isfinite(gap)
    return _numeric_integral(gap[finite], grid[finite]), grid, a, b


def _surface_function(meta, eq):
    lhs = (meta.get("lhs_str") or "").strip().casefold()
    rhs = meta.get("rhs_expr")
    if lhs == "z" and rhs is not None:
        return rhs.subs(sp.Symbol("t"), 0) if sp.Symbol("t") in rhs.free_symbols else rhs
    return None

@router.post("/plot")
def generate_plot_data(req: GraphRequest):
    try:
        if len(req.expressions) > MAX_EXPRESSIONS:
            raise ValueError(f"Too many expressions (maximum {MAX_EXPRESSIONS}).")
        if len(req.commands) > MAX_COMMANDS:
            raise ValueError(f"Too many commands (maximum {MAX_COMMANDS}).")
        for command in req.commands:
            if len(command) > MAX_COMMAND_LENGTH:
                raise ValueError(f"Command is too long (maximum {MAX_COMMAND_LENGTH} characters).")
        for key in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "theta_min", "theta_max", "phi_min", "phi_max"):
            if key in req.axis and not np.isfinite(float(req.axis[key])):
                raise ValueError(f"Axis value {key} must be finite.")
        if req.coordinate_system in {"cartesian2d", "cartesian3d"} and req.axis.get("x_scale") == "log":
            if _axis(req, "x_min", -20) <= 0 or _axis(req, "x_max", 20) <= 0:
                raise ValueError("Logarithmic X scale requires a positive X range.")
        if req.coordinate_system in {"cartesian2d", "cartesian3d"} and req.axis.get("y_scale") == "log":
            if _axis(req, "y_min", -20) <= 0 or _axis(req, "y_max", 20) <= 0:
                raise ValueError("Logarithmic Y scale requires a positive Y range.")
        preset = QUALITY_PRESETS.get(req.quality, QUALITY_PRESETS["medium"])
        t_sym = sp.Symbol("t")
        parsed, plots, parsed_eqs, parsed_meta = [], [], [], []
        special_points: List[Dict[str, Any]] = []

        # ---------------- Parse once ----------------
        for expr_str in req.expressions:
            if not expr_str.strip():
                parsed.append(None)
                continue
            try:
                base_eq, implicit, lhs, rhs = _cached_parse(expr_str)
                parsed.append({
                    "base_eq": base_eq,
                    "is_implicit": implicit,
                    "lhs_str": lhs,
                    "rhs_expr": rhs,
                    "syms": {s.name for s in base_eq.free_symbols},
                })
            except Exception as exc:
                parsed.append({"error": f"Syntax Error: {exc}"})

        # ---------------- Build plots ----------------
        for idx, p in enumerate(parsed):
            if p is None:
                if req.include_plots:
                    plots.append({"empty": True})
                parsed_eqs.append(None)
                parsed_meta.append(None)
                continue
            if "error" in p:
                if req.include_plots:
                    plots.append({"error": p["error"]})
                parsed_eqs.append(None)
                parsed_meta.append(p)
                continue

            base_eq = p["base_eq"]
            parsed_eqs.append(base_eq)
            parsed_meta.append(p)
            if not req.include_plots:
                continue

            try:
                eq = base_eq.subs(t_sym, req.t_val) if t_sym in base_eq.free_symbols else base_eq
                syms = {s.name for s in eq.free_symbols}
                lhs = p["lhs_str"]
                rhs = p["rhs_expr"]

                if req.coordinate_system == "complex":
                    if syms & {"x", "y"}:
                        data = _plot_complex_geometry(eq, req, preset)
                    else:
                        data = _plot_complex(req.expressions[idx], req, preset)
                elif req.coordinate_system == "cartesian2d":
                    data = _plot_2d_cartesian(eq, p["is_implicit"], lhs, rhs, syms, req, preset, t_sym)
                elif req.coordinate_system == "polar2d":
                    data = _plot_2d_polar(eq, p["is_implicit"], lhs, rhs, syms, req, preset, t_sym)
                else:
                    data = _plot_3d(req, eq, p["is_implicit"], lhs, rhs, syms, preset, t_sym)

                data["latex_eq"] = data.get("latex_eq") or (sp.latex(eq) + (" = 0" if p["is_implicit"] else ""))
                plots.append({**data, "error": None})

                if req.coordinate_system != "complex":
                    is3d = req.coordinate_system in {"cartesian3d", "cylindrical3d", "spherical3d"}
                    for point in _axis_points(eq, p["is_implicit"], frozenset(syms), is3d):
                        point["owner"] = idx
                        special_points.append(point)
            except Exception as exc:
                plots.append({"error": f"Plot Error: {exc}"})

        special_points = _unique_points(special_points)

        # ---------------- Command/analysis setup ----------------
        command_results: List[str] = []
        command_plots: List[Dict[str, Any]] = []
        command_selection: List[int] = []
        expression_names = list(req.expression_names)
        if len(expression_names) < len(req.expressions):
            expression_names.extend([f"Eq {i + 1}" for i in range(len(expression_names), len(req.expressions))])

        def valid(i: Optional[int]) -> bool:
            return i is not None and 0 <= i < len(parsed_eqs) and parsed_eqs[i] is not None

        def is2d(i: Optional[int]) -> bool:
            return (
                valid(i)
                and parsed_meta[i] is not None
                and set(parsed_meta[i].get("syms", set())) <= {"x", "y", "r", "theta", "t"}
            )

        def is3d_expr(i: Optional[int]) -> bool:
            return valid(i) and parsed_meta[i] is not None and _is_3d_symbols(set(parsed_meta[i].get("syms", set())))

        def resolve_pair(args: List[str], usage: str) -> Tuple[int, int]:
            if len(args) != 2:
                raise ValueError(f"Usage: {usage}")
            i1 = _resolve_expression_ref(args[0], expression_names, len(parsed_eqs))
            i2 = _resolve_expression_ref(args[1], expression_names, len(parsed_eqs))
            if i1 is None or i2 is None:
                raise ValueError("Could not resolve one or both equation names/numbers.")
            return i1, i2

        def resolve_one(args: List[str], usage: str) -> int:
            if len(args) != 1:
                raise ValueError(f"Usage: {usage}")
            index = _resolve_expression_ref(args[0], expression_names, len(parsed_eqs))
            if index is None:
                raise ValueError("Could not resolve the equation name/number.")
            return index

        def add_point_plots(points, label_prefix, name):
            if not points:
                return
            command_plots.append({
                "kind": "points",
                "x": [p["x"] for p in points],
                "y": [p["y"] for p in points],
                "z": [p.get("z", 0.0) for p in points],
                "mode": "markers+text",
                "marker": {"size": 10, "color": "#ff2d95", "line": {"width": 2, "color": "#ffffff"}, "symbol": "star"},
                "text": [f"{label_prefix}{i+1}" for i in range(len(points))],
                "name": name,
            })

        def add_area_plot(xv, yv, name, fill_color="#60a5fa"):
            command_plots.append({
                "kind": "area",
                "x": list(map(float, xv)),
                "y": list(map(float, yv)),
                "z": [0.0] * len(xv),
                "mode": "lines",
                "line": {"color": fill_color, "width": 2},
                "fill": "tozeroy",
                "fillcolor": "rgba(96,165,250,0.24)",
                "name": name,
            })

        def current_eq(i):
            return _analysis_eq(parsed_meta[i], parsed_eqs[i], t_sym, req)

        def selection_preface(entries) -> str:
            """entries: list of (label, index_or_None, eq_expr_or_None). Records
            which expression indices the command actually used — sent back to the
            frontend as command_selection so it can visually highlight exactly
            those curves/surfaces — and returns a LaTeX line naming each selected
            object together with its equation, so 'what did I just compute on' is
            answered up front rather than left implicit."""
            command_selection.clear()
            parts = []
            for label, idx, eq_expr in entries:
                if idx is not None and idx >= 0:
                    command_selection.append(idx)
                if eq_expr is not None:
                    try:
                        parts.append(f"\\text{{{label}: }}{sp.latex(eq_expr)}")
                    except Exception:
                        parts.append(f"\\text{{{label}}}")
                else:
                    parts.append(f"\\text{{{label}}}")
            return "\\text{Selected — }" + ",\\quad ".join(parts)

        def _solve_for_z_height(base_eq, is_implicit, lhs, rhs):
            """Return z=f(x,y) (or a constant) for a surface/plane written either
            explicitly (z = ...) or implicitly (F(x,y,z)=0), by solving for z."""
            z_s = sp.Symbol("z")
            if not is_implicit and (lhs or "").strip().casefold() == "z" and rhs is not None:
                return rhs
            if z_s not in base_eq.free_symbols:
                return None
            try:
                sols = [s for s in sp.solve(sp.Eq(base_eq, 0), z_s) if not (s.free_symbols & {z_s})]
                return sols[0] if sols else None
            except Exception:
                return None

        def resolve_ref_to_eq(ref: str):
            """Resolve a command argument to (implicit_equation, label, index).
            Accepts an existing expression's name/number, OR a raw axis-plane / ad
            hoc equation typed directly in the command, e.g. 'z=0', 'x=3', 'theta=45'.
            index is None for ad hoc refs that don't correspond to a plotted curve."""
            idx = _resolve_expression_ref(ref, expression_names, len(parsed_eqs))
            if idx is not None and valid(idx):
                return current_eq(idx), expression_names[idx], idx
            try:
                norm = _normalize_math_text(ref)
                if "=" not in norm:
                    return None, None, None
                base_eq, implicit, lhs, rhs = _cached_parse(ref)
                meta_tmp = {
                    "is_implicit": implicit, "lhs_str": lhs, "rhs_expr": rhs,
                    "syms": {s.name for s in base_eq.free_symbols},
                }
                return _analysis_eq(meta_tmp, base_eq, t_sym, req), ref.strip(), None
            except Exception:
                return None, None, None

        def resolve_ref_to_surface_height(ref: str):
            """Like resolve_ref_to_eq, but returns (z=f(x,y) height, label, index),
            resolving implicit forms via sp.solve(...,'z') where needed."""
            idx = _resolve_expression_ref(ref, expression_names, len(parsed_eqs))
            if idx is not None and valid(idx):
                meta = parsed_meta[idx]
                height = _solve_for_z_height(parsed_eqs[idx], meta["is_implicit"], meta.get("lhs_str"), meta.get("rhs_expr"))
                return height, expression_names[idx], idx
            try:
                base_eq, implicit, lhs, rhs = _cached_parse(ref)
                height = _solve_for_z_height(base_eq, implicit, lhs, rhs)
                return height, ref.strip(), None
            except Exception:
                return None, None, None

        for command in req.commands:
            function_name, args = _split_command(command)
            try:
                canonical = {
                    "intersection": "intersection", "intersect": "intersection", "intersections": "intersection",
                    "intersectall": "intersectall", "locus": "locus", "distance": "distance",
                    "roots": "roots", "root": "roots", "intercepts": "intercepts", "axisintercepts": "intercepts",
                    "derive": "derive", "derivative": "derive", "differentiate": "derive",
                    "integrate": "integrate", "integral": "integrate",
                    "area": "area_under", "area_under": "area_under", "under": "area_under",
                    "area_between": "area_between", "between": "area_between", "region": "area_region", "area_region": "area_region",
                    "area_closed": "area_closed", "closed_area": "area_closed",
                    "arc_length": "arc_length", "length": "arc_length",
                    "polar_area": "polar_area", "polar_arc_length": "polar_arc_length",
                    "volume_under": "volume_under", "volume": "volume_under", "surface_area": "surface_area",
                    "volume_between": "volume_between", "volume_intersect": "volume_intersect",
                    "intersect_volume": "volume_intersect", "region_volume": "volume_intersect",
                    "evaluate": "evaluate", "value": "evaluate", "tangent": "tangent", "normal": "normal",
                    "extrema": "extrema", "convert": "convert", "conversion": "convert", "help": "help",
                }.get(function_name, function_name)

                if canonical == "intersectall":
                    if args: raise ValueError("Usage: intersectall()")
                    indices = [i for i in range(len(parsed_eqs)) if is2d(i)]
                    pairs = []; total = 0
                    for pos, i1 in enumerate(indices):
                        for i2 in indices[pos + 1:]:
                            pts = _intersection_points(current_eq(i1), current_eq(i2), 32, _range(req, "x_min", "x_max", (-20, 20)), _range(req, "y_min", "y_max", (-20, 20)))
                            if pts:
                                pairs.append(f"({expression_names[i1]}, {expression_names[i2]}): " + ", ".join(_format_point_latex(pt) for pt in pts))
                                total += len(pts)
                                add_point_plots(pts, f"I {i1+1},{i2+1}-", f"{expression_names[i1]} × {expression_names[i2]}")
                    command_results.append("\\text{Pairwise intersections: }" + "\\\\ ".join(pairs) if total else "\\text{No real pairwise intersections found.}")

                elif canonical == "intersection":
                    i1, i2 = resolve_pair(args, "intersection(eq1, eq2)")
                    if not (is2d(i1) and is2d(i2)):
                        raise ValueError("Intersection is currently a 2D curve operation.")
                    command_results.append(selection_preface([
                        (expression_names[i1], i1, parsed_eqs[i1]),
                        (expression_names[i2], i2, parsed_eqs[i2]),
                    ]))
                    points = _intersection_points(current_eq(i1), current_eq(i2), 32, _range(req, "x_min", "x_max", (-20, 20)), _range(req, "y_min", "y_max", (-20, 20)))
                    if not points:
                        command_results.append("\\text{No real intersection found.}")
                    else:
                        add_point_plots(points, "I", f"Intersection {expression_names[i1]} × {expression_names[i2]}")
                        command_results.append("\\text{Intersections: }" + "\\\\ ".join(f"I_{{{n}}}={_format_point_latex(pt)}" for n, pt in enumerate(points, 1)))

                elif canonical == "locus":
                    i1, i2 = resolve_pair(args, "locus(eq1, eq2)")
                    if not (is2d(i1) and is2d(i2)): raise ValueError("Locus requires two valid 2D equations.")
                    command_results.append(selection_preface([
                        (expression_names[i1], i1, parsed_eqs[i1]),
                        (expression_names[i2], i2, parsed_eqs[i2]),
                    ]))
                    points = _intersection_points(current_eq(i1), current_eq(i2), 32, _range(req, "x_min", "x_max", (-20, 20)), _range(req, "y_min", "y_max", (-20, 20)))
                    if len(points) < 2:
                        command_results.append("\\text{At least two real intersection points are required for a fitted locus.}")
                    else:
                        fit = _fit_locus(points)
                        if not fit["success"]:
                            command_results.append(f"\\text{{Locus fit failed: }} {fit['message']}")
                        else:
                            command_results.append(f"\\text{{Fitted locus ({fit['orientation']}): }} {sp.latex(fit['equation'])}")
                            command_plots.append({"kind":"locus","x":fit["x"],"y":fit["y"],"z":[0.0]*len(fit["x"]),"mode":"lines","line":{"color":"#a855f7","width":3,"dash":"dash"},"name":f"Locus fit {expression_names[i1]} × {expression_names[i2]}"})
                            add_point_plots(points, "L", "Locus points")

                elif canonical == "convert":
                    i=resolve_one(args,"convert(eq1)")
                    command_results.append(selection_preface([(expression_names[i], i, parsed_eqs[i])]))
                    eq=parsed_eqs[i].subs(t_sym,req.t_val) if t_sym in parsed_eqs[i].free_symbols else parsed_eqs[i]
                    forms=_coordinate_forms(parsed_meta[i],eq)
                    if not forms: raise ValueError("No coordinate conversion is available for this expression.")
                    command_results.append("\\begin{aligned}"+"\\\\".join([f"\\text{{{k}: }} {v}" for k,v in forms.items()])+"\\end{aligned}")

                elif canonical == "intercepts":
                    i = resolve_one(args, "intercepts(eq1)")
                    command_results.append(selection_preface([(expression_names[i], i, parsed_eqs[i])]))
                    meta = parsed_meta[i]
                    eq = parsed_eqs[i].subs(t_sym, req.t_val) if t_sym in parsed_eqs[i].free_symbols else parsed_eqs[i]
                    # Reuse axis-point engine; for polar forms it reports physical x/y axis intersections.
                    points = _axis_points(eq, meta["is_implicit"], frozenset(meta.get("syms", set())), False)
                    if points:
                        add_point_plots(points, "A", f"Axis intercepts — {expression_names[i]}")
                        command_results.append("\\text{Axis intercepts: }" + "\\\\ ".join(_format_point_latex(pt) for pt in points))
                    else:
                        command_results.append("\\text{No finite axis intercepts found.}")

                elif canonical == "distance":
                    i1, i2 = resolve_pair(args, "distance(eq1, eq2)")
                    command_results.append(selection_preface([
                        (expression_names[i1], i1, parsed_eqs[i1]),
                        (expression_names[i2], i2, parsed_eqs[i2]),
                    ]))
                    distance, method = _curve_distance(
                        current_eq(i1), current_eq(i2),
                        _range(req, "x_min", "x_max", (-20, 20)),
                        _range(req, "y_min", "y_max", (-20, 20)),
                    )
                    if distance is None:
                        raise ValueError("Distance is supported for parallel lines and explicit y=f(x) curves.")
                    if method == "intersecting":
                        command_results.append("\\text{Minimum distance = } 0\\quad(\\text{curves intersect})")
                    elif method == "exact":
                        command_results.append(f"\\text{{Distance (exact) = }} {sp.N(distance, 10)}")
                    else:
                        command_results.append(f"\\text{{Distance (sampled approximation) = }} {sp.N(distance, 10)}")

                elif canonical in {"area_under", "area_closed", "polar_area", "arc_length", "polar_arc_length", "area_region", "area_between"}:
                    if canonical == "area_under":
                        if not (1 <= len(args) <= 3): raise ValueError("Usage: area_under(eq1, a, b)")
                        i = resolve_one(args[:1], "area_under(eq1, a, b)")
                        command_results.append(selection_preface([(expression_names[i], i, parsed_eqs[i])]))
                        lo = float(args[1]) if len(args) >= 2 else None; hi = float(args[2]) if len(args) >= 3 else None
                        meta = parsed_meta[i]; eq = parsed_eqs[i]
                        area, formula = _area_under(meta, eq, req, lo, hi)
                        if (meta.get("lhs_str") or "").strip().casefold() == "r" and len(args) < 3:
                            lo, hi = _axis(req, "theta_min", 0.0), _axis(req, "theta_max", 360.0)
                        kind, variable, function = _function_data(meta, eq, req)
                        if kind == "cartesian":
                            lo = float(args[1]); hi = float(args[2])
                            gx, gy = _eval_explicit_function(function, variable, lo, hi, 1800)
                            add_area_plot(gx, gy, f"Area under {expression_names[i]}")
                        elif kind == "polar":
                            aa = np.deg2rad(lo) if req.angle_unit == "degrees" else lo
                            bb = np.deg2rad(hi) if req.angle_unit == "degrees" else hi
                            gg = np.linspace(aa, bb, 1800)
                            ff = sp.lambdify(variable, function, modules=["numpy"])
                            rr = np.broadcast_to(_safe_array(ff(gg)), gg.shape); add_area_plot(rr*np.cos(gg), rr*np.sin(gg), f"Polar region {expression_names[i]}")
                        command_results.append(f"\\text{{Area over [{lo:g}, {hi:g}] = }} {sp.N(area, 10)}")

                    elif canonical == "area_between":
                        # Generalized: works for ANY mix of coordinate forms (polar,
                        # Cartesian, implicit, explicit) — no more "both curves must
                        # be the same explicit kind" restriction. Every curve is
                        # reduced to one or more numeric (x, y) sample arrays over
                        # [lo, hi], then compared with one shared routine:
                        #   - explicit y=f(x)            -> evaluate directly
                        #   - implicit F(x,y)=0           -> sp.solve(F, y) branches
                        #   - polar r=f(theta)            -> traced parametrically
                        # The polar case is NOT solved symbolically for y=f(x): for
                        # a general r=f(theta), that inversion has no closed form in
                        # the first place (it's not a shortcut being skipped, it's
                        # genuinely not algebraically available), so it is traced
                        # over theta and binned by x — the mathematically correct
                        # tool for this case, not just a fallback.
                        i1, i2 = resolve_pair(args[:2], "area_between(eq1, eq2, a, b)")
                        if len(args) != 4: raise ValueError("Usage: area_between(eq1, eq2, a, b)")
                        command_results.append(selection_preface([
                            (expression_names[i1], i1, parsed_eqs[i1]),
                            (expression_names[i2], i2, parsed_eqs[i2]),
                        ]))
                        lo, hi = float(args[2]), float(args[3])
                        x_s, y_s, theta_s = sp.symbols("x y theta")
                        grid = np.linspace(lo, hi, 4000)

                        def _xy_branches_of(idx):
                            """-> list of y-value arrays (each aligned with `grid`)."""
                            meta = parsed_meta[idx]
                            lhs_name = (meta.get("lhs_str") or "").strip().casefold()
                            rhs_expr = meta.get("rhs_expr")

                            if lhs_name == "r" and rhs_expr is not None:
                                rhs_e = rhs_expr.subs(t_sym, req.t_val) if t_sym in rhs_expr.free_symbols else rhs_expr
                                th = np.linspace(0.0, 2 * np.pi, 8000)
                                r_fn = sp.lambdify(theta_s, rhs_e, modules=["numpy"])
                                r_vals = _safe_array(np.broadcast_to(r_fn(th), th.shape))
                                xs_p = r_vals * np.cos(th); ys_p = r_vals * np.sin(th)
                                finite = np.isfinite(xs_p) & np.isfinite(ys_p)
                                xs_p, ys_p = xs_p[finite], ys_p[finite]
                                if xs_p.size == 0:
                                    return []
                                bins = np.clip(np.searchsorted(grid, xs_p), 0, len(grid) - 1)
                                y_lo = np.full(grid.shape, np.nan); y_hi = np.full(grid.shape, np.nan)
                                order = np.argsort(bins)
                                bins_sorted = bins[order]; ys_sorted = ys_p[order]
                                split_at = np.searchsorted(bins_sorted, np.arange(len(grid) + 1))
                                for b in range(len(grid)):
                                    chunk = ys_sorted[split_at[b]:split_at[b + 1]]
                                    if chunk.size:
                                        y_lo[b] = chunk.min(); y_hi[b] = chunk.max()
                                branches = [y_lo]
                                if not np.allclose(np.nan_to_num(y_hi), np.nan_to_num(y_lo)):
                                    branches.append(y_hi)
                                return branches

                            if lhs_name == "y" and rhs_expr is not None:
                                rhs_e = rhs_expr.subs(t_sym, req.t_val) if t_sym in rhs_expr.free_symbols else rhs_expr
                                return [np.broadcast_to(_safe_array(sp.lambdify(x_s, rhs_e, modules=["numpy"])(grid)), grid.shape)]

                            try:
                                sym_branches = [sp.simplify(b) for b in sp.solve(current_eq(idx), y_s) if x_s in getattr(b, "free_symbols", set())]
                            except Exception:
                                sym_branches = []
                            return [np.broadcast_to(_safe_array(sp.lambdify(x_s, b, modules=["numpy"])(grid)), grid.shape) for b in sym_branches]

                        b1, b2 = _xy_branches_of(i1), _xy_branches_of(i2)
                        if not b1 or not b2:
                            raise ValueError("Neither curve produced a resolvable y-value over that interval — check the equation and bounds.")

                        # At each x, pair every branch of curve 1 against every
                        # branch of curve 2 and take the closest pair — vectorized,
                        # no per-point Python/SymPy loop, and identical regardless
                        # of whether a branch came from an explicit function, an
                        # implicit solve, or a traced polar curve.
                        vals1 = np.stack(b1); vals2 = np.stack(b2)
                        diffs = np.abs(vals1[:, None, :] - vals2[None, :, :])
                        safe_diffs = np.where(np.isfinite(diffs), diffs, np.inf)
                        flat_idx = np.argmin(safe_diffs.reshape(-1, safe_diffs.shape[-1]), axis=0)
                        gaps = diffs.reshape(-1, diffs.shape[-1])[flat_idx, np.arange(len(grid))]
                        gaps = np.where(np.isfinite(gaps), gaps, np.nan)
                        b1_idx, b2_idx = np.unravel_index(flat_idx, (len(b1), len(b2)))
                        cols = np.arange(len(grid))
                        upper = np.maximum(vals1[b1_idx, cols], vals2[b2_idx, cols])
                        lower = np.minimum(vals1[b1_idx, cols], vals2[b2_idx, cols])
                        area = _numeric_integral(gaps, grid)
                        if len(b1) == 1:
                            add_area_plot(grid, vals1[0], f"{expression_names[i1]}")
                        if len(b2) == 1:
                            command_plots.append({"kind": "area-bound", "x": grid.tolist(), "y": vals2[0].tolist(), "z": [0.0] * len(grid), "mode": "lines", "line": {"color": "#f59e0b", "width": 2}, "name": expression_names[i2]})
                        command_plots.append({"kind": "area-between", "x": grid.tolist() + grid[::-1].tolist(), "y": upper.tolist() + lower[::-1].tolist(), "z": [0.0] * (2 * len(grid)), "mode": "lines", "line": {"color": "#60a5fa", "width": 1}, "fill": "toself", "fillcolor": "rgba(96,165,250,0.20)", "name": "Area between (closest branch pair)"})
                        command_results.append(f"\\text{{Area between over [{lo:g}, {hi:g}] = }} {sp.N(area, 10)}")

                    elif canonical == "area_region":
                        i1, i2 = resolve_pair(args, "area_region(eq1, eq2)")
                        command_results.append(selection_preface([
                            (expression_names[i1], i1, parsed_eqs[i1]),
                            (expression_names[i2], i2, parsed_eqs[i2]),
                        ]))
                        k1, v1, f1 = _function_data(parsed_meta[i1], parsed_eqs[i1], req)
                        k2, v2, f2 = _function_data(parsed_meta[i2], parsed_eqs[i2], req)
                        if k1 == "cartesian" and k2 == "cartesian" and f1 is not None and f2 is not None and v1 == sp.Symbol("x") and v2 == sp.Symbol("x"):
                            diff = sp.simplify(f1 - f2); roots = _real_solutions(diff, sp.Symbol("x"), 20)
                            if len(roots) < 2:
                                raise ValueError("Could not find at least two bounding intersections; use area_between(eq1,eq2,a,b).")
                            total_area = 0.0
                            for a,b in zip(roots[:-1], roots[1:]):
                                grid = np.linspace(a,b,1600)
                                va = np.broadcast_to(_safe_array(sp.lambdify(v1,f1,modules=["numpy"])(grid)), grid.shape)
                                vb = np.broadcast_to(_safe_array(sp.lambdify(v2,f2,modules=["numpy"])(grid)), grid.shape)
                                total_area += _numeric_integral(np.abs(va-vb), grid)
                            command_results.append(f"\\text{{Closed region area = }} {sp.N(total_area,10)}")
                            continue
                        if k1 == "polar" and k2 == "polar" and f1 is not None and f2 is not None:
                            diff = sp.simplify(f1-f2); roots = _real_solutions(diff, sp.Symbol("theta"), 20)
                            if len(roots) < 2:
                                raise ValueError("Could not find angular bounds; use area_between(eq1,eq2,theta1,theta2).")
                            total_area = 0.0
                            for a,b in zip(roots[:-1],roots[1:]):
                                gg=np.linspace(a,b,1600)
                                va=np.broadcast_to(_safe_array(sp.lambdify(v1,f1,modules=["numpy"])(gg)),gg.shape)
                                vb=np.broadcast_to(_safe_array(sp.lambdify(v2,f2,modules=["numpy"])(gg)),gg.shape)
                                total_area += 0.5*_numeric_integral(np.abs(va*va-vb*vb),gg)
                            command_results.append(f"\\text{{Closed polar region area = }} {sp.N(total_area,10)}")
                            continue
                        # Mixed implicit/explicit Cartesian curves: approximate the minor bounded region
                        # by resolving vertical branches and integrating the smallest vertical gap.
                        F1 = current_eq(i1); F2 = current_eq(i2); points = _intersection_points(F1,F2,32)
                        if len(points) < 2:
                            raise ValueError("Could not find bounding intersections.")
                        lo = min(point["x"] for point in points); hi = max(point["x"] for point in points)
                        x, y = sp.symbols("x y")
                        def branches_for(F, meta):
                            lhs_name=(meta.get("lhs_str") or "").strip().casefold()
                            if lhs_name == "y" and meta.get("rhs_expr") is not None:
                                rhs_expr=meta["rhs_expr"]; return [rhs_expr.subs(sp.Symbol("t"),req.t_val) if sp.Symbol("t") in rhs_expr.free_symbols else rhs_expr]
                            if not meta.get("is_implicit"): return []
                            try:
                                return [sp.simplify(item) for item in sp.solve(F,y) if x in item.free_symbols]
                            except Exception:
                                return []
                        b1=branches_for(F1,parsed_meta[i1]); b2=branches_for(F2,parsed_meta[i2])
                        if not b1 or not b2:
                            raise ValueError("Could not resolve vertical branches; use area_between(eq1,eq2,a,b).")
                        grid=np.linspace(lo,hi,2400); gaps=[]; upper=[]; lower=[]
                        for xv in grid:
                            vals1=[_as_real_number(branch.subs(x,xv)) for branch in b1]; vals2=[_as_real_number(branch.subs(x,xv)) for branch in b2]
                            vals1=[v for v in vals1 if v is not None and np.isfinite(v)]; vals2=[v for v in vals2 if v is not None and np.isfinite(v)]
                            if not vals1 or not vals2:
                                gaps.append(np.nan); upper.append(np.nan); lower.append(np.nan); continue
                            _, a_y, b_y = min((abs(a-b),a,b) for a in vals1 for b in vals2)
                            gaps.append(abs(a_y-b_y)); upper.append(max(a_y,b_y)); lower.append(min(a_y,b_y))
                        gaps=np.asarray(gaps,dtype=float); area=_numeric_integral(gaps,grid)
                        command_plots.append({"kind":"area-between","x":grid.tolist()+grid[::-1].tolist(),"y":upper+lower[::-1],"z":[0.0]*(2*len(grid)),"mode":"lines","line":{"color":"#60a5fa","width":1},"fill":"toself","fillcolor":"rgba(96,165,250,0.20)","name":"Minor bounded region"})
                        command_results.append(f"\\text{{Approx. minor bounded region area = }} {sp.N(area,10)}")

                    elif canonical == "area_closed":
                        i = resolve_one(args[:1], "area_closed(eq1, a, b)")
                        command_results.append(selection_preface([(expression_names[i], i, parsed_eqs[i])]))
                        lo = float(args[1]) if len(args) >= 2 else None; hi = float(args[2]) if len(args) >= 3 else None
                        meta=parsed_meta[i]; eq=parsed_eqs[i]
                        kind,var,fun=_function_data(meta,eq,req)
                        if kind == "polar":
                            area,_=_area_under(meta,eq,req,lo,hi); command_results.append(f"\\text{{Closed polar area = }} {sp.N(area,10)}")
                        elif meta.get("is_implicit"):
                            area=_closed_implicit_area(_analysis_eq(meta,eq,t_sym,req),lo,hi)
                            if area is None: raise ValueError("Could not resolve a closed implicit boundary automatically. Try area_closed(eq1, x_min, x_max).")
                            command_results.append(f"\\text{{Closed region area = }} {sp.N(area,10)}")
                        else:
                            raise ValueError("area_closed needs a polar r(theta) curve or an implicit closed Cartesian curve.")

                    elif canonical == "arc_length":
                        i=resolve_one(args[:1],"arc_length(eq1,a,b)")
                        command_results.append(selection_preface([(expression_names[i], i, parsed_eqs[i])]))
                        if len(args)!=3: raise ValueError("Usage: arc_length(eq1, a, b)")
                        lo,hi=float(args[1]),float(args[2]); meta=parsed_meta[i]; eq=parsed_eqs[i]; kind,var,fun=_function_data(meta,eq,req)
                        if kind!="cartesian" or var!=sp.Symbol("x"): raise ValueError("arc_length currently uses y=f(x).")
                        d=sp.diff(fun,var); grid=np.linspace(lo,hi,3500); fv=_safe_array(sp.lambdify(var,fun,modules=["numpy"])(grid)); dv=_safe_array(sp.lambdify(var,d,modules=["numpy"])(grid)); length=_numeric_integral(np.sqrt(1+dv*dv),grid); command_results.append(f"\\text{{Arc length = }} {sp.N(length,10)}")

                    elif canonical == "polar_arc_length":
                        i=resolve_one(args[:1],"polar_arc_length(eq1,theta1,theta2)")
                        command_results.append(selection_preface([(expression_names[i], i, parsed_eqs[i])]))
                        if len(args)!=3: raise ValueError("Usage: polar_arc_length(eq1, theta1, theta2)")
                        lo,hi=float(args[1]),float(args[2]); meta=parsed_meta[i]; eq=parsed_eqs[i]; kind,var,fun=_function_data(meta,eq,req)
                        if kind!="polar": raise ValueError("polar_arc_length requires r=f(theta).")
                        lo_r=np.deg2rad(lo) if req.angle_unit=="degrees" else lo; hi_r=np.deg2rad(hi) if req.angle_unit=="degrees" else hi; grid=np.linspace(lo_r,hi_r,3500); f=sp.lambdify(var,fun,modules=["numpy"]); rvals=_safe_array(f(grid)); d=sp.diff(fun,var); dr=_safe_array(sp.lambdify(var,d,modules=["numpy"])(grid)); length=_numeric_integral(np.sqrt(rvals*rvals+dr*dr),grid); command_results.append(f"\\text{{Polar arc length = }} {sp.N(length,10)}")

                    elif canonical == "polar_area":
                        i=resolve_one(args[:1],"polar_area(eq1,theta1,theta2)")
                        command_results.append(selection_preface([(expression_names[i], i, parsed_eqs[i])]))
                        lo=float(args[1]) if len(args)>=2 else _axis(req,"theta_min",0.0); hi=float(args[2]) if len(args)>=3 else _axis(req,"theta_max",360.0); area,_=_area_under(parsed_meta[i],parsed_eqs[i],req,lo,hi); command_results.append(f"\\text{{Polar area over [{lo:g}, {hi:g}] = }} {sp.N(area,10)}")

                elif canonical == "roots":
                    i=resolve_one(args,"roots(eq1)"); meta=parsed_meta[i]; equation=parsed_eqs[i].subs(t_sym,req.t_val) if t_sym in parsed_eqs[i].free_symbols else parsed_eqs[i]
                    if (meta.get("lhs_str") or "").strip().casefold()=="r":
                        rhs=meta.get("rhs_expr"); theta=sp.Symbol("theta"); roots=_real_solutions((rhs.subs(t_sym,req.t_val) if t_sym in rhs.free_symbols else rhs),theta,32); command_results.append(f"\\text{{Polar roots (r=0) of {expression_names[i]}: }} {sp.latex(roots)}")
                    else:
                        x_symbol,y_symbol=sp.symbols("x y"); target=sp.simplify(_analysis_eq(meta,equation,t_sym,req).subs(y_symbol,0)); roots=[root for root in sp.solve(target,x_symbol) if _as_real_number(root) is not None]; command_results.append(f"\\text{{Roots of {expression_names[i]} (x-intercepts): }} {sp.latex(roots)}")

                elif canonical == "derive":
                    if not (1 <= len(args) <= 2): raise ValueError("Usage: derive(eq1[, order])")
                    i=resolve_one(args[:1],"derive(eq1[,order])"); order=int(args[1]) if len(args)==2 else 1
                    if order < 1 or order > 8: raise ValueError("Derivative order must be between 1 and 8.")
                    meta=parsed_meta[i]; eq=parsed_eqs[i].subs(t_sym,req.t_val) if t_sym in parsed_eqs[i].free_symbols else parsed_eqs[i]
                    kind,var,fun=_function_data(meta,eq,req)
                    if kind in {"cartesian","polar"}:
                        derivative=sp.diff(fun,var,order)
                        command_results.append(f"\\frac{{d^{{{order}}}}}{{d {var.name}^{{{order}}}}}({expression_names[i]}) = {sp.latex(derivative)}")
                    else: raise ValueError("Could not identify a single differentiable function variable.")

                elif canonical == "integrate":
                    i=resolve_one(args,"integrate(eq1)"); meta=parsed_meta[i]; eq=parsed_eqs[i].subs(t_sym,req.t_val) if t_sym in parsed_eqs[i].free_symbols else parsed_eqs[i]
                    kind,var,fun=_function_data(meta,eq,req)
                    target=fun if kind in {"cartesian","polar"} else eq; variable=var if kind in {"cartesian","polar"} else sp.Symbol("x"); integral=sp.integrate(target,variable); command_results.append(f"\\int {expression_names[i]}\\,d{variable.name} = {sp.latex(integral)} + C")

                elif canonical == "evaluate":
                    if len(args)!=2: raise ValueError("Usage: evaluate(eq1, value)")
                    i=resolve_one(args[:1],"evaluate(eq1,value)"); value=float(args[1]); meta=parsed_meta[i]; eq=parsed_eqs[i]; kind,var,fun=_function_data(meta,eq,req)
                    if kind is None: raise ValueError("evaluate needs an explicit function.")
                    result=_as_real_number(fun.subs(var,value)); if_result = result
                    if result is None: raise ValueError("The function is not finite at that value.")
                    command_results.append(f"\\text{{{expression_names[i]}({value}) = }} {sp.N(result,10)}")

                elif canonical in {"tangent","normal"}:
                    if len(args)!=2: raise ValueError(f"Usage: {canonical}(eq1, x0)")
                    i=resolve_one(args[:1],f"{canonical}(eq1,x0)"); x0=float(args[1]); meta=parsed_meta[i]; eq=parsed_eqs[i]; kind,var,fun=_function_data(meta,eq,req)
                    if kind!="cartesian" or var!=sp.Symbol("x"): raise ValueError(f"{canonical} currently uses y=f(x).")
                    y0=_as_real_number(fun.subs(var,x0)); slope=_as_real_number(sp.diff(fun,var).subs(var,x0))
                    if y0 is None or slope is None: raise ValueError("The function is not finite/differentiable there.")
                    if canonical=="normal": slope=0.0 if abs(slope)>1e100 else (-1.0/slope if abs(slope)>1e-12 else float("inf"))
                    if np.isfinite(slope):
                        intercept=y0-slope*x0; expr=slope*sp.Symbol("x")+intercept; command_results.append(f"\\text{{{canonical.title()}: }} y = {sp.latex(sp.N(expr,8))}")
                        grid=np.linspace(_axis(req,"x_min",-20),_axis(req,"x_max",20),800); yy=slope*grid+intercept; command_plots.append({"kind":"line","x":grid.tolist(),"y":yy.tolist(),"z":[0.0]*len(grid),"mode":"lines","line":{"color":"#22d3ee","width":2,"dash":"dot"},"name":f"{canonical.title()} at x={x0:g}"})
                    else:
                        command_results.append(f"\\text{{{canonical.title()}: }} x = {sp.N(x0,8)}")

                elif canonical == "extrema":
                    i=resolve_one(args[:1],"extrema(eq1,a,b)")
                    meta=parsed_meta[i]; eq=parsed_eqs[i]; kind,var,fun=_function_data(meta,eq,req)
                    if kind!="cartesian" or var!=sp.Symbol("x"): raise ValueError("extrema currently uses y=f(x).")
                    lo=float(args[1]) if len(args)>=2 else _axis(req,"x_min",-20); hi=float(args[2]) if len(args)>=3 else _axis(req,"x_max",20); d=sp.diff(fun,var); roots=_real_solutions(d,var,32); roots=[v for v in roots if lo-1e-9<=v<=hi+1e-9]
                    vals=[]
                    for xv in roots:
                        yv=_as_real_number(fun.subs(var,xv));
                        if yv is not None: vals.append((xv,yv))
                    if vals: add_point_plots([{"x":a,"y":b,"z":0.0} for a,b in vals],"E","Extrema"); command_results.append("\\text{Stationary points: }"+"\\\\ ".join(f"({sp.N(a,7)},{sp.N(b,7)})" for a,b in vals))
                    else: command_results.append("\\text{No real stationary points in the requested interval.}")

                elif canonical == "volume_under":
                    if len(args) not in {1,5}: raise ValueError("Usage: volume_under(surface[, xmin, xmax, ymin, ymax])")
                    i=resolve_one(args[:1],"volume_under(surface,xmin,xmax,ymin,ymax)"); meta=parsed_meta[i]; fn_expr=_surface_function(meta,parsed_eqs[i]);
                    if fn_expr is None: raise ValueError("volume_under requires z=f(x,y).")
                    command_results.append(selection_preface([(expression_names[i], i, parsed_eqs[i])]))
                    x0=float(args[1]) if len(args)==5 else _axis(req,"x_min",-20); x1=float(args[2]) if len(args)==5 else _axis(req,"x_max",20); y0=float(args[3]) if len(args)==5 else _axis(req,"y_min",-20); y1=float(args[4]) if len(args)==5 else _axis(req,"y_max",20); n=220; gx=np.linspace(x0,x1,n); gy=np.linspace(y0,y1,n); X,Y=np.meshgrid(gx,gy); f=sp.lambdify((sp.Symbol("x"),sp.Symbol("y")),fn_expr.subs(sp.Symbol("t"),req.t_val),modules=["numpy"]); V=np.nan_to_num(_safe_array(f(X,Y)),nan=0.0,posinf=0.0,neginf=0.0); volume=float(np.trapezoid(np.trapezoid(V,gy,axis=0),gx)); command_results.append(f"\\text{{Signed volume over }} [{x0:g},{x1:g}]\\times[{y0:g},{y1:g}] \\text{{ = }} {sp.N(volume,10)}")

                elif canonical == "surface_area":
                    if len(args) not in {1,5}: raise ValueError("Usage: surface_area(surface[, xmin, xmax, ymin, ymax])")
                    i=resolve_one(args[:1],"surface_area(surface,xmin,xmax,ymin,ymax)"); meta=parsed_meta[i]; fn_expr=_surface_function(meta,parsed_eqs[i]);
                    if fn_expr is None: raise ValueError("surface_area requires z=f(x,y).")
                    command_results.append(selection_preface([(expression_names[i], i, parsed_eqs[i])]))
                    x0=float(args[1]) if len(args)==5 else _axis(req,"x_min",-20); x1=float(args[2]) if len(args)==5 else _axis(req,"x_max",20); y0=float(args[3]) if len(args)==5 else _axis(req,"y_min",-20); y1=float(args[4]) if len(args)==5 else _axis(req,"y_max",20); gx=np.linspace(x0,x1,180); gy=np.linspace(y0,y1,180); X,Y=np.meshgrid(gx,gy); sx=sp.diff(fn_expr,sp.Symbol("x")); sy=sp.diff(fn_expr,sp.Symbol("y")); fx=sp.lambdify((sp.Symbol("x"),sp.Symbol("y")),sx,modules=["numpy"]); fy=sp.lambdify((sp.Symbol("x"),sp.Symbol("y")),sy,modules=["numpy"]); integrand=np.sqrt(1+np.nan_to_num(_safe_array(fx(X,Y)),nan=0.0,posinf=0.0,neginf=0.0)**2+np.nan_to_num(_safe_array(fy(X,Y)),nan=0.0,posinf=0.0,neginf=0.0)**2); area=float(np.trapezoid(np.trapezoid(integrand,gy,axis=0),gx)); command_results.append(f"\\text{{Surface area over }} [{x0:g},{x1:g}]\\times[{y0:g},{y1:g}] \\text{{ = }} {sp.N(area,10)}")

                elif canonical == "volume_between":
                    # Generalizes volume_under from "one surface vs. implicit z=0" to
                    # ANY two surfaces (or a surface vs. a named plane like z=0, z=3).
                    # Still exact-as-possible: both sides are solved for z=f(x,y) and
                    # integrated via a trapezoidal double integral, not Monte Carlo.
                    if not (2 <= len(args) <= 6):
                        raise ValueError("Usage: volume_between(A, B[, xmin, xmax, ymin, ymax])  — A, B: surface names or planes like z=0")
                    hA, labelA, idxA = resolve_ref_to_surface_height(args[0])
                    hB, labelB, idxB = resolve_ref_to_surface_height(args[1])
                    if hA is None or hB is None:
                        raise ValueError("Both objects must be expressible as z=f(x,y) (a surface) or a constant z-plane, e.g. z=0.")
                    command_results.append(selection_preface([
                        (labelA, idxA, sp.Eq(sp.Symbol("z"), hA)),
                        (labelB, idxB, sp.Eq(sp.Symbol("z"), hB)),
                    ]))
                    x0 = float(args[2]) if len(args) >= 3 else _axis(req, "x_min", -20)
                    x1 = float(args[3]) if len(args) >= 4 else _axis(req, "x_max", 20)
                    y0 = float(args[4]) if len(args) >= 5 else _axis(req, "y_min", -20)
                    y1 = float(args[5]) if len(args) >= 6 else _axis(req, "y_max", 20)
                    n = 220
                    gx = np.linspace(x0, x1, n); gy = np.linspace(y0, y1, n)
                    X, Y = np.meshgrid(gx, gy)
                    x_s, y_s = sp.symbols("x y")
                    hA_t = hA.subs(t_sym, req.t_val) if t_sym in getattr(hA, "free_symbols", set()) else hA
                    hB_t = hB.subs(t_sym, req.t_val) if t_sym in getattr(hB, "free_symbols", set()) else hB
                    fA = sp.lambdify((x_s, y_s), hA_t, modules=["numpy"])
                    fB = sp.lambdify((x_s, y_s), hB_t, modules=["numpy"])
                    ZA = np.nan_to_num(np.broadcast_to(_safe_array(fA(X, Y)), X.shape), nan=0.0)
                    ZB = np.nan_to_num(np.broadcast_to(_safe_array(fB(X, Y)), X.shape), nan=0.0)
                    gap = np.abs(ZA - ZB)
                    volume = float(np.trapezoid(np.trapezoid(gap, gy, axis=0), gx))
                    # Draw both selected surfaces (translucent, distinct from the main
                    # plot's own surfaces) so it's visually obvious which two bounded
                    # the computed volume, not just their names in the text result.
                    command_plots.append({"kind": "selected-surface", "type": "surface3d_command", "x": gx.tolist(), "y": gy.tolist(), "z": ZA.tolist(), "name": f"{labelA} (selected)", "opacity": 0.55, "colorscale": [[0, "#fbbf24"], [1, "#fbbf24"]]})
                    command_plots.append({"kind": "selected-surface", "type": "surface3d_command", "x": gx.tolist(), "y": gy.tolist(), "z": ZB.tolist(), "name": f"{labelB} (selected)", "opacity": 0.55, "colorscale": [[0, "#38bdf8"], [1, "#38bdf8"]]})
                    command_results.append(f"\\text{{Volume between over }} [{x0:g},{x1:g}]\\times[{y0:g},{y1:g}] \\text{{ = }} {sp.N(volume, 10)}")

                elif canonical == "volume_intersect":
                    # Generalized volume of the region common to 2-3 arbitrary implicit
                    # objects (surfaces, planes, or solids), using the standard sign
                    # convention that F(x,y,z)=0 is the boundary and F(x,y,z)<=0 is
                    # "inside" (matches how implicit equations are normally written,
                    # e.g. a sphere x^2+y^2+z^2-9=0 is inside for x^2+y^2+z^2-9<=0).
                    # This does NOT require any object to be solvable for z, which is
                    # exactly the case that symbolic/explicit integration cannot cover
                    # in general. It trades exactness for generality: the result is a
                    # Monte Carlo estimate, and is reported as such.
                    if not (2 <= len(args) <= 3):
                        raise ValueError("Usage: volume_intersect(A, B[, C])  — A,B,C: expression names or plane refs like z=0, x=3")
                    eqs, labels, idxs = [], [], []
                    for ref in args:
                        eq_expr, label, idx = resolve_ref_to_eq(ref)
                        if eq_expr is None:
                            raise ValueError(f"Could not resolve '{ref}' to an equation, plane, or named expression.")
                        eqs.append(eq_expr); labels.append(label); idxs.append(idx)
                    command_results.append(selection_preface(list(zip(labels, idxs, eqs))))
                    x0, x1 = _range(req, "x_min", "x_max", (-20, 20))
                    y0, y1 = _range(req, "y_min", "y_max", (-20, 20))
                    z0, z1 = _range(req, "z_min", "z_max", (-20, 20))
                    n = 60000
                    rng = np.random.default_rng(0)
                    xs = rng.uniform(x0, x1, n); ys = rng.uniform(y0, y1, n); zs = rng.uniform(z0, z1, n)
                    x_s, y_s, z_s = sp.symbols("x y z")
                    inside = np.ones(n, dtype=bool)
                    for eq_expr in eqs:
                        eq_t = eq_expr.subs(t_sym, req.t_val) if t_sym in eq_expr.free_symbols else eq_expr
                        fn = sp.lambdify((x_s, y_s, z_s), eq_t, modules=["numpy"])
                        vals = _safe_array(np.broadcast_to(fn(xs, ys, zs), xs.shape))
                        inside &= np.isfinite(vals) & (vals <= 0)
                    box_volume = (x1 - x0) * (y1 - y0) * (z1 - z0)
                    fraction = float(np.count_nonzero(inside)) / n
                    volume = box_volume * fraction
                    command_results.append(
                        f"\\text{{Volume}}(" + ",".join(labels) + f") \\approx {sp.N(volume, 6)} "
                        f"\\quad(\\text{{Monte Carlo, N={n}, box }}[{x0:g},{x1:g}]\\times[{y0:g},{y1:g}]\\times[{z0:g},{z1:g}])"
                    )
                    hit_idx = np.where(inside)[0]
                    if hit_idx.size:
                        sample = hit_idx[:1500]
                        command_plots.append({
                            "kind": "volume-points",
                            "x": xs[sample].tolist(), "y": ys[sample].tolist(), "z": zs[sample].tolist(),
                            "mode": "markers",
                            "marker": {"size": 2, "color": "#22d3ee", "opacity": 0.35},
                            "name": "Intersection region (sample)",
                        })
                    else:
                        command_results.append("\\text{The sampled bounding box found no shared region — check the axis ranges.}")

                elif canonical == "help":
                    if args: raise ValueError("Usage: help()")
                    command_results.append("\\begin{aligned}\\text{2D: }&intersection(A,B),\\ intersectall(),\\ locus(A,B),\\ intercepts(A)\\\\&area_under(A,a,b),\\ area_between(A,B,a,b),\\ area_region(A,B)\\\\&area_closed(A[,a,b]),\\ polar_area(A[,a,b]),\\ arc_length(A,a,b),\\ polar_arc_length(A,a,b)\\\\&roots(A),\\ derive(A[,n]),\\ integrate(A),\\ evaluate(A,x),\\ tangent(A,x_0),\\ normal(A,x_0),\\ extrema(A[,a,b])\\\\\\text{3D: }&volume_under(S,x_1,x_2,y_1,y_2),\\ surface_area(S,x_1,x_2,y_1,y_2)\\\\&volume_between(A,B[,x_1,x_2,y_1,y_2])\\ \\text{(2 surfaces/planes, exact grid integral)}\\\\&volume_intersect(A,B[,C])\\ \\text{(2-3 arbitrary solids/planes, Monte Carlo)}\\\\\\text{Refs A,B,C may be an equation name/number OR a raw plane, e.g. }&z{=}0,\\ x{=}3,\\ \\theta{=}45\\end{aligned}")

                else:
                    command_results.append("\\text{Unknown command. Use help() for the full command list.}")

            except Exception as exc:
                command_results.append(f"\\text{{Command failed: }} {str(exc)}")

        # coordinate_forms is intentionally NOT auto-computed here anymore (see
        # convert(Name) above) — it used to run a SymPy simplify pass across 6
        # coordinate systems for every visible curve on every single plot request,
        # regardless of whether anyone had the panel open. That was the dominant
        # cost in this endpoint. The frontend now fetches it on demand, the same
        # way any other analysis command works.
        coordinate_forms = []
        response = {
            "plots": plots,
            "special_points": special_points,
            "command_results": command_results,
            "command_plots": command_plots,
            "command_selection": command_selection,
            "coordinate_forms": coordinate_forms,
            "suggestions": _smart_suggestions(parsed_meta, parsed_eqs, expression_names, req),
            "error": None,
            "coordinate_system": req.coordinate_system,
            "suggested_view": _suggested_view(plots, req),
        }
        return _json_safe(response)
    except Exception as exc:
        return {"error": f"Graphing Error: {exc}"}