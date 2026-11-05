from __future__ import annotations

import asyncio
import json
import math
import os
import re
import tempfile
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field, field_validator

router = APIRouter()


_XY_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_.])[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
_XY_LABEL_RE = re.compile(r"\b(?:x|y|time|value|index)\b\s*=?\s*", re.IGNORECASE)


def _normalize_xy_pairs(text: str) -> str:
    """Validate and normalise pasted curve rows before passing them to C++."""
    points: list[tuple[float, float]] = []
    for line_number, raw_line in enumerate(text.lstrip("\ufeff").splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith("//"):
            continue
        numeric_text = _XY_LABEL_RE.sub(" ", line)
        # A remaining word indicates a heading/annotation, not a measurement.
        if re.search(r"[A-DF-Za-df-z_]", numeric_text):
            if not points:
                continue
            raise ValueError(f"Row {line_number}: expected exactly two numeric values (x and y).")
        tokens = _XY_NUMBER_RE.findall(numeric_text)
        if not tokens:
            if not points:
                continue
            raise ValueError(f"Row {line_number}: no numeric x-y pair was found.")
        if len(tokens) != 2:
            raise ValueError(f"Row {line_number}: expected exactly two numeric values, found {len(tokens)}.")
        x_value, y_value = map(float, tokens)
        if not (math.isfinite(x_value) and math.isfinite(y_value)):
            raise ValueError(f"Row {line_number}: x and y must be finite numbers.")
        points.append((x_value, y_value))

    if len(points) < 3:
        raise ValueError("Enter at least three valid x-y pairs.")
    ordered_x = sorted(x for x, _ in points)
    for left, right in zip(ordered_x, ordered_x[1:]):
        if math.isclose(left, right, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("Duplicate x values are not supported. Each x value must be unique.")

    return "\n".join(f"{x:.17g} {y:.17g}" for x, y in points)

def _tol_arg(tolerance: float | None) -> str:
    # Keep the native protocol backward compatible: -1 means adaptive/automatic.
    return "-1" if tolerance is None else str(tolerance)


# Resolve the native executable without forcing a particular IDE/CMake output name.
# The C++ source still remains the actual engine; this file only transports data.
#
# Text-based endpoints remain backward compatible.  Binary spreadsheet uploads
# use /file-upload and are written to a temporary file so the native C++ loader
# can inspect XLSX/XLSM/XLS/ODS directly; XLS falls back through LibreOffice
# when the Python xlrd reader is unavailable.
def _resolve_binary() -> str | None:
    configured = os.getenv("ANALYSER_BIN")
    module_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.dirname(module_dir)
    project_dir = os.path.dirname(backend_dir)

    candidates: list[str | None] = [configured]
    search_dirs = [module_dir, backend_dir, project_dir]
    for d in search_dirs:
        candidates.extend([
            os.path.join(d, "analyser_bin.exe"),
            os.path.join(d, "analyser_bin"),
            os.path.join(d, "analyser.exe"),
            os.path.join(d, "analyser"),
            os.path.join(d, "sequence_engine_final.exe"),
            os.path.join(d, "sequence_engine_final"),
        ])
        candidates.extend([
            os.path.join(d, "cmake-build-debug", "analyser.exe"),
            os.path.join(d, "cmake-build-debug", "analyser"),
            os.path.join(d, "build", "analyser.exe"),
            os.path.join(d, "build", "analyser"),
        ])

    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


class SequenceRequest(BaseModel):
    data: str = Field(min_length=1)
    horizon: int = Field(default=4, ge=1, le=1000)
    # None means "auto"; the C++ engine computes adaptive tolerance.
    tolerance: float | None = Field(default=None, ge=-1.0, le=100.0)


class CurveRequest(BaseModel):
    pairs: str = Field(min_length=1)
    horizon: int = Field(default=4, ge=1, le=1000)
    tolerance: float | None = Field(default=None, ge=-1.0, le=100.0, allow_inf_nan=False)
    step: float = Field(default=1.0, gt=0.0, le=1_000_000.0, allow_inf_nan=False)

    @field_validator("pairs")
    @classmethod
    def validate_pairs(cls, value: str) -> str:
        return _normalize_xy_pairs(value)


class MultivariateRequest(BaseModel):
    rows: str = Field(min_length=1)
    variables: int = Field(ge=2, le=64)
    horizon: int = Field(default=3, ge=1, le=1000)
    tolerance: float | None = Field(default=None, ge=-1.0, le=100.0)


class CoordinatesRequest(BaseModel):
    points: str = Field(min_length=1)
    mode: int = Field(ge=1, le=5)
    response: int = Field(default=1, ge=0, le=2)
    angle_unit: int = Field(default=0, ge=0, le=1)  # 0 radians, 1 degrees
    horizon: int = Field(default=4, ge=1, le=1000)
    tolerance: float | None = Field(default=None, ge=-1.0, le=100.0)


class FileAnalysisRequest(BaseModel):
    text: str = Field(min_length=1)
    mode: int = Field(ge=0, le=6)
    # A 1D sequence legitimately contains a single selected variable.
    variables: int = Field(default=3, ge=1, le=64)
    angle_unit: int = Field(default=0, ge=0, le=1)
    horizon: int = Field(default=4, ge=1, le=1000)
    tolerance: float | None = Field(default=None, ge=-1.0, le=100.0)


class FamilyFitRequest(BaseModel):
    pairs: str = Field(min_length=1)
    family: str = Field(min_length=1)
    degree: int = Field(default=1, ge=1, le=18)
    param: float = Field(default=0.0, allow_inf_nan=False)
    param2: float = Field(default=0.0, allow_inf_nan=False)
    horizon: int = Field(default=4, ge=1, le=1000)
    step: float = Field(default=1.0, gt=0.0, le=1_000_000.0, allow_inf_nan=False)

    @field_validator("pairs")
    @classmethod
    def validate_pairs(cls, value: str) -> str:
        return _normalize_xy_pairs(value)


class CipherRequest(BaseModel):
    operation: str = Field(min_length=1)
    lines: list[str] = Field(default_factory=list)


class RandomGenerateRequest(BaseModel):
    generator: int = Field(ge=1, le=4)
    seed: int = Field(ge=0)
    count: int = Field(ge=1, le=1_000_000)
    modulus: int | None = Field(default=None, ge=2)
    multiplier: int | None = Field(default=None, ge=0)
    increment: int | None = Field(default=None, ge=0)


class PuzzleCubeRequest(BaseModel):
    size: int = Field(default=3, ge=2, le=4)
    scramble: str = ""
    depth: int = Field(default=8, ge=0, le=24)
    node_limit: int = Field(default=3_000_000, ge=1, le=10_000_000)


class PuzzleGenericRequest(BaseModel):
    state_size: int = Field(default=8, ge=2, le=128)
    start: list[int]
    goal: list[int]
    depth: int = Field(default=10, ge=0, le=24)
    node_limit: int = Field(default=3_000_000, ge=1, le=10_000_000)
    # Each move is {name, group, perm} and is converted into the compact
    # line protocol understood by the C++ engine.
    moves: list[dict[str, Any]]


async def _run_engine(
    args: list[str],
    stdin_text: str = "",
    timeout: float = 180.0,
) -> dict[str, Any]:
    binary = _resolve_binary()
    if binary is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "Native analyser executable not found. Build analyser.cpp with "
                "g++ -O3 -std=c++17 analyser.cpp -o analyser_bin "
                "or set ANALYSER_BIN to the executable path."
            ),
        )

    try:
        proc = await asyncio.create_subprocess_exec(
            binary,
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to start native C++ engine: {exc}",
        ) from exc

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(stdin_text.encode("utf-8")),
            timeout=timeout,
        )
    except asyncio.TimeoutError as exc:
        if proc.returncode is None:
            proc.kill()
        await proc.wait()
        raise HTTPException(
            status_code=504,
            detail=f"Native engine exceeded the {int(timeout)} second execution limit.",
        ) from exc

    stdout_text = stdout.decode("utf-8", errors="replace").strip()
    stderr_text = stderr.decode("utf-8", errors="replace").strip()

    if not stdout_text:
        detail = stderr_text or f"Native engine exited with code {proc.returncode}."
        raise HTTPException(status_code=500, detail=detail)

    try:
        result = json.loads(stdout_text)
    except json.JSONDecodeError as exc:
        detail = stderr_text or stdout_text[:4000]
        raise HTTPException(
            status_code=500,
            detail=f"Native engine returned invalid JSON: {detail}",
        ) from exc

    if proc.returncode not in (0, None):
        detail = result.get("error") if isinstance(result, dict) else None
        raise HTTPException(
            status_code=400,
            detail=detail or stderr_text or f"Native engine exited with code {proc.returncode}.",
        )

    if not isinstance(result, dict):
        raise HTTPException(status_code=500, detail="Native engine returned an unexpected JSON value.")

    if result.get("ok") is False:
        raise HTTPException(
            status_code=400,
            detail=result.get("error", "Native engine rejected the request."),
        )

    return result


@router.post("/auto")
async def analyze_auto(req: SequenceRequest):
    """Automatically infer sequence vs x-y vs multivariate layout in the native engine."""
    return await _run_engine(
        ["--web-auto", str(req.horizon), _tol_arg(req.tolerance)],
        req.data,
    )


@router.post("/analyze")
async def analyze_sequence(req: SequenceRequest):
    return await _run_engine(
        ["--web-sequence", str(req.horizon), _tol_arg(req.tolerance)],
        req.data,
    )


@router.post("/curve")
async def analyze_curve(req: CurveRequest):
    return await _run_engine(
        ["--web-curve", str(req.horizon), _tol_arg(req.tolerance), str(req.step)],
        req.pairs,
    )


@router.post("/families")
async def compare_curve_families(req: CurveRequest):
    return await _run_engine(["--web-families"], req.pairs)


@router.post("/family")
async def fit_curve_family(req: FamilyFitRequest):
    return await _run_engine(
        [
            "--web-family",
            req.family,
            str(req.degree),
            str(req.param),
            str(req.param2),
            str(req.horizon),
            str(req.step),
        ],
        req.pairs,
    )


@router.post("/multivariate")
async def analyze_multivariate(req: MultivariateRequest):
    return await _run_engine(
        ["--web-multivariate", str(req.variables), str(req.horizon), _tol_arg(req.tolerance)],
        req.rows,
    )


@router.post("/coordinates")
async def analyze_coordinates(req: CoordinatesRequest):
    return await _run_engine(
        [
            "--web-coordinates",
            str(req.mode),
            str(req.response),
            str(req.angle_unit),
            str(req.horizon),
            _tol_arg(req.tolerance),
        ],
        req.points,
    )


@router.post("/file-upload")
async def analyze_file_upload(
    file: UploadFile = File(...),
    mode: int = 0,
    variables: int = 3,
    angle_unit: int = 0,
    horizon: int = 4,
    tolerance: float | None = None,
):
    """Accept a real binary spreadsheet/text upload without forcing the browser to parse it."""
    filename = os.path.basename(file.filename or "upload.dat")
    suffix = os.path.splitext(filename)[1].lower()
    allowed = {".txt", ".csv", ".tsv", ".xlsx", ".xlsm", ".xls", ".xlx", ".xlsb", ".ods"}
    if suffix not in allowed:
        raise HTTPException(status_code=422, detail=f"Unsupported file type: {suffix or '<none>'}.")
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")
    if len(payload) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Uploaded file exceeds the 50 MiB limit.")

    fd, temp_path = tempfile.mkstemp(prefix="analyser-upload-", suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(payload)
        return await _run_engine(
            [
                "--web-file-path",
                temp_path,
                str(mode),
                str(variables),
                str(angle_unit),
                str(horizon),
                _tol_arg(tolerance),
            ]
        )
    finally:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass


@router.post("/upload")
async def upload_file_alias(
    file: UploadFile = File(...),
    mode: int = 0,
    variables: int = 3,
    angle_unit: int = 0,
    horizon: int = 4,
    tolerance: float | None = None,
):
    return await analyze_file_upload(file, mode, variables, angle_unit, horizon, tolerance)


@router.post("/file")
async def analyze_file(req: FileAnalysisRequest):
    return await _run_engine(
        [
            "--web-file",
            str(req.mode),
            str(req.variables),
            str(req.angle_unit),
            str(req.horizon),
            _tol_arg(req.tolerance),
        ],
        req.text,
    )


@router.post("/stats")
async def stats(req: dict[str, str]):
    text = req.get("data", "")
    if not text.strip():
        raise HTTPException(status_code=422, detail="No data supplied.")
    return await _run_engine(["--web-stats"], text)


@router.post("/cipher")
async def cipher(req: CipherRequest):
    payload = "\n".join(req.lines)
    return await _run_engine(["--web-cipher", req.operation], payload)


@router.post("/random/analyze")
async def random_analyze(req: dict[str, str]):
    text = req.get("data", "")
    if not text.strip():
        raise HTTPException(status_code=422, detail="No data supplied.")
    return await _run_engine(["--web-prng", "analyze"], text)


@router.post("/random/generate")
async def random_generate(req: RandomGenerateRequest):
    args = ["--web-prng", "generate", str(req.generator), str(req.seed), str(req.count)]
    if req.generator == 2:
        if req.modulus is None or req.multiplier is None or req.increment is None:
            raise HTTPException(status_code=422, detail="LCG requires modulus, multiplier and increment.")
        args.extend([str(req.modulus), str(req.multiplier), str(req.increment)])
    return await _run_engine(args)


@router.post("/selftest")
async def selftest():
    return await _run_engine(["--web-selftest"])


@router.post("/puzzle/cube")
async def puzzle_cube(req: PuzzleCubeRequest):
    return await _run_engine(
        ["--web-puzzle-cube", str(req.size), str(req.depth), str(req.node_limit)],
        req.scramble,
    )


@router.post("/puzzle/generic")
async def puzzle_generic(req: PuzzleGenericRequest):
    if len(req.start) != req.state_size or len(req.goal) != req.state_size:
        raise HTTPException(status_code=422, detail="Start and goal must match state_size.")

    lines = [
        " ".join(str(v) for v in req.start),
        " ".join(str(v) for v in req.goal),
        str(len(req.moves)),
    ]

    for move in req.moves:
        name = str(move.get("name", "move"))
        group = int(move.get("group", -1))
        perm = move.get("perm")
        if not isinstance(perm, list) or len(perm) != req.state_size:
            raise HTTPException(status_code=422, detail=f"Move '{name}' has an invalid permutation.")
        lines.append(f"{name}|{group}|" + " ".join(str(v) for v in perm))

    return await _run_engine(
        ["--web-puzzle-generic", str(req.state_size), str(req.depth), str(req.node_limit)],
        "\n".join(lines),
    )