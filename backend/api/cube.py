"""
Cube solver API endpoints.
"""

from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import time

from core.algorithms.cube import (
    CubeState, Algorithm, generate_scramble, solve_cube, solve_cube_async,
    solve_scramble_async, apply_moves, generate_solution_states, FACE_NAMES, FACE_COLORS
)


router = APIRouter(tags=["Cube Solver"])


class ScrambleRequest(BaseModel):
    size: int = Field(ge=2, le=16, default=3)
    length: int = Field(ge=1, le=100, default=20)
    seed: Optional[int] = None


class ScrambleResponse(BaseModel):
    size: int
    scramble: str
    moves: List[str]
    facelets: str  # 6*N*N chars '0'-'5'
    state: List[int]


class SolveRequest(BaseModel):
    size: int = Field(ge=2, le=16)
    facelets: str  # 6*N*N chars '0'-'5'
    algorithm: Literal["auto", "kociemba", "thistlethwaite", "korf", "native_ida", "pocket"] = "auto"
    depth: int = Field(ge=1, le=30, default=20)
    node_limit: int = Field(ge=1000, le=20000000, default=5000000)


class SolveScrambleRequest(BaseModel):
    size: int = Field(ge=2, le=16)
    scramble: str  # space-separated moves like "R U R' U'"
    algorithm: Literal["auto", "kociemba", "thistlethwaite", "korf", "native_ida", "pocket"] = "auto"
    depth: int = Field(ge=1, le=30, default=20)
    node_limit: int = Field(ge=1000, le=20000000, default=5000000)


class MoveApplyRequest(BaseModel):
    size: int = Field(ge=2, le=16)
    facelets: str
    moves: List[str]


class ValidateRequest(BaseModel):
    size: int = Field(ge=2, le=16)
    facelets: str


class SolveResponse(BaseModel):
    algorithm: str
    moves: List[str]
    move_count: int
    solved: bool
    nodes: int
    time_ms: int
    states: List[str]  # facelet strings for each step (initial + after each move)
    error: Optional[str] = None


class CapabilitiesResponse(BaseModel):
    algorithms: List[str]
    supported_sizes: List[int]
    color_scheme: List[List[int]]  # RGB per face


@router.get("/capabilities", response_model=CapabilitiesResponse)
async def get_capabilities():
    return CapabilitiesResponse(
        algorithms=["auto", "kociemba", "thistlethwaite", "korf", "pocket", "native_ida"],
        supported_sizes=list(range(2, 17)),
        color_scheme=[list(c) for c in FACE_COLORS],
    )


@router.post("/scramble", response_model=ScrambleResponse)
async def scramble(req: ScrambleRequest):
    if req.size < 2 or req.size > 16:
        raise HTTPException(422, "Size must be 2-16")
    scramble_str, moves = generate_scramble(req.size, req.length, req.seed)
    state = CubeState.solved(req.size)
    state = apply_moves(state, moves)
    return ScrambleResponse(
        size=req.size,
        scramble=scramble_str,
        moves=moves,
        facelets=state.to_facelet_string(),
        state=state.facelets,
    )


@router.post("/solve", response_model=SolveResponse)
async def solve(req: SolveRequest):
    start = time.time()
    try:
        state = CubeState.from_facelet_string(req.facelets, req.size)
    except Exception as e:
        raise HTTPException(422, f"Invalid facelets: {e}")

    valid, errors = state.validate()
    if not valid:
        raise HTTPException(422, f"Invalid cube state: {'; '.join(errors)}")

    algo = Algorithm(req.algorithm)
    result = await solve_cube_async(state, algorithm=algo, depth=req.depth, node_limit=req.node_limit)

    elapsed_ms = int((time.time() - start) * 1000)
    result["time_ms"] = result.get("time_ms", elapsed_ms)

    if not result.get("solved", False) and "error" not in result:
        result["error"] = "Solver did not find a solution"

    # Generate intermediate states for animation
    states = []
    if result.get("solved") and result.get("moves"):
        sol_states = generate_solution_states(state, result["moves"])
        states = [s.to_facelet_string() for s in sol_states]

    return SolveResponse(
        algorithm=result.get("algorithm", req.algorithm),
        moves=result.get("moves", []),
        move_count=result.get("move_count", 0),
        solved=result.get("solved", False),
        nodes=result.get("nodes", 0),
        time_ms=result["time_ms"],
        states=states,
        error=result.get("error"),
    )


@router.post("/solve-scramble", response_model=SolveResponse)
async def solve_scramble(req: SolveScrambleRequest):
    """
    Solve a cube given a scramble string.
    This endpoint uses the best available solver for the given size.
    """
    start = time.time()
    try:
        # Validate scramble format
        moves = req.scramble.split()
        for m in moves:
            if not m or m[0].upper() not in FACE_NAMES:
                raise HTTPException(422, f"Invalid move in scramble: {m}")
    except Exception as e:
        raise HTTPException(422, f"Invalid scramble: {e}")

    algo = Algorithm(req.algorithm)
    result = await solve_scramble_async(req.scramble, req.size, algorithm=algo, depth=req.depth, node_limit=req.node_limit)

    elapsed_ms = int((time.time() - start) * 1000)
    result["time_ms"] = result.get("time_ms", elapsed_ms)

    if not result.get("solved", False) and "error" not in result:
        result["error"] = "Solver did not find a solution"

    # Generate intermediate states for animation
    states = []
    if result.get("solved") and result.get("moves"):
        state = CubeState.solved(req.size)
        sol_states = generate_solution_states(state, result["moves"])
        states = [s.to_facelet_string() for s in sol_states]

    return SolveResponse(
        algorithm=result.get("algorithm", req.algorithm),
        moves=result.get("moves", []),
        move_count=result.get("move_count", 0),
        solved=result.get("solved", False),
        nodes=result.get("nodes", 0),
        time_ms=result["time_ms"],
        states=states,
        error=result.get("error"),
    )


@router.post("/apply", response_model=ScrambleResponse)
async def apply_moves_endpoint(req: MoveApplyRequest):
    try:
        state = CubeState.from_facelet_string(req.facelets, req.size)
    except Exception as e:
        raise HTTPException(422, f"Invalid facelets: {e}")

    new_state = apply_moves(state, req.moves)
    return ScrambleResponse(
        size=req.size,
        scramble=" ".join(req.moves),
        moves=req.moves,
        facelets=new_state.to_facelet_string(),
        state=new_state.facelets,
    )


@router.post("/validate")
async def validate(req: ValidateRequest):
    try:
        state = CubeState.from_facelet_string(req.facelets, req.size)
    except Exception as e:
        return {"valid": False, "errors": [str(e)]}
    valid, errors = state.validate()
    return {"valid": valid, "errors": errors, "color_counts": state.color_counts()}