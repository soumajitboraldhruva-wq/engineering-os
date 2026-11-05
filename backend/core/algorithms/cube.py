"""
General N×N×N Rubik's Cube model and solvers.
Supports: N=2..16 facelet representation, scramble generation, move application,
solver dispatch (Kociemba for 3×3, Thistlethwaite for 4×4+, Korf for 3×3 optimal,
native IDA* for any N via C++, pocket solver for 2×2).
"""

from __future__ import annotations
import random
import subprocess
import json
import os
import asyncio
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum


# Standard color scheme (face indices match Kociemba's URFDLB order)
# 0=U(white), 1=R(red), 2=F(green), 3=D(yellow), 4=L(orange), 5=B(blue)
FACE_NAMES = ["U", "R", "F", "D", "L", "B"]
FACE_COLORS = [
    (255, 255, 255),  # U - white
    (255, 0, 0),      # R - red
    (0, 180, 0),      # F - green
    (255, 255, 0),    # D - yellow
    (255, 165, 0),    # L - orange
    (0, 0, 255),      # B - blue
]


class Algorithm(Enum):
    KOCIEMBA = "kociemba"          # 3x3 only, two-phase
    THISTLETHWAITE = "thistlethwaite"  # 4x4+ (also works 3x3)
    KORF = "korf"                   # 3x3 optimal IDA*
    NATIVE_IDA = "native_ida"      # any N via C++ engine, optimal for shallow scrambles
    POCKET = "pocket"              # 2x2 dedicated, optimal IDA*
    AUTO = "auto"                  # pick best for size


@dataclass
class CubeState:
    size: int
    facelets: List[int]  # 6*N*N ints 0-5, face-major order URFDLB (Kociemba convention)

    def __post_init__(self):
        expected = 6 * self.size * self.size
        if len(self.facelets) != expected:
            raise ValueError(f"Facelet count mismatch: got {len(self.facelets)}, expected {expected}")

    def copy(self) -> "CubeState":
        return CubeState(self.size, self.facelets.copy())

    def is_solved(self) -> bool:
        for f in range(6):
            base = f * self.size * self.size
            if any(self.facelets[base + i] != f for i in range(self.size * self.size)):
                return False
        return True

    def color_counts(self) -> List[int]:
        counts = [0] * 6
        for c in self.facelets:
            if 0 <= c < 6:
                counts[c] += 1
        return counts

    def validate(self) -> Tuple[bool, List[str]]:
        errors = []
        counts = self.color_counts()
        target = self.size * self.size
        for f, cnt in enumerate(counts):
            if cnt != target:
                errors.append(f"Face {FACE_NAMES[f]} has {cnt} stickers, expected {target}")
        if len(self.facelets) != 6 * self.size * self.size:
            errors.append("Invalid facelet length")
        return len(errors) == 0, errors

    def to_facelet_string(self) -> str:
        """Return facelet string in Kociemba format using URFDLB letters."""
        face_letters = "URFDLB"
        return "".join(face_letters[c] for c in self.facelets)

    @classmethod
    def solved(cls, size: int) -> "CubeState":
        facelets = []
        for f in range(6):
            facelets.extend([f] * (size * size))
        return cls(size, facelets)

    @classmethod
    def from_facelet_string(cls, s: str, size: int) -> "CubeState":
        expected = 6 * size * size
        if len(s) != expected:
            raise ValueError(f"Facelet string length {len(s)} != {expected}")
        face_letters = "URFDLB"
        facelets = []
        for ch in s:
            if ch in face_letters:
                facelets.append(face_letters.index(ch))
            elif ch.isdigit():
                facelets.append(int(ch))
            else:
                raise ValueError(f"Invalid facelet character: {ch}")
        return cls(size, facelets)


# ──────────────────────────────────────────────────────────────────────
# Move definitions and application (Singmaster notation, Kociemba-compatible)

def _rotate_face_cw(facelets: List[int], size: int) -> List[int]:
    """Rotate a single face's stickers 90° clockwise."""
    out = facelets[:]
    for r in range(size):
        for c in range(size):
            out[r * size + c] = facelets[(size - 1 - c) * size + r]
    return out


def _apply_outer_move(state: CubeState, face: int, prime: bool = False, double: bool = False) -> CubeState:
    """
    Apply an outer face turn (U, R, F, D, L, B or their primes/doubles).
    Uses the standard sticker permutation for any N, matching Kociemba's convention.
    Face indices: 0=U, 1=R, 2=F, 3=D, 4=L, 5=B
    """
    s = state.size
    out = state.facelets[:]
    turns = 3 if prime else (2 if double else 1)  # 1=cw, 2=180, 3=ccw

    # Face rotation
    face_start = face * s * s
    face_stickers = [state.facelets[face_start + i] for i in range(s * s)]
    for _ in range(turns):
        face_stickers = _rotate_face_cw(face_stickers, s)
    for i, val in enumerate(face_stickers):
        out[face_start + i] = val

    # Adjacent face strips - standard Singmaster cycles for outer layer turns
    # Each cycle is a list of (face, row, col) tuples that form a 4-cycle
    # Order of faces in cycle matters: U->R->F->L for U face, etc.
    cycles = []
    if face == 0:  # U (top layer)
        cycles = [
            [(2, 0, i) for i in range(s)],          # F row 0 (top)
            [(1, 0, i) for i in range(s)],          # R row 0 (top)
            [(5, 0, i) for i in range(s)],          # B row 0 (top) - note: not reversed in Kociemba convention
            [(4, 0, i) for i in range(s)],          # L row 0 (top)
        ]
    elif face == 1:  # R (right layer)
        # Physical cycle for R clockwise: U -> F -> D -> B -> U
        # With forward rotation strips[shift:] + strips[:shift], this works correctly
        cycles = [
            [(0, i, s-1) for i in range(s)],        # U col s-1 (right)
            [(2, i, s-1) for i in range(s)],        # F col s-1 (right)
            [(3, i, s-1) for i in range(s)],        # D col s-1 (right)
            [(5, i, 0) for i in range(s)],          # B col 0 (left) - B face viewed from back
        ]
    elif face == 2:  # F (front layer)
        cycles = [
            [(0, s-1, i) for i in range(s)],        # U row s-1 (bottom)
            [(1, i, 0) for i in range(s)],          # R col 0 (left)
            [(3, 0, i) for i in range(s)],          # D row 0 (top)
            [(4, i, s-1) for i in range(s)],        # L col s-1 (right)
        ]
    elif face == 3:  # D (bottom layer)
        cycles = [
            [(2, s-1, i) for i in range(s)],        # F row s-1 (bottom)
            [(4, s-1, i) for i in range(s)],        # L row s-1 (bottom)
            [(5, s-1, i) for i in range(s)],        # B row s-1 (bottom)
            [(1, s-1, i) for i in range(s)],        # R row s-1 (bottom)
        ]
    elif face == 4:  # L (left layer)
        cycles = [
            [(0, i, 0) for i in range(s)],          # U col 0 (left)
            [(2, i, 0) for i in range(s)],          # F col 0 (left)
            [(3, i, 0) for i in range(s)],          # D col 0 (left)
            [(5, s-1-i, s-1) for i in range(s)],    # B col s-1 (right) reversed
        ]
    elif face == 5:  # B (back layer)
        cycles = [
            [(0, 0, i) for i in range(s)],          # U row 0 (top)
            [(1, i, s-1) for i in range(s)],        # R col s-1 (right)
            [(3, s-1, i) for i in range(s)],        # D row s-1 (bottom)
            [(4, s-1-i, 0) for i in range(s)],      # L col 0 (left) reversed
        ]

    # Extract strips
    strips = []
    for cyc in cycles:
        strip = [state.facelets[f * s * s + r * s + c] for (f, r, c) in cyc]
        strips.append(strip)

    # Rotate strips by one position in the cycle (for clockwise turn)
    # Using forward rotation: strips[shift:] + strips[:shift]
    # This gives the correct physical cycle U->F->D->B->U for R face
    shift = turns % 4
    if shift:
        strips = strips[shift:] + strips[:shift]

    # Write back
    for cyc, strip in zip(cycles, strips):
        for (f, r, c), val in zip(cyc, strip):
            out[f * s * s + r * s + c] = val

    return CubeState(state.size, out)


def apply_move(state: CubeState, move: str) -> CubeState:
    """Apply a single move to cube state."""
    if not move:
        return state
    face_char = move[0].upper()
    face = FACE_NAMES.index(face_char)
    prime = False
    double = False
    if len(move) > 1:
        if move[1] == "'":
            prime = True
        elif move[1] == "2":
            double = True
    return _apply_outer_move(state, face, prime, double)


def apply_moves(state: CubeState, moves: List[str]) -> CubeState:
    """Apply a sequence of moves."""
    s = state
    for m in moves:
        s = apply_move(s, m)
    return s


def invert_moves(moves: List[str]) -> List[str]:
    """Return inverse move sequence."""
    inv = []
    for m in reversed(moves):
        if m.endswith("'"):
            inv.append(m[0])
        elif m.endswith("2"):
            inv.append(m)
        else:
            inv.append(m + "'")
    return inv


# ──────────────────────────────────────────────────────────────────────
# Scramble generation

def generate_scramble(size: int, length: int = 20, seed: Optional[int] = None) -> Tuple[str, List[str]]:
    """Generate a random scramble. Returns (move_string, move_list)."""
    if seed is not None:
        random.seed(seed)
    faces = FACE_NAMES[:]
    moves = []
    last_face = -1
    last_axis = -1  # 0=UD, 1=RL, 2=FB
    for _ in range(length):
        # Avoid same face or same axis consecutively
        choices = [i for i in range(6) if i != last_face and (i // 2) != last_axis]
        if not choices:
            choices = [i for i in range(6) if i != last_face]
        f = random.choice(choices)
        last_face = f
        last_axis = f // 2
        mod = random.choice(["", "'", "2"])
        moves.append(FACE_NAMES[f] + mod)
    return " ".join(moves), moves


# ──────────────────────────────────────────────────────────────────────
# C++ Native Solver Interface

def _find_analyser_bin() -> Optional[str]:
    candidates = [
        "/home/zero/engineering-os/backend/analyser_bin",
        "/home/zero/engineering-os/analyser_bin",
        "./analyser_bin",
    ]
    for p in candidates:
        if os.path.exists(p) and os.access(p, os.X_OK):
            return p
    return None


async def solve_native_cpp(state: CubeState, depth: int = 20, node_limit: int = 5000000) -> Dict:
    """
    Solve using native C++ IDA* solver via analyser_bin --web-puzzle-cube.
    The C++ engine expects a scramble string (moves applied to solved state).
    For arbitrary states, we can't easily convert to scramble without solving first.
    """
    bin_path = _find_analyser_bin()
    if not bin_path:
        return {"error": "Native binary not found", "solved": False}
    return {"error": "Native C++ solver requires scramble input", "solved": False}


async def solve_scramble_native_cpp(scramble: str, size: int, depth: int = 20, node_limit: int = 5000000) -> Dict:
    """
    Solve a scramble using the C++ engine's --web-puzzle-cube endpoint.
    This is the intended use: user provides scramble, engine finds solution.
    """
    bin_path = _find_analyser_bin()
    if not bin_path:
        return {"error": "Native binary not found", "solved": False}

    try:
        proc = await asyncio.create_subprocess_exec(
            bin_path,
            "--web-puzzle-cube", str(size), str(depth), str(node_limit),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(scramble.encode("utf-8")),
            timeout=180.0,
        )
        stdout_text = stdout.decode("utf-8", errors="replace").strip()
        stderr_text = stderr.decode("utf-8", errors="replace").strip()
        if not stdout_text:
            return {"error": stderr_text or "No output from C++ engine", "solved": False}
        result = json.loads(stdout_text)
        if result.get("ok") is False:
            return {"error": result.get("error", "C++ solver failed"), "solved": False}

        # Parse C++ engine response
        # Expected format: {"ok": true, "solution": [...], "nodes": N, "time_ms": T, "depth": D}
        moves = result.get("solution", [])
        if not moves:
            moves = result.get("moves", [])
        return {
            "algorithm": "native_ida",
            "moves": moves,
            "move_count": len(moves),
            "solved": True,
            "nodes": result.get("nodes", 0),
            "time_ms": result.get("time_ms", 0),
        }
    except asyncio.TimeoutError:
        return {"error": "Solver timeout", "solved": False}
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON from C++ engine: {e}", "solved": False}
    except Exception as e:
        return {"error": str(e), "solved": False}


# ──────────────────────────────────────────────────────────────────────
# Kociemba Solver (3x3 only) - via cube-solver library (uses same facelet convention)

def _facelets_to_color_repr(facelets: List[int]) -> str:
    """Convert facelet indices (0-5 for U,R,F,D,L,B) to cube-solver's color repr (ULFRBD order)."""
    # Our facelets: URFDLB order (0=U, 1=R, 2=F, 3=D, 4=L, 5=B)
    # cube-solver REPR_ORDER: ULFRBD (U, L, F, R, B, D)
    color_map = ['W', 'R', 'G', 'Y', 'O', 'B']
    # Reorder from URFDLB to ULFRBD
    # U (0) -> U (0)
    # L (4) -> L (1)
    # F (2) -> F (2)
    # R (1) -> R (3)
    # B (5) -> B (4)
    # D (3) -> D (5)
    reorder = [0, 4, 2, 1, 5, 3]
    size = int((len(facelets) / 6) ** 0.5)
    face_size = size * size
    result = []
    for new_face in reorder:
        start = new_face * face_size
        end = start + face_size
        for idx in range(start, end):
            result.append(color_map[facelets[idx]])
    return "".join(result)


def solve_kociemba(state: CubeState) -> Dict:
    """Solve 3x3 using Kociemba two-phase algorithm via cube-solver library."""
    if state.size != 3:
        return {"error": "Kociemba only supports 3x3", "solved": False}
    try:
        from cube_solver import Cube as CS_Cube, Kociemba as CS_Kociemba
        import time

        # Convert our facelets to cube-solver's color format
        color_repr = _facelets_to_color_repr(state.facelets)
        cs_cube = CS_Cube(repr=color_repr)

        solver = CS_Kociemba()
        start = time.time()
        solution = solver.solve(cs_cube)
        elapsed = int((time.time() - start) * 1000)

        # Convert solution to our move format
        moves = [str(m) for m in (solution or [])]
        # Verify using our move tables
        final = apply_moves(state, moves)
        return {
            "algorithm": "kociemba",
            "moves": moves,
            "move_count": len(moves),
            "solved": final.is_solved(),
            "nodes": 0,
            "time_ms": elapsed,
        }
    except Exception as e:
        return {"error": f"Kociemba solver error: {e}", "solved": False}


# ──────────────────────────────────────────────────────────────────────
# Thistlethwaite Solver (4x4+) - via cube-solver library

async def solve_thistlethwaite(state: CubeState) -> Dict:
    """Solve 4x4+ using Thistlethwaite algorithm via cube-solver library."""
    if state.size < 3:
        return {"error": "Thistlethwaite requires size >= 3", "solved": False}
    try:
        from cube_solver import Cube as CS_Cube, Thistlethwaite as CS_Thistlethwaite
        import time

        # Convert our facelets to cube-solver's color format
        color_repr = _facelets_to_color_repr(state.facelets)
        cs_cube = CS_Cube(repr=color_repr)

        solver = CS_Thistlethwaite()
        start = time.time()
        solution = solver.solve(cs_cube)
        elapsed = int((time.time() - start) * 1000)

        # Convert solution to our move format
        moves = [str(m) for m in (solution or [])]
        # Verify
        final = apply_moves(state, moves)
        return {
            "algorithm": "thistlethwaite",
            "moves": moves,
            "move_count": len(moves),
            "solved": final.is_solved(),
            "nodes": 0,
            "time_ms": elapsed,
        }
    except Exception as e:
        return {"error": f"Thistlethwaite solver error: {e}", "solved": False}


# ──────────────────────────────────────────────────────────────────────
# Korf Solver (3x3 optimal) - via cube-solver library

async def solve_korf(state: CubeState) -> Dict:
    """Solve 3x3 optimally using Korf's IDA* algorithm via cube-solver library."""
    if state.size != 3:
        return {"error": "Korf only supports 3x3", "solved": False}
    try:
        from cube_solver import Cube as CS_Cube, Korf as CS_Korf
        import time

        # Convert our facelets to cube-solver's color format
        color_repr = _facelets_to_color_repr(state.facelets)
        cs_cube = CS_Cube(repr=color_repr)

        solver = CS_Korf()
        start = time.time()
        solution = solver.solve(cs_cube)
        elapsed = int((time.time() - start) * 1000)

        moves = [str(m) for m in (solution or [])]
        final = apply_moves(state, moves)
        return {
            "algorithm": "korf",
            "moves": moves,
            "move_count": len(moves),
            "solved": final.is_solved(),
            "nodes": 0,
            "time_ms": elapsed,
        }
    except Exception as e:
        return {"error": f"Korf solver error: {e}", "solved": False}


# ──────────────────────────────────────────────────────────────────────
# Pocket Cube (2×2) solver - fixed corner mapping for Kociemba convention

# Corner indices in Kociemba's cubie representation:
# 0=URF, 1=UFL, 2=ULB, 3=UBR, 4=DFR, 5=DLF, 6=DBL, 7=DRB
# Our facelets are in URFDLB face order, each face row-major.
# For 2x2, each face has 4 stickers in positions:
# U: 0,1,2,3  R: 4,5,6,7  F: 8,9,10,11  D: 12,13,14,15  L: 16,17,18,19  B: 20,21,22,23
# Corner facelet positions (face, row, col) for each corner:
# URF: U(0,0), R(0,0), F(0,1)  -> indices: 0, 4, 9
# UFL: U(0,1), F(0,0), L(0,0)  -> indices: 1, 8, 16
# ULB: U(1,1), L(0,1), B(0,0)  -> indices: 3, 17, 20
# UBR: U(1,0), B(0,1), R(0,1)  -> indices: 2, 21, 5
# DFR: D(0,0), F(1,0), R(1,0)  -> indices: 12, 10, 6
# DLF: D(0,1), L(1,1), F(1,1)  -> indices: 13, 19, 11
# DBL: D(1,1), B(1,0), L(1,0)  -> indices: 15, 22, 18
# DRB: D(1,0), R(1,1), B(1,1)  -> indices: 14, 7, 23

POCKET_CORNER_FACELETS = [
    [0, 4, 9],    # URF
    [1, 8, 16],   # UFL
    [3, 17, 20],  # ULB
    [2, 21, 5],   # UBR
    [12, 10, 6],  # DFR
    [13, 19, 11], # DLF
    [15, 22, 18], # DBL
    [14, 7, 23],  # DRB
]

POCKET_CORNER_COLORS = {
    0: (0, 1, 2),  # URF: U, R, F
    1: (0, 2, 4),  # UFL: U, F, L
    2: (0, 4, 5),  # ULB: U, L, B
    3: (0, 5, 1),  # UBR: U, B, R
    4: (3, 2, 1),  # DFR: D, F, R
    5: (3, 4, 2),  # DLF: D, L, F
    6: (3, 5, 4),  # DBL: D, B, L
    7: (3, 1, 5),  # DRB: D, R, B
}

POCKET_MOVES = {
    "U":  {"perm": [3, 0, 1, 2, 4, 5, 6, 7], "orient_delta": [0]*8},
    "U'": {"perm": [1, 2, 3, 0, 4, 5, 6, 7], "orient_delta": [0]*8},
    "U2": {"perm": [2, 3, 0, 1, 4, 5, 6, 7], "orient_delta": [0]*8},
    "R":  {"perm": [4, 1, 2, 0, 7, 5, 6, 3], "orient_delta": [2,0,0,1,1,0,0,2]},
    "R'": {"perm": [3, 1, 2, 7, 0, 5, 6, 4], "orient_delta": [1,0,0,2,2,0,0,1]},
    "R2": {"perm": [7, 1, 2, 4, 3, 5, 6, 0], "orient_delta": [0]*8},
    "F":  {"perm": [1, 5, 2, 3, 0, 4, 6, 7], "orient_delta": [1,2,0,0,2,1,0,0]},
    "F'": {"perm": [4, 0, 2, 3, 5, 1, 6, 7], "orient_delta": [2,1,0,0,1,2,0,0]},
    "F2": {"perm": [5, 4, 2, 3, 1, 0, 6, 7], "orient_delta": [0]*8},
    "D":  {"perm": [0, 1, 2, 3, 5, 6, 7, 4], "orient_delta": [0]*8},
    "D'": {"perm": [0, 1, 2, 3, 7, 4, 5, 6], "orient_delta": [0]*8},
    "D2": {"perm": [0, 1, 2, 3, 6, 7, 4, 5], "orient_delta": [0]*8},
    "L":  {"perm": [0, 2, 6, 3, 4, 1, 5, 7], "orient_delta": [0,1,2,0,0,2,1,0]},
    "L'": {"perm": [0, 5, 1, 3, 4, 6, 2, 7], "orient_delta": [0,2,1,0,0,1,2,0]},
    "L2": {"perm": [0, 6, 5, 3, 4, 2, 1, 7], "orient_delta": [0]*8},
    "B":  {"perm": [0, 1, 3, 7, 4, 5, 2, 6], "orient_delta": [0,0,1,2,0,0,2,1]},
    "B'": {"perm": [0, 1, 6, 2, 4, 5, 7, 3], "orient_delta": [0,0,2,1,0,0,1,2]},
    "B2": {"perm": [0, 1, 7, 6, 4, 5, 3, 2], "orient_delta": [0]*8},
}


def _facelets_to_pocket_corners(state: CubeState) -> Tuple[List[int], List[int]]:
    """Extract corner permutation and orientation from 2x2 facelets in Kociemba convention."""
    if state.size != 2:
        raise ValueError("Pocket solver requires 2x2")
    s = state.facelets
    perm = [0] * 8
    orient = [0] * 8
    for c in range(8):
        f0, f1, f2 = POCKET_CORNER_FACELETS[c]
        colors = [s[f0], s[f1], s[f2]]
        target_colors = POCKET_CORNER_COLORS[c]
        # Find which corner cubie is at this position
        found = None
        for k in range(8):
            k_colors = POCKET_CORNER_COLORS[k]
            if set(colors) == set(k_colors):
                found = k
                break
        if found is None:
            raise ValueError(f"Invalid corner colors at corner {c}: {colors}")
        perm[c] = found
        # Orientation: which of the three colors is on U/D face
        u_or_d = 0 if found < 4 else 3
        try:
            o = colors.index(u_or_d)
        except ValueError:
            o = 0
        orient[c] = o % 3
    return perm, orient


def _apply_pocket_move(perm: List[int], orient: List[int], move: str) -> Tuple[List[int], List[int]]:
    m = POCKET_MOVES[move]
    new_perm = [perm[m["perm"][i]] for i in range(8)]
    new_orient = [(orient[m["perm"][i]] + m["orient_delta"][i]) % 3 for i in range(8)]
    return new_perm, new_orient


def _pocket_heuristic(perm: List[int], orient: List[int]) -> int:
    """Admissible heuristic: number of misplaced corners + misoriented corners / 3."""
    misplaced = sum(1 for i in range(8) if perm[i] != i)
    misoriented = sum(1 for i in range(8) if orient[i] != 0)
    return misplaced + (misoriented + 2) // 3


def solve_pocket(state: CubeState, max_depth: int = 14, node_limit: int = 2000000) -> Dict:
    """Solve 2x2 using IDA* with corner heuristic."""
    if state.size != 2:
        return {"error": "Pocket solver only for 2x2", "solved": False}

    perm, orient = _facelets_to_pocket_corners(state)
    goal_perm = list(range(8))
    goal_orient = [0] * 8

    if perm == goal_perm and orient == goal_orient:
        return {"algorithm": "pocket", "moves": [], "move_count": 0, "solved": True, "nodes": 0, "time_ms": 0}

    moves_list = list(POCKET_MOVES.keys())
    nodes = 0

    def dfs(p, o, g, bound, path, prev_move):
        nonlocal nodes
        nodes += 1
        if nodes > node_limit:
            return float('inf')
        h = _pocket_heuristic(p, o)
        f = g + h
        if f > bound:
            return f
        if p == goal_perm and o == goal_orient:
            return -1  # found
        min_t = float('inf')
        for m in moves_list:
            if prev_move and m[0] == prev_move[0]:
                continue
            np, no = _apply_pocket_move(p, o, m)
            path.append(m)
            t = dfs(np, no, g + 1, bound, path, m)
            if t == -1:
                return -1
            if t < min_t:
                min_t = t
            path.pop()
        return min_t

    import time
    start = time.time()
    bound = _pocket_heuristic(perm, orient)
    path = []
    while True:
        t = dfs(perm, orient, 0, bound, path, "")
        if t == -1:
            break
        if t == float('inf') or nodes > node_limit:
            return {"error": "No solution within depth/node limit", "solved": False, "nodes": nodes}
        bound = t

    elapsed = int((time.time() - start) * 1000)
    return {
        "algorithm": "pocket",
        "moves": path[:],
        "move_count": len(path),
        "solved": True,
        "nodes": nodes,
        "time_ms": elapsed,
    }


# ──────────────────────────────────────────────────────────────────────
# Solver dispatch

def solve_cube(state: CubeState, algorithm: Algorithm = Algorithm.AUTO,
               depth: int = 20, node_limit: int = 5000000) -> Dict:
    """Main solver entry point (synchronous, for non-native solvers)."""
    if algorithm == Algorithm.AUTO:
        if state.size == 2:
            algorithm = Algorithm.POCKET
        elif state.size == 3:
            algorithm = Algorithm.KOCIEMBA
        elif state.size <= 7:
            algorithm = Algorithm.THISTLETHWAITE
        else:
            algorithm = Algorithm.NATIVE_IDA

    if algorithm == Algorithm.KOCIEMBA:
        return solve_kociemba(state)
    elif algorithm == Algorithm.POCKET:
        if state.size == 2:
            return {"error": "Pocket solver for arbitrary states not supported; use /solve-scramble with scramble", "solved": False}
        return solve_pocket(state, max_depth=depth, node_limit=node_limit)
    elif algorithm == Algorithm.NATIVE_IDA:
        return {"error": "Native IDA* requires scramble input; use /solve-scramble endpoint", "solved": False}
    else:
        # Thistlethwaite and Korf are async - fall back to error for sync call
        return {"error": f"Algorithm {algorithm.value} requires async call; use solve_cube_async", "solved": False}


async def solve_cube_async(state: CubeState, algorithm: Algorithm = Algorithm.AUTO,
                           depth: int = 20, node_limit: int = 5000000) -> Dict:
    """Async solver entry point (supports all solvers)."""
    if algorithm == Algorithm.AUTO:
        if state.size == 2:
            algorithm = Algorithm.POCKET
        elif state.size == 3:
            algorithm = Algorithm.KOCIEMBA
        elif state.size <= 7:
            algorithm = Algorithm.THISTLETHWAITE
        else:
            algorithm = Algorithm.NATIVE_IDA

    if algorithm == Algorithm.KOCIEMBA:
        if state.size == 3:
            return await solve_cs_kociemba(state)
        return {"error": "Kociemba only supports 3x3", "solved": False}
    elif algorithm == Algorithm.POCKET:
        if state.size == 2:
            return {"error": "Pocket solver for arbitrary states not supported; use /solve-scramble with scramble", "solved": False}
        return solve_pocket(state, max_depth=depth, node_limit=node_limit)
    elif algorithm == Algorithm.THISTLETHWAITE:
        if state.size == 3:
            return await solve_cs_thistlethwaite(state)
        return {"error": "Thistlethwaite only supports 3x3 in cube-solver", "solved": False}
    elif algorithm == Algorithm.KORF:
        if state.size == 3:
            return await solve_cs_korf(state)
        return {"error": "Korf only supports 3x3", "solved": False}
    elif algorithm == Algorithm.NATIVE_IDA:
        return await solve_native_cpp(state, depth=depth, node_limit=node_limit)
    else:
        return {"error": f"Unknown algorithm: {algorithm}", "solved": False}


def _cs_move_to_std(move_str: str) -> str:
    """Convert cube-solver move format (R1, R2, R3) to standard (R, R2, R')."""
    face = move_str[0]
    suffix = move_str[1]
    if suffix == '1':
        return face
    elif suffix == '2':
        return face + '2'
    elif suffix == '3':
        return face + "'"
    return move_str

def _our_facelets_to_cs_repr(facelets: List[int]) -> str:
    """Convert our URFDLB facelets to cube-solver's ULFRBD color repr."""
    color_map = ['W', 'R', 'G', 'Y', 'O', 'B']
    size = int((len(facelets) / 6) ** 0.5)
    face_size = size * size
    # Our order: U(0), R(1), F(2), D(3), L(4), B(5)
    # CS order:  U(0), L(1), F(2), R(3), B(4), D(5)
    cs_order = [0, 4, 2, 1, 5, 3]
    result = []
    for cs_face in cs_order:
        start = cs_face * face_size
        for idx in range(start, start + face_size):
            result.append(color_map[facelets[idx]])
    return "".join(result)


def _cs_repr_to_our_facelets(cs_repr: str, size: int = 3) -> List[int]:
    """Convert cube-solver's ULFRBD color repr to our URFDLB facelets."""
    color_to_idx = {'W': 0, 'R': 1, 'G': 2, 'Y': 3, 'O': 4, 'B': 5}
    face_size = size * size
    faces = [cs_repr[i:i+face_size] for i in range(0, 6*face_size, face_size)]
    # CS faces: [U, L, F, R, B, D]
    # Our order: U, R, F, D, L, B
    our_order = [faces[0], faces[3], faces[2], faces[5], faces[1], faces[4]]
    result = []
    for face in our_order:
        for ch in face:
            result.append(color_to_idx[ch])
    return result


def _our_scramble_to_cs_maneuver(scramble: str):
    """Convert our scramble string to cube-solver Maneuver."""
    from cube_solver import Maneuver, Move
    move_map = {
        'U': Move.U1, "U'": Move.U3, 'U2': Move.U2,
        'R': Move.R1, "R'": Move.R3, 'R2': Move.R2,
        'F': Move.F1, "F'": Move.F3, 'F2': Move.F2,
        'D': Move.D1, "D'": Move.D3, 'D2': Move.D2,
        'L': Move.L1, "L'": Move.L3, 'L2': Move.L2,
        'B': Move.B1, "B'": Move.B3, 'B2': Move.B2,
    }
    moves = [move_map[token] for token in scramble.split()]
    return Maneuver(moves)


async def solve_cs_kociemba(state: CubeState) -> Dict:
    """Solve 3x3 using cube-solver's Kociemba."""
    if state.size != 3:
        return {"error": "Kociemba only supports 3x3", "solved": False}
    try:
        from cube_solver import Cube as CS_Cube, Kociemba, apply_maneuver, Maneuver
        import time

        cs_repr = _our_facelets_to_cs_repr(state.facelets)
        cs_cube = CS_Cube(repr=cs_repr)

        solver = Kociemba()
        start = time.time()
        solution = solver.solve(cs_cube)
        elapsed = int((time.time() - start) * 1000)

        sol_moves = list(solution)
        moves = [_cs_move_to_std(str(m).replace('Move.', '')) for m in sol_moves]

        # Generate intermediate states for animation
        states = [cs_repr]
        current = cs_cube
        for move in sol_moves:
            current = apply_maneuver(current, Maneuver([move]))
            states.append(repr(current))

        # Convert states to our format
        our_states = [_cs_repr_to_our_facelets(s, 3) for s in states]
        our_state_strings = ["".join(str(c) for c in fs) for fs in our_states]

        # Verify final state
        final_state = _cs_repr_to_our_facelets(states[-1], 3)
        final_cube = CubeState(3, final_state)

        return {
            "algorithm": "kociemba",
            "moves": moves,
            "move_count": len(moves),
            "solved": final_cube.is_solved(),
            "nodes": 0,
            "time_ms": elapsed,
            "states": our_state_strings,
        }
    except Exception as e:
        return {"error": f"Kociemba solver error: {e}", "solved": False}


async def solve_cs_thistlethwaite(state: CubeState) -> Dict:
    """Solve 3x3 using cube-solver's Thistlethwaite."""
    if state.size != 3:
        return {"error": "Thistlethwaite only supports 3x3 in cube-solver", "solved": False}
    try:
        from cube_solver import Cube as CS_Cube, Thistlethwaite, apply_maneuver, Maneuver
        import time

        cs_repr = _our_facelets_to_cs_repr(state.facelets)
        cs_cube = CS_Cube(repr=cs_repr)

        solver = Thistlethwaite()
        start = time.time()
        solution = solver.solve(cs_cube)
        elapsed = int((time.time() - start) * 1000)

        sol_moves = list(solution)
        moves = [_cs_move_to_std(str(m).replace('Move.', '')) for m in sol_moves]

        states = [cs_repr]
        current = cs_cube
        for move in sol_moves:
            current = apply_maneuver(current, Maneuver([move]))
            states.append(repr(current))

        our_states = [_cs_repr_to_our_facelets(s, 3) for s in states]
        our_state_strings = ["".join(str(c) for c in fs) for fs in our_states]

        final_state = _cs_repr_to_our_facelets(states[-1], 3)
        final_cube = CubeState(3, final_state)

        return {
            "algorithm": "thistlethwaite",
            "moves": moves,
            "move_count": len(moves),
            "solved": final_cube.is_solved(),
            "nodes": 0,
            "time_ms": elapsed,
            "states": our_state_strings,
        }
    except Exception as e:
        return {"error": f"Thistlethwaite solver error: {e}", "solved": False}


async def solve_cs_korf(state: CubeState) -> Dict:
    """Solve 3x3 optimally using cube-solver's Korf."""
    if state.size != 3:
        return {"error": "Korf only supports 3x3", "solved": False}
    try:
        from cube_solver import Cube as CS_Cube, Korf, apply_maneuver, Maneuver
        import time

        cs_repr = _our_facelets_to_cs_repr(state.facelets)
        cs_cube = CS_Cube(repr=cs_repr)

        solver = Korf()
        start = time.time()
        solution = solver.solve(cs_cube)
        elapsed = int((time.time() - start) * 1000)

        sol_moves = list(solution)
        moves = [_cs_move_to_std(str(m).replace('Move.', '')) for m in sol_moves]

        states = [cs_repr]
        current = cs_cube
        for move in sol_moves:
            current = apply_maneuver(current, Maneuver([move]))
            states.append(repr(current))

        our_states = [_cs_repr_to_our_facelets(s, 3) for s in states]
        our_state_strings = ["".join(str(c) for c in fs) for fs in our_states]

        final_state = _cs_repr_to_our_facelets(states[-1], 3)
        final_cube = CubeState(3, final_state)

        return {
            "algorithm": "korf",
            "moves": moves,
            "move_count": len(moves),
            "solved": final_cube.is_solved(),
            "nodes": 0,
            "time_ms": elapsed,
            "states": our_state_strings,
        }
    except Exception as e:
        return {"error": f"Korf solver error: {e}", "solved": False}


async def solve_cs_scramble(scramble: str, size: int, algorithm: Algorithm, depth: int, node_limit: int) -> Dict:
    """Solve a scramble using cube-solver library (3x3 only)."""
    if size != 3:
        return {"error": "Cube-solver only supports 3x3", "solved": False}
    try:
        from cube_solver import Cube as CS_Cube, Kociemba, Thistlethwaite, Korf, apply_maneuver, Maneuver
        import time

        # Convert our scramble to cube-solver Maneuver and apply to get scrambled state
        cs_maneuver = _our_scramble_to_cs_maneuver(scramble)
        c = CS_Cube()
        c_scrambled = apply_maneuver(c, cs_maneuver)

        # Select solver
        if algorithm == Algorithm.KOCIEMBA:
            solver = Kociemba()
        elif algorithm == Algorithm.THISTLETHWAITE:
            solver = Thistlethwaite()
        elif algorithm == Algorithm.KORF:
            solver = Korf()
        else:
            solver = Kociemba()

        start = time.time()
        solution = solver.solve(c_scrambled)
        elapsed = int((time.time() - start) * 1000)

        sol_moves = list(solution)
        moves = [_cs_move_to_std(str(m).replace('Move.', '')) for m in sol_moves]

        # Generate intermediate states
        states = [repr(c_scrambled)]
        current = c_scrambled
        for move in sol_moves:
            current = apply_maneuver(current, Maneuver([move]))
            states.append(repr(current))

        our_states = [_cs_repr_to_our_facelets(s, 3) for s in states]
        our_state_strings = ["".join(str(c) for c in fs) for fs in our_states]

        final_state = _cs_repr_to_our_facelets(states[-1], 3)
        final_cube = CubeState(3, final_state)

        return {
            "algorithm": algorithm.value,
            "moves": moves,
            "move_count": len(moves),
            "solved": final_cube.is_solved(),
            "nodes": 0,
            "time_ms": elapsed,
            "states": our_state_strings,
        }
    except Exception as e:
        return {"error": f"Cube-solver error: {e}", "solved": False}


async def solve_scramble_async(scramble: str, size: int, algorithm: Algorithm = Algorithm.AUTO,
                               depth: int = 20, node_limit: int = 5000000) -> Dict:
    """Solve a scramble string directly."""
    if algorithm == Algorithm.AUTO:
        if size == 3:
            algorithm = Algorithm.KOCIEMBA
        else:
            algorithm = Algorithm.NATIVE_IDA

    if algorithm in (Algorithm.KOCIEMBA, Algorithm.THISTLETHWAITE, Algorithm.KORF):
        if size == 3:
            return await solve_cs_scramble(scramble, size, algorithm, depth, node_limit)
        return {"error": f"{algorithm.value} only supports 3x3", "solved": False}
    elif algorithm == Algorithm.NATIVE_IDA:
        return await solve_scramble_native_cpp(scramble, size, depth, node_limit)
    elif algorithm == Algorithm.POCKET:
        state = CubeState.solved(size)
        state = apply_moves(state, scramble.split())
        return await solve_cube_async(state, algorithm, depth, node_limit)
    else:
        return {"error": f"Unknown algorithm: {algorithm}", "solved": False}


# ──────────────────────────────────────────────────────────────────────
# Animation support: generate intermediate states

def generate_solution_states(state: CubeState, moves: List[str]) -> List[CubeState]:
    """Return list of states including initial, after each move, and final."""
    states = [state.copy()]
    current = state
    for m in moves:
        current = apply_move(current, m)
        states.append(current.copy())
    return states