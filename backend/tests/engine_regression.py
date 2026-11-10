"""Regression checks for the native analyser engine (curve lab + exact-rule detectors).

Run from the repo root:   python backend/tests/engine_regression.py
Uses ANALYSER_BIN or backend/analyser_bin. Exits non-zero if any check fails.
"""
from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.environ.get("ANALYSER_BIN") or os.path.join(HERE, "..", "analyser_bin")


def run(args, stdin=""):
    p = subprocess.run([BIN, *map(str, args)], input=stdin, capture_output=True, text=True, timeout=180)
    return json.loads(p.stdout)


def curve(xs, ys, horizon=4):
    step = xs[1] - xs[0]
    return run(["--web-curve", horizon, -1, step], "\n".join(f"{x} {y}" for x, y in zip(xs, ys)))


FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail and not ok else ""))
    if not ok:
        FAILS.append(name)


def forecast_ok(name, xs, ys, truth, tol=1e-3, horizon=4):
    j = curve(xs, ys, horizon)
    a = j["analysis"]
    fx = j["futureX"]
    errs = [abs(p - truth(x)) / max(abs(truth(x)), 0.5) for p, x in zip(a["predictions"], fx)]
    check(name, bool(a["predictions"]) and max(errs) < tol, f"{a['category']} err={max(errs, default=float('nan')):.3g}")


def sequence_ok(name, ys, future):
    j = curve(list(range(1, len(ys) + 1)), ys, len(future))
    p = j["analysis"]["predictions"]
    check(name, len(p) == len(future) and all(abs(a - b) <= 1e-3 * max(1, abs(b)) for a, b in zip(p, future)),
          f"{j['analysis']['category']} -> {p}")


def main() -> int:
    t0 = time.time()
    print("Smooth curves")
    xs = [i * 0.5 - 3 for i in range(21)]
    forecast_ok("cubic (exact)", xs, [2 * x**3 - 5 * x + 1 for x in xs], lambda x: 2 * x**3 - 5 * x + 1)
    forecast_ok("quadratic is NOT chopped into pieces", xs, [0.5 * x * x - 2 * x + 3 for x in xs], lambda x: 0.5 * x * x - 2 * x + 3)
    forecast_ok("sine + trend (non-integer period)", xs, [3 * math.sin(1.3 * x) + x / 2 for x in xs], lambda x: 3 * math.sin(1.3 * x) + x / 2)
    xs2 = [i * 0.25 for i in range(30)]
    forecast_ok("exponential growth", xs2, [2 * math.exp(0.3 * x) for x in xs2], lambda x: 2 * math.exp(0.3 * x))
    forecast_ok("damped sine (RLC ring-down)", xs2, [5 * math.exp(-0.2 * x) * math.sin(2 * x) for x in xs2], lambda x: 5 * math.exp(-0.2 * x) * math.sin(2 * x))
    xs3 = [0.5 * i for i in range(40)]
    forecast_ok("decay towards an offset", xs3, [4 + 6 * math.exp(-0.4 * x) for x in xs3], lambda x: 4 + 6 * math.exp(-0.4 * x))
    forecast_ok("charging curve (asymptote above data)", xs3, [5 * (1 - math.exp(-0.5 * x)) for x in xs3], lambda x: 5 * (1 - math.exp(-0.5 * x)))
    forecast_ok("sqrt", [0.5 * i for i in range(25)], [math.sqrt(0.5 * i) for i in range(25)], lambda x: math.sqrt(x))
    xl = [2000 + 0.5 * i for i in range(30)]
    forecast_ok("large x offset", xl, [3 * math.sin(0.9 * (x - 2000)) + 1 for x in xl], lambda x: 3 * math.sin(0.9 * (x - 2000)) + 1)
    random.seed(1)
    xn = [i * 0.25 for i in range(30)]
    j = curve(xn, [1.5 * x + 2 + random.gauss(0, 0.3) for x in xn])
    check("noisy line stays a line (not a step signal)", "zero-order" not in j["analysis"]["model"]["name"].lower()
          and abs(j["analysis"]["predictions"][0] - (1.5 * (xn[-1] + 0.25) + 2)) < 0.6, j["analysis"]["model"]["name"])

    print("Deterministic but erratic sequences")
    c = [27]
    while len(c) < 40:
        c.append(c[-1] // 2 if c[-1] % 2 == 0 else 3 * c[-1] + 1)
    sequence_ok("Collatz trajectory from 27", c[:30], c[30:35])
    r = [0]
    seen = {0}
    for n in range(1, 60):
        v = r[-1] - n
        r.append(v if v > 0 and v not in seen else r[-1] + n)
        seen.add(r[-1])
    sequence_ok("Recaman", r[1:21], r[21:26])
    q = [0, 1, 1]
    for n in range(3, 60):
        q.append(q[n - q[n - 1]] + q[n - q[n - 2]])
    sequence_ok("Hofstadter Q", q[1:31], q[31:36])
    g = [0, 1]
    for n in range(2, 80):
        g.append(1 + g[n - g[g[n - 1]]])
    sequence_ok("Golomb", g[1:31], g[31:36])
    y = [0.5]
    for _ in range(40):
        y.append(3.9 * y[-1] * (1 - y[-1]))
    sequence_ok("logistic map r=3.9 (6-digit data)", [round(v, 6) for v in y[:15]], y[15:19])
    trib = [0, 0, 1]
    for _ in range(30):
        trib.append(trib[-1] + trib[-2] + trib[-3])
    sequence_ok("Tribonacci", trib[:14], trib[14:19])
    sequence_ok("staircase is not a recurrence", [2, 2, 2, 2, 5, 5, 5, 5, 9, 9, 9, 9], [9, 9])

    print("Honesty")
    random.seed(7)
    xr = [0.2 * i for i in range(40)]
    j = curve(xr, [random.gauss(0, 1) for _ in xr])
    check("pure noise is not reported as a pattern", not j["analysis"]["found"])

    print("Family explorer")
    j = run(["--web-families"], "\n".join(f"{x} {math.sin(x) + 0.1 * x}" for x in [i * 0.5 for i in range(20)]))
    check("families are ranked by AICc and none has a zero RMSE artefact",
          all(f["model"]["metrics"]["rmse"] > 1e-12 for f in j["families"] if f["family"] in ("holt", "damped_holt", "ar", "differenced_ar", "prony")))
    j = run(["--web-family", "sin", 1, 0, 0, 3, 0.5], "\n".join(f"{x} {math.sin(1.3 * x)}" for x in [i * 0.5 for i in range(20)]))
    check("manual sin fit with frequency 0 estimates it automatically", j.get("ok") is True)

    print(f"\n{len(FAILS)} failed, finished in {time.time() - t0:.1f}s")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
