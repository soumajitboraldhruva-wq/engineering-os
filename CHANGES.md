# Changes in this update

## Advanced Analyser - engine (`backend/analyser.cpp`, rebuild: `g++ -std=c++20 -O2 -o analyser_bin analyser.cpp`)
Root causes found by testing against known functions and fixed:
- **Validation R-squared was wrong.** It was averaged per fold; 1-step folds always gave 0, so a perfect linear fit scored R2 = -640 and every real model was crushed. Now pooled across folds.
- **Zero-order-hold ("step") model scored unfairly** (hard-coded direction/correlation = 1.0) and won on noisy trending data. Now scored honestly and only tried when the data really has plateaus.
- **Smooth curves were chopped into straight pieces** (parabola, sqrt, ...) because the piecewise-linear route ran before the polynomial check and accepted loose fits. Polynomial/closed-form check now runs first; heuristic routes must beat the best smooth model.
- **Validation only looked 1-2 steps ahead**, so models that are hopeless at your requested horizon could win. Validation now covers the requested horizon.
- **Sinusoid periods were in samples but applied in x units** (wrong for any non-unit x spacing). Fixed, plus fine frequency refinement (non-integer periods), joint (omega, alpha) refinement for damped sinusoids, a multi-tone family (sum of unrelated sinusoids) and a shifted-exponential family (offset above or below the data).
- Spikes/outliers are no longer mistaken for domain breaks; models need >= 3 observations per parameter; parsimony (simplest model within the noise band); evidence score is never 0 for a reported model.
- **New exact detectors** (forecast by running the rule, not by curve fitting): Collatz / modular piecewise-affine maps, iterated maps (logistic map etc.), linear recurrences up to order 4, and a library of self-referential / arithmetic sequences (Recaman, Hofstadter Q/G/H/Conway, Golomb, divisor/sigma/phi/Moebius, popcount, Thue-Morse, Stern, Josephus, partitions, ...). A false-positive in the "eventual cycle" detector was tightened.
- Bounded run time (25 s sweep budget with cheap families first; staged fast path that validates the best closed-form candidate before the exhaustive sweep).
- **Family explorer** ranked by raw in-sample RMSE, so the most flexible curve always won, and Holt/AR/Prony reported `fitted = data` (RMSE 0). Now ranked by AICc, recursive families use one-step-ahead error. Manual fit with frequency/width/penalty = 0 now auto-estimates it instead of failing.

## Advanced Analyser - data import (`backend/api/data_import.py`, `/inspect-file`, `/inspect-text`)
CSV, TSV, `;` and `|` separated, whitespace, labelled (`x=1, y=2`), bracketed lists, decimal commas, thousands separators, BOM/UTF-16/Latin-1, comment lines, header + units rows, dates/times (converted to seconds), several tables per file or per Excel sheet, ragged / single-line data, row-labelled wide tables, XLSX/XLSM (+ XLS/ODS/XLSB with optional readers). Columns are grouped by heading family (`Temp_1`, `Temp_2`), by unit and by X/Y pairs; roles (X / Series / Skip) and the analysis mode (curve / sequence / pairs / multivariate) are suggested automatically and can be changed. Missing values can be skipped or interpolated.

## Linear Algebra
- UI rewritten: two-pane layout (inputs / sticky results), grouped operations, matrix editor with keyboard navigation and spreadsheet paste, structured result view (status banner, matrices, diagnostics, "Use as A"), history, and a console tab with variables and command recall. Light theme only (the old mix of `dark:` classes followed the OS setting and looked broken).
- Fixed: loading an example with different dimensions wiped it to zeros; `rank` crashed (`A` undefined); indefinite integration failed in `/calculus` and `/symbolic`; unknown console functions silently "succeeded". New: null-space basis for rank-deficient systems, multi-statement console input, matrix expressions (`A*B+2*A`, `A'`, `A^-1`).

## Cube solver
Hidden, not deleted. Backend router only loads with `ENABLE_CUBE_SOLVER=1`; frontend nav and the analyser's puzzle tile only show with `VITE_SHOW_CUBE_SOLVER=true` (see `frontend/.env.example`).

## Other
- Tailwind `darkMode: 'class'` so the analyser's dark-mode toggle actually controls dark styles.
- `backend/requirements.txt`: added `openpyxl`, `xlrd`, `odfpy` for spreadsheet import.
- `backend/tests/engine_regression.py`: 20 regression checks for the engine (`python backend/tests/engine_regression.py`).
