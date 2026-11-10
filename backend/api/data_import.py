"""Dataset inspector for the Advanced Analyser import screen.

Turns an uploaded file (or pasted text) into one or more *tables* of named columns so the UI can
let the user choose which columns are X, which are series, and how to analyse them. All
mathematics stays in the C++ engine; this module only parses and describes data.

Supported input
---------------
* Text: CSV, TSV, ';' or '|' separated, whitespace separated, labelled values (``x=1, y=2``),
  decimal commas (``1,5;2,3``), BOM / UTF-16 / Latin-1 encodings, ``#`` comment lines.
* Several tables in one file separated by blank lines (each with its own header).
* Header row and optional units row (``time,temp`` / ``s,degC``) detection.
* Ragged or single-line numeric files are treated as one sequence.
* Row-labelled wide files (``temp, 1, 2, 3 ...``) are transposed into columns.
* Spreadsheets: XLSX / XLSM (openpyxl); XLS / ODS / XLSB through pandas when its readers exist.
* Date / time columns are converted to seconds since the first value.
* Columns are grouped by heading family (``Temp_1``, ``Temp_2`` -> ``Temp``), by unit, and
  by X/Y pairs (``X1,Y1,X2,Y2``).
"""
from __future__ import annotations

import csv
import io
import math
import re
from datetime import date, datetime, time
from typing import Any

MAX_ROWS = 200_000
MAX_PREVIEW_ROWS = 8

_X_NAMES = re.compile(
    r"^(x|t|time|timestamp|date|datetime|index|idx|n|no|sr\.?no|sample|samples|step|frame|"
    r"frequency|freq|f|angle|theta|distance|dist|position|pos|depth|length|l|age|year|day)$",
    re.I,
)
_INDEX_NAMES = re.compile(r"^(index|idx|n|no|sr\.?\s*no|s\.?\s*no|#|id|row|sample|step)$", re.I)

_DATE_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d",
    "%d-%m-%Y %H:%M:%S",
    "%d-%m-%Y",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y",
    "%H:%M:%S",
    "%H:%M",
)


# --------------------------------------------------------------------------------------
# scalar parsing
# --------------------------------------------------------------------------------------
def _parse_number(cell: Any, decimal_comma: bool = False) -> float | None:
    if cell is None:
        return None
    if isinstance(cell, bool):
        return None
    if isinstance(cell, (int, float)):
        v = float(cell)
        return v if math.isfinite(v) else None
    s = str(cell).strip()
    if not s or s.lower() in {"nan", "na", "n/a", "null", "none", "-", "--", "?", "#n/a", "inf", "-inf"}:
        return None
    s = s.replace("\u2212", "-").replace("\u00a0", " ")
    neg = False
    if s.startswith("(") and s.endswith(")"):  # accounting negative
        neg, s = True, s[1:-1]
    s = s.rstrip("%").strip()
    s = s.replace(" ", "")
    if decimal_comma:
        s = s.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"[-+]?\d{1,3}(,\d{3})+(\.\d+)?", s):  # thousands separators
        s = s.replace(",", "")
    try:
        v = float(s)
    except ValueError:
        return None
    if not math.isfinite(v):
        return None
    return -v if neg else v


def _parse_datetime(cell: Any) -> datetime | None:
    if isinstance(cell, datetime):
        return cell
    if isinstance(cell, date):
        return datetime(cell.year, cell.month, cell.day)
    if isinstance(cell, time):
        return datetime(1970, 1, 1, cell.hour, cell.minute, cell.second)
    if cell is None:
        return None
    s = str(cell).strip()
    if len(s) < 5 or not re.search(r"\d", s):
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", ""))
    except ValueError:
        pass
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


_MISSING = {"nan", "na", "n/a", "null", "none", "-", "--", "?", "#n/a", "inf", "-inf", "#div/0!", "#value!"}


def _is_blank(cell: Any) -> bool:
    """Empty cell or an explicit 'missing value' token (NaN, N/A, -, ...)."""
    if cell is None:
        return True
    if isinstance(cell, float) and math.isnan(cell):
        return True
    return isinstance(cell, str) and (not cell.strip() or cell.strip().lower() in _MISSING)


# --------------------------------------------------------------------------------------
# text -> raw rows
# --------------------------------------------------------------------------------------
def _decode(data: bytes) -> str:
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        return data.decode("utf-16", errors="replace")
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1", errors="replace")


def _split_blocks(lines: list[str]) -> list[list[str]]:
    blocks: list[list[str]] = []
    cur: list[str] = []
    for line in lines:
        if not line.strip():
            if cur:
                blocks.append(cur)
                cur = []
        else:
            cur.append(line)
    if cur:
        blocks.append(cur)
    return blocks


def _sniff_delimiter(lines: list[str]) -> str | None:
    """Return the delimiter that splits most lines into the same number (>=2) of fields."""
    sample = lines[: min(len(lines), 60)]
    best: tuple[float, str | None] = (0.0, None)
    for delim in (",", "\t", ";", "|"):
        counts = [len(next(csv.reader([ln], delimiter=delim))) for ln in sample]
        if max(counts) < 2:
            continue
        mode = max(set(counts), key=counts.count)
        if mode < 2:
            continue
        consistency = counts.count(mode) / len(counts)
        score = consistency * 10 + (0.5 if delim == "\t" else 0) + min(mode, 8) * 0.01
        if consistency >= 0.6 and score > best[0]:
            best = (score, delim)
    return best[1]


_LABELLED = re.compile(r"([A-Za-z_][\w\.\- ]*?)\s*[=:]\s*([-+]?\d[\d.eE+\-]*)")


def _rows_from_text_block(lines: list[str]) -> tuple[list[list[str]], bool]:
    """Split one text block into rows of cell strings. Returns (rows, decimal_comma)."""
    # labelled values: "x=1, y=2"
    labelled = [list(_LABELLED.findall(ln)) for ln in lines]
    if sum(1 for m in labelled if len(m) >= 2) >= max(2, int(0.7 * len(lines))):
        names: list[str] = []
        for m in labelled:
            for k, _ in m:
                k = k.strip()
                if k not in names:
                    names.append(k)
        rows = [names]
        for m in labelled:
            d = {k.strip(): v for k, v in m}
            rows.append([d.get(n, "") for n in names])
        return rows, False

    # "[1, 2, 3]" / "(4, 5)" / "{6 7}": unwrap lines that are one bracketed list of numbers
    unwrap = re.compile(r"^\s*[\[\(\{]([\s\d.,;eE+\-]*)[\]\)\}]\s*,?\s*$")
    lines = [unwrap.sub(lambda m: m.group(1), ln) for ln in lines]

    delim = _sniff_delimiter(lines)
    decimal_comma = False
    if delim in (";", "\t", "|", None):
        comma_cells = sum(len(re.findall(r"(?<![\d.])-?\d+,\d+(?![\d.,])", ln)) for ln in lines)
        dot_cells = sum(len(re.findall(r"(?<![\d.])-?\d+\.\d+", ln)) for ln in lines)
        decimal_comma = delim is not None and comma_cells > 0 and dot_cells == 0
    if delim:
        rows = [next(csv.reader([ln], delimiter=delim, skipinitialspace=True)) for ln in lines]
    else:
        # whitespace / single column; also tolerate bracketed or comma-glued numbers
        rows = []
        for ln in lines:
            ln = re.sub(r"[\[\]\(\)\{\}]", " ", ln)
            rows.append([t for t in re.split(r"[\s,]+", ln.strip()) if t != ""])
    return rows, decimal_comma


# --------------------------------------------------------------------------------------
# rows -> table
# --------------------------------------------------------------------------------------
def _row_numeric_fraction(row: list[Any], dc: bool) -> float:
    cells = [c for c in row if not _is_blank(c)]
    if not cells:
        return 0.0
    return sum(1 for c in cells if _parse_number(c, dc) is not None) / len(cells)


def _detect_header(rows: list[list[Any]], dc: bool) -> tuple[bool, bool]:
    """(has_header, has_unit_row)."""
    if len(rows) < 2:
        return False, False
    first = rows[0]
    first_num = _row_numeric_fraction(first, dc)
    # name row + units row + numeric body (checked first: the units row would otherwise
    # be mistaken for a data row)
    if len(rows) >= 3 and first_num <= 0.34 and _row_numeric_fraction(rows[1], dc) <= 0.0:
        rest = [_row_numeric_fraction(r, dc) for r in rows[2 : min(len(rows), 12)]]
        if rest and sum(rest) / len(rest) >= 0.8 and any(not _is_blank(c) for c in rows[1]):
            return True, True
    body_num = [_row_numeric_fraction(r, dc) for r in rows[1 : min(len(rows), 12)]]
    if first_num <= 0.34 and body_num and sum(body_num) / len(body_num) >= 0.6:
        return True, False
    # a header made only of text over text-heavy body (e.g. a table of labels)
    if first_num == 0.0 and body_num and sum(body_num) / len(body_num) >= 0.3:
        return True, False
    return False, False


def _clean_name(name: Any, idx: int) -> str:
    s = str(name).strip() if name is not None else ""
    s = re.sub(r"\s+", " ", s)
    return s or f"Column {idx + 1}"


_UNIT_RX = [
    re.compile(r"^(?P<n>.*?)\s*\[(?P<u>[^\]]+)\]\s*$"),
    re.compile(r"^(?P<n>.*?)\s*\((?P<u>[^)]+)\)\s*$"),
    re.compile(r"^(?P<n>.*?)\s*/\s*(?P<u>[^/\s]+)\s*$"),
]


def _split_unit(name: str) -> tuple[str, str]:
    for rx in _UNIT_RX:
        m = rx.match(name)
        if m and m.group("n").strip():
            return m.group("n").strip(), m.group("u").strip()
    return name, ""


def _family_key(name: str) -> str:
    base, _ = _split_unit(name)
    base = re.sub(r"[\s_\-\.]*\d+$", "", base)  # trailing counter
    base = re.sub(r"^\d+[\s_\-\.]*", "", base)  # leading counter
    base = re.sub(r"[\s_\-\.]+", " ", base).strip().lower()
    return base


def _column_stats(values: list[float | None]) -> dict[str, Any]:
    nums = [v for v in values if v is not None]
    out: dict[str, Any] = {"numericCount": len(nums), "missing": len(values) - len(nums)}
    if not nums:
        return out
    out.update(min=min(nums), max=max(nums), mean=sum(nums) / len(nums))
    diffs = [b - a for a, b in zip(nums, nums[1:])]
    if diffs and all(d > 0 for d in diffs):
        out["monotonic"] = "increasing"
    elif diffs and all(d < 0 for d in diffs):
        out["monotonic"] = "decreasing"
    else:
        out["monotonic"] = "none"
    if len(diffs) >= 2:
        s = sorted(diffs)
        med = s[len(s) // 2]
        out["step"] = med
        out["uniformStep"] = bool(med != 0 and all(abs(d - med) <= 1e-6 * max(1.0, abs(med)) for d in diffs))
    return out


def _build_table(rows: list[list[Any]], name: str, source: str, dc: bool = False,
                 force_header: bool | None = None) -> dict[str, Any] | None:
    rows = [r for r in rows if any(not _is_blank(c) for c in r)]
    if not rows:
        return None
    notes: list[str] = []

    # Row-labelled wide table: text label first, then many numbers -> transpose.
    if (
        len(rows) <= 24
        and len(rows) >= 1
        and all(len(r) >= 6 for r in rows)
        and all(_parse_number(r[0], dc) is None and not _is_blank(r[0]) for r in rows)
        and all(_row_numeric_fraction(r[1:], dc) >= 0.8 for r in rows)
    ):
        width = max(len(r) for r in rows)
        header = [str(r[0]).strip() for r in rows]
        body = [[(rows[i][j] if j < len(rows[i]) else None) for i in range(len(rows))] for j in range(1, width)]
        rows = [header] + body
        notes.append("Each input line was a labelled series; the table was transposed into columns.")
        force_header = True

    # Ragged / single-line numeric data -> one flat sequence.
    widths = {len(r) for r in rows}
    numeric_rows = all(_row_numeric_fraction(r, dc) >= 0.8 for r in rows)
    if numeric_rows and (len(rows) == 1 or len(widths) > 1) and max(widths) >= 2:
        flat = [c for r in rows for c in r if not _is_blank(c)]
        rows = [["values"]] + [[c] for c in flat]
        force_header = True
        notes.append("Lines with different lengths were read as a single sequence of values.")

    has_header, has_units = _detect_header(rows, dc)
    if force_header is not None:
        has_header = force_header
        has_units = False if force_header and not has_units else has_units
    width = max(len(r) for r in rows)
    names: list[str] = []
    units: list[str] = []
    if has_header:
        head = rows[0]
        names = [_clean_name(head[i] if i < len(head) else None, i) for i in range(width)]
        body = rows[1:]
        if has_units and body:
            unit_row = body[0]
            units = [str(unit_row[i]).strip() if i < len(unit_row) and not _is_blank(unit_row[i]) else "" for i in range(width)]
            body = body[1:]
    else:
        names = [f"Column {i + 1}" for i in range(width)]
        body = rows
    if len(body) > MAX_ROWS:
        notes.append(f"Only the first {MAX_ROWS:,} rows were read.")
        body = body[:MAX_ROWS]

    columns: list[dict[str, Any]] = []
    for i in range(width):
        raw = [r[i] if i < len(r) else None for r in body]
        non_blank = [c for c in raw if not _is_blank(c)]
        nums = [_parse_number(c, dc) for c in raw]
        num_ok = sum(1 for c, v in zip(raw, nums) if not _is_blank(c) and v is not None)
        kind = "text"
        values: list[float | None] = []
        t0: datetime | None = None
        if non_blank and num_ok / len(non_blank) >= 0.8:
            kind, values = "numeric", nums
        else:
            dts = [_parse_datetime(c) for c in raw]
            dt_ok = sum(1 for c, d in zip(raw, dts) if not _is_blank(c) and d is not None)
            if non_blank and dt_ok / len(non_blank) >= 0.8:
                first = next(d for d in dts if d is not None)
                t0 = first
                kind = "datetime"
                values = [(d - first).total_seconds() if d is not None else None for d in dts]
        base_name, unit_from_name = _split_unit(names[i])
        unit = units[i] if units and units[i] else unit_from_name
        col: dict[str, Any] = {
            "index": i,
            "name": names[i],
            "label": base_name,
            "unit": unit,
            "kind": kind,
            "family": _family_key(names[i]),
        }
        if kind in ("numeric", "datetime"):
            col["values"] = values
            col.update(_column_stats(values))
            if t0 is not None:
                col["origin"] = t0.isoformat(sep=" ")
                col["unit"] = "s"
        else:
            col["samples"] = [str(c) for c in non_blank[:5]]
            col["distinct"] = len({str(c) for c in non_blank})
        columns.append(col)

    numeric_cols = [c for c in columns if c["kind"] in ("numeric", "datetime")]
    if not numeric_cols:
        return None

    # suggested roles
    x_idx: int | None = None
    for c in numeric_cols:
        if _INDEX_NAMES.match(c["label"]) or _X_NAMES.match(c["label"]) or c["kind"] == "datetime":
            if c.get("monotonic") in ("increasing", "decreasing") or c["kind"] == "datetime":
                x_idx = c["index"]
                break
    if x_idx is None and len(numeric_cols) >= 2:
        c0 = numeric_cols[0]
        if c0.get("monotonic") == "increasing" and c0.get("uniformStep"):
            x_idx = c0["index"]
    for c in columns:
        if c["kind"] == "text":
            c["suggestedRole"] = "ignore"
        elif c["index"] == x_idx:
            c["suggestedRole"] = "x"
        else:
            c["suggestedRole"] = "series"

    # grouping: heading families, units and x/y pairs
    groups: list[dict[str, Any]] = []
    fam: dict[str, list[int]] = {}
    for c in numeric_cols:
        if c["family"] and not re.match(r"^column \d+$", c["name"], re.I) and c["name"] != "values":
            fam.setdefault(c["family"], []).append(c["index"])
    for key, idxs in fam.items():
        if len(idxs) >= 2:
            groups.append({"key": f"family:{key}", "label": key.title(), "kind": "family", "columns": idxs})
    by_unit: dict[str, list[int]] = {}
    for c in numeric_cols:
        if c["unit"] and c["index"] != x_idx:
            by_unit.setdefault(c["unit"], []).append(c["index"])
    for u, idxs in by_unit.items():
        if len(idxs) >= 2:
            groups.append({"key": f"unit:{u}", "label": f"Unit: {u}", "kind": "unit", "columns": idxs})
    pairs: list[dict[str, Any]] = []
    xs = {}
    ys = {}
    for c in numeric_cols:
        m = re.match(r"^([xy])[\s_\-\.]*(.*)$", c["label"], re.I)
        if m:
            (xs if m.group(1).lower() == "x" else ys)[m.group(2).lower()] = c["index"]
    for suffix, xi in xs.items():
        if suffix in ys and len(xs) >= 2:
            pairs.append({"x": xi, "y": ys[suffix], "label": f"{names[ys[suffix]]} vs {names[xi]}"})
    for p in pairs:
        groups.append({"key": f"pair:{p['x']}:{p['y']}", "label": p["label"], "kind": "pair", "columns": [p["x"], p["y"]]})
    groups = [g for g in groups if not (g["kind"] == "family" and g["key"] in ("family:x", "family:y") and pairs)]
    if len(pairs) >= 2:  # X1,Y1,X2,Y2,... : every X belongs to its own Y
        for c in columns:
            if c["kind"] != "text":
                c["suggestedRole"] = "x" if c["index"] in {p["x"] for p in pairs} else "series"
        x_idx = None

    ys_cols = [c for c in numeric_cols if c["suggestedRole"] == "series"]
    if len(pairs) >= 2:
        analysis = "pairs"
    elif x_idx is not None and ys_cols:
        analysis = "curve"
    elif len(ys_cols) == 1:
        analysis = "sequence"
    elif len(ys_cols) >= 2:
        analysis = "multivariate"
    else:
        analysis = "sequence"

    preview = []
    for r in rows[: MAX_PREVIEW_ROWS + (2 if has_header else 0)]:
        preview.append([("" if c is None else str(c)) for c in (list(r) + [None] * (width - len(r)))[:width]])

    return {
        "name": name,
        "source": source,
        "headerDetected": has_header,
        "unitRowDetected": bool(units),
        "decimalComma": dc,
        "rowCount": len(body),
        "columns": columns,
        "groups": groups,
        "pairs": pairs,
        "suggestedX": x_idx,
        "suggestedAnalysis": analysis,
        "preview": preview,
        "notes": notes,
    }


# --------------------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------------------
def inspect_text(text: str, filename: str = "pasted data") -> dict[str, Any]:
    lines = [ln.rstrip("\r\n") for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    # keep '#' comment lines out, but remember one as a possible header
    comment_header: str | None = None
    kept: list[str] = []
    for ln in lines:
        stripped = ln.strip()
        if stripped.startswith(("#", "//")):
            if comment_header is None and re.search(r"[A-Za-z]", stripped):
                comment_header = stripped.lstrip("#/ ").strip()
            continue
        kept.append(ln)
    blocks = _split_blocks(kept)
    tables: list[dict[str, Any]] = []
    warnings: list[str] = []
    for bi, block in enumerate(blocks):
        rows, dc = _rows_from_text_block(block)
        if bi == 0 and comment_header and rows and _row_numeric_fraction(rows[0], dc) >= 0.8:
            delim = _sniff_delimiter([comment_header])
            head = (next(csv.reader([comment_header], delimiter=delim)) if delim else comment_header.split())
            if len(head) == len(rows[0]):
                rows = [head] + rows
        name = f"Table {len(tables) + 1}" if len(blocks) > 1 else "Data"
        t = _build_table(rows, name, filename, dc)
        if t:
            tables.append(t)
        elif len(block) >= 2:
            warnings.append(f"Block {bi + 1} had no numeric columns and was skipped.")
    # blocks that are just a lone header line belong to the table that follows
    if not tables:
        return {"ok": False, "error": "No numeric data was found in this input.", "filename": filename, "warnings": warnings}
    return {"ok": True, "filename": filename, "format": "text", "tables": tables, "warnings": warnings}


def _excel_rows_openpyxl(data: bytes) -> list[tuple[str, list[list[Any]]]]:
    import openpyxl  # noqa: PLC0415

    wb = openpyxl.load_workbook(io.BytesIO(data), data_only=True, read_only=True)
    out: list[tuple[str, list[list[Any]]]] = []
    try:
        for ws in wb.worksheets:
            rows = [list(r) for r in ws.iter_rows(values_only=True)]
            # trim trailing empty columns
            width = 0
            for r in rows:
                for i in range(len(r) - 1, -1, -1):
                    if not _is_blank(r[i]):
                        width = max(width, i + 1)
                        break
            rows = [r[:width] for r in rows]
            out.append((ws.title, rows))
    finally:
        wb.close()
    return out


def _excel_rows_pandas(data: bytes, suffix: str) -> list[tuple[str, list[list[Any]]]]:
    import pandas as pd  # noqa: PLC0415

    engine = {".xls": "xlrd", ".xlx": "xlrd", ".ods": "odf", ".xlsb": "pyxlsb"}.get(suffix)
    sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, header=None, engine=engine)
    out = []
    for name, df in sheets.items():
        df = df.astype(object).where(df.notna(), None)
        out.append((str(name), df.values.tolist()))
    return out


def inspect_bytes(filename: str, data: bytes) -> dict[str, Any]:
    suffix = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    # a ".xls"/".xlx" file that is really text (very common: CSV saved with an Excel extension)
    binary_magic = data[:2] == b"PK" or data[:4] == b"\xd0\xcf\x11\xe0"
    if suffix in {".xlsx", ".xlsm", ".xls", ".xlx", ".xlsb", ".ods"} and not binary_magic and b"\x00" not in data[:512]:
        suffix = ".txt"
    if suffix in {".xlsx", ".xlsm", ".xls", ".xlx", ".xlsb", ".ods"}:
        sheets: list[tuple[str, list[list[Any]]]]
        err: str | None = None
        sheets = []
        try:
            if data[:2] == b"PK":  # OOXML container, whatever the extension claims
                sheets = _excel_rows_openpyxl(data)
            else:
                sheets = _excel_rows_pandas(data, suffix)
        except ImportError as exc:
            err = (
                f"Reading {suffix} files needs an extra Python package ({exc}). "
                "Install it with: pip install openpyxl xlrd odfpy pyxlsb  - or export the sheet as CSV."
            )
        except Exception as exc:  # corrupt / password protected / wrong format
            err = f"Could not read this spreadsheet: {exc}"
        if err:
            return {"ok": False, "error": err, "filename": filename}
        tables = []
        warnings: list[str] = []
        for sname, rows in sheets:
            # split a sheet into blocks separated by fully blank rows
            blocks: list[list[list[Any]]] = []
            cur: list[list[Any]] = []
            for r in rows:
                if all(_is_blank(c) for c in r):
                    if cur:
                        blocks.append(cur)
                        cur = []
                else:
                    cur.append(r)
            if cur:
                blocks.append(cur)
            for bi, block in enumerate(blocks):
                label = sname if len(blocks) == 1 else f"{sname} - block {bi + 1}"
                t = _build_table(block, label, filename)
                if t:
                    tables.append(t)
        if not tables:
            return {"ok": False, "error": "No numeric columns were found in this workbook.", "filename": filename}
        return {"ok": True, "filename": filename, "format": suffix.lstrip("."), "tables": tables, "warnings": warnings}

    result = inspect_text(_decode(data), filename)
    if result.get("ok"):
        result["format"] = suffix.lstrip(".") or "text"
    return result
