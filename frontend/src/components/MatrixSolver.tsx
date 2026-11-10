/* eslint-disable @typescript-eslint/no-explicit-any -- backend payloads are dynamic JSON */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { ClipboardEvent, KeyboardEvent, ReactNode } from 'react';
import 'katex/dist/katex.min.css';
import { BlockMath } from 'react-katex';

/* ------------------------------------------------------------------------------------------
 * Linear Algebra workspace
 *
 *  - Operations tab : pick an operation -> enter A (and b / B / options) -> Run -> structured result.
 *  - Console tab    : MATLAB-style scratchpad (A = [[1,2],[3,4]]; A*B + 2*A; det(A) ...).
 *
 * All maths is done by the FastAPI/NumPy/SymPy backend (/api/matrix/*); this file only edits
 * inputs and presents results.
 * ------------------------------------------------------------------------------------------ */

type Matrix = string[][];
type Vector = string[];

type Operation =
  | 'solve' | 'determinant' | 'inverse' | 'rank' | 'rref' | 'nullspace' | 'eigenvalues' | 'eigenvectors'
  | 'svd' | 'qr' | 'lu' | 'cholesky' | 'pseudoinverse'
  | 'norm' | 'transpose' | 'multiply' | 'power' | 'trace'
  | 'differentiate' | 'integrate' | 'symbolic';

type Category = 'Core' | 'Decompositions' | 'Transforms' | 'Calculus';

interface OperationInfo {
  id: Operation;
  label: string;
  hint: string;
  category: Category;
}

interface HistoryItem {
  id: number;
  label: string;
  result: any;
  time: string;
}

interface ConsoleEntry {
  id: number;
  command: string;
  result: any;
  error?: string;
}

const API_URL = (import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000').replace(/\/+$/, '');
const API_BASE = `${API_URL}/api/matrix`;
const MAX_DIM = 12;

const OPERATIONS: OperationInfo[] = [
  { id: 'solve', label: 'Solve Ax = b', hint: 'Linear system / least squares', category: 'Core' },
  { id: 'determinant', label: 'Determinant', hint: 'det(A)', category: 'Core' },
  { id: 'inverse', label: 'Inverse', hint: 'A⁻¹', category: 'Core' },
  { id: 'rank', label: 'Rank', hint: 'Number of independent rows', category: 'Core' },
  { id: 'rref', label: 'RREF', hint: 'Reduced row-echelon form', category: 'Core' },
  { id: 'nullspace', label: 'Null space', hint: 'Solutions of Ax = 0', category: 'Core' },
  { id: 'eigenvalues', label: 'Eigenvalues', hint: 'Spectrum of A', category: 'Core' },
  { id: 'eigenvectors', label: 'Eigenvectors', hint: 'Eigenpairs of A', category: 'Core' },
  { id: 'svd', label: 'SVD', hint: 'A = U Σ Vᵀ', category: 'Decompositions' },
  { id: 'qr', label: 'QR', hint: 'A = Q R', category: 'Decompositions' },
  { id: 'lu', label: 'LU', hint: 'A = P L U (pivoted)', category: 'Decompositions' },
  { id: 'cholesky', label: 'Cholesky', hint: 'A = L Lᵀ (symmetric positive definite)', category: 'Decompositions' },
  { id: 'pseudoinverse', label: 'Pseudoinverse', hint: 'Moore–Penrose A⁺', category: 'Decompositions' },
  { id: 'transpose', label: 'Transpose', hint: 'Aᵀ', category: 'Transforms' },
  { id: 'trace', label: 'Trace', hint: 'tr(A)', category: 'Transforms' },
  { id: 'norm', label: 'Norms', hint: 'Frobenius, spectral, 1, ∞', category: 'Transforms' },
  { id: 'multiply', label: 'Multiply A × B', hint: 'Matrix product', category: 'Transforms' },
  { id: 'power', label: 'Matrix power', hint: 'Aⁿ', category: 'Transforms' },
  { id: 'differentiate', label: 'Differentiate', hint: 'Derivative of every entry', category: 'Calculus' },
  { id: 'integrate', label: 'Integrate', hint: 'Indefinite or definite, every entry', category: 'Calculus' },
  { id: 'symbolic', label: 'Symbolic expression', hint: 'Simplify / differentiate / integrate', category: 'Calculus' },
];

const CATEGORIES: Category[] = ['Core', 'Decompositions', 'Transforms', 'Calculus'];

const EXAMPLES: Record<Operation, { A: Matrix; b?: Vector; B?: Matrix; power?: number }> = {
  solve: { A: [['2', '1'], ['1', '3']], b: ['5', '7'] },
  determinant: { A: [['4', '2'], ['1', '3']] },
  inverse: { A: [['4', '7'], ['2', '6']] },
  rank: { A: [['1', '2', '3'], ['2', '4', '6'], ['1', '0', '1']] },
  rref: { A: [['1', '2', '3'], ['2', '4', '7'], ['1', '1', '1']] },
  nullspace: { A: [['1', '2', '3'], ['2', '4', '6']] },
  eigenvalues: { A: [['2', '1'], ['1', '2']] },
  eigenvectors: { A: [['2', '1'], ['1', '2']] },
  svd: { A: [['1', '2'], ['3', '4'], ['5', '6']] },
  qr: { A: [['1', '2'], ['3', '4'], ['5', '6']] },
  lu: { A: [['4', '3'], ['6', '3']] },
  cholesky: { A: [['4', '2'], ['2', '5']] },
  pseudoinverse: { A: [['1', '2', '3'], ['4', '5', '6']] },
  norm: { A: [['1', '-2'], ['3', '4']] },
  transpose: { A: [['1', '2', '3'], ['4', '5', '6']] },
  multiply: { A: [['1', '2', '3'], ['4', '5', '6']], B: [['1', '2'], ['3', '4'], ['5', '6']] },
  power: { A: [['1', '1'], ['0', '1']], power: 5 },
  trace: { A: [['1', '2'], ['3', '4']] },
  differentiate: { A: [['sin(x)', 'x^3'], ['log(x)', 'exp(x)']] },
  integrate: { A: [['sin(x)', 'x^2'], ['cos(x)', 'exp(x)']] },
  symbolic: { A: [['0']] },
};

const CONSOLE_CHIPS = ['help()', 'A = [[2,1],[1,3]]', 'b = [5,7]', 'solve(A,b)', 'det(A)', "A'", 'A^-1', 'A*A', 'eigenvalues(A)', 'symbols()'];

/* ------------------------------------------------------------------------------------------
 * helpers
 * ------------------------------------------------------------------------------------------ */
const makeMatrix = (rows: number, cols: number, fill = '0'): Matrix =>
  Array.from({ length: rows }, () => Array.from({ length: cols }, () => fill));

/** Resize while keeping every entry that still fits (changing a dimension never wipes the data). */
function resizeMatrix(m: Matrix, rows: number, cols: number): Matrix {
  return Array.from({ length: rows }, (_, r) => Array.from({ length: cols }, (_, c) => m[r]?.[c] ?? '0'));
}

function resizeVector(v: Vector, n: number): Vector {
  return Array.from({ length: n }, (_, i) => v[i] ?? '0');
}

const clampDim = (n: number) => Math.max(1, Math.min(MAX_DIM, Math.round(Number.isFinite(n) ? n : 1)));

function splitTopLevel(text: string, separators: RegExp): string[] {
  const out: string[] = [];
  let depth = 0;
  let cur = '';
  for (const ch of text) {
    if ('([{'.includes(ch)) depth += 1;
    else if (')]}'.includes(ch)) depth = Math.max(0, depth - 1);
    if (depth === 0 && separators.test(ch)) {
      out.push(cur);
      cur = '';
    } else cur += ch;
  }
  out.push(cur);
  return out;
}

/** Accepts JSON ([[1,2],[3,4]]), MATLAB ([1 2; 3 4]), or plain grids (spreadsheet paste, CSV, TSV, spaces). */
function parseMatrixText(raw: string): Matrix {
  let text = raw.trim();
  if (!text) throw new Error('Nothing to import.');
  if (text.startsWith('[') && text.includes('"')) {
    const parsed = JSON.parse(text);
    if (!Array.isArray(parsed)) throw new Error('Expected a list of rows.');
    const rows = Array.isArray(parsed[0]) ? parsed : [parsed];
    return rows.map((r: any[]) => r.map((c) => String(c)));
  }
  text = text.replace(/^\[+/, '').replace(/\]+$/, '');
  const rowTexts = text.includes(';') || text.includes('\n') || text.includes('],')
    ? text.split(/;|\r?\n|\]\s*,\s*\[|\]\s*\[/).map((s) => s.replace(/[[\]]/g, ''))
    : [text.replace(/[[\]]/g, '')];
  const rows = rowTexts
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      const sep = line.includes('\t') ? /\t/ : line.includes(',') ? /,/ : /\s/;
      return splitTopLevel(line, sep).map((c) => c.trim()).filter((c) => c !== '');
    })
    .filter((r) => r.length);
  if (!rows.length) throw new Error('No numbers found.');
  const width = Math.max(...rows.map((r) => r.length));
  return rows.map((r) => Array.from({ length: width }, (_, i) => r[i] ?? '0'));
}

function pretty(value: any): string {
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return '—';
    if (Math.abs(value) < 1e-10) return '0';
    const abs = Math.abs(value);
    if (abs >= 1e7 || abs < 1e-4) return value.toExponential(5).replace(/\.?0+e/, 'e');
    return Number(value.toPrecision(8)).toString();
  }
  if (value && typeof value === 'object' && 'real' in value && 'imag' in value) {
    const re = pretty(value.real);
    const im = pretty(Math.abs(value.imag));
    return `${re} ${value.imag >= 0 ? '+' : '−'} ${im}i`;
  }
  if (value === null || value === undefined) return '—';
  return String(value);
}

const PRETTY_KEY: Record<string, string> = {
  Vt: 'Vᵀ', U: 'U', Q: 'Q', R: 'R', L: 'L', P: 'P',
  singular_values: 'Singular values', residual_vector: 'Residual vector', residual_norm: 'Residual norm',
  condition_number: 'Condition number', augmented_rank: 'Augmented rank', null_space: 'Null-space basis',
  one_norm: '1-norm', infinity_norm: '∞-norm', orthogonality_error: 'Orthogonality error',
};
const keyLabel = (key: string) => PRETTY_KEY[key] ?? key.replaceAll('_', ' ').replace(/^./, (c) => c.toUpperCase());

const HIDDEN_RESULT_KEYS = new Set(['error', 'metadata', 'operation', 'status', 'explanation', 'method', 'latex', 'type']);
/** Result fields that can be sent back into the A editor. */
const MATRIX_KEYS = new Set(['result', 'inverse', 'U', 'Vt', 'Q', 'R', 'L', 'P', 'reconstruction', 'vectors']);

const isMatrixValue = (v: any): v is any[][] => Array.isArray(v) && v.length > 0 && Array.isArray(v[0]);

/* ------------------------------------------------------------------------------------------
 * small UI atoms
 * ------------------------------------------------------------------------------------------ */
function Card({ title, step, right, children }: { title: string; step?: number; right?: ReactNode; children: ReactNode }) {
  return (
    <section className="rounded-2xl border border-slate-200 bg-white shadow-sm">
      <header className="flex items-center justify-between gap-3 border-b border-slate-100 px-5 py-3.5">
        <h2 className="flex items-center gap-2.5 text-sm font-bold text-slate-800">
          {step !== undefined && (
            <span className="flex h-6 w-6 items-center justify-center rounded-full bg-blue-600 text-[11px] font-bold text-white">{step}</span>
          )}
          {title}
        </h2>
        {right}
      </header>
      <div className="p-5">{children}</div>
    </section>
  );
}

const ghostButton =
  'inline-flex items-center justify-center rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 text-xs font-semibold text-slate-600 transition hover:border-slate-300 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-40';
const fieldClass =
  'w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-800 outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-100';

function Stepper({ label, value, onChange }: { label: string; value: number; onChange: (n: number) => void }) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs font-semibold text-slate-500">{label}</span>
      <div className="flex items-center overflow-hidden rounded-lg border border-slate-300 bg-white">
        <button type="button" aria-label={`Decrease ${label}`} onClick={() => onChange(clampDim(value - 1))} className="px-2.5 py-1 text-slate-500 hover:bg-slate-100">−</button>
        <input
          aria-label={label}
          type="number"
          min={1}
          max={MAX_DIM}
          value={value}
          onChange={(e) => onChange(clampDim(Number(e.target.value)))}
          className="w-10 border-x border-slate-200 py-1 text-center text-sm font-semibold outline-none"
        />
        <button type="button" aria-label={`Increase ${label}`} onClick={() => onChange(clampDim(value + 1))} className="px-2.5 py-1 text-slate-500 hover:bg-slate-100">+</button>
      </div>
    </div>
  );
}

/** Editable grid. `lockRows` locks the number of rows (used for B whose rows must equal columns of A). */
function MatrixEditor({
  matrix, onChange, name, onResize, lockRows, lockCols, onPasteError,
}: {
  matrix: Matrix;
  onChange: (m: Matrix) => void;
  name: string;
  onResize: (rows: number, cols: number) => void;
  lockRows?: boolean;
  lockCols?: boolean;
  onPasteError: (message: string) => void;
}) {
  const rows = matrix.length;
  const cols = matrix[0]?.length ?? 1;
  const refs = useRef<Record<string, HTMLInputElement | null>>({});

  const focusCell = (r: number, c: number) => {
    const el = refs.current[`${r}:${c}`];
    if (el) {
      el.focus();
      el.select();
    }
  };

  const setCell = (r: number, c: number, value: string) => {
    const next = matrix.map((row) => [...row]);
    next[r][c] = value;
    onChange(next);
  };

  const onKey = (e: KeyboardEvent<HTMLInputElement>, r: number, c: number) => {
    const atEnd = e.currentTarget.selectionStart === e.currentTarget.value.length;
    const atStart = e.currentTarget.selectionEnd === 0;
    if (e.key === 'ArrowDown' || e.key === 'Enter') { e.preventDefault(); focusCell(Math.min(rows - 1, r + 1), c); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); focusCell(Math.max(0, r - 1), c); }
    else if (e.key === 'ArrowRight' && atEnd) { e.preventDefault(); focusCell(r, Math.min(cols - 1, c + 1)); }
    else if (e.key === 'ArrowLeft' && atStart) { e.preventDefault(); focusCell(r, Math.max(0, c - 1)); }
  };

  /** Pasting a spreadsheet block into any cell fills the grid starting there (growing it if needed). */
  const onPaste = (e: ClipboardEvent<HTMLInputElement>, r: number, c: number) => {
    const text = e.clipboardData.getData('text');
    if (!/[\t\n;]/.test(text.trim())) return; // single value: normal paste
    e.preventDefault();
    try {
      const block = parseMatrixText(text);
      const needRows = lockRows ? rows : clampDim(Math.max(rows, r + block.length));
      const needCols = lockCols ? cols : clampDim(Math.max(cols, c + block[0].length));
      const next = resizeMatrix(matrix, needRows, needCols);
      block.forEach((br, i) => br.forEach((v, j) => {
        if (r + i < needRows && c + j < needCols) next[r + i][c + j] = v;
      }));
      onChange(next);
    } catch (err: any) {
      onPasteError(err?.message || 'Could not paste that block.');
    }
  };

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-4">
          <span className="rounded-md bg-slate-900 px-2 py-0.5 font-mono text-xs font-bold text-white">{name}</span>
          {!lockRows && <Stepper label="Rows" value={rows} onChange={(n) => onResize(n, cols)} />}
          {!lockCols && <Stepper label="Cols" value={cols} onChange={(n) => onResize(rows, n)} />}
          {lockRows && <span className="text-xs text-slate-500">{rows} rows (= columns of A)</span>}
        </div>
        <div className="flex flex-wrap gap-1.5">
          <button type="button" className={ghostButton} onClick={() => onChange(makeMatrix(rows, cols))}>Zeros</button>
          <button type="button" className={ghostButton} onClick={() => onChange(matrix.map((row, r) => row.map((_, c) => (r === c ? '1' : '0'))))}>Identity</button>
          <button type="button" className={ghostButton} onClick={() => onChange(matrix.map((row) => row.map(() => String(Math.floor(Math.random() * 19) - 9))))}>Random</button>
        </div>
      </div>
      <div className="max-w-full overflow-auto rounded-xl border border-slate-200 bg-slate-50 p-3">
        <div className="inline-block">
          {matrix.map((row, r) => (
            <div key={r} className="flex gap-1.5 pb-1.5 last:pb-0">
              {row.map((cell, c) => (
                <input
                  key={c}
                  ref={(el) => { refs.current[`${r}:${c}`] = el; }}
                  aria-label={`${name}[${r + 1},${c + 1}]`}
                  value={cell}
                  spellCheck={false}
                  onFocus={(e) => e.currentTarget.select()}
                  onChange={(e) => setCell(r, c, e.target.value)}
                  onKeyDown={(e) => onKey(e, r, c)}
                  onPaste={(e) => onPaste(e, r, c)}
                  className="h-10 w-[4.5rem] rounded-lg border border-slate-300 bg-white px-1.5 text-center font-mono text-sm text-slate-800 outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
                />
              ))}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function VectorEditor({ vector, onChange }: { vector: Vector; onChange: (v: Vector) => void }) {
  return (
    <div>
      <div className="mb-3 flex items-center gap-3">
        <span className="rounded-md bg-slate-900 px-2 py-0.5 font-mono text-xs font-bold text-white">b</span>
        <span className="text-xs text-slate-500">{vector.length} values (one per row of A)</span>
      </div>
      <div className="flex flex-wrap gap-1.5 rounded-xl border border-slate-200 bg-slate-50 p-3">
        {vector.map((v, i) => (
          <input
            key={i}
            aria-label={`b[${i + 1}]`}
            value={v}
            spellCheck={false}
            onFocus={(e) => e.currentTarget.select()}
            onChange={(e) => onChange(vector.map((old, j) => (j === i ? e.target.value : old)))}
            className="h-10 w-[4.5rem] rounded-lg border border-slate-300 bg-white px-1.5 text-center font-mono text-sm outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
          />
        ))}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------------------------------
 * result presentation
 * ------------------------------------------------------------------------------------------ */
function DataTable({ value, rowLabel }: { value: any[][]; rowLabel?: string }) {
  return (
    <div className="max-w-full overflow-auto rounded-xl border border-slate-200">
      <table className="min-w-full border-collapse text-sm">
        <tbody>
          {value.map((row, r) => (
            <tr key={r} className="odd:bg-white even:bg-slate-50/60">
              {rowLabel && <td className="border-r border-slate-200 bg-slate-100 px-3 py-2 text-xs font-semibold text-slate-500">{rowLabel}{r + 1}</td>}
              {row.map((cell, c) => (
                <td key={c} className="border-r border-slate-100 px-3 py-2 text-right font-mono tabular-nums last:border-r-0">{pretty(cell)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ValueView({ field, value }: { field: string; value: any }) {
  if (isMatrixValue(value)) {
    // a list of vectors (null-space basis, eigenvector list) reads best as one vector per row
    const asVectors = field === 'basis' || field === 'null_space';
    return <DataTable value={value} rowLabel={asVectors ? 'v' : undefined} />;
  }
  if (Array.isArray(value)) {
    if (!value.length) return <span className="text-sm text-slate-400">none</span>;
    return (
      <div className="flex flex-wrap gap-1.5">
        {value.map((item, i) => (
          <span key={i} className="rounded-lg bg-slate-100 px-2.5 py-1 font-mono text-sm tabular-nums text-slate-800">{pretty(item)}</span>
        ))}
      </div>
    );
  }
  if (value && typeof value === 'object' && !('real' in value && 'imag' in value)) {
    return <pre className="max-h-60 overflow-auto rounded-xl bg-slate-900 p-3 text-xs text-slate-100">{JSON.stringify(value, null, 2)}</pre>;
  }
  return <span className="font-mono text-lg font-semibold tabular-nums text-slate-900">{pretty(value)}</span>;
}

function StatusBanner({ result }: { result: any }) {
  const status: string = result.status;
  const tone =
    status === 'unique_solution' ? 'border-emerald-200 bg-emerald-50 text-emerald-800'
      : status === 'infinitely_many_solutions' ? 'border-amber-200 bg-amber-50 text-amber-800'
        : 'border-rose-200 bg-rose-50 text-rose-800';
  return (
    <div className={`rounded-xl border p-4 ${tone}`}>
      <div className="text-[11px] font-bold uppercase tracking-wider">{status.replaceAll('_', ' ')}</div>
      {result.explanation && <p className="mt-1 text-sm leading-relaxed">{result.explanation}</p>}
      {result.method && <p className="mt-1.5 text-xs opacity-75">Method: {String(result.method).replace('_', ' ')}</p>}
    </div>
  );
}

function ResultPanel({ result, onUseAsA }: { result: any; onUseAsA: (m: Matrix) => void }) {
  const [showLatex, setShowLatex] = useState(true);
  if (result?.error) {
    return <div className="rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm leading-relaxed text-rose-800">{result.error}</div>;
  }
  const entries = Object.entries(result).filter(([k, v]) => !HIDDEN_RESULT_KEYS.has(k) && v !== null && v !== undefined);
  const scalars = entries.filter(([, v]) => !Array.isArray(v) && (typeof v !== 'object' || (v && 'real' in (v as any))));
  const blocks = entries.filter(([, v]) => Array.isArray(v) || (typeof v === 'object' && v !== null && !('real' in (v as any))));
  const metadata: Record<string, any> | undefined = result.metadata;

  return (
    <div className="space-y-5">
      {result.status && <StatusBanner result={result} />}

      {scalars.length > 0 && (
        <div className="grid grid-cols-2 gap-3">
          {scalars.map(([k, v]) => (
            <div key={k} className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3">
              <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">{keyLabel(k)}</div>
              <div className="mt-0.5"><ValueView field={k} value={v} /></div>
            </div>
          ))}
        </div>
      )}

      {blocks.map(([k, v]) => (
        <section key={k} className="space-y-2">
          <div className="flex items-center justify-between">
            <h3 className="text-[11px] font-bold uppercase tracking-wider text-slate-400">{keyLabel(k)}</h3>
            {MATRIX_KEYS.has(k) && isMatrixValue(v) && (
              <button
                type="button"
                className="text-xs font-semibold text-blue-600 hover:text-blue-800"
                onClick={() => onUseAsA((v as any[][]).map((row) => row.map((c) => (typeof c === 'number' ? String(Number(c.toPrecision(12))) : String(c)))))}
              >
                Use as matrix A →
              </button>
            )}
          </div>
          <ValueView field={k} value={v} />
        </section>
      ))}

      {result.latex && (
        <section>
          <button type="button" className="mb-2 text-[11px] font-bold uppercase tracking-wider text-slate-400 hover:text-slate-600" onClick={() => setShowLatex((s) => !s)}>
            {showLatex ? '▾' : '▸'} Typeset
          </button>
          {showLatex && (
            <div className="overflow-auto rounded-xl border border-slate-200 bg-white p-3">
              <BlockMath math={String(result.latex)} renderError={() => <code className="text-xs text-slate-500">{String(result.latex)}</code>} />
            </div>
          )}
        </section>
      )}

      {metadata && (
        <section>
          <h3 className="mb-2 text-[11px] font-bold uppercase tracking-wider text-slate-400">Matrix diagnostics</h3>
          <div className="grid grid-cols-3 gap-2">
            {Object.entries(metadata)
              .filter(([k]) => !['rows', 'columns', 'singular_values'].includes(k))
              .map(([k, v]) => (
                <div key={k} className="rounded-lg bg-slate-50 px-3 py-2">
                  <div className="text-[9px] font-semibold uppercase tracking-wide text-slate-400">{keyLabel(k)}</div>
                  <div className="mt-0.5 truncate text-xs font-semibold text-slate-700">
                    {Array.isArray(v) ? v.join(' × ') : typeof v === 'boolean' ? (v ? 'yes' : 'no') : pretty(v)}
                  </div>
                </div>
              ))}
          </div>
        </section>
      )}
    </div>
  );
}

/** Compact rendering of one console result (assignment, matrix, value, operation result, help). */
function ConsoleResult({ result }: { result: any }) {
  if (!result) return null;
  if (result.type === 'help') {
    return <ul className="space-y-0.5 font-mono text-xs text-slate-300">{result.commands.map((c: string) => <li key={c}>{c}</li>)}</ul>;
  }
  if (result.type === 'assignment') {
    const v = result.value;
    return (
      <div className="text-sm">
        <span className="font-mono font-bold text-emerald-300">{result.name}</span>
        <span className="text-slate-400"> = </span>
        {isMatrixValue(v) ? <div className="mt-2 text-slate-900"><DataTable value={v} /></div> : <span className="font-mono">{Array.isArray(v) ? `[${v.join(', ')}]` : String(v)}</span>}
      </div>
    );
  }
  if (result.type === 'clear') return <span className="text-sm text-slate-300">Variables cleared.</span>;
  if (result.type === 'variables') return <span className="font-mono text-sm">{Object.keys(result.variables || {}).join(', ') || '(none)'}</span>;
  if (result.type === 'shape') return <span className="font-mono text-sm">{result.value.join(' × ')}</span>;
  if (result.type === 'value') return <span className="font-mono text-base">{result.value}</span>;
  return (
    <div className="text-slate-900">
      <ResultPanel result={result} onUseAsA={() => undefined} />
    </div>
  );
}

/* ------------------------------------------------------------------------------------------
 * main component
 * ------------------------------------------------------------------------------------------ */
export default function MatrixSolver() {
  const [tab, setTab] = useState<'operations' | 'console'>('operations');
  const [category, setCategory] = useState<Category>('Core');
  const [operation, setOperation] = useState<Operation>('solve');

  const [matrixA, setMatrixA] = useState<Matrix>(EXAMPLES.solve.A);
  const [vectorB, setVectorB] = useState<Vector>(EXAMPLES.solve.b ?? ['0', '0']);
  const [matrixB, setMatrixB] = useState<Matrix>(makeMatrix(2, 2));
  const [power, setPower] = useState(2);
  const [variable, setVariable] = useState('x');
  const [order, setOrder] = useState(1);
  const [lower, setLower] = useState('');
  const [upper, setUpper] = useState('');
  const [symbolicExpression, setSymbolicExpression] = useState('sin(x)^2 + cos(x)^2');
  const [symbolicOp, setSymbolicOp] = useState<'simplify' | 'differentiate' | 'integrate'>('simplify');
  const [tolerance, setTolerance] = useState('1e-10');
  const [showImport, setShowImport] = useState(false);
  const [pasteText, setPasteText] = useState('');

  const [result, setResult] = useState<any>(null);
  const [resultLabel, setResultLabel] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [online, setOnline] = useState<boolean | null>(null);
  const [backendInfo, setBackendInfo] = useState('');
  const [history, setHistory] = useState<HistoryItem[]>([]);

  const [variables, setVariables] = useState<Record<string, any>>({});
  const [consoleInput, setConsoleInput] = useState('');
  const [consoleLog, setConsoleLog] = useState<ConsoleEntry[]>([]);
  const [recall, setRecall] = useState<string[]>([]);
  const [recallIndex, setRecallIndex] = useState(-1);
  const consoleEnd = useRef<HTMLDivElement | null>(null);

  const info = useMemo(() => OPERATIONS.find((o) => o.id === operation)!, [operation]);
  const rows = matrixA.length;
  const cols = matrixA[0]?.length ?? 1;
  const needsMatrix = operation !== 'symbolic';
  const showCalculusOptions = operation === 'differentiate' || operation === 'integrate' || (operation === 'symbolic' && symbolicOp !== 'simplify');
  const showBounds = operation === 'integrate' || (operation === 'symbolic' && symbolicOp === 'integrate');

  /* ---- backend health ---- */
  const checkHealth = useCallback(() => {
    fetch(`${API_BASE}/health`, { cache: 'no-store' })
      .then(async (response) => {
        setOnline(response.ok);
        if (response.ok) {
          const data = await response.json().catch(() => ({}));
          setBackendInfo([data.numpy && `NumPy ${data.numpy}`, data.sympy && `SymPy ${data.sympy}`, data.scipy === false && 'SciPy missing'].filter(Boolean).join(' · '));
        }
      })
      .catch(() => setOnline(false));
  }, []);

  useEffect(() => {
    const first = window.setTimeout(checkHealth, 0);
    const timer = window.setInterval(checkHealth, 30000);
    return () => { window.clearTimeout(first); window.clearInterval(timer); };
  }, [checkHealth]);

  /* ---- data editing ---- */
  const resizeA = (r: number, c: number) => {
    setMatrixA((m) => resizeMatrix(m, clampDim(r), clampDim(c)));
    setVectorB((v) => resizeVector(v, clampDim(r)));
  };

  const resizeB = (_r: number, c: number) => setMatrixB(resizeMatrix(matrixBView, cols, clampDim(c)));

  // B must have as many rows as A has columns: derive the grid instead of syncing state in an effect
  const matrixBView = matrixB.length === cols ? matrixB : resizeMatrix(matrixB, cols, matrixB[0]?.length ?? 1);

  const selectOperation = (id: Operation) => {
    setOperation(id);
    setResult(null);
    setError(null);
    if (id === 'symbolic') setSymbolicOp('simplify');
    if (id === 'differentiate' || id === 'integrate') {
      // calculus works on symbolic entries: offer the example if the grid is still plain numbers
      const plain = matrixA.every((row) => row.every((c) => /^-?\d*\.?\d+$/.test(c.trim())));
      if (plain) setMatrixA(EXAMPLES[id].A);
    }
  };

  const loadExample = () => {
    const ex = EXAMPLES[operation];
    if (operation === 'symbolic') {
      setSymbolicExpression('x^2 * exp(x)');
      return;
    }
    const r = ex.A.length;
    const c = ex.A[0].length;
    setMatrixA(ex.A.map((row) => [...row]));
    setVectorB(ex.b ? [...ex.b] : makeMatrix(1, r)[0]);
    setMatrixB(ex.B ? ex.B.map((row) => [...row]) : makeMatrix(c, r));
    if (ex.power !== undefined) setPower(ex.power);
    setResult(null);
    setError(null);
  };

  const importText = () => {
    try {
      const m = parseMatrixText(pasteText);
      if (m.length > MAX_DIM || m[0].length > MAX_DIM) throw new Error(`Matrices up to ${MAX_DIM}×${MAX_DIM} are supported.`);
      setMatrixA(m);
      setVectorB((v) => resizeVector(v, m.length));
      setPasteText('');
      setShowImport(false);
      setError(null);
    } catch (err: any) {
      setError(err?.message || 'Could not read that matrix.');
    }
  };

  /* ---- API ---- */
  const tol = Number(tolerance) > 0 ? Number(tolerance) : 1e-10;

  const post = useCallback(async (path: string, body: unknown) => {
    const response = await fetch(`${API_BASE}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await response.json().catch(() => ({ error: `HTTP ${response.status}` }));
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : data.error || `HTTP ${response.status}`);
    if (data.error) throw new Error(data.error);
    setOnline(true);
    return data;
  }, []);

  const execute = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      let data: any;
      let label = info.label;
      if (operation === 'solve') {
        data = await post('/solve', { matrix_a: matrixA, vector_b: vectorB, tolerance: tol });
      } else if (operation === 'differentiate' || operation === 'integrate') {
        data = await post('/calculus', {
          operation, matrix: matrixA, variable, order,
          lower: operation === 'integrate' && lower.trim() ? lower : null,
          upper: operation === 'integrate' && upper.trim() ? upper : null,
        });
      } else if (operation === 'symbolic') {
        data = await post('/symbolic', {
          expression: symbolicExpression, operation: symbolicOp, variable, order,
          lower: symbolicOp === 'integrate' && lower.trim() ? lower : null,
          upper: symbolicOp === 'integrate' && upper.trim() ? upper : null,
        });
        label = `Symbolic · ${symbolicOp}`;
      } else {
        data = await post('/operation', {
          operation, matrix_a: matrixA,
          matrix_b: operation === 'multiply' ? matrixBView : undefined,
          power: operation === 'power' ? power : undefined,
          tolerance: tol,
        });
      }
      setResult(data);
      setResultLabel(label);
      setHistory((items) => [{ id: Date.now(), label, result: data, time: new Date().toLocaleTimeString() }, ...items].slice(0, 12));
    } catch (err: any) {
      const unreachable = err instanceof TypeError;
      if (unreachable) setOnline(false);
      setResult(null);
      setError(unreachable ? 'Cannot reach the backend. Is the FastAPI server running?' : err?.message || 'Calculation failed.');
    } finally {
      setLoading(false);
    }
  }, [operation, matrixA, vectorB, matrixBView, power, variable, order, lower, upper, symbolicExpression, symbolicOp, tol, info.label, post]);

  useEffect(() => {
    if (tab !== 'operations') return;
    const handler = (event: globalThis.KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
        event.preventDefault();
        void execute();
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [tab, execute]);

  async function runCommand(command = consoleInput) {
    const trimmed = command.trim();
    if (!trimmed) return;
    setLoading(true);
    const id = Date.now();
    try {
      const response = await fetch(`${API_BASE}/command`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command: trimmed, variables, tolerance: tol }),
      });
      const data = await response.json();
      setOnline(true);
      if (data.variables) setVariables(data.variables);
      const results: Array<{ command: string; result: any }> = data.results?.length ? data.results : [{ command: trimmed, result: data.result }];
      const entries: ConsoleEntry[] = results.map((r, i) => ({ id: id + i, command: r.command, result: r.result }));
      if (data.error) entries.push({ id: id + entries.length, command: data.results?.length ? '' : trimmed, result: null, error: data.error });
      setConsoleLog((log) => [...log, ...entries].slice(-60));
      setRecall((r) => [trimmed, ...r.filter((x) => x !== trimmed)].slice(0, 40));
      setRecallIndex(-1);
      if (!data.error) setConsoleInput('');
    } catch {
      setOnline(false);
      setConsoleLog((log) => [...log, { id, command: trimmed, result: null, error: 'Cannot reach the backend. Is the FastAPI server running?' }]);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    consoleEnd.current?.scrollIntoView?.({ block: 'end' });
  }, [consoleLog]);

  const onConsoleKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') { e.preventDefault(); void runCommand(); }
    else if (e.key === 'ArrowUp' && recall.length) {
      e.preventDefault();
      const next = Math.min(recall.length - 1, recallIndex + 1);
      setRecallIndex(next);
      setConsoleInput(recall[next]);
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      const next = recallIndex - 1;
      setRecallIndex(next);
      setConsoleInput(next >= 0 ? recall[next] : '');
    }
  };

  const sendEditorToConsole = (target: 'A' | 'B') => {
    const m = target === 'A' ? matrixA : matrixBView;
    setConsoleInput(`${target} = [${m.map((r) => `[${r.join(',')}]`).join(',')}]`);
    setTab('console');
  };

  const clearAll = () => {
    setMatrixA(makeMatrix(rows, cols));
    setVectorB(makeMatrix(1, rows)[0]);
    setMatrixB(makeMatrix(cols, matrixBView[0]?.length ?? 1));
    setResult(null);
    setError(null);
  };

  /* ---------------------------------------------------------------------------------------- */
  return (
    <div className="space-y-5 text-slate-900">
      {/* header */}
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Linear Algebra</h1>
          <p className="mt-1 text-sm text-slate-500">Systems, decompositions, eigenproblems and matrix calculus — computed by NumPy &amp; SymPy.</p>
        </div>
        <div className="flex items-center gap-3">
          <label className="flex items-center gap-2 text-xs font-semibold text-slate-500" title="Numerical tolerance used for rank decisions">
            Tolerance
            <input value={tolerance} onChange={(e) => setTolerance(e.target.value)} className="w-24 rounded-lg border border-slate-300 bg-white px-2 py-1 font-mono text-xs outline-none focus:border-blue-500" />
          </label>
          <button
            type="button"
            onClick={checkHealth}
            title={backendInfo || 'Check backend connection'}
            className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-semibold ${
              online === null ? 'border-slate-200 bg-white text-slate-500'
                : online ? 'border-emerald-200 bg-emerald-50 text-emerald-700' : 'border-rose-200 bg-rose-50 text-rose-700'
            }`}
          >
            <span className={`h-2 w-2 rounded-full ${online === null ? 'bg-slate-400' : online ? 'bg-emerald-500' : 'bg-rose-500'}`} />
            {online === null ? 'Checking…' : online ? 'Backend online' : 'Backend offline — retry'}
          </button>
        </div>
      </div>

      {/* tabs */}
      <div className="inline-flex rounded-xl border border-slate-200 bg-white p-1 shadow-sm">
        {([['operations', 'Operations'], ['console', 'Console']] as const).map(([id, label]) => (
          <button
            key={id}
            type="button"
            onClick={() => setTab(id)}
            className={`rounded-lg px-5 py-2 text-sm font-semibold transition ${tab === id ? 'bg-blue-600 text-white shadow-sm' : 'text-slate-500 hover:text-slate-800'}`}
          >
            {label}
          </button>
        ))}
      </div>

      {tab === 'operations' && (
        <div className="grid items-start gap-5 xl:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
          {/* ------------------------------------------------ inputs */}
          <div className="space-y-5">
            <Card title="Choose an operation" step={1}>
              <div className="mb-4 flex flex-wrap gap-1.5">
                {CATEGORIES.map((cat) => (
                  <button
                    key={cat}
                    type="button"
                    onClick={() => setCategory(cat)}
                    className={`rounded-full px-3.5 py-1.5 text-xs font-semibold transition ${
                      category === cat ? 'bg-slate-900 text-white' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                    }`}
                  >
                    {cat}
                  </button>
                ))}
              </div>
              <div className="grid grid-cols-2 gap-2 lg:grid-cols-3">
                {OPERATIONS.filter((o) => o.category === category).map((o) => (
                  <button
                    key={o.id}
                    type="button"
                    onClick={() => selectOperation(o.id)}
                    className={`rounded-xl border p-3 text-left transition ${
                      operation === o.id ? 'border-blue-600 bg-blue-50 ring-1 ring-blue-600' : 'border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50'
                    }`}
                  >
                    <div className={`text-sm font-semibold ${operation === o.id ? 'text-blue-700' : 'text-slate-800'}`}>{o.label}</div>
                    <div className="mt-0.5 text-[11px] leading-snug text-slate-500">{o.hint}</div>
                  </button>
                ))}
              </div>
              {info.category !== category && (
                <p className="mt-3 text-xs text-slate-500">Selected: <span className="font-semibold text-slate-700">{info.label}</span> (in {info.category})</p>
              )}
            </Card>

            <Card
              title={operation === 'symbolic' ? 'Enter the expression' : `Enter the data for ${info.label}`}
              step={2}
              right={
                <div className="flex gap-1.5">
                  <button type="button" className={ghostButton} onClick={loadExample}>Load example</button>
                  {needsMatrix && <button type="button" className={ghostButton} onClick={() => setShowImport((s) => !s)}>{showImport ? 'Close import' : 'Import / paste'}</button>}
                  {needsMatrix && <button type="button" className={ghostButton} onClick={clearAll}>Clear</button>}
                </div>
              }
            >
              <div className="space-y-6">
                {showImport && needsMatrix && (
                  <div className="rounded-xl border border-blue-100 bg-blue-50/60 p-4">
                    <p className="mb-2 text-xs leading-relaxed text-slate-600">
                      Paste a matrix in any common format — a spreadsheet selection, <code className="rounded bg-white px-1">1 2; 3 4</code>,
                      CSV/TSV, or JSON <code className="rounded bg-white px-1">[[1,2],[3,4]]</code>. You can also paste a block straight into any cell of the grid.
                    </p>
                    <textarea value={pasteText} onChange={(e) => setPasteText(e.target.value)} rows={4} spellCheck={false} placeholder={'1 2 3\n4 5 6'} className={`${fieldClass} font-mono text-xs`} />
                    <div className="mt-2 flex justify-end gap-2">
                      <button type="button" className={ghostButton} onClick={() => { setPasteText(''); setShowImport(false); }}>Cancel</button>
                      <button type="button" className="rounded-lg bg-blue-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-blue-700" onClick={importText}>Use as matrix A</button>
                    </div>
                  </div>
                )}

                {needsMatrix && (
                  <MatrixEditor name="A" matrix={matrixA} onChange={setMatrixA} onResize={resizeA} onPasteError={setError} />
                )}

                {operation === 'solve' && <VectorEditor vector={vectorB} onChange={setVectorB} />}

                {operation === 'multiply' && (
                  <MatrixEditor name="B" matrix={matrixBView} onChange={setMatrixB} onResize={resizeB} lockRows onPasteError={setError} />
                )}

                {operation === 'power' && (
                  <label className="flex flex-wrap items-center gap-3 text-sm font-semibold text-slate-700">
                    Exponent n
                    <input type="number" value={power} onChange={(e) => setPower(Math.round(Number(e.target.value)) || 0)} className="w-24 rounded-lg border border-slate-300 px-3 py-2 text-center font-mono outline-none focus:border-blue-500" />
                    <span className="text-xs font-normal text-slate-500">A must be square; negative n uses the inverse</span>
                  </label>
                )}

                {operation === 'symbolic' && (
                  <div className="space-y-4">
                    <div className="flex flex-wrap gap-1.5">
                      {(['simplify', 'differentiate', 'integrate'] as const).map((op) => (
                        <button key={op} type="button" onClick={() => setSymbolicOp(op)} className={`rounded-full px-3.5 py-1.5 text-xs font-semibold capitalize transition ${symbolicOp === op ? 'bg-slate-900 text-white' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'}`}>{op}</button>
                      ))}
                    </div>
                    <textarea value={symbolicExpression} onChange={(e) => setSymbolicExpression(e.target.value)} rows={3} spellCheck={false} className={`${fieldClass} font-mono`} placeholder="sin(x)^2 + cos(x)^2" />
                  </div>
                )}

                {showCalculusOptions && (
                  <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                    <label className="text-[11px] font-bold uppercase tracking-wide text-slate-400">Variable
                      <select value={variable} onChange={(e) => setVariable(e.target.value)} className={`${fieldClass} mt-1 font-mono`}>
                        {['x', 'y', 'z', 't', 'u', 'v', 's'].map((v) => <option key={v}>{v}</option>)}
                      </select>
                    </label>
                    <label className="text-[11px] font-bold uppercase tracking-wide text-slate-400">Order
                      <input type="number" min={1} max={20} value={order} onChange={(e) => setOrder(Math.max(1, Math.min(20, Number(e.target.value) || 1)))} className={`${fieldClass} mt-1`} />
                    </label>
                    {showBounds && (
                      <>
                        <label className="text-[11px] font-bold uppercase tracking-wide text-slate-400">Lower <span className="font-normal normal-case">(optional)</span>
                          <input value={lower} onChange={(e) => setLower(e.target.value)} placeholder="0" className={`${fieldClass} mt-1 font-mono`} />
                        </label>
                        <label className="text-[11px] font-bold uppercase tracking-wide text-slate-400">Upper <span className="font-normal normal-case">(optional)</span>
                          <input value={upper} onChange={(e) => setUpper(e.target.value)} placeholder="1" className={`${fieldClass} mt-1 font-mono`} />
                        </label>
                      </>
                    )}
                  </div>
                )}
                {showBounds && (
                  <p className="-mt-3 text-xs text-slate-500">Leave both bounds empty for an indefinite integral (repeated <em>order</em> times).</p>
                )}
              </div>

              <div className="mt-6 flex flex-wrap items-center justify-between gap-3 border-t border-slate-100 pt-4">
                <span className="text-xs text-slate-400">Shortcut: <kbd className="rounded border border-slate-300 bg-slate-50 px-1.5 py-0.5 font-sans text-[10px] font-semibold text-slate-600">Ctrl</kbd> + <kbd className="rounded border border-slate-300 bg-slate-50 px-1.5 py-0.5 font-sans text-[10px] font-semibold text-slate-600">Enter</kbd></span>
                <button
                  type="button"
                  onClick={() => void execute()}
                  disabled={loading}
                  className="inline-flex min-w-40 items-center justify-center gap-2 rounded-xl bg-blue-600 px-6 py-2.5 text-sm font-bold text-white shadow-sm transition hover:bg-blue-700 disabled:cursor-wait disabled:opacity-60"
                >
                  {loading && <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-white/40 border-t-white" />}
                  {loading ? 'Computing…' : `Run ${info.label}`}
                </button>
              </div>
            </Card>
          </div>

          {/* ------------------------------------------------ results */}
          <div className="space-y-5 xl:sticky xl:top-0">
            <Card
              title={result ? `Result · ${resultLabel}` : 'Result'}
              right={result && !result.error ? (
                <button type="button" className={ghostButton} onClick={() => void navigator.clipboard?.writeText(JSON.stringify(result, null, 2))}>Copy JSON</button>
              ) : undefined}
            >
              {error && <div className="mb-4 rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm leading-relaxed text-rose-800">{error}</div>}
              {result ? (
                <ResultPanel result={result} onUseAsA={(m) => { setMatrixA(m); setVectorB((v) => resizeVector(v, m.length)); setResult(null); }} />
              ) : !error ? (
                <div className="flex min-h-56 flex-col items-center justify-center text-center">
                  <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-2xl bg-slate-100 font-mono text-lg font-bold text-slate-400">[ ]</div>
                  <p className="text-sm font-semibold text-slate-600">No result yet</p>
                  <p className="mt-1 max-w-xs text-xs leading-relaxed text-slate-400">Pick an operation, fill in the matrix and press <strong>Run</strong>. Your result and its diagnostics appear here.</p>
                </div>
              ) : null}
            </Card>

            {history.length > 0 && (
              <Card title="Recent calculations" right={<button type="button" className={ghostButton} onClick={() => setHistory([])}>Clear</button>}>
                <ul className="-my-1 divide-y divide-slate-100">
                  {history.map((h) => (
                    <li key={h.id}>
                      <button type="button" onClick={() => { setResult(h.result); setResultLabel(h.label); setError(null); }} className="flex w-full items-center justify-between gap-3 py-2 text-left text-sm hover:text-blue-700">
                        <span className="truncate font-medium">{h.label}</span>
                        <span className="shrink-0 text-xs text-slate-400">{h.time}</span>
                      </button>
                    </li>
                  ))}
                </ul>
              </Card>
            )}
          </div>
        </div>
      )}

      {/* ------------------------------------------------ console */}
      {tab === 'console' && (
        <div className="grid items-start gap-5 xl:grid-cols-[minmax(0,1fr)_320px]">
          <Card title="Console" right={<button type="button" className={ghostButton} onClick={() => setConsoleLog([])}>Clear output</button>}>
            <div className="h-[min(60vh,520px)] overflow-auto rounded-xl bg-slate-900 p-4 font-mono text-sm text-slate-100">
              {consoleLog.length === 0 && (
                <div className="text-slate-400">
                  <p>Type commands below. Variables persist between lines.</p>
                  <p className="mt-2 text-slate-500">A = [[2,1],[1,3]]&nbsp;&nbsp;b = [5,7]&nbsp;&nbsp;solve(A,b)&nbsp;&nbsp;A*A + 2*A&nbsp;&nbsp;A'&nbsp;&nbsp;A^-1&nbsp;&nbsp;eigenvalues(A)</p>
                  <p className="mt-1 text-slate-500">Several statements can be separated with <span className="text-slate-300">;</span> or new lines. Use ↑ / ↓ to recall earlier commands.</p>
                </div>
              )}
              {consoleLog.map((entry) => (
                <div key={entry.id} className="mb-3">
                  {entry.command && <div className="text-sky-300"><span className="select-none text-slate-500">&gt;&nbsp;</span>{entry.command}</div>}
                  {entry.error ? (
                    <div className="mt-1 whitespace-pre-wrap text-rose-300">{entry.error}</div>
                  ) : (
                    <div className="mt-1.5 rounded-lg bg-slate-800/70 p-3 font-sans text-slate-100"><ConsoleResult result={entry.result} /></div>
                  )}
                </div>
              ))}
              <div ref={consoleEnd} />
            </div>
            <div className="mt-3 flex gap-2">
              <input
                value={consoleInput}
                onChange={(e) => setConsoleInput(e.target.value)}
                onKeyDown={onConsoleKey}
                spellCheck={false}
                placeholder="solve(A,b)   |   A*B   |   det(A)   |   help()"
                className={`${fieldClass} font-mono`}
              />
              <button type="button" disabled={loading} onClick={() => void runCommand()} className="rounded-lg bg-blue-600 px-5 text-sm font-bold text-white hover:bg-blue-700 disabled:opacity-60">Run</button>
            </div>
            <div className="mt-3 flex flex-wrap gap-1.5">
              {CONSOLE_CHIPS.map((c) => (
                <button key={c} type="button" onClick={() => setConsoleInput(c)} className="rounded-full bg-slate-100 px-3 py-1 font-mono text-[11px] font-medium text-slate-600 hover:bg-slate-200">{c}</button>
              ))}
            </div>
          </Card>

          <div className="space-y-5">
            <Card title="Variables" right={<button type="button" className={ghostButton} onClick={() => setVariables({})}>Clear</button>}>
              {Object.keys(variables).length === 0 ? (
                <p className="text-xs leading-relaxed text-slate-400">No variables yet. Assign one — <code className="rounded bg-slate-100 px-1">A = [[1,2],[3,4]]</code> — or send the matrices from the Operations tab.</p>
              ) : (
                <ul className="space-y-2">
                  {Object.entries(variables).map(([name, value]) => {
                    const dims = Array.isArray(value) ? (Array.isArray(value[0]) ? `${value.length}×${value[0].length}` : `${value.length}×1`) : 'scalar';
                    return (
                      <li key={name} className="flex items-center justify-between rounded-lg bg-slate-50 px-3 py-2">
                        <button type="button" onClick={() => setConsoleInput((c) => c + name)} className="font-mono text-sm font-bold text-slate-800 hover:text-blue-700">{name}</button>
                        <span className="text-xs text-slate-500">{dims}</span>
                      </li>
                    );
                  })}
                </ul>
              )}
              <div className="mt-4 flex gap-2 border-t border-slate-100 pt-4">
                <button type="button" className={`${ghostButton} flex-1`} onClick={() => sendEditorToConsole('A')}>Send A to console</button>
                <button type="button" className={`${ghostButton} flex-1`} onClick={() => sendEditorToConsole('B')}>Send B</button>
              </div>
            </Card>
          </div>
        </div>
      )}
    </div>
  );
}
