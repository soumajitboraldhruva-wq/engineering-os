import { useEffect, useMemo, useState } from 'react';
import 'katex/dist/katex.min.css';
import { BlockMath, InlineMath } from 'react-katex';

type Cell = string;
type Matrix = Cell[][];
type Vector = Cell[];

type Operation =
  | 'solve'
  | 'determinant'
  | 'inverse'
  | 'rank'
  | 'rref'
  | 'nullspace'
  | 'eigenvalues'
  | 'eigenvectors'
  | 'svd'
  | 'qr'
  | 'lu'
  | 'cholesky'
  | 'pseudoinverse'
  | 'norm'
  | 'transpose'
  | 'multiply'
  | 'power'
  | 'trace'
  | 'differentiate'
  | 'integrate'
  | 'symbolic';

type HistoryItem = {
  id: number;
  label: string;
  result: any;
  timestamp: string;
};

const API_URL = (import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000').replace(/\/+$/, '');
const API_BASE = `${API_URL}/api/matrix`;

const operations: Array<{
  id: Operation;
  label: string;
  description: string;
  group: string;
}> = [
  { id: 'solve', label: 'Solve Ax = b', description: 'Linear systems and least squares', group: 'Core' },
  { id: 'determinant', label: 'Determinant', description: 'Compute det(A)', group: 'Core' },
  { id: 'inverse', label: 'Inverse', description: 'Compute A⁻¹', group: 'Core' },
  { id: 'rank', label: 'Rank', description: 'Matrix rank', group: 'Core' },
  { id: 'rref', label: 'RREF', description: 'Reduced row-echelon form', group: 'Core' },
  { id: 'nullspace', label: 'Nullspace', description: 'Null-space basis', group: 'Core' },
  { id: 'eigenvalues', label: 'Eigenvalues', description: 'Spectrum of A', group: 'Core' },
  { id: 'eigenvectors', label: 'Eigenvectors', description: 'Eigenpairs', group: 'Core' },
  { id: 'svd', label: 'SVD', description: 'U Σ Vᵀ', group: 'Decompositions' },
  { id: 'qr', label: 'QR', description: 'QR factorization', group: 'Decompositions' },
  { id: 'lu', label: 'LU', description: 'Pivoted LU', group: 'Decompositions' },
  { id: 'cholesky', label: 'Cholesky', description: 'LLᵀ for SPD matrices', group: 'Decompositions' },
  { id: 'pseudoinverse', label: 'Pseudoinverse', description: 'Moore–Penrose A⁺', group: 'Decompositions' },
  { id: 'norm', label: 'Norms', description: 'Frobenius, spectral, 1, ∞', group: 'Transforms' },
  { id: 'transpose', label: 'Transpose', description: 'Aᵀ', group: 'Transforms' },
  { id: 'multiply', label: 'Multiply A × B', description: 'Matrix product', group: 'Transforms' },
  { id: 'power', label: 'Matrix Power', description: 'Aⁿ', group: 'Transforms' },
  { id: 'trace', label: 'Trace', description: 'tr(A)', group: 'Transforms' },
  { id: 'differentiate', label: 'Differentiate', description: 'Differentiate each cell', group: 'Calculus' },
  { id: 'integrate', label: 'Integrate', description: 'Indefinite or definite', group: 'Calculus' },
  { id: 'symbolic', label: 'Symbolic', description: 'Simplify an expression', group: 'Calculus' },
];

const examples: Record<Operation, { A: Matrix; b?: Vector; B?: Matrix; power?: number }> = {
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
  symbolic: { A: [['x^2 + 2*x + 1']] },
};

const functionButtons = [
  ['sin(', ')'], ['cos(', ')'], ['tan(', ')'], ['log(', ')'],
  ['ln(', ')'], ['sqrt(', ')'], ['exp(', ')'], ['abs(', ')'],
  ['diff(', ',x)'], ['integrate(', ',x)'], ['pi', ''], ['x', ''],
  ['y', ''], ['z', ''],
];

function resizeMatrix(matrix: Matrix, rows: number, columns: number): Matrix {
  return Array.from({ length: rows }, (_, r) =>
    Array.from({ length: columns }, (_, c) => matrix[r]?.[c] ?? '0')
  );
}

function resizeVector(vector: Vector, size: number): Vector {
  return Array.from({ length: size }, (_, i) => vector[i] ?? '0');
}
function zeros(rows: number, cols: number): Matrix {
  return Array.from({ length: rows }, () => Array.from({ length: cols }, () => '0'));
}

function zeroVector(size: number): Vector {
  return Array.from({ length: size }, () => '0');
}

function pretty(value: any): string {
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return '—';
    if (Math.abs(value) < 1e-10) return '0';
    return Number(value.toPrecision(8)).toString();
  }
  if (typeof value === 'object' && value !== null && 'real' in value && 'imag' in value) {
    return `${pretty(value.real)} ${value.imag >= 0 ? '+' : '-'} ${pretty(Math.abs(value.imag))}i`;
  }
  return String(value);
}

function ResultTable({ value }: { value: any[][] }) {
  return (
    <div className="overflow-auto rounded-xl border border-slate-200 dark:border-slate-800">
      <table className="min-w-full border-collapse text-sm">
        <tbody>
          {value.map((row, r) => (
            <tr key={r}>
              {row.map((cell, c) => (
                <td key={c} className="border-b border-r border-slate-100 px-3 py-2 text-right font-mono last:border-r-0 dark:border-slate-800">
                  {pretty(cell)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ValueView({ value }: { value: any }) {
  if (Array.isArray(value)) {
    if (!value.length) return <span>[]</span>;
    if (Array.isArray(value[0])) return <ResultTable value={value} />;
    return (
      <div className="flex flex-wrap gap-2">
        {value.map((item, index) => (
          <span key={index} className="rounded-lg bg-slate-100 px-3 py-1 font-mono text-sm dark:bg-slate-800">
            {pretty(item)}
          </span>
        ))}
      </div>
    );
  }
  if (typeof value === 'object' && value !== null) {
    return <pre className="max-h-72 overflow-auto rounded-xl bg-slate-950 p-3 text-xs text-slate-200">{JSON.stringify(value, null, 2)}</pre>;
  }
  return <span className="font-mono text-base">{pretty(value)}</span>;
}

function ResultView({ result }: { result: any }) {
  if (!result) {
    return <div className="rounded-xl border border-dashed border-slate-300 p-5 text-sm text-slate-500 dark:border-slate-700">Run a calculation to see the result.</div>;
  }
  if (result.error) {
    return <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700 dark:border-red-900 dark:bg-red-950/30 dark:text-red-300">{result.error}</div>;
  }
  return (
    <div className="space-y-4 animate-[fadeIn_.25s_ease-out]">
      {result.status && (
        <div className="rounded-xl border border-blue-200 bg-blue-50 p-4 dark:border-blue-900 dark:bg-blue-950/30">
          <div className="text-xs font-bold uppercase tracking-wide text-blue-600 dark:text-blue-300">{result.status.replaceAll('_', ' ')}</div>
          <div className="mt-1 text-sm">{result.explanation}</div>
          {result.method && <div className="mt-2 text-xs text-slate-500">Method: {result.method}</div>}
        </div>
      )}
      {result.latex && (
        <div className="overflow-auto rounded-xl border border-slate-200 bg-white p-3 dark:border-slate-800 dark:bg-slate-950">
          <BlockMath math={String(result.latex)} />
        </div>
      )}
      {Object.entries(result).map(([key, value]) => {
        if (['error', 'metadata', 'operation', 'status', 'explanation', 'method', 'latex'].includes(key)) return null;
        return (
          <section key={key} className="space-y-2">
            <div className="text-[10px] font-bold uppercase tracking-widest text-slate-400">{key.replaceAll('_', ' ')}</div>
            <ValueView value={value} />
          </section>
        );
      })}
      {result.metadata && (
        <div className="border-t border-slate-200 pt-4 dark:border-slate-800">
          <div className="mb-2 text-[10px] font-bold uppercase tracking-widest text-slate-400">Diagnostics</div>
          <div className="grid grid-cols-2 gap-2">
            {Object.entries(result.metadata).map(([key, value]) => (
              <div key={key} className="rounded-xl bg-slate-50 p-3 dark:bg-slate-950">
                <div className="text-[9px] uppercase tracking-wide text-slate-400">{key.replaceAll('_', ' ')}</div>
                <div className="mt-1 text-xs font-semibold">{Array.isArray(value) ? `${value.length} values` : pretty(value)}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function MatrixEditor({ matrix, title, onChange }: { matrix: Matrix; title: string; onChange: (m: Matrix) => void }) {
  function update(r: number, c: number, value: string) {
    const next = matrix.map((row) => [...row]);
    next[r][c] = value;
    onChange(next);
  }
  function focusCell(r: number, c: number) {
    document.querySelector<HTMLInputElement>(`input[data-cell="${r}-${c}"]`)?.focus();
  }
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div className="text-sm font-semibold">{title}</div>
        <div className="text-[10px] uppercase tracking-widest text-slate-400">{matrix.length} × {matrix[0]?.length || 0}</div>
      </div>
      <div className="overflow-auto rounded-2xl border border-slate-200 bg-slate-50 p-3 dark:border-slate-800 dark:bg-slate-950">
        <table className="border-collapse">
          <tbody>
            {matrix.map((row, r) => (
              <tr key={r}>
                {row.map((value, c) => (
                  <td key={`${r}-${c}`} className="p-1">
                    <input
                      data-cell={`${r}-${c}`}
                      type="text"
                      value={value}
                      onChange={(e) => update(r, c, e.target.value)}
                      onKeyDown={(e) => {
                        const move: Record<string, [number, number]> = {
                          ArrowUp: [r - 1, c], ArrowDown: [r + 1, c],
                          ArrowLeft: [r, c - 1], ArrowRight: [r, c + 1],
                        };
                        if (move[e.key]) {
                          e.preventDefault();
                          focusCell(...move[e.key]);
                        }
                      }}
                      className="h-11 w-28 rounded-lg border border-slate-300 bg-white px-2 text-center font-mono text-sm outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-200 dark:border-slate-700 dark:bg-slate-900 dark:focus:ring-blue-900"
                      placeholder="0"
                    />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function VectorEditor({ vector, title, onChange }: { vector: Vector; title: string; onChange: (v: Vector) => void }) {
  return (
    <div className="space-y-2">
      <div className="text-sm font-semibold">{title}</div>
      <div className="flex flex-wrap gap-2 rounded-2xl border border-slate-200 bg-slate-50 p-3 dark:border-slate-800 dark:bg-slate-950">
        {vector.map((value, i) => (
          <input key={i} type="text" value={value} onChange={(e) => {
            const next = [...vector];
            next[i] = e.target.value;
            onChange(next);
          }} className="h-11 w-24 rounded-lg border border-slate-300 bg-white px-2 text-center font-mono text-sm dark:border-slate-700 dark:bg-slate-900" placeholder="0" />
        ))}
      </div>
    </div>
  );
}

export default function MatrixSolver() {
  const [rows, setRows] = useState(3);
  const [columns, setColumns] = useState(3);
  const [matrixA, setMatrixA] = useState<Matrix>(zeros(3, 3));
  const [vectorB, setVectorB] = useState<Vector>(zeroVector(3));
  const [bColumns, setBColumns] = useState(2);
  const [matrixB, setMatrixB] = useState<Matrix>(zeros(3, 2));
  const [operation, setOperation] = useState<Operation>('solve');
  const [power, setPower] = useState(2);
  const [variable, setVariable] = useState('x');
  const [order, setOrder] = useState(1);
  const [lower, setLower] = useState('');
  const [upper, setUpper] = useState('');
  const [symbolicExpression, setSymbolicExpression] = useState('sin(x)^2 + cos(x)^2');
  const [tolerance, setTolerance] = useState(1e-10);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [online, setOnline] = useState<boolean | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [variables, setVariables] = useState<Record<string, any>>({});
  const [consoleInput, setConsoleInput] = useState('');
  const [consoleOutput, setConsoleOutput] = useState<any[]>([]);
  const [pasteText, setPasteText] = useState('');

  const selected = useMemo(() => operations.find((item) => item.id === operation)!, [operation]);

  useEffect(() => {
    // Resizing should preserve existing cells instead of wiping the user's work.
    setMatrixA((current) => resizeMatrix(current, rows, columns));
    setVectorB((current) => resizeVector(current, rows));
    setMatrixB((current) => resizeMatrix(current, columns, bColumns));
    setResult(null);
    setError(null);
  }, [rows, columns, bColumns]);

  useEffect(() => {
    checkHealth();
    const timer = window.setInterval(checkHealth, 30000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
        event.preventDefault();
        execute();
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  });

  async function checkHealth() {
    try {
      const response = await fetch(`${API_BASE}/health`, { cache: 'no-store' });
      setOnline(response.ok);
    } catch {
      setOnline(false);
    }
  }

  function loadExample() {
    const example = examples[operation];
    if (!example) return;
    setRows(example.A.length);
    setColumns(example.A[0]?.length || 1);
    setMatrixA(example.A.map((row) => [...row]));
    setVectorB(example.b ? [...example.b] : zeroVector(example.A.length));
    const nextB = example.B ? example.B.map((row) => [...row]) : zeros(example.A[0]?.length || 1, bColumns);
    setBColumns(example.B?.[0]?.length || bColumns);
    setMatrixB(nextB);
    if (example.power !== undefined) setPower(example.power);
    setResult(null);
    setError(null);
  }

  function addHistory(label: string, data: any) {
    setHistory((items) => [{ id: Date.now(), label, result: data, timestamp: new Date().toLocaleTimeString() }, ...items].slice(0, 16));
  }

  async function post(path: string, body: any) {
    const response = await fetch(`${API_BASE}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await response.json().catch(() => ({ error: `HTTP ${response.status}` }));
    if (!response.ok || data.error) {
      const detail = Array.isArray(data.detail)
        ? data.detail.map((item: any) => item.msg || JSON.stringify(item)).join('; ')
        : data.detail;
      throw new Error(data.error || detail || `HTTP ${response.status}`);
    }
    setOnline(true);
    return data;
  }

  async function execute() {
    setLoading(true);
    setError(null);
    try {
      let data: any;
      if (operation === 'solve') {
        data = await post('/solve', { matrix_a: matrixA, vector_b: vectorB, tolerance });
      } else if (operation === 'differentiate' || operation === 'integrate') {
        data = await post('/calculus', {
          operation,
          matrix: matrixA,
          variable,
          order,
          lower: lower || null,
          upper: upper || null,
        });
      } else if (operation === 'symbolic') {
        data = await post('/symbolic', {
          expression: symbolicExpression,
          operation: 'simplify',
          variable,
        });
      } else {
        data = await post('/operation', {
          operation,
          matrix_a: matrixA,
          matrix_b: operation === 'multiply' ? matrixB : undefined,
          power: operation === 'power' ? power : undefined,
          tolerance,
        });
      }
      setResult(data);
      addHistory(selected.label, data);
    } catch (err: any) {
      setOnline(false);
      setResult(null);
      setError(err?.message || 'Calculation failed.');
    } finally {
      setLoading(false);
    }
  }

  async function runCommand(command = consoleInput) {
    const trimmed = command.trim();
    if (!trimmed) return;
    setLoading(true);
    try {
      const data = await post('/command', { command: trimmed, variables, tolerance });
      setVariables(data.variables || {});
      setConsoleOutput((items) => [{ command: trimmed, result: data.result }, ...items].slice(0, 30));
      setConsoleInput('');
      addHistory(trimmed, data.result);
    } catch (err: any) {
      setConsoleOutput((items) => [{ command: trimmed, result: { error: err?.message || 'Command failed.' } }, ...items].slice(0, 30));
    } finally {
      setLoading(false);
    }
  }

  function importMatrix() {
    try {
      const parsed = JSON.parse(pasteText);
      if (!Array.isArray(parsed) || !parsed.length || !Array.isArray(parsed[0])) throw new Error('Expected a 2D matrix.');
      const width = parsed[0].length;
      if (!width || parsed.some((row: any) => !Array.isArray(row) || row.length !== width)) throw new Error('All rows must have equal length.');
      setRows(parsed.length);
      setColumns(width);
      setMatrixA(parsed.map((row: any[]) => row.map((cell) => String(cell))));
      setPasteText('');
    } catch (err: any) {
      setError(err?.message || 'Invalid matrix.');
    }
  }

  function insertToken(prefix: string, suffix: string) {
    const cell = document.activeElement as HTMLInputElement | null;
    if (!cell?.dataset.cell) return;
    const [r, c] = cell.dataset.cell.split('-').map(Number);
    if (!Number.isInteger(r) || !Number.isInteger(c)) return;
    const next = matrixA.map((row) => [...row]);
    next[r][c] = `${next[r][c] === '0' ? '' : next[r][c]}${prefix}${suffix}`;
    setMatrixA(next);
  }

  const groups = Array.from(new Set(operations.map((op) => op.group)));

  return (
    <div className="min-h-screen bg-slate-100 text-slate-900 dark:bg-slate-950 dark:text-slate-100">
      <style>{`@keyframes fadeIn{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:translateY(0)}}`}</style>
      <div className="mx-auto max-w-[1800px] p-3 md:p-5">
        <header className="mb-3 rounded-2xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900">
          <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
            <div className="flex items-center gap-3">
              <div className="rounded-xl bg-blue-600 px-3 py-2 text-sm font-black text-white">LA</div>
              <div>
                <h1 className="text-2xl font-bold tracking-tight">ENG-OS Linear Algebra</h1>
                <p className="text-xs text-slate-500">Numerical · symbolic · calculus · decompositions</p>
              </div>
            </div>
            <div className="flex flex-wrap items-center gap-2 text-xs">
              <span className="rounded-full border border-slate-200 px-3 py-1.5 dark:border-slate-700">NumPy</span>
              <span className="rounded-full border border-slate-200 px-3 py-1.5 dark:border-slate-700">SciPy</span>
              <span className="rounded-full border border-slate-200 px-3 py-1.5 dark:border-slate-700">SymPy</span>
              <span className={`rounded-full px-3 py-1.5 font-semibold ${online ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-300' : online === false ? 'bg-red-100 text-red-700 dark:bg-red-950/40 dark:text-red-300' : 'bg-slate-100 text-slate-600 dark:bg-slate-800'}`}>
                {online ? '● Backend online' : online === false ? '● Backend offline' : '● Checking backend'}
              </span>
            </div>
          </div>
        </header>

        <div className="grid grid-cols-1 gap-3 xl:grid-cols-[210px_minmax(0,1fr)] 2xl:grid-cols-[210px_minmax(0,1fr)_310px]">
          <aside className="max-h-[32vh] overflow-y-auto rounded-2xl border border-slate-200 bg-white p-3 shadow-sm dark:border-slate-800 dark:bg-slate-900 xl:sticky xl:top-3 xl:max-h-[calc(100vh-1.5rem)]">
            <div className="mb-3 px-2">
              <div className="text-[10px] font-bold uppercase tracking-widest text-slate-400">Operations</div>
              <div className="mt-1 text-xs text-slate-500">Choose what the matrix workspace should do.</div>
            </div>
            <div className="space-y-3">
              {groups.map((group) => (
                <div key={group}>
                  <div className="mb-1 px-2 text-[9px] font-bold uppercase tracking-widest text-slate-400">{group}</div>
                  <div className="space-y-1">
                    {operations.filter((item) => item.group === group).map((item) => (
                      <button key={item.id} onClick={() => { setOperation(item.id); setError(null); }} className={`w-full rounded-xl px-3 py-2 text-left transition ${operation === item.id ? 'bg-blue-600 text-white shadow-sm' : 'hover:bg-slate-100 dark:hover:bg-slate-800'}`}>
                        <div className="text-sm font-semibold">{item.label}</div>
                        <div className={`text-[10px] ${operation === item.id ? 'text-blue-100' : 'text-slate-500'}`}>{item.description}</div>
                      </button>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </aside>

          <main className="min-w-0 space-y-3">
            <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900">
              <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
                <div>
                  <div className="text-[10px] font-bold uppercase tracking-widest text-blue-600">Active operation</div>
                  <h2 className="mt-1 text-xl font-bold">{selected.label}</h2>
                  <p className="mt-1 text-sm text-slate-500">{selected.description}</p>
                </div>
                <div className="flex gap-2">
                  <button onClick={loadExample} className="rounded-xl border border-slate-300 px-4 py-2 text-sm font-semibold hover:bg-slate-50 dark:border-slate-700 dark:hover:bg-slate-800">Example</button>
                  <button onClick={execute} disabled={loading} className="rounded-xl bg-blue-600 px-5 py-2 text-sm font-semibold text-white hover:bg-blue-700 disabled:opacity-50">{loading ? 'Working…' : 'Calculate'}</button>
                </div>
              </div>

              {operation !== 'symbolic' && (
                <div className="mt-4 flex flex-wrap gap-2 border-t border-slate-200 pt-4 dark:border-slate-800">
                  <label className="text-[10px] font-bold uppercase tracking-wide text-slate-400">Rows<input type="number" min={1} max={50} value={rows} onChange={(e) => setRows(Math.max(1, Math.min(50, Number(e.target.value) || 1)))} className="mt-1 block w-20 rounded-lg border border-slate-300 bg-white px-2 py-2 text-sm dark:border-slate-700 dark:bg-slate-950" /></label>
                  <label className="text-[10px] font-bold uppercase tracking-wide text-slate-400">Columns<input type="number" min={1} max={50} value={columns} onChange={(e) => setColumns(Math.max(1, Math.min(50, Number(e.target.value) || 1)))} className="mt-1 block w-20 rounded-lg border border-slate-300 bg-white px-2 py-2 text-sm dark:border-slate-700 dark:bg-slate-950" /></label>
                  <label className="text-[10px] font-bold uppercase tracking-wide text-slate-400">Tolerance<input type="number" step="any" value={tolerance} onChange={(e) => setTolerance(Number(e.target.value) || 1e-10)} className="mt-1 block w-28 rounded-lg border border-slate-300 bg-white px-2 py-2 text-sm dark:border-slate-700 dark:bg-slate-950" /></label>
                  {operation === 'multiply' && <label className="text-[10px] font-bold uppercase tracking-wide text-slate-400">B columns<input type="number" min={1} max={50} value={bColumns} onChange={(e) => setBColumns(Math.max(1, Math.min(50, Number(e.target.value) || 1)))} className="mt-1 block w-24 rounded-lg border border-slate-300 bg-white px-2 py-2 text-sm dark:border-slate-700 dark:bg-slate-950" /></label>}
                  {operation === 'power' && <label className="text-[10px] font-bold uppercase tracking-wide text-slate-400">Power n<input type="number" min={-100} max={100} value={power} onChange={(e) => setPower(Number(e.target.value))} className="mt-1 block w-24 rounded-lg border border-slate-300 bg-white px-2 py-2 text-sm dark:border-slate-700 dark:bg-slate-950" /></label>}
                </div>
              )}

              {operation === 'differentiate' || operation === 'integrate' ? (
                <div className="mt-4 grid gap-3 border-t border-slate-200 pt-4 md:grid-cols-4 dark:border-slate-800">
                  <label className="text-[10px] font-bold uppercase tracking-wide text-slate-400">Variable<select value={variable} onChange={(e) => setVariable(e.target.value)} className="mt-1 block w-full rounded-lg border border-slate-300 bg-white px-2 py-2 text-sm dark:border-slate-700 dark:bg-slate-950"><option>x</option><option>y</option><option>z</option><option>t</option><option>u</option><option>v</option><option>s</option></select></label>
                  <label className="text-[10px] font-bold uppercase tracking-wide text-slate-400">Order<input type="number" min={1} max={20} value={order} onChange={(e) => setOrder(Number(e.target.value) || 1)} className="mt-1 block w-full rounded-lg border border-slate-300 bg-white px-2 py-2 text-sm dark:border-slate-700 dark:bg-slate-950" /></label>
                  {operation === 'integrate' && <label className="text-[10px] font-bold uppercase tracking-wide text-slate-400">Lower<input value={lower} onChange={(e) => setLower(e.target.value)} placeholder="0 (optional)" className="mt-1 block w-full rounded-lg border border-slate-300 bg-white px-2 py-2 text-sm dark:border-slate-700 dark:bg-slate-950" /></label>}
                  {operation === 'integrate' && <label className="text-[10px] font-bold uppercase tracking-wide text-slate-400">Upper<input value={upper} onChange={(e) => setUpper(e.target.value)} placeholder="π (optional)" className="mt-1 block w-full rounded-lg border border-slate-300 bg-white px-2 py-2 text-sm dark:border-slate-700 dark:bg-slate-950" /></label>}
                </div>
              ) : null}

              {operation === 'symbolic' ? (
                <div className="mt-4 rounded-xl bg-slate-50 p-4 dark:bg-slate-950">
                  <label className="text-[10px] font-bold uppercase tracking-widest text-slate-400">Expression</label>
                  <input value={symbolicExpression} onChange={(e) => setSymbolicExpression(e.target.value)} className="mt-1 w-full rounded-xl border border-slate-300 bg-white px-3 py-3 font-mono text-sm dark:border-slate-700 dark:bg-slate-900" />
                  <div className="mt-2 text-xs text-slate-500">Try <InlineMath math="\sin(x)^2+\cos(x)^2" /> or <InlineMath math="\frac{d}{dx}x^3" />.</div>
                </div>
              ) : (
                <div className="mt-4 grid gap-5 lg:grid-cols-2">
                  <MatrixEditor matrix={matrixA} title="Matrix A" onChange={setMatrixA} />
                  {operation === 'solve' && <VectorEditor vector={vectorB} title="Vector b" onChange={setVectorB} />}
                  {operation === 'multiply' && <MatrixEditor matrix={matrixB} title="Matrix B" onChange={setMatrixB} />}
                </div>
              )}
            </section>

            {operation !== 'symbolic' && (
              <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900">
                <div className="mb-2 flex items-center justify-between">
                  <div>
                    <div className="font-semibold">Expression palette</div>
                    <div className="text-xs text-slate-500">Matrix cells accept symbolic expressions, not just numbers.</div>
                  </div>
                  <button onClick={() => setPasteText(JSON.stringify(matrixA))} className="rounded-lg border border-slate-300 px-3 py-1.5 text-xs dark:border-slate-700">Matrix JSON</button>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {functionButtons.map(([prefix, suffix]) => (
                    <button key={`${prefix}${suffix}`} onMouseDown={(e) => e.preventDefault()} onClick={() => insertToken(prefix, suffix)} className="rounded-lg border border-slate-200 bg-slate-50 px-2.5 py-1.5 font-mono text-xs hover:border-blue-400 hover:text-blue-600 dark:border-slate-700 dark:bg-slate-950">
                      {prefix}{suffix}
                    </button>
                  ))}
                </div>
                <div className="mt-3 grid gap-2 md:grid-cols-2">
                  <div className="rounded-xl bg-slate-950 p-3 text-xs text-slate-400">Examples: <span className="text-slate-200">sin(x)</span> · <span className="text-slate-200">log(10)</span> · <span className="text-slate-200">x^2</span> · <span className="text-slate-200">diff(x^3,x)</span> · <span className="text-slate-200">integrate(sin(x),x)</span></div>
                  <div className="rounded-xl bg-blue-50 p-3 text-xs text-blue-700 dark:bg-blue-950/30 dark:text-blue-300">Numeric operations automatically reject unresolved symbols and tell you which symbols need substitution.</div>
                </div>
              </section>
            )}

            <section className="rounded-2xl border border-slate-800 bg-slate-950 p-4 text-slate-100 shadow-sm">
              <div className="flex items-center justify-between">
                <div><div className="font-semibold">Command console</div><div className="text-xs text-slate-500">Ctrl/Cmd + Enter executes the workspace.</div></div>
                <button onClick={() => runCommand('help()')} className="rounded-lg bg-slate-800 px-3 py-1.5 text-xs">help()</button>
              </div>
              <div className="mt-3 max-h-64 overflow-auto rounded-xl border border-slate-800 bg-black/20 p-3 font-mono text-xs">
                {consoleOutput.length === 0 ? <div className="text-slate-600">Try A = [[1,2],[3,4]] · b = [5,6] · solve(A,b) · diff(x^3,x)</div> : consoleOutput.map((item, index) => <div key={index} className="mb-4"><div className="text-blue-400">$ {item.command}</div><pre className="mt-1 whitespace-pre-wrap text-slate-300">{JSON.stringify(item.result, null, 2)}</pre></div>)}
              </div>
              <div className="mt-3 flex gap-2">
                <input value={consoleInput} onChange={(e) => setConsoleInput(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter') runCommand(); }} placeholder="A = [[1,2],[3,4]]" className="min-w-0 flex-1 rounded-xl border border-slate-700 bg-slate-900 px-3 py-3 font-mono text-sm outline-none focus:ring-2 focus:ring-blue-500" />
                <button onClick={() => runCommand()} className="rounded-xl bg-blue-600 px-5 font-semibold">Run</button>
              </div>
            </section>

            <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900">
              <div className="mb-2 font-semibold">Paste/import matrix</div>
              <textarea value={pasteText} onChange={(e) => setPasteText(e.target.value)} placeholder='[["sin(x)", "x^2"], ["log(10)", "3"]]' className="min-h-24 w-full rounded-xl border border-slate-300 bg-slate-50 p-3 font-mono text-sm dark:border-slate-700 dark:bg-slate-950" />
              <button onClick={importMatrix} className="mt-2 rounded-lg bg-slate-900 px-4 py-2 text-sm font-semibold text-white dark:bg-slate-100 dark:text-slate-900">Import into A</button>
            </section>
          </main>

          <aside className="space-y-3">
            <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900">
              <div className="mb-3 flex items-center justify-between"><div className="font-semibold">Result</div><button onClick={() => { setResult(null); setError(null); }} className="text-xs text-slate-500 hover:text-red-500">Clear</button></div>
              {error && <div className="mb-3 rounded-xl border border-red-200 bg-red-50 p-3 text-sm text-red-700 dark:border-red-900 dark:bg-red-950/30 dark:text-red-300">{error}</div>}
              <ResultView result={result} />
            </section>

            <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900">
              <div className="mb-2 flex items-center justify-between"><div className="font-semibold">Variables</div><span className="text-[10px] uppercase tracking-widest text-slate-400">{Object.keys(variables).length}</span></div>
              {Object.keys(variables).length === 0 ? <div className="text-xs text-slate-500">Use A = [[...]] in the console.</div> : <div className="space-y-1">{Object.entries(variables).map(([name, value]) => <div key={name} className="rounded-lg bg-slate-50 p-2 font-mono text-xs dark:bg-slate-950"><b>{name}</b> = {JSON.stringify(value)}</div>)}</div>}
            </section>

            <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900">
              <div className="mb-2 flex items-center justify-between"><div className="font-semibold">History</div><button onClick={() => setHistory([])} className="text-xs text-slate-500">Clear</button></div>
              <div className="max-h-56 space-y-1 overflow-auto">
                {history.length === 0 ? <div className="text-xs text-slate-500">No calculations yet.</div> : history.map((item) => <button key={item.id} onClick={() => setResult(item.result)} className="w-full rounded-lg p-2 text-left hover:bg-slate-100 dark:hover:bg-slate-800"><div className="truncate text-xs font-mono">{item.label}</div><div className="text-[9px] text-slate-500">{item.timestamp}</div></button>)}
              </div>
            </section>
          </aside>
        </div>
      </div>
    </div>
  );
}

