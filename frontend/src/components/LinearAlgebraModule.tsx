import { useEffect, useMemo, useRef, useState } from 'react';
import axios from 'axios';
import 'katex/dist/katex.min.css';
import { BlockMath } from 'react-katex';

type Matrix = string[][];
type Vector = string[];

type MatrixOperation =
  | 'solve' | 'determinant' | 'inverse' | 'rank' | 'rref' | 'nullspace'
  | 'eigenvalues' | 'eigenvectors' | 'svd' | 'qr' | 'lu' | 'cholesky'
  | 'pseudoinverse' | 'norm' | 'transpose' | 'multiply' | 'power' | 'trace';
type ViewMode = 'matrix' | 'calculus' | 'symbolic';
type CalculusOperation = 'differentiate' | 'integrate';

type HistoryItem = {
  id: number;
  title: string;
  timestamp: string;
  result: any;
};

const API_URL = (import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000').replace(/\/+$/, '');
const API = `${API_URL}/api/linalg`;

const MATRIX_OPS: Array<{ id: MatrixOperation; label: string; hint: string; group: string }> = [
  { id: 'solve', label: 'Solve Ax = b', hint: 'Linear systems', group: 'Core' },
  { id: 'determinant', label: 'Determinant', hint: 'det(A)', group: 'Core' },
  { id: 'inverse', label: 'Inverse', hint: 'A⁻¹', group: 'Core' },
  { id: 'rank', label: 'Rank', hint: 'rank(A)', group: 'Core' },
  { id: 'rref', label: 'RREF', hint: 'Row reduction', group: 'Core' },
  { id: 'nullspace', label: 'Nullspace', hint: 'Basis of null(A)', group: 'Core' },
  { id: 'eigenvalues', label: 'Eigenvalues', hint: 'λ spectrum', group: 'Spectral' },
  { id: 'eigenvectors', label: 'Eigenvectors', hint: 'λ, v pairs', group: 'Spectral' },
  { id: 'svd', label: 'SVD', hint: 'UΣVᵀ', group: 'Decompositions' },
  { id: 'qr', label: 'QR', hint: 'A = QR', group: 'Decompositions' },
  { id: 'lu', label: 'LU', hint: 'A = PLU', group: 'Decompositions' },
  { id: 'cholesky', label: 'Cholesky', hint: 'A = LLᵀ', group: 'Decompositions' },
  { id: 'pseudoinverse', label: 'Pseudoinverse', hint: 'A⁺', group: 'Transforms' },
  { id: 'norm', label: 'Norms', hint: '1, 2, ∞, F', group: 'Transforms' },
  { id: 'transpose', label: 'Transpose', hint: 'Aᵀ', group: 'Transforms' },
  { id: 'trace', label: 'Trace', hint: 'tr(A)', group: 'Transforms' },
  { id: 'multiply', label: 'Multiply A × B', hint: 'Matrix product', group: 'Transforms' },
  { id: 'power', label: 'Matrix Power', hint: 'Aⁿ', group: 'Transforms' },
];

const EXAMPLES: Record<string, { A: Matrix; b?: Vector; B?: Matrix; power?: number }> = {
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
  trace: { A: [['1', '2'], ['3', '4']] },
  multiply: { A: [['1', '2', '3'], ['4', '5', '6']], B: [['1', '2'], ['3', '4'], ['5', '6']] },
  power: { A: [['1', '1'], ['0', '1']], power: 5 },
};

const FUNCTION_PALETTE = ['sin(', 'cos(', 'tan(', 'log(', 'ln(', 'exp(', 'sqrt(', 'diff(', 'integrate(', 'limit(', 'x', 'y', 'z', 'pi', 'e'];

function makeMatrix(rows: number, cols: number, fill = '0'): Matrix {
  return Array.from({ length: rows }, () => Array.from({ length: cols }, () => fill));
}

function makeVector(size: number, fill = '0'): Vector {
  return Array.from({ length: size }, () => fill);
}

function isMatrix(value: unknown): value is unknown[][] {
  return Array.isArray(value) && Array.isArray(value[0]);
}

function prettyValue(value: any): string {
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return '—';
    if (Math.abs(value) < 1e-12) return '0';
    return Number(value.toPrecision(8)).toString();
  }
  if (typeof value === 'object' && value !== null && 'real' in value && 'imag' in value) {
    const real = prettyValue(value.real);
    const imag = prettyValue(Math.abs(value.imag));
    return `${real} ${value.imag >= 0 ? '+' : '-'} ${imag}i`;
  }
  return String(value);
}

function ResultData({ value }: { value: any }) {
  if (value == null) return <span className="text-slate-400">—</span>;
  if (isMatrix(value)) {
    return (
      <div className="overflow-auto rounded-xl border border-slate-200 dark:border-slate-800">
        <table className="min-w-full border-collapse font-mono text-sm">
          <tbody>
            {value.map((row, r) => (
              <tr key={r}>
                {row.map((cell, c) => (
                  <td key={c} className="border-b border-r border-slate-100 px-3 py-2 text-right last:border-r-0 dark:border-slate-800">
                    {prettyValue(cell)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }
  if (Array.isArray(value)) {
    return <div className="flex flex-wrap gap-2">{value.map((x, i) => <span key={i} className="rounded-lg bg-slate-100 px-2.5 py-1 font-mono text-xs dark:bg-slate-800">{prettyValue(x)}</span>)}</div>;
  }
  if (typeof value === 'object') {
    return <pre className="max-h-80 overflow-auto rounded-xl bg-slate-950 p-3 text-xs text-slate-200">{JSON.stringify(value, null, 2)}</pre>;
  }
  return <span className="font-mono text-base">{prettyValue(value)}</span>;
}

function MatrixEditor({
  matrix,
  title,
  onChange,
  onFocusCell,
  focusedCell,
}: {
  matrix: Matrix;
  title: string;
  onChange: (next: Matrix) => void;
  onFocusCell: (r: number, c: number) => void;
  focusedCell: [number, number] | null;
}) {
  const setCell = (r: number, c: number, value: string) => {
    onChange(matrix.map((row, ri) => row.map((cell, ci) => ri === r && ci === c ? value : cell)));
  };

  return (
    <div className="min-w-0">
      <div className="mb-2 flex items-center justify-between">
        <div className="text-sm font-semibold">{title}</div>
        <div className="text-[11px] text-slate-400">{matrix.length} × {matrix[0]?.length ?? 0}</div>
      </div>
      <div className="overflow-auto rounded-2xl border border-slate-200 bg-slate-50 p-3 shadow-inner dark:border-slate-800 dark:bg-slate-950">
        <table className="border-collapse">
          <tbody>
            {matrix.map((row, r) => (
              <tr key={r}>
                {row.map((cell, c) => {
                  const focused = focusedCell?.[0] === r && focusedCell?.[1] === c;
                  return (
                    <td key={`${r}-${c}`} className="p-1.5">
                      <input
                        data-cell={`${r}-${c}`}
                        value={cell}
                        onFocus={() => onFocusCell(r, c)}
                        onChange={(e) => setCell(r, c, e.target.value)}
                        onKeyDown={(e) => {
                          const move: Record<string, [number, number]> = {
                            ArrowUp: [r - 1, c], ArrowDown: [r + 1, c], ArrowLeft: [r, c - 1], ArrowRight: [r, c + 1],
                          };
                          const target = move[e.key];
                          if (!target) return;
                          e.preventDefault();
                          (document.querySelector(`input[data-cell="${target[0]}-${target[1]}"]`) as HTMLInputElement | null)?.focus();
                        }}
                        className={`h-11 w-[88px] rounded-xl border bg-white px-2 text-center text-sm outline-none transition-all dark:bg-slate-900 ${focused ? 'border-blue-500 ring-2 ring-blue-200 dark:ring-blue-900' : 'border-slate-300 dark:border-slate-700'}`}
                        placeholder="0"
                        aria-label={`${title} row ${r + 1} column ${c + 1}`}
                      />
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function VectorEditor({ vector, title, onChange }: { vector: Vector; title: string; onChange: (next: Vector) => void }) {
  return (
    <div>
      <div className="mb-2 flex items-center justify-between"><div className="text-sm font-semibold">{title}</div><div className="text-[11px] text-slate-400">{vector.length} entries</div></div>
      <div className="flex flex-col gap-2 rounded-2xl border border-slate-200 bg-slate-50 p-3 dark:border-slate-800 dark:bg-slate-950">
        {vector.map((value, i) => (
          <input key={i} value={value} onChange={(e) => onChange(vector.map((v, j) => j === i ? e.target.value : v))} className="h-11 rounded-xl border border-slate-300 bg-white px-3 text-center text-sm outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-200 dark:border-slate-700 dark:bg-slate-900" placeholder="0" />
        ))}
      </div>
    </div>
  );
}

export default function LinearAlgebraModule() {
  const [viewMode, setViewMode] = useState<ViewMode>('matrix');
  const [operation, setOperation] = useState<MatrixOperation>('solve');
  const [calculusOperation, setCalculusOperation] = useState<CalculusOperation>('differentiate');
  const [rows, setRows] = useState(3);
  const [cols, setCols] = useState(3);
  const [bCols, setBCols] = useState(2);
  const [matrixA, setMatrixA] = useState<Matrix>(makeMatrix(3, 3));
  const [matrixB, setMatrixB] = useState<Matrix>(makeMatrix(3, 2));
  const [vectorB, setVectorB] = useState<Vector>(makeVector(3));
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
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [focusedCell, setFocusedCell] = useState<[number, number] | null>([0, 0]);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [consoleInput, setConsoleInput] = useState('');
  const [consoleOutput, setConsoleOutput] = useState<any[]>([]);
  const [variables, setVariables] = useState<Record<string, any>>({});
  const consoleRef = useRef<HTMLInputElement>(null);

  const groupedOps = useMemo(() => ['Core', 'Spectral', 'Decompositions', 'Transforms'].map((g) => ({ group: g, items: MATRIX_OPS.filter((x) => x.group === g) })), []);
  const selected = MATRIX_OPS.find((x) => x.id === operation);

  useEffect(() => {
    setMatrixA(makeMatrix(rows, cols));
    setVectorB(makeVector(rows));
    setMatrixB(makeMatrix(cols, bCols));
    setResult(null);
  }, [rows, cols]);

  useEffect(() => {
    setMatrixB((prev) => {
      const next = makeMatrix(cols, bCols);
      for (let r = 0; r < Math.min(prev.length, next.length); r++) {
        for (let c = 0; c < Math.min(prev[0]?.length ?? 0, next[0]?.length ?? 0); c++) next[r][c] = prev[r][c];
      }
      return next;
    });
  }, [bCols]);

  useEffect(() => {
    checkHealth();
    const timer = window.setInterval(checkHealth, 30000);
    return () => window.clearInterval(timer);
  }, []);

  async function checkHealth() {
    try {
      const response = await axios.get(`${API}/health`, { timeout: 2500 });
      setBackendOnline(response.status === 200);
    } catch {
      setBackendOnline(false);
    }
  }

  function pushHistory(title: string, data: any) {
    setHistory((old) => [{ id: Date.now(), title, timestamp: new Date().toLocaleTimeString(), result: data }, ...old].slice(0, 8));
  }

  function loadExample(nextOperation: MatrixOperation = operation) {
    const example = EXAMPLES[nextOperation];
    if (!example) return;
    setViewMode('matrix');
    setOperation(nextOperation);
    setRows(example.A.length);
    setCols(example.A[0]?.length ?? 1);
    setMatrixA(example.A.map((r) => [...r]));
    if (example.b) setVectorB([...example.b]);
    if (example.B) {
      setMatrixB(example.B.map((r) => [...r]));
      setBCols(example.B[0]?.length ?? 1);
    }
    if (example.power != null) setPower(example.power);
    setError(null);
    setResult(null);
  }

  async function runMatrixOperation() {
    setLoading(true); setError(null);
    try {
      let endpoint = `${API}/operation`;
      let body: any = { operation, matrix_a: matrixA, tolerance };
      if (operation === 'solve') { endpoint = `${API}/solve`; body = { matrix_a: matrixA, vector_b: vectorB, tolerance }; }
      if (operation === 'multiply') body.matrix_b = matrixB;
      if (operation === 'power') body.power = power;
      const response = await axios.post(endpoint, body, { timeout: 15000 });
      if (response.data?.error) throw new Error(response.data.error);
      setResult(response.data); setBackendOnline(true); pushHistory(selected?.label || operation, response.data);
    } catch (err: any) {
      setBackendOnline(err?.code === 'ERR_NETWORK' ? false : backendOnline);
      setError(err?.response?.data?.detail || err?.response?.data?.error || err?.message || 'Calculation failed.');
    } finally { setLoading(false); }
  }

  async function runCalculus() {
    setLoading(true); setError(null);
    try {
      const response = await axios.post(`${API}/calculus`, { matrix: matrixA, operation: calculusOperation, variable, order, lower: lower || null, upper: upper || null }, { timeout: 15000 });
      if (response.data?.error) throw new Error(response.data.error);
      setResult(response.data); setBackendOnline(true); pushHistory(`${calculusOperation} / ${variable}`, response.data);
    } catch (err: any) {
      setError(err?.response?.data?.error || err?.message || 'Calculus operation failed.');
    } finally { setLoading(false); }
  }

  async function runSymbolic() {
    setLoading(true); setError(null);
    try {
      const response = await axios.post(`${API}/symbolic`, { expression: symbolicExpression, variable, order }, { timeout: 15000 });
      if (response.data?.error) throw new Error(response.data.error);
      setResult(response.data); setBackendOnline(true); pushHistory('Symbolic expression', response.data);
    } catch (err: any) {
      setError(err?.response?.data?.error || err?.message || 'Symbolic operation failed.');
    } finally { setLoading(false); }
  }

  async function runCommand(command = consoleInput) {
    const clean = command.trim();
    if (!clean) return;
    try {
      const response = await axios.post(`${API}/command`, { command: clean, variables, tolerance }, { timeout: 15000 });
      const data = response.data;
      setConsoleOutput((old) => [{ command: clean, result: data.result ?? data.error }, ...old].slice(0, 10));
      if (data.error) setError(data.error); else setError(null);
      setVariables(data.variables || variables);
      setBackendOnline(true);
      setConsoleInput('');
      pushHistory(clean, data.result ?? data.error);
    } catch (err: any) {
      setBackendOnline(false);
      setConsoleOutput((old) => [{ command: clean, result: err?.message || 'Command failed.' }, ...old].slice(0, 10));
    }
  }

  function insertFunction(token: string) {
    if (!focusedCell) return;
    const [r, c] = focusedCell;
    const next = matrixA.map((row) => [...row]);
    const current = next[r]?.[c] ?? '';
    next[r][c] = token.endsWith('(') ? `${current}${token}` : `${current}${token}`;
    setMatrixA(next);
    requestAnimationFrame(() => (document.querySelector(`input[data-cell="${r}-${c}"]`) as HTMLInputElement | null)?.focus());
  }

  function importMatrix() {
    try {
      const parsed = JSON.parse(prompt('Paste matrix JSON, e.g. [[sin(x),2],[3,log(10)]]') || '');
      if (!Array.isArray(parsed) || !parsed.length || !Array.isArray(parsed[0])) throw new Error('Expected a non-empty 2D matrix.');
      const width = parsed[0].length;
      if (parsed.some((row: any) => !Array.isArray(row) || row.length !== width)) throw new Error('All rows must have the same length.');
      const next = parsed.map((row: any[]) => row.map((x) => String(x)));
      setRows(next.length); setCols(width); setMatrixA(next); setError(null); setResult(null);
    } catch (e: any) {
      if (e?.message !== 'Unexpected end of JSON input') setError(e?.message || 'Invalid matrix.');
    }
  }

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.ctrlKey && event.key === 'Enter') {
        event.preventDefault();
        if (viewMode === 'matrix') runMatrixOperation();
        if (viewMode === 'calculus') runCalculus();
        if (viewMode === 'symbolic') runSymbolic();
      }
      if (event.key === '/' && document.activeElement !== consoleRef.current) {
        if (event.ctrlKey) { event.preventDefault(); consoleRef.current?.focus(); }
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  return (
    <div className="min-h-screen bg-slate-100 text-slate-900 dark:bg-slate-950 dark:text-slate-100">
      <style>{`
        @keyframes engSlideIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes engPulseLine { 0%,100% { opacity:.45 } 50% { opacity:1 } }
        .eng-enter { animation: engSlideIn .28s ease-out both; }
        .eng-pulse { animation: engPulseLine 1.8s ease-in-out infinite; }
      `}</style>

      <div className="mx-auto max-w-[1800px] p-3 md:p-5">
        <header className="eng-enter mb-3 rounded-2xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <div className="flex items-center gap-3">
              <div className="rounded-xl bg-blue-600 px-3 py-2 text-sm font-black text-white shadow-sm">LA</div>
              <div>
                <h1 className="text-2xl font-bold tracking-tight">ENG-OS Linear Algebra</h1>
                <p className="text-xs text-slate-500">Symbolic + numerical engineering mathematics</p>
              </div>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <div className="rounded-full border border-slate-200 px-3 py-1.5 text-[11px] dark:border-slate-700">NumPy</div>
              <div className="rounded-full border border-slate-200 px-3 py-1.5 text-[11px] dark:border-slate-700">SciPy</div>
              <div className="rounded-full border border-slate-200 px-3 py-1.5 text-[11px] dark:border-slate-700">SymPy</div>
              <div className={`rounded-full px-3 py-1.5 text-[11px] font-semibold ${backendOnline === true ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-300' : backendOnline === false ? 'bg-red-100 text-red-700 dark:bg-red-950/40 dark:text-red-300' : 'bg-slate-100 text-slate-500 dark:bg-slate-800'}`}>
                <span className={backendOnline === null ? 'eng-pulse' : ''}>●</span> {backendOnline === true ? 'Backend online' : backendOnline === false ? 'Backend offline' : 'Checking backend'}
              </div>
            </div>
          </div>
        </header>

        <div className="grid gap-3 lg:grid-cols-[250px_minmax(0,1fr)_340px]">
          {/* LEFT SIDEBAR */}
          <aside className="eng-enter rounded-2xl border border-slate-200 bg-white p-3 shadow-sm dark:border-slate-800 dark:bg-slate-900">
            <div className="mb-3 px-2">
              <div className="text-[10px] font-bold uppercase tracking-[.18em] text-slate-400">Workspace</div>
              <div className="mt-1 text-xs text-slate-500">Pick a mathematical mode.</div>
            </div>

            <div className="mb-3 grid grid-cols-3 gap-1 rounded-xl bg-slate-100 p-1 dark:bg-slate-950">
              {(['matrix', 'calculus', 'symbolic'] as ViewMode[]).map((mode) => (
                <button key={mode} onClick={() => setViewMode(mode)} className={`rounded-lg px-2 py-2 text-[11px] font-semibold capitalize ${viewMode === mode ? 'bg-white shadow-sm dark:bg-slate-800' : 'text-slate-500'}`}>{mode}</button>
              ))}
            </div>

            {viewMode === 'matrix' && groupedOps.map(({ group, items }) => (
              <div key={group} className="mb-3 last:mb-0">
                <div className="mb-1 px-2 text-[9px] font-bold uppercase tracking-[.16em] text-slate-400">{group}</div>
                <div className="space-y-1">
                  {items.map((item) => (
                    <button key={item.id} onClick={() => setOperation(item.id)} className={`w-full rounded-xl px-2.5 py-2 text-left transition-all ${operation === item.id ? 'bg-blue-600 text-white shadow-sm' : 'hover:bg-slate-100 dark:hover:bg-slate-800'}`}>
                      <div className="text-xs font-semibold">{item.label}</div>
                      <div className={`text-[10px] ${operation === item.id ? 'text-blue-100' : 'text-slate-400'}`}>{item.hint}</div>
                    </button>
                  ))}
                </div>
              </div>
            ))}

            {viewMode === 'calculus' && (
              <div className="space-y-2">
                <button onClick={() => setCalculusOperation('differentiate')} className={`w-full rounded-xl px-3 py-2 text-left text-xs font-semibold ${calculusOperation === 'differentiate' ? 'bg-blue-600 text-white' : 'hover:bg-slate-100 dark:hover:bg-slate-800'}`}>Differentiate matrix</button>
                <button onClick={() => setCalculusOperation('integrate')} className={`w-full rounded-xl px-3 py-2 text-left text-xs font-semibold ${calculusOperation === 'integrate' ? 'bg-blue-600 text-white' : 'hover:bg-slate-100 dark:hover:bg-slate-800'}`}>Integrate matrix</button>
                <div className="rounded-xl border border-slate-200 p-2 dark:border-slate-800">
                  <div className="text-[10px] font-semibold text-slate-500">Variable</div>
                  <input value={variable} onChange={(e) => setVariable(e.target.value)} className="mt-1 h-9 w-full rounded-lg border border-slate-300 bg-transparent px-2 text-xs dark:border-slate-700" />
                  <div className="mt-2 text-[10px] font-semibold text-slate-500">Order</div>
                  <input type="number" min={1} max={20} value={order} onChange={(e) => setOrder(Number(e.target.value) || 1)} className="mt-1 h-9 w-full rounded-lg border border-slate-300 bg-transparent px-2 text-xs dark:border-slate-700" />
                </div>
                {calculusOperation === 'integrate' && <div className="rounded-xl border border-slate-200 p-2 dark:border-slate-800"><div className="text-[10px] font-semibold text-slate-500">Optional bounds</div><div className="mt-1 grid grid-cols-2 gap-2"><input value={lower} onChange={(e) => setLower(e.target.value)} placeholder="lower" className="h-8 rounded-lg border border-slate-300 px-2 text-xs dark:border-slate-700 dark:bg-slate-900" /><input value={upper} onChange={(e) => setUpper(e.target.value)} placeholder="upper" className="h-8 rounded-lg border border-slate-300 px-2 text-xs dark:border-slate-700 dark:bg-slate-900" /></div></div>}
              </div>
            )}

            {viewMode === 'symbolic' && <div className="space-y-2 rounded-xl border border-slate-200 p-2 dark:border-slate-800"><div className="text-[10px] font-semibold uppercase tracking-wide text-slate-500">Expression</div><textarea value={symbolicExpression} onChange={(e) => setSymbolicExpression(e.target.value)} className="min-h-28 w-full resize-y rounded-lg border border-slate-300 bg-transparent p-2 font-mono text-xs dark:border-slate-700" /><div className="text-[10px] text-slate-400">Examples: <code>sin(x)</code>, <code>x^2 + log(x)</code>, <code>sqrt(1-x^2)</code></div></div>}

            <div className="mt-3 rounded-xl border border-slate-200 p-2 dark:border-slate-800">
              <div className="mb-2 flex items-center justify-between"><div className="text-[10px] font-semibold uppercase tracking-wide text-slate-500">Function palette</div><div className="text-[9px] text-slate-400">focused cell</div></div>
              <div className="flex flex-wrap gap-1">
                {FUNCTION_PALETTE.map((fn) => <button key={fn} onClick={() => insertFunction(fn)} className="rounded-md border border-slate-200 px-2 py-1 text-[10px] hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800">{fn}</button>)}
              </div>
            </div>

            <button onClick={() => loadExample(operation)} className="mt-3 w-full rounded-xl border border-slate-300 px-3 py-2 text-xs font-semibold hover:bg-slate-50 dark:border-slate-700 dark:hover:bg-slate-800">Load example</button>
          </aside>

          {/* CENTER */}
          <main className="min-w-0 space-y-3">
            {viewMode === 'matrix' && (
              <section className="eng-enter rounded-2xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900">
                <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
                  <div>
                    <div className="text-[10px] font-bold uppercase tracking-[.18em] text-blue-600">Active operation</div>
                    <h2 className="mt-1 text-xl font-bold">{selected?.label}</h2>
                    <p className="mt-1 text-xs text-slate-500">{selected?.hint}. Matrix entries accept numbers and symbolic functions.</p>
                  </div>
                  <div className="flex flex-wrap gap-2"><button onClick={importMatrix} className="rounded-xl border border-slate-300 px-3 py-2 text-xs font-semibold dark:border-slate-700">Import JSON</button><button onClick={runMatrixOperation} disabled={loading} className="rounded-xl bg-blue-600 px-5 py-2 text-xs font-semibold text-white shadow-sm disabled:opacity-50">{loading ? 'Working…' : 'Calculate'}</button></div>
                </div>

                <div className="mt-4 flex flex-wrap gap-2 border-t border-slate-200 pt-4 dark:border-slate-800">
                  <label className="text-[10px] font-semibold text-slate-500">Rows<input type="number" min={1} max={50} value={rows} onChange={(e) => setRows(Math.max(1, Math.min(50, Number(e.target.value) || 1)))} className="mt-1 block h-9 w-16 rounded-lg border border-slate-300 bg-transparent px-2 text-xs dark:border-slate-700" /></label>
                  <label className="text-[10px] font-semibold text-slate-500">Columns<input type="number" min={1} max={50} value={cols} onChange={(e) => setCols(Math.max(1, Math.min(50, Number(e.target.value) || 1)))} className="mt-1 block h-9 w-16 rounded-lg border border-slate-300 bg-transparent px-2 text-xs dark:border-slate-700" /></label>
                  {operation === 'multiply' && <label className="text-[10px] font-semibold text-slate-500">B columns<input type="number" min={1} max={50} value={bCols} onChange={(e) => setBCols(Math.max(1, Math.min(50, Number(e.target.value) || 1)))} className="mt-1 block h-9 w-20 rounded-lg border border-slate-300 bg-transparent px-2 text-xs dark:border-slate-700" /></label>}
                  {operation === 'power' && <label className="text-[10px] font-semibold text-slate-500">Power n<input type="number" min={-100} max={100} value={power} onChange={(e) => setPower(Number(e.target.value))} className="mt-1 block h-9 w-20 rounded-lg border border-slate-300 bg-transparent px-2 text-xs dark:border-slate-700" /></label>}
                  <label className="text-[10px] font-semibold text-slate-500">Tolerance<input type="number" step="any" value={tolerance} onChange={(e) => setTolerance(Number(e.target.value) || 1e-10)} className="mt-1 block h-9 w-24 rounded-lg border border-slate-300 bg-transparent px-2 text-xs dark:border-slate-700" /></label>
                </div>

                <div className="mt-5 grid gap-5 xl:grid-cols-2">
                  <MatrixEditor matrix={matrixA} title="Matrix A" onChange={setMatrixA} onFocusCell={(r, c) => setFocusedCell([r, c])} focusedCell={focusedCell} />
                  {operation === 'solve' ? <VectorEditor vector={vectorB} title="Vector b" onChange={setVectorB} /> : operation === 'multiply' ? <MatrixEditor matrix={matrixB} title="Matrix B" onChange={setMatrixB} onFocusCell={() => setFocusedCell(null)} focusedCell={null} /> : <div className="rounded-2xl border border-dashed border-slate-300 p-6 text-xs text-slate-500 dark:border-slate-700"><div className="font-semibold">Expression cells enabled</div><p className="mt-1">Try <code className="rounded bg-slate-100 px-1 dark:bg-slate-800">sin(x)</code>, <code className="rounded bg-slate-100 px-1 dark:bg-slate-800">log(10)</code> or <code className="rounded bg-slate-100 px-1 dark:bg-slate-800">x^2</code> in Matrix A.</p></div>}
                </div>

                <div className="mt-4 rounded-xl bg-slate-50 p-3 text-[10px] text-slate-500 dark:bg-slate-950"><b>Tip:</b> click a matrix cell, then use the function palette to insert symbolic functions. Ctrl + Enter runs the active tool.</div>
              </section>
            )}

            {viewMode === 'calculus' && (
              <section className="eng-enter rounded-2xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900">
                <div className="flex items-start justify-between gap-4"><div><div className="text-[10px] font-bold uppercase tracking-[.18em] text-blue-600">Matrix calculus</div><h2 className="mt-1 text-xl font-bold">{calculusOperation === 'differentiate' ? 'Differentiate each matrix entry' : 'Integrate each matrix entry'}</h2><p className="mt-1 text-xs text-slate-500">Uses SymPy elementwise differentiation/integration.</p></div><button onClick={runCalculus} disabled={loading} className="rounded-xl bg-blue-600 px-5 py-2 text-xs font-semibold text-white disabled:opacity-50">{loading ? 'Working…' : 'Run calculus'}</button></div>
                <div className="mt-4 grid gap-4 xl:grid-cols-2"><MatrixEditor matrix={matrixA} title="Expression Matrix" onChange={setMatrixA} onFocusCell={(r, c) => setFocusedCell([r, c])} focusedCell={focusedCell} /><div className="rounded-2xl border border-slate-200 p-4 dark:border-slate-800"><div className="text-xs font-semibold">Current settings</div><div className="mt-3 grid grid-cols-2 gap-2 text-xs"><div className="rounded-xl bg-slate-50 p-3 dark:bg-slate-950"><div className="text-slate-400">Variable</div><div className="mt-1 font-mono">{variable}</div></div><div className="rounded-xl bg-slate-50 p-3 dark:bg-slate-950"><div className="text-slate-400">Order</div><div className="mt-1 font-mono">{order}</div></div></div>{calculusOperation === 'integrate' && <div className="mt-3 text-[11px] text-slate-500">Bounds are optional. With no bounds, an indefinite integral is returned.</div>}<div className="mt-4"><div className="text-[10px] uppercase tracking-wide text-slate-400">Example</div><div className="mt-1 font-mono text-xs">[[sin(x), x^2], [exp(x), log(x)]]</div></div></div></div>
              </section>
            )}

            {viewMode === 'symbolic' && (
              <section className="eng-enter rounded-2xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900"><div className="flex items-start justify-between gap-4"><div><div className="text-[10px] font-bold uppercase tracking-[.18em] text-blue-600">Symbolic engine</div><h2 className="mt-1 text-xl font-bold">Expression workspace</h2></div><button onClick={runSymbolic} disabled={loading} className="rounded-xl bg-blue-600 px-5 py-2 text-xs font-semibold text-white disabled:opacity-50">{loading ? 'Working…' : 'Evaluate'}</button></div><textarea value={symbolicExpression} onChange={(e) => setSymbolicExpression(e.target.value)} className="mt-4 min-h-40 w-full rounded-2xl border border-slate-300 bg-slate-50 p-4 font-mono text-sm outline-none focus:border-blue-500 dark:border-slate-700 dark:bg-slate-950" /><div className="mt-3 text-[11px] text-slate-500">For derivatives or integrals of a single expression, use <code>diff:</code> or <code>integrate:</code> prefixes, or use the Matrix Calculus tab for elementwise matrix calculus.</div></section>
            )}

            {/* Command console */}
            <section className="eng-enter rounded-2xl border border-slate-800 bg-slate-950 p-4 text-slate-100 shadow-sm"><div className="flex items-center justify-between"><div><div className="font-semibold text-sm">Command Console</div><div className="text-[10px] text-slate-500">A = [[sin(x),2],[3,log(10)]] · solve(A,b) · diff(x^3,x)</div></div><button onClick={() => runCommand('help()')} className="rounded-lg bg-slate-800 px-3 py-1.5 text-[10px]">help()</button></div><div className="mt-3 max-h-44 overflow-auto rounded-xl border border-slate-800 p-3 font-mono text-[10px]">{consoleOutput.length === 0 ? <span className="text-slate-600">Enter a command to keep reusable workspace variables.</span> : consoleOutput.map((x, i) => <div key={i} className="mb-3 last:mb-0"><div className="text-blue-400">$ {x.command}</div><pre className="mt-1 whitespace-pre-wrap text-slate-300">{JSON.stringify(x.result, null, 2)}</pre></div>)}</div><div className="mt-2 flex gap-2"><input ref={consoleRef} value={consoleInput} onChange={(e) => setConsoleInput(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter') runCommand(); }} placeholder="A = [[1,2],[3,4]]" className="min-w-0 flex-1 rounded-xl border border-slate-700 bg-slate-900 px-3 py-2.5 font-mono text-xs outline-none focus:ring-2 focus:ring-blue-500"/><button onClick={() => runCommand()} className="rounded-xl bg-blue-600 px-4 text-xs font-semibold">Run</button></div></section>
          </main>

          {/* RIGHT SIDE */}
          <aside className="space-y-3">
            <section className="eng-enter rounded-2xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900"><div className="mb-3 flex items-center justify-between"><div><div className="text-sm font-semibold">Result</div><div className="text-[10px] text-slate-400">Live calculation output</div></div><button onClick={() => { setResult(null); setError(null); }} className="text-[10px] text-slate-500 hover:text-red-500">Clear</button></div>{error && <div className="mb-3 rounded-xl border border-red-200 bg-red-50 p-3 text-xs text-red-700 dark:border-red-900 dark:bg-red-950/30 dark:text-red-300">{error}</div>}{result ? <div className="eng-enter space-y-3">{result.latex && <div className="overflow-auto rounded-xl bg-slate-50 p-3 dark:bg-slate-950"><BlockMath math={result.latex} /></div>}{Object.entries(result).map(([key, value]) => { if (['latex','error','metadata','operation','mode'].includes(key)) return null; return <div key={key} className="space-y-1"><div className="text-[9px] font-bold uppercase tracking-wide text-slate-400">{key.replaceAll('_', ' ')}</div><ResultData value={value} /></div>; })}{result.metadata && <div className="grid grid-cols-2 gap-2 border-t border-slate-200 pt-3 dark:border-slate-800">{Object.entries(result.metadata).slice(0, 8).map(([k, v]) => <div key={k} className="rounded-xl bg-slate-50 p-2 dark:bg-slate-950"><div className="text-[9px] text-slate-400">{k.replaceAll('_', ' ')}</div><div className="mt-1 truncate text-xs font-semibold">{Array.isArray(v) ? `[${v.length}]` : prettyValue(v)}</div></div>)}</div>}</div> : <div className="rounded-xl border border-dashed border-slate-300 p-4 text-xs text-slate-500 dark:border-slate-700">No result yet.</div>}</section>

            <section className="rounded-2xl border border-slate-200 bg-white p-3 shadow-sm dark:border-slate-800 dark:bg-slate-900"><div className="mb-2 flex items-center justify-between"><div className="text-sm font-semibold">Workspace</div><span className="text-[9px] text-slate-400">{Object.keys(variables).length} vars</span></div>{Object.keys(variables).length === 0 ? <div className="text-[11px] text-slate-500">No variables yet.</div> : <div className="max-h-28 space-y-1 overflow-auto">{Object.entries(variables).map(([k, v]) => <div key={k} className="truncate rounded-lg bg-slate-50 p-2 font-mono text-[10px] dark:bg-slate-950"><b>{k}</b> = {JSON.stringify(v)}</div>)}</div>}</section>

            <section className="rounded-2xl border border-slate-200 bg-white p-3 shadow-sm dark:border-slate-800 dark:bg-slate-900"><div className="mb-2 flex items-center justify-between"><div className="text-sm font-semibold">History</div><button onClick={() => setHistory([])} className="text-[10px] text-slate-500">Clear</button></div>{history.length === 0 ? <div className="text-[11px] text-slate-500">No calculations yet.</div> : <div className="max-h-44 space-y-1 overflow-auto">{history.map((item) => <button key={item.id} onClick={() => setResult(item.result)} className="w-full rounded-lg p-2 text-left hover:bg-slate-100 dark:hover:bg-slate-800"><div className="truncate text-[10px] font-semibold">{item.title}</div><div className="text-[9px] text-slate-400">{item.timestamp}</div></button>)}</div>}</section>

            <section className="rounded-2xl border border-slate-200 bg-white p-3 shadow-sm dark:border-slate-800 dark:bg-slate-900"><div className="text-sm font-semibold">Matrix shortcuts</div><div className="mt-2 grid grid-cols-2 gap-1 text-[10px]"><button onClick={() => setMatrixA(EXAMPLES.solve.A)} className="rounded-lg bg-slate-100 px-2 py-2 dark:bg-slate-950">2×2 solve</button><button onClick={() => setMatrixA([['sin(x)', 'cos(x)'], ['log(x)', 'sqrt(x)']])} className="rounded-lg bg-slate-100 px-2 py-2 dark:bg-slate-950">Function matrix</button><button onClick={() => setMatrixA([['x^2', '2*x'], ['sin(x)', 'exp(x)']])} className="rounded-lg bg-slate-100 px-2 py-2 dark:bg-slate-950">Calculus matrix</button><button onClick={() => { setViewMode('symbolic'); setSymbolicExpression('diff: x^3*sin(x)'); }} className="rounded-lg bg-slate-100 px-2 py-2 dark:bg-slate-950">Symbolic demo</button></div></section>
          </aside>
        </div>
      </div>
    </div>
  );
}
