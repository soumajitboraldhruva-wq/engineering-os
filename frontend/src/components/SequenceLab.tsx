import { useMemo, useState } from 'react';
import type { ChangeEvent, ReactNode } from 'react';
import axios from 'axios';
import PlotComponent from 'react-plotly.js';

const Plot = (() => {
  const moduleValue = PlotComponent as any;

  if (typeof moduleValue === 'function') {
    return moduleValue;
  }

  if (typeof moduleValue?.default === 'function') {
    return moduleValue.default;
  }

  if (typeof moduleValue?.default?.default === 'function') {
    return moduleValue.default.default;
  }

  throw new Error(
    'react-plotly.js did not resolve to a React component.'
  );
})();
import {
  ArrowLeft,
  BarChart3,
  Binary,
  Boxes,
  CheckCircle2,
  ChevronRight,
  CircleHelp,
  Cpu,
  Database,
  FileText,
  FlaskConical,
  GitBranch,
  Grid3X3,
  KeyRound,
  LineChart,
  LockKeyhole,
  Moon,
  Play,
  RefreshCcw,
  Search,
  Sigma,
  Sparkles,
  Sun,
  Table2,
  TerminalSquare,
  TestTube2,
  TrendingUp,
  Upload,
  Workflow,
  Zap,
} from 'lucide-react';

type AnyRecord = Record<string, any>;

const API = (import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000').replace(/\/+$/, '') + '/api/analyser';

const MODEL_FAMILIES = [
  ['mean', 'Mean baseline'],
  ['linear', 'Linear'],
  ['poly', 'Polynomial'],
  ['chebyshev', 'Chebyshev polynomial'],
  ['taylor', 'Taylor / Newton'],
  ['exp', 'Exponential'],
  ['log', 'Logarithmic'],
  ['power', 'Power law'],
  ['rational', 'Rational'],
  ['sin', 'Sinusoid'],
  ['fourier', 'Fourier / harmonics'],
  ['damped_sine', 'Damped sinusoid (RLC transient)'],
  ['ar', 'Autoregressive'],
  ['differenced_ar', 'Differenced AR'],
  ['prony', 'Prony / Z-transform'],
  ['holt', 'Holt trend'],
  ['damped_holt', 'Damped Holt'],
  ['piecewise', 'Piecewise linear'],
  ['piecewise_const', 'Piecewise constant / ZOH'],
  ['segmented', 'Segmented linear'],
  ['chirp', 'Chirp / quadratic phase'],
  ['root_poly', 'Root-guided polynomial'],
  ['logistic', 'Logistic + trend'],
  ['gompertz', 'Gompertz + trend'],
  ['symbolic', 'Bounded symbolic regression'],
  ['single_feature', 'Single-feature search'],
  ['rbf', 'RBF / kernel ridge'],
  ['ridge_basis', 'Ridge basis'],
  ['lasso_basis', 'Lasso basis'],
  ['elastic_basis', 'Elastic Net basis'],
  ['huber_basis', 'Huber basis'],
] as const;

const familyLabel = (family: string) =>
  MODEL_FAMILIES.find(([id]) => id === family)?.[1] ?? family;

const safeNumber = (value: unknown, digits = 4) => {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(digits) : '—';
};

// ----------------------------------------------------------------------
// Flexible data-entry helpers.
//
// Every native engine command already reads whitespace/comma/tab-separated
// tokens per line (see parseNumberLine in the C++ engine), so a "row" can be
// typed with spaces or commas interchangeably without any change here or on
// the server. The one axis that previously was NOT flexible is orientation:
// a multivariate/coordinate dataset had to be pasted with one point per line
// (each line = one row of variables). Some users have data the other way
// around -- one variable's full column pasted per line. These helpers let a
// screen offer both without touching the Python bridge or the C++ parser:
// "columns" input is simply transposed into row form in the browser before
// it is sent, so the wire format the backend already understands is unchanged.
// ----------------------------------------------------------------------
function parseRoughMatrix(text: string): number[][] {
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
    .map((line) =>
      line
        .split(/[\s,]+/)
        .filter((tok) => tok.length > 0)
        .map(Number)
    );
}

function matrixToLines(matrix: number[][]): string {
  return matrix.map((row) => row.join(' ')).join('\n');
}

function transposeRagged(matrix: number[][]): number[][] {
  if (!matrix.length) return [];
  const cols = Math.max(...matrix.map((row) => row.length));
  const out: number[][] = Array.from({ length: cols }, () => [] as number[]);
  matrix.forEach((row) => {
    row.forEach((value, i) => {
      out[i].push(value);
    });
  });
  return out;
}

type ImportedDataGroup = { id: string; title: string; headers: string[]; rows: number[][] };

function splitImportLine(line: string, delimiter: string): string[] {
  if (delimiter === " ") return line.trim().split(/\\s+/);
  const cells: string[] = [];
  let cell = "";
  let quoted = false;
  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    if (char === '"' && quoted && line[i + 1] === '"') { cell += '"'; i += 1; }
    else if (char === '"') quoted = !quoted;
    else if (char === delimiter && !quoted) { cells.push(cell.trim()); cell = ""; }
    else cell += char;
  }
  cells.push(cell.trim());
  return cells;
}

function parseImportedDataset(text: string): ImportedDataGroup[] {
  const lines = text.replace(/^\\uFEFF/, "").split(/\\r?\\n/);
  const meaningful = lines.map((line) => line.trim()).filter((line) => line && !line.startsWith("#") && !line.startsWith("//"));
  if (!meaningful.length) return [];
  const sample = meaningful.slice(0, 12).join("\\n");
  const counts = [",", "\\t", ";"].map((delimiter) => ({ delimiter, count: (sample.match(new RegExp(delimiter === "\\t" ? "\\t" : `\\${delimiter}`, "g")) ?? []).length }));
  const best = counts.sort((a, b) => b.count - a.count)[0];
  const delimiter = best.count > 0 ? best.delimiter : " ";
  const groups: ImportedDataGroup[] = [];
  let current: ImportedDataGroup = { id: "section-1", title: "Dataset", headers: [], rows: [] };
  let sectionNumber = 1;
  const beginSection = (title?: string) => {
    if (current.rows.length || current.headers.length) groups.push(current);
    sectionNumber += 1;
    current = { id: `section-${sectionNumber}`, title: title?.trim() || `Dataset ${sectionNumber}`, headers: [], rows: [] };
  };
  for (const raw of lines) {
    const line = raw.trim();
    if (!line || line.startsWith("#") || line.startsWith("//")) continue;
    const cells = splitImportLine(line, delimiter);
    if (!cells.length || cells.every((cell) => !cell)) continue;
    const numeric = cells.map((cell) => cell === "" ? Number.NaN : Number(cell));
    const allNumeric = cells.every((cell, index) => cell !== "" && Number.isFinite(numeric[index]));
    const hasText = cells.some((cell, index) => cell !== "" && !Number.isFinite(numeric[index]));
    if (cells.length === 1 && hasText) {
      if (current.rows.length || current.headers.length) beginSection(cells[0]);
      else current.title = cells[0];
      continue;
    }
    if (hasText) {
      const header = cells.map((cell, index) => cell || `Column ${index + 1}`);
      if (current.rows.length) beginSection();
      current.headers = header;
      continue;
    }
    if (!allNumeric) continue;
    if (!current.headers.length) current.headers = cells.map((_, index) => `Column ${index + 1}`);
    if (cells.length > current.headers.length) {
      for (let i = current.headers.length; i < cells.length; i += 1) current.headers.push(`Column ${i + 1}`);
    }
    const row = Array.from({ length: current.headers.length }, (_, index) => numeric[index] ?? Number.NaN);
    current.rows.push(row);
  }
  if (current.rows.length || current.headers.length) groups.push(current);
  return groups
    .filter((group) => group.rows.some((row) => row.some((value) => Number.isFinite(value))))
    .map((group, index) => ({ ...group, id: `section-${index + 1}` }));
}

function numericColumns(group?: ImportedDataGroup): number[] {
  if (!group) return [];
  return group.headers.map((_, index) => index).filter((index) => group.rows.some((row) => Number.isFinite(row[index])));
}

type Layout = 'rows' | 'columns';

/**
 * Normalizes user-typed text into the row-per-point format every backend
 * command expects. In "rows" layout the text is passed straight through
 * (each line is already one point / one row of variables). In "columns"
 * layout each line is treated as one full variable's values across every
 * point, and the matrix is transposed before being sent.
 */
function normalizeByLayout(text: string, layout: Layout): string {
  if (layout === 'rows') return text;
  const matrix = parseRoughMatrix(text);
  return matrixToLines(transposeRagged(matrix));
}

function LayoutToggle({
  value,
  onChange,
  rowsHint,
  columnsHint,
}: {
  value: Layout;
  onChange: (next: Layout) => void;
  rowsHint: string;
  columnsHint: string;
}) {
  return (
    <div>
      <div className="mb-1.5 text-xs font-bold uppercase tracking-wide text-slate-500 dark:text-slate-400">
        Data layout
      </div>
      <div className="grid grid-cols-2 gap-2">
        <button
          type="button"
          onClick={() => onChange('rows')}
          title={rowsHint}
          className={`rounded-xl border px-3 py-2 text-xs font-semibold transition ${
            value === 'rows'
              ? 'border-blue-600 bg-blue-600 text-white'
              : 'border-slate-300 bg-white text-slate-700 hover:bg-slate-50 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-200 dark:hover:bg-slate-700'
          }`}
        >
          Rows (one point per line)
        </button>
        <button
          type="button"
          onClick={() => onChange('columns')}
          title={columnsHint}
          className={`rounded-xl border px-3 py-2 text-xs font-semibold transition ${
            value === 'columns'
              ? 'border-blue-600 bg-blue-600 text-white'
              : 'border-slate-300 bg-white text-slate-700 hover:bg-slate-50 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-200 dark:hover:bg-slate-700'
          }`}
        >
          Columns (one variable per line)
        </button>
      </div>
      <p className="mt-1.5 text-[11px] leading-4 text-slate-400 dark:text-slate-500">
        {value === 'rows' ? rowsHint : columnsHint} Spaces or commas both work either way.
      </p>
    </div>
  );
}

const shellCard =
  'rounded-2xl border border-slate-200 bg-white shadow-sm dark:border-slate-700 dark:bg-slate-800';
const controlClass =
  'w-full rounded-xl border border-slate-300 bg-white px-3 py-2.5 text-sm text-slate-800 outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-100 dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100 dark:focus:border-blue-400 dark:focus:ring-blue-900';
const primaryButton =
  'inline-flex items-center justify-center gap-2 rounded-xl bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50';
const secondaryButton =
  'inline-flex items-center justify-center gap-2 rounded-xl border border-slate-300 bg-white px-4 py-2.5 text-sm font-semibold text-slate-700 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-200 dark:hover:bg-slate-700';

function PageShell({
  title,
  subtitle,
  children,
  onBack,
}: {
  title: string;
  subtitle?: string;
  children: ReactNode;
  onBack: () => void;
}) {
  return (
    <div className="space-y-5 text-slate-900 dark:text-slate-100">
      <div className="flex items-start justify-between gap-4">
        <div>
          <button
            type="button"
            onClick={onBack}
            className="mb-3 inline-flex items-center gap-2 text-sm font-semibold text-slate-500 hover:text-slate-900 dark:text-slate-400 dark:hover:text-white"
          >
            <ArrowLeft size={16} /> Back to Analyser
          </button>
          <h1 className="text-2xl font-bold tracking-tight">{title}</h1>
          {subtitle && <p className="mt-1 max-w-4xl text-sm text-slate-500 dark:text-slate-400">{subtitle}</p>}
        </div>
      </div>
      {children}
    </div>
  );
}

function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    <label className="block">
      <div className="mb-1.5 flex items-baseline justify-between gap-2">
        <span className="text-xs font-bold uppercase tracking-wide text-slate-500 dark:text-slate-400">{label}</span>
        {hint && <span className="text-[11px] text-slate-400 dark:text-slate-500">{hint}</span>}
      </div>
      {children}
    </label>
  );
}

function InfoStrip({ children }: { children: ReactNode }) {
  return (
    <div className="rounded-xl border border-blue-100 bg-blue-50 px-3 py-2.5 text-xs leading-5 text-blue-900 dark:border-blue-900/50 dark:bg-blue-950/40 dark:text-blue-200">
      {children}
    </div>
  );
}

function AnalysisSummary({ result }: { result: AnyRecord }) {
  const analysis = result?.analysis ?? result;
  const model = analysis?.model ?? {};
  const stats = result?.stats ?? {};
  return (
    <div className="grid gap-4 xl:grid-cols-[1.2fr_1fr]">
      <div className={`${shellCard} p-5`}>
        <div className="mb-3 flex items-center gap-2">
          <CheckCircle2 className="text-blue-600 dark:text-blue-400" size={19} />
          <h3 className="font-bold">Engine Result</h3>
        </div>
        <div className="grid gap-3 sm:grid-cols-2">
          <div className="rounded-xl bg-slate-50 p-3 dark:bg-slate-900/60">
            <div className="text-[10px] font-bold uppercase tracking-wide text-slate-400 dark:text-slate-500">Category</div>
            <div className="mt-1 text-sm font-semibold">{analysis?.category || '—'}</div>
          </div>
          <div className="rounded-xl bg-slate-50 p-3 dark:bg-slate-900/60">
            <div className="text-[10px] font-bold uppercase tracking-wide text-slate-400 dark:text-slate-500">Evidence score</div>
            <div className="mt-1 font-mono text-sm font-semibold">
              {safeNumber(analysis?.evidenceScore, 2)}%
            </div>
          </div>
        </div>
        <div className="mt-3 rounded-xl border border-slate-200 bg-white p-3 dark:border-slate-700 dark:bg-slate-900/40">
          <div className="text-[10px] font-bold uppercase tracking-wide text-slate-400 dark:text-slate-500">Formula / model</div>
          <div className="mt-1 break-words font-mono text-sm text-slate-800 dark:text-slate-200">
            {analysis?.formula || model?.formula || 'No model formula returned.'}
          </div>
        </div>
        <div className="mt-3 rounded-xl border border-slate-200 bg-white p-3 dark:border-slate-700 dark:bg-slate-900/40">
          <div className="text-[10px] font-bold uppercase tracking-wide text-slate-400 dark:text-slate-500">Core logic</div>
          <div className="mt-1 text-sm leading-6 text-slate-600 dark:text-slate-300">{analysis?.logic || '—'}</div>
        </div>
        {!!analysis?.warning && (
          <div className="mt-3 rounded-xl border border-amber-200 bg-amber-50 p-3 text-xs leading-5 text-amber-900 dark:border-amber-900/50 dark:bg-amber-950/40 dark:text-amber-200">
            {analysis.warning}
          </div>
        )}
      </div>

      <div className={`${shellCard} p-5`}>
        <div className="mb-3 flex items-center gap-2">
          <Sigma className="text-blue-600 dark:text-blue-400" size={19} />
          <h3 className="font-bold">Diagnostics</h3>
        </div>
        <div className="grid grid-cols-2 gap-3">
          {[
            ['N', stats?.n],
            ['Mean', stats?.mean],
            ['Median', stats?.median],
            ['Std dev', stats?.sd],
            ['Skew', stats?.skew],
            ['Excess kurtosis', stats?.excessKurtosis],
            ['Durbin-Watson', stats?.durbinWatson],
            ['Tolerance %', analysis?.tolerancePct],
            ['RMSE', model?.metrics?.rmse],
            ['R²', model?.metrics?.r2],
          ].map(([label, value]) => (
            <div key={String(label)} className="rounded-xl bg-slate-50 p-3 dark:bg-slate-900/60">
              <div className="text-[10px] font-bold uppercase tracking-wide text-slate-400 dark:text-slate-500">{label}</div>
              <div className="mt-1 font-mono text-sm font-semibold">{safeNumber(value, 4)}</div>
            </div>
          ))}
        </div>
        <div className="mt-3 grid gap-2 text-xs text-slate-600 dark:text-slate-300">
          <div>
            <span className="font-semibold text-slate-900 dark:text-slate-100">Real roots:</span>{' '}
            {model?.approxRealRoots?.length ? model.approxRealRoots.map((v: number) => safeNumber(v, 5)).join(', ') : 'None detected'}
          </div>
          <div>
            <span className="font-semibold text-slate-900 dark:text-slate-100">Turning points:</span>{' '}
            {model?.approxDerivativeRoots?.length ? model.approxDerivativeRoots.map((v: number) => safeNumber(v, 5)).join(', ') : 'None detected'}
          </div>
        </div>
      </div>
    </div>
  );
}

function SequencePlot({ result, title = 'Sequence analysis', dark = false }: { result: AnyRecord; title?: string; dark?: boolean }) {
  const analysis = result?.analysis ?? result;
  const x = result?.x ?? analysis?.x ?? [];
  const y = result?.y ?? analysis?.y ?? [];
  const fitted = analysis?.model?.fitted ?? result?.fitted ?? [];
  const gridX = result?.gridX ?? [];
  const gridY = result?.gridY ?? [];
  const predictions = result?.futureY ?? analysis?.predictions ?? [];
  const futureX = result?.futureX ?? predictions.map((_: number, i: number) => {
    const last = x.length ? x[x.length - 1] : 0;
    return last + i + 1;
  });
  const low = analysis?.predictionLow ?? [];
  const high = analysis?.predictionHigh ?? [];

  // The last observed/fitted point is prepended to the forecast trace (and to
  // the uncertainty band) so the forecast line visually continues on from the
  // fitted curve instead of leaving a blank gap between "last real point" and
  // "first predicted point" -- Plotly does not join separate traces on its own.
  const lastX = x.length ? x[x.length - 1] : (gridX.length ? gridX[gridX.length - 1] : 0);
  const lastFittedY = fitted.length ? fitted[fitted.length - 1] : (gridY.length ? gridY[gridY.length - 1] : (y.length ? y[y.length - 1] : undefined));

  const traces: AnyRecord[] = [
    {
      x,
      y,
      mode: 'markers',
      type: 'scatter',
      name: 'Observed',
      marker: { size: 8 },
    },
  ];

  if (gridX.length && gridY.length) {
    traces.push({
      x: gridX,
      y: gridY,
      mode: 'lines',
      type: 'scatter',
      name: 'Fitted model',
      line: { width: 2.5 },
    });
  } else if (fitted.length) {
    traces.push({
      x,
      y: fitted,
      mode: 'lines',
      type: 'scatter',
      name: 'Fitted model',
      line: { width: 2.5 },
    });
  }

  if (predictions.length) {
    const joinedX = lastFittedY !== undefined ? [lastX, ...futureX] : futureX;
    const joinedY = lastFittedY !== undefined ? [lastFittedY, ...predictions] : predictions;
    traces.push({
      x: joinedX,
      y: joinedY,
      mode: 'lines+markers',
      type: 'scatter',
      name: 'Forecast',
      line: { dash: 'dot', width: 2 },
      marker: { size: 8, symbol: 'star' },
    });
  }

  if (low.length && high.length && futureX.length) {
    const bandX = lastFittedY !== undefined ? [lastX, ...futureX] : futureX;
    const bandLow = lastFittedY !== undefined ? [lastFittedY, ...low] : low;
    const bandHigh = lastFittedY !== undefined ? [lastFittedY, ...high] : high;
    traces.push({
      x: [...bandX, ...bandX.slice().reverse()],
      y: [...bandHigh, ...bandLow.slice().reverse()],
      fill: 'toself',
      fillcolor: dark ? 'rgba(96, 165, 250, 0.18)' : 'rgba(37, 99, 235, 0.12)',
      line: { color: 'transparent' },
      name: '90% empirical band',
    });
  }

  const gridColor = dark ? '#334155' : '#e2e8f0';
  const paperColor = dark ? '#1e293b' : 'white';
  const fontColor = dark ? '#e2e8f0' : '#1e293b';

  return (
    <div className={`${shellCard} overflow-hidden`}>
      <div className="border-b border-slate-200 px-5 py-4 dark:border-slate-700">
        <h3 className="font-bold">{title}</h3>
        <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">Plotly view of the native C++ result.</p>
      </div>
      <div className="h-[520px] p-2">
        <Plot
          data={traces}
          layout={{
            autosize: true,
            margin: { l: 60, r: 25, t: 30, b: 55 },
            hovermode: 'closest',
            xaxis: { title: 'x / sequence index', zeroline: false, gridcolor: gridColor },
            yaxis: { title: 'y', zeroline: false, gridcolor: gridColor },
            legend: { orientation: 'h', y: 1.08 },
            paper_bgcolor: paperColor,
            plot_bgcolor: paperColor,
            font: { color: fontColor },
          }}
          style={{ width: '100%', height: '100%' }}
          useResizeHandler
          config={{ responsive: true, displaylogo: false }}
        />
      </div>
    </div>
  );
}

function FamiliesTable({
  families,
  filter,
  onSelect,
}: {
  families: AnyRecord[];
  filter: string;
  onSelect: (candidate: AnyRecord) => void;
}) {
  const filtered = families.filter((item) => {
    const q = filter.trim().toLowerCase();
    if (!q) return true;
    return `${item.family} ${item.model?.name} ${item.model?.category}`.toLowerCase().includes(q);
  });

  return (
    <div className={`${shellCard} overflow-hidden`}>
      <div className="border-b border-slate-200 px-5 py-4 dark:border-slate-700">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="font-bold">Curve-family candidates</h3>
            <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
              Native `makeSpecs()` + `fitByFamily()` sweep. These are in-sample fit diagnostics; the sequence engine separately performs rolling validation.
            </p>
          </div>
          <span className="rounded-full bg-slate-100 px-2.5 py-1 text-xs font-semibold text-slate-600 dark:bg-slate-700 dark:text-slate-200">
            {filtered.length} shown / {families.length} fitted
          </span>
        </div>
      </div>
      <div className="max-h-[520px] overflow-auto">
        <table className="w-full text-left text-xs">
          <thead className="sticky top-0 bg-slate-100 text-[10px] uppercase tracking-wide text-slate-500 dark:bg-slate-900 dark:text-slate-400">
            <tr>
              <th className="px-4 py-3">Family</th>
              <th className="px-4 py-3">Model</th>
              <th className="px-4 py-3">RMSE</th>
              <th className="px-4 py-3">R²</th>
              <th className="px-4 py-3"></th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((item, index) => (
              <tr key={`${item.family}-${item.model?.name}-${index}`} className="border-t border-slate-100 hover:bg-slate-50 dark:border-slate-700 dark:hover:bg-slate-700/40">
                <td className="px-4 py-3 font-semibold text-slate-700 dark:text-slate-200">{familyLabel(item.family)}</td>
                <td className="max-w-[360px] px-4 py-3">
                  <div className="truncate font-medium text-slate-800 dark:text-slate-100">{item.model?.name}</div>
                  <div className="mt-0.5 truncate font-mono text-[11px] text-slate-400 dark:text-slate-500">{item.model?.formula}</div>
                </td>
                <td className="px-4 py-3 font-mono">{safeNumber(item.model?.metrics?.rmse, 5)}</td>
                <td className="px-4 py-3 font-mono">{safeNumber(item.model?.metrics?.r2, 5)}</td>
                <td className="px-4 py-3 text-right">
                  <button type="button" onClick={() => onSelect(item)} className="font-semibold text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300">
                    Plot / fit
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function SequenceScreen({ go, dark }: { go: (screen: string) => void; dark: boolean }) {
  const [data, setData] = useState('10, 15, 23, 35, 53, 80, 121, 183');
  const [horizon, setHorizon] = useState(4);
  const [tolerance, setTolerance] = useState(-1);
  const [result, setResult] = useState<AnyRecord | null>(null);
  const [loading, setLoading] = useState(false);

  const execute = async () => {
    setLoading(true);
    try {
      const { data: payload } = await axios.post(`${API}/analyze`, { data, horizon, tolerance });
      setResult(payload);
    } catch (error: any) {
      alert(error.response?.data?.detail ?? 'Sequence analysis failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <PageShell
      title="Single 1D Sequence"
      subtitle="The browser sends the observations directly to the native sequence engine. Exact rules, finite differences, nonlinear families, symbolic/basis models, rolling-origin validation, mirror retry, roots, change points and forecast bands stay in C++."
      onBack={() => go('home')}
    >
      <div className="grid gap-5 xl:grid-cols-[380px_1fr]">
        <div className={`${shellCard} p-5`}>
          <Field label="Observations" hint="commas, spaces, brackets accepted">
            <textarea value={data} onChange={(e) => setData(e.target.value)} rows={8} className={`${controlClass} resize-y font-mono text-xs`} />
          </Field>
          <div className="mt-4 grid grid-cols-2 gap-3">
            <Field label="Forecast horizon">
              <input type="number" min={1} max={1000} value={horizon} onChange={(e) => setHorizon(Number(e.target.value) || 1)} className={controlClass} />
            </Field>
            <Field label="Tolerance %" hint="-1 = auto">
              <input type="number" step="0.1" min={-1} max={100} value={tolerance} onChange={(e) => setTolerance(Number(e.target.value))} className={controlClass} />
            </Field>
          </div>
          <button type="button" onClick={execute} disabled={loading} className={`${primaryButton} mt-4 w-full`}>
            {loading ? <RefreshCcw size={16} className="animate-spin" /> : <Play size={16} />}
            {loading ? 'Running native engine…' : 'Analyze sequence'}
          </button>
          <button type="button" onClick={() => go('curve')} className={`${secondaryButton} mt-2 w-full`}>
            <LineChart size={16} /> Open Curve / Polynomial Lab
          </button>
        </div>

        <div className="space-y-5">
          {!result ? (
            <div className={`${shellCard} flex min-h-[460px] items-center justify-center p-8 text-center text-slate-400 dark:text-slate-500`}>
              <div>
                <Cpu size={42} className="mx-auto mb-3 opacity-40" />
                <div className="font-semibold">Waiting for C++ analysis</div>
                <p className="mt-1 text-sm">The result appears here without reproducing the model-selection logic in React.</p>
              </div>
            </div>
          ) : (
            <>
              <AnalysisSummary result={result} />
              <SequencePlot result={result} title="Native sequence fit + forecast" dark={dark} />
              {!!result.analysis?.leaderboard?.length && (
                <div className={`${shellCard} overflow-hidden`}>
                  <div className="border-b border-slate-200 px-5 py-4 dark:border-slate-700">
                    <h3 className="font-bold">Validated candidate models</h3>
                    <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">Candidates retained by the native rolling-origin selection path.</p>
                  </div>
                  <div className="max-h-[360px] overflow-auto">
                    <table className="w-full text-left text-xs">
                      <thead className="sticky top-0 bg-slate-100 text-[10px] uppercase tracking-wide text-slate-500 dark:bg-slate-900 dark:text-slate-400"><tr><th className="px-4 py-3">Model</th><th className="px-4 py-3">Score</th><th className="px-4 py-3">R²</th><th className="px-4 py-3">P90</th></tr></thead>
                      <tbody>{result.analysis.leaderboard.slice(0, 25).map((m: AnyRecord, i: number) => <tr key={i} className="border-t border-slate-100 dark:border-slate-700"><td className="px-4 py-3"><div className="font-semibold">{m.name}</div><div className="mt-0.5 truncate font-mono text-[11px] text-slate-400 dark:text-slate-500">{m.formula}</div></td><td className="px-4 py-3 font-mono">{safeNumber(m.metrics?.score, 2)}</td><td className="px-4 py-3 font-mono">{safeNumber(m.metrics?.r2, 4)}</td><td className="px-4 py-3 font-mono">{safeNumber(m.metrics?.p90DeviationPct, 3)}%</td></tr>)}</tbody>
                    </table>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </PageShell>
  );
}

function CurveScreen({ go, dark }: { go: (screen: string) => void; dark: boolean }) {
  const [pairs, setPairs] = useState(`0.0 10.000\n0.5 15.207\n1.0 11.246\n1.5 5.272\n2.0 6.939\n2.5 9.683\n3.0 3.151\n3.5 -5.870\n4.0 -3.141\n4.5 9.762\n5.0 20.343\n5.5 30.125\n6.0 40.811\n6.5 54.288\n7.0 51.124\n7.5 27.146\n8.0 -3.839\n8.5 -31.564\n9.0 -59.011\n9.5 -80.885\n10.0 -78.840`);
  const [layout, setLayout] = useState<Layout>('rows');
  const [horizon, setHorizon] = useState(4);
  const [step, setStep] = useState(0.5);
  const [tolerance, setTolerance] = useState(-1);
  const [result, setResult] = useState<AnyRecord | null>(null);
  const [families, setFamilies] = useState<AnyRecord[] | null>(null);
  const [familyFilter, setFamilyFilter] = useState('');
  const [family, setFamily] = useState('poly');
  const [degree, setDegree] = useState(8);
  const [param, setParam] = useState(0);
  const [param2, setParam2] = useState(0);
  const [familyFit, setFamilyFit] = useState<AnyRecord | null>(null);
  const [loading, setLoading] = useState(false);
  const [familyLoading, setFamilyLoading] = useState(false);

  const normalizedPairs = () => normalizeByLayout(pairs, layout);

  const analyze = async () => {
    setLoading(true);
    try {
      const { data: payload } = await axios.post(`${API}/curve`, { pairs: normalizedPairs(), horizon, tolerance, step });
      setResult(payload);
    } catch (error: any) {
      alert(error.response?.data?.detail ?? 'Curve analysis failed.');
    } finally {
      setLoading(false);
    }
  };

  const compareFamilies = async () => {
    setFamilyLoading(true);
    try {
      const { data: payload } = await axios.post(`${API}/families`, { pairs: normalizedPairs(), horizon, tolerance, step });
      setFamilies(payload.families ?? []);
      setFamilyFilter('');
    } catch (error: any) {
      alert(error.response?.data?.detail ?? 'Family sweep failed.');
    } finally {
      setFamilyLoading(false);
    }
  };

  const fitFamily = async (override?: AnyRecord) => {
    const selectedFamily = override?.family ?? family;
    const selectedDegree = override?.degree ?? degree;
    const selectedParam = override?.param ?? param;
    const selectedParam2 = override?.param2 ?? param2;
    setFamilyLoading(true);
    try {
      const { data: payload } = await axios.post(`${API}/family`, {
        pairs: normalizedPairs(),
        family: selectedFamily,
        degree: selectedDegree,
        param: selectedParam,
        param2: selectedParam2,
        horizon,
        step,
      });
      setFamilyFit(payload);
    } catch (error: any) {
      alert(error.response?.data?.detail ?? 'Family fit failed.');
    } finally {
      setFamilyLoading(false);
    }
  };

  const familyPlot = familyFit ? (
    <div className={`${shellCard} overflow-hidden`}>
      <div className="border-b border-slate-200 px-5 py-4 dark:border-slate-700">
        <h3 className="font-bold">Manual family fit</h3>
        <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">{familyFit.model?.name} · {familyFit.model?.category}</p>
      </div>
      <div className="h-[500px] p-2">
        <Plot
          data={[
            { x: familyFit?.x ?? [], y: familyFit?.y ?? [], mode: 'markers', type: 'scatter', name: 'Observed', marker: { size: 8 } },
            { x: familyFit.gridX ?? [], y: familyFit.gridY ?? [], mode: 'lines', type: 'scatter', name: 'Selected family', line: { width: 3 } },
            ...(familyFit.futureX?.length ? [{ x: familyFit.futureX, y: familyFit.futureY, mode: 'lines+markers', type: 'scatter', name: 'Forecast', line: { dash: 'dot', width: 2 } }] : []),
          ]}
          layout={{
            autosize: true,
            margin: { l: 60, r: 25, t: 30, b: 55 },
            xaxis: { title: 'x', gridcolor: dark ? '#334155' : '#e2e8f0' },
            yaxis: { title: 'y', gridcolor: dark ? '#334155' : '#e2e8f0' },
            legend: { orientation: 'h', y: 1.08 },
            paper_bgcolor: dark ? '#1e293b' : 'white',
            plot_bgcolor: dark ? '#1e293b' : 'white',
            font: { color: dark ? '#e2e8f0' : '#1e293b' },
          }}
          style={{ width: '100%', height: '100%' }}
          useResizeHandler
          config={{ responsive: true, displaylogo: false }}
        />
      </div>
      <div className="grid gap-3 border-t border-slate-200 p-5 sm:grid-cols-4 dark:border-slate-700">
        <div className="rounded-xl bg-slate-50 p-3 dark:bg-slate-900/60"><div className="text-[10px] font-bold uppercase text-slate-400 dark:text-slate-500">RMSE</div><div className="mt-1 font-mono text-sm font-semibold">{safeNumber(familyFit.model?.metrics?.rmse, 5)}</div></div>
        <div className="rounded-xl bg-slate-50 p-3 dark:bg-slate-900/60"><div className="text-[10px] font-bold uppercase text-slate-400 dark:text-slate-500">MAE</div><div className="mt-1 font-mono text-sm font-semibold">{safeNumber(familyFit.model?.metrics?.mae, 5)}</div></div>
        <div className="rounded-xl bg-slate-50 p-3 dark:bg-slate-900/60"><div className="text-[10px] font-bold uppercase text-slate-400 dark:text-slate-500">R²</div><div className="mt-1 font-mono text-sm font-semibold">{safeNumber(familyFit.model?.metrics?.r2, 5)}</div></div>
        <div className="rounded-xl bg-slate-50 p-3 dark:bg-slate-900/60"><div className="text-[10px] font-bold uppercase text-slate-400 dark:text-slate-500">Complexity</div><div className="mt-1 font-mono text-sm font-semibold">{familyFit.model?.complexity ?? '—'}</div></div>
      </div>
    </div>
  ) : null;

  return (
    <PageShell
      title="Graph / Curve & Polynomial Lab"
      subtitle="This is the dedicated physical x/y workspace. The C++ engine's complete model-family generator is exposed here instead of being hidden behind a generic sequence form: polynomial, Chebyshev, Taylor, rational, Fourier, damped-sinusoid/RLC transient, chirp, AR/Prony, segmented, root-guided, symbolic/basis, RBF and more."
      onBack={() => go('home')}
    >
      <div className="grid gap-5 xl:grid-cols-[390px_1fr]">
        <div className={`${shellCard} p-5`}>
          <LayoutToggle
            value={layout}
            onChange={setLayout}
            rowsHint="Each line is one (x, y) pair."
            columnsHint="Line 1 = every x value, line 2 = every y value."
          />
          <Field label="x y pairs" hint={layout === 'rows' ? 'one pair per line' : 'one variable per line'}>
            <textarea value={pairs} onChange={(e) => setPairs(e.target.value)} rows={16} className={`${controlClass} mt-3 resize-y font-mono text-xs`} />
          </Field>
          <div className="mt-4 grid grid-cols-3 gap-3">
            <Field label="Horizon"><input type="number" min={1} max={1000} value={horizon} onChange={(e) => setHorizon(Number(e.target.value) || 1)} className={controlClass} /></Field>
            <Field label="x step"><input type="number" step="0.01" min="0.000001" value={step} onChange={(e) => setStep(Number(e.target.value) || 0.5)} className={controlClass} /></Field>
            <Field label="Tolerance"><input type="number" step="0.1" min={-1} max={100} value={tolerance} onChange={(e) => setTolerance(Number(e.target.value))} className={controlClass} /></Field>
          </div>
          <div className="mt-4 grid grid-cols-2 gap-2">
            <button type="button" onClick={analyze} disabled={loading} className={primaryButton}>
              {loading ? <RefreshCcw size={16} className="animate-spin" /> : <Play size={16} />} Analyze curve
            </button>
            <button type="button" onClick={compareFamilies} disabled={familyLoading} className={secondaryButton}>
              {familyLoading ? <RefreshCcw size={16} className="animate-spin" /> : <GitBranch size={16} />} Compare families
            </button>
          </div>
          <InfoStrip>
            The normal Analyze action keeps the native rolling-validation selector. Compare families is a separate, faster exploratory sweep so you can inspect what the polynomial / combination model space contains without paying that cost on every click.
          </InfoStrip>

          <div className="mt-5 border-t border-slate-200 pt-5 dark:border-slate-700">
            <div className="mb-3 flex items-center gap-2">
              <Sparkles size={18} className="text-blue-600 dark:text-blue-400" />
              <div className="font-bold">Manual family fit</div>
            </div>
            <div className="grid gap-3">
              <Field label="Family">
                <select value={family} onChange={(e) => setFamily(e.target.value)} className={controlClass}>
                  {MODEL_FAMILIES.map(([id, label]) => <option key={id} value={id}>{label}</option>)}
                </select>
              </Field>
              <div className="grid grid-cols-3 gap-2">
                <Field label="Degree / order"><input type="number" min={1} max={18} value={degree} onChange={(e) => setDegree(Number(e.target.value) || 1)} className={controlClass} /></Field>
                <Field label="Param"><input type="number" step="0.001" value={param} onChange={(e) => setParam(Number(e.target.value))} className={controlClass} /></Field>
                <Field label="Param 2"><input type="number" step="0.001" value={param2} onChange={(e) => setParam2(Number(e.target.value))} className={controlClass} /></Field>
              </div>
              <button type="button" onClick={() => fitFamily()} disabled={familyLoading} className={`${secondaryButton} w-full`}>
                <FlaskConical size={16} /> Fit selected family
              </button>
            </div>
          </div>
        </div>

        <div className="space-y-5">
          {result ? <><AnalysisSummary result={result} /><SequencePlot result={result} title="Physical x/y curve analysis" dark={dark} /></> : (
            <div className={`${shellCard} flex min-h-[310px] items-center justify-center p-8 text-center text-slate-400 dark:text-slate-500`}>
              <div><LineChart size={40} className="mx-auto mb-3 opacity-40" /><div className="font-semibold">Curve plot appears here</div><p className="mt-1 text-sm">Use physical x values; the engine will not silently replace them with sequence indices.</p></div>
            </div>
          )}
          {familyFit && familyPlot}
          {families && (
            <>
              <div className="flex items-center justify-between gap-3">
                <div className="text-sm font-bold text-slate-700 dark:text-slate-200">Family explorer</div>
                <div className="relative w-full max-w-sm">
                  <Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 dark:text-slate-500" />
                  <input value={familyFilter} onChange={(e) => setFamilyFilter(e.target.value)} placeholder="Filter family / model" className={`${controlClass} pl-9`} />
                </div>
              </div>
              <FamiliesTable families={families} filter={familyFilter} onSelect={(candidate) => {
                const selected = {
                  family: candidate.family,
                  degree: Number(candidate.degree) || 1,
                  param: Number(candidate.param) || 0,
                  param2: Number(candidate.param2) || 0,
                };
                setFamily(selected.family);
                setDegree(selected.degree);
                setParam(selected.param);
                setParam2(selected.param2);
                void fitFamily(selected);
              }} />
            </>
          )}
        </div>
      </div>
    </PageShell>
  );
}

function MultivariateScreen({ go, dark }: { go: (screen: string) => void; dark: boolean }) {
  const [variables, setVariables] = useState(3);
  const [rows, setRows] = useState('4 8 12\n5 10 15\n6 12 18\n7 14 21\n8 16 24\n9 18 27\n10 20 30\n11 22 33\n12 24 36\n13 26 39\n14 28 42\n15 30 45');
  const [layout, setLayout] = useState<Layout>('rows');
  const [horizon, setHorizon] = useState(4);
  const [tolerance, setTolerance] = useState(-1);
  const [result, setResult] = useState<AnyRecord | null>(null);
  const [loading, setLoading] = useState(false);

  const execute = async () => {
    setLoading(true);
    try {
      const { data } = await axios.post(`${API}/multivariate`, { rows: normalizeByLayout(rows, layout), variables, horizon, tolerance });
      setResult(data);
    } catch (error: any) {
      alert(error.response?.data?.detail ?? 'Multivariate analysis failed.');
    } finally {
      setLoading(false);
    }
  };

  const plot = result?.rows?.length ? (
    <div className={`${shellCard} overflow-hidden`}>
      <div className="border-b border-slate-200 px-5 py-4 dark:border-slate-700"><h3 className="font-bold">Multivariate trajectories</h3><p className="mt-1 text-xs text-slate-500 dark:text-slate-400">Observed series plus chained future rows from the native joint lag model.</p></div>
      <div className="h-[500px] p-2">
        <Plot
          data={Array.from({ length: variables }).map((_, j) => ({
            x: result.rows.map((_: number[], i: number) => i + 1),
            y: result.rows.map((row: number[]) => row[j]),
            mode: 'lines+markers',
            type: 'scatter',
            name: `Variable ${j + 1}`,
          })).concat((result.analysis.predictions ?? []).length ? Array.from({ length: variables }).map((_, j) => ({
            x: (result.analysis.predictions ?? []).map((_: number[], i: number) => result.rows.length + i + 1),
            y: (result.analysis.predictions ?? []).map((row: number[]) => row[j]),
            mode: 'lines+markers',
            type: 'scatter',
            name: `Forecast ${j + 1}`,
            line: { dash: 'dot' },
          })) : [])}
          layout={{
            autosize: true,
            margin: { l: 60, r: 25, t: 30, b: 55 },
            xaxis: { title: 'row', gridcolor: dark ? '#334155' : '#e2e8f0' },
            yaxis: { title: 'value', gridcolor: dark ? '#334155' : '#e2e8f0' },
            legend: { orientation: 'h', y: 1.08 },
            paper_bgcolor: dark ? '#1e293b' : 'white',
            plot_bgcolor: dark ? '#1e293b' : 'white',
            font: { color: dark ? '#e2e8f0' : '#1e293b' },
          }}
          style={{ width: '100%', height: '100%' }} useResizeHandler config={{ responsive: true, displaylogo: false }}
        />
      </div>
    </div>
  ) : null;

  return (
    <PageShell title="N-Dimensional / Multivariate Dataset" subtitle="Joint regularized lag modeling, rolling validation, chained fallback and per-variable dependency analysis stay in the C++ engine." onBack={() => go('home')}>
      <div className="grid gap-5 xl:grid-cols-[390px_1fr]">
        <div className={`${shellCard} p-5`}>
          <div className="grid grid-cols-2 gap-3">
            <Field label="Variables"><input type="number" min={2} max={64} value={variables} onChange={(e) => setVariables(Number(e.target.value) || 2)} className={controlClass} /></Field>
            <Field label="Forecast horizon"><input type="number" min={1} max={1000} value={horizon} onChange={(e) => setHorizon(Number(e.target.value) || 1)} className={controlClass} /></Field>
          </div>
          <Field label="Tolerance %" hint="-1 = auto"><input type="number" step="0.1" min={-1} max={100} value={tolerance} onChange={(e) => setTolerance(Number(e.target.value))} className={`${controlClass} mt-3`} /></Field>
          <div className="mt-4">
            <LayoutToggle
              value={layout}
              onChange={setLayout}
              rowsHint="Each line is one full row, e.g. one timestamp's values for every variable."
              columnsHint="Each line is one variable's entire history across all rows."
            />
          </div>
          <div className="mt-4"><Field label="Rows" hint={layout === 'rows' ? 'one row per line' : 'one variable per line'}><textarea value={rows} onChange={(e) => setRows(e.target.value)} rows={18} className={`${controlClass} resize-y font-mono text-xs`} /></Field></div>
          <button type="button" onClick={execute} disabled={loading} className={`${primaryButton} mt-4 w-full`}>{loading ? <RefreshCcw size={16} className="animate-spin" /> : <Play size={16} />} Run multivariate engine</button>
        </div>
        <div className="space-y-5">
          {result ? <>
            <AnalysisSummary result={{ analysis: result.analysis?.driver, stats: result.analysis?.driver?.stats }} />
            {plot}
            {!!result.analysis?.links?.length && <div className={`${shellCard} overflow-hidden`}><div className="border-b border-slate-200 px-5 py-4 font-bold dark:border-slate-700">Variable links</div><div className="divide-y divide-slate-100 dark:divide-slate-700">{result.analysis.links.map((link: AnyRecord, i: number) => <div key={i} className="p-4"><div className="text-sm font-semibold">Variable {i + 2}: {link.category}</div><div className="mt-1 font-mono text-xs text-slate-500 dark:text-slate-400">{link.formula}</div><div className="mt-2 text-xs text-slate-500 dark:text-slate-400">Evidence {safeNumber(link.evidenceScore, 2)}%</div></div>)}</div></div>}
          </> : <div className={`${shellCard} flex min-h-[460px] items-center justify-center p-8 text-center text-slate-400 dark:text-slate-500`}><div><Table2 size={40} className="mx-auto mb-3 opacity-40" /><div className="font-semibold">Multivariate result will appear here</div></div></div>}
        </div>
      </div>
    </PageShell>
  );
}

function CoordinatesScreen({ go, dark }: { go: (screen: string) => void; dark: boolean }) {
  const [mode, setMode] = useState(1);
  const [response, setResponse] = useState(1);
  const [angleUnit, setAngleUnit] = useState(0);
  const [points, setPoints] = useState('0 10.000\n0.5 15.207\n1.0 11.246\n1.5 5.272\n2.0 6.939\n2.5 9.683\n3.0 3.151\n3.5 -5.870\n4.0 -3.141\n4.5 9.762\n5.0 20.343\n5.5 30.125\n6.0 40.811\n6.5 54.288\n7.0 51.124\n7.5 27.146\n8.0 -3.839\n8.5 -31.564\n9.0 -59.011\n9.5 -80.885\n10.0 -78.840');
  const [layout, setLayout] = useState<Layout>('rows');
  const [horizon, setHorizon] = useState(4);
  const [tolerance, setTolerance] = useState(-1);
  const [result, setResult] = useState<AnyRecord | null>(null);
  const [loading, setLoading] = useState(false);

  const execute = async () => {
    setLoading(true);
    try {
      const { data } = await axios.post(`${API}/coordinates`, { mode, response, angle_unit: angleUnit, points: normalizeByLayout(points, layout), horizon, tolerance });
      setResult(data);
    } catch (error: any) {
      alert(error.response?.data?.detail ?? 'Coordinate analysis failed.');
    } finally { setLoading(false); }
  };

  const is3D = mode >= 3;

  return (
    <PageShell title="Manual Coordinates" subtitle="2D Cartesian, polar, 3D Cartesian, cylindrical and spherical coordinates. The web screen makes the independent/response coordinate explicit so a 2D data set is not accidentally analyzed as X versus itself." onBack={() => go('home')}>
      <div className="grid gap-5 xl:grid-cols-[390px_1fr]">
        <div className={`${shellCard} p-5`}>
          <Field label="Coordinate representation">
            <select value={mode} onChange={(e) => setMode(Number(e.target.value))} className={controlClass}>
              <option value={1}>2D Cartesian (x y)</option>
              <option value={2}>2D Polar (r theta)</option>
              <option value={3}>3D Cartesian (x y z)</option>
              <option value={4}>3D Cylindrical (r theta z)</option>
              <option value={5}>3D Spherical (r theta phi)</option>
            </select>
          </Field>
          {(mode === 2 || mode >= 4) && <Field label="Angle units"><select value={angleUnit} onChange={(e) => setAngleUnit(Number(e.target.value))} className={`${controlClass} mt-3`}><option value={0}>Radians</option><option value={1}>Degrees</option></select></Field>}
          <Field label="Analyzed response" hint={mode < 3 ? 'default Y' : 'response coordinate'}>
            <select value={response} onChange={(e) => setResponse(Number(e.target.value))} className={`${controlClass} mt-3`}>
              <option value={0}>X</option>
              <option value={1}>Y</option>
              {is3D && <option value={2}>Z</option>}
            </select>
          </Field>
          <div className="mt-4 grid grid-cols-2 gap-3">
            <Field label="Horizon"><input type="number" min={1} value={horizon} onChange={(e) => setHorizon(Number(e.target.value) || 1)} className={controlClass} /></Field>
            <Field label="Tolerance"><input type="number" min={-1} step="0.1" value={tolerance} onChange={(e) => setTolerance(Number(e.target.value))} className={controlClass} /></Field>
          </div>
          <div className="mt-4">
            <LayoutToggle
              value={layout}
              onChange={setLayout}
              rowsHint="Each line is one point's coordinates."
              columnsHint="Each line is one coordinate's values across every point."
            />
          </div>
          <Field label={mode === 1 ? 'Points: x y' : mode === 2 ? 'Points: r theta' : mode === 3 ? 'Points: x y z' : mode === 4 ? 'Points: r theta z' : 'Points: r theta phi'}>
            <textarea value={points} onChange={(e) => setPoints(e.target.value)} rows={15} className={`${controlClass} mt-3 resize-y font-mono text-xs`} />
          </Field>
          <button type="button" onClick={execute} disabled={loading} className={`${primaryButton} mt-4 w-full`}>{loading ? <RefreshCcw size={16} className="animate-spin" /> : <Play size={16} />} Analyze coordinates</button>
        </div>
        <div className="space-y-5">
          {result ? <>
            <AnalysisSummary result={result} />
            <div className={`${shellCard} overflow-hidden`}>
              <div className="border-b border-slate-200 px-5 py-4 dark:border-slate-700"><h3 className="font-bold">Geometry</h3></div>
              <div className="h-[520px] p-2">
                {result.is3D ? <Plot data={[{ x: result.points.map((p: AnyRecord) => p.x), y: result.points.map((p: AnyRecord) => p.y), z: result.points.map((p: AnyRecord) => p.z), mode: 'markers+lines', type: 'scatter3d', name: 'Observed points', marker: { size: 4 } }]} layout={{ autosize: true, margin: { l: 0, r: 0, t: 30, b: 0 }, scene: { xaxis: { title: 'X' }, yaxis: { title: 'Y' }, zaxis: { title: 'Z' } }, paper_bgcolor: dark ? '#1e293b' : 'white', font: { color: dark ? '#e2e8f0' : '#1e293b' } }} style={{ width: '100%', height: '100%' }} useResizeHandler config={{ responsive: true, displaylogo: false }} /> : <Plot data={[{ x: result.points.map((p: AnyRecord) => p.x), y: result.points.map((p: AnyRecord) => p.y), mode: 'markers+lines', type: 'scatter', name: 'Observed points', marker: { size: 8 } }, ...(result.gridX?.length ? [{ x: result.gridX, y: result.gridY, mode: 'lines', type: 'scatter', name: 'Fitted response', line: { width: 2.5 } }] : [])]} layout={{ autosize: true, margin: { l: 60, r: 25, t: 30, b: 55 }, xaxis: { title: 'X / transformed coordinate', gridcolor: dark ? '#334155' : '#e2e8f0' }, yaxis: { title: 'Y', gridcolor: dark ? '#334155' : '#e2e8f0' }, legend: { orientation: 'h', y: 1.08 }, paper_bgcolor: dark ? '#1e293b' : 'white', plot_bgcolor: dark ? '#1e293b' : 'white', font: { color: dark ? '#e2e8f0' : '#1e293b' } }} style={{ width: '100%', height: '100%' }} useResizeHandler config={{ responsive: true, displaylogo: false }} />}
              </div>
            </div>
          </> : <div className={`${shellCard} flex min-h-[460px] items-center justify-center p-8 text-center text-slate-400 dark:text-slate-500`}><div><Grid3X3 size={40} className="mx-auto mb-3 opacity-40" /><div className="font-semibold">Coordinate plot will appear here</div></div></div>}
        </div>
      </div>
    </PageShell>
  );
}

function FileScreen({ go, dark }: { go: (screen: string) => void; dark: boolean }) {
  const [fileName, setFileName] = useState('');
  const [text, setText] = useState('');
  const [groupIndex, setGroupIndex] = useState(0);
  const [selectedColumns, setSelectedColumns] = useState<number[]>([]);
  const [xColumn, setXColumn] = useState(0);
  const [yColumn, setYColumn] = useState(1);
  const [mode, setMode] = useState(0);
  const [angleUnit, setAngleUnit] = useState(0);
  const [horizon, setHorizon] = useState(4);
  const [tolerance, setTolerance] = useState(-1);
  const [result, setResult] = useState<AnyRecord | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const groups = useMemo(() => parseImportedDataset(text), [text]);
  const activeGroup = groups[groupIndex] ?? groups[0];
  const columns = numericColumns(activeGroup);

  const handleFile = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const contents = await file.text();
    const parsed = parseImportedDataset(contents);
    setFileName(file.name);
    setText(contents);
    setGroupIndex(0);
    setSelectedColumns(parsed[0] ? numericColumns(parsed[0]) : []);
    setXColumn(0);
    setYColumn(1);
    setResult(null);
    setError(parsed.length ? "" : "No numeric table was detected. Check the delimiter, headers and data rows.");
    event.target.value = "";
  };

  const handleTextChange = (value: string) => {
    setText(value);
    setGroupIndex(0);
    const parsed = parseImportedDataset(value);
    setSelectedColumns(parsed[0] ? numericColumns(parsed[0]) : []);
    setXColumn(0);
    setYColumn(1);
    setResult(null);
    setError("");
  };

  const selectGroup = (index: number) => {
    setGroupIndex(index);
    setSelectedColumns(numericColumns(groups[index]));
    setXColumn(columns[0] ?? 0);
    setYColumn(columns[1] ?? columns[0] ?? 0);
    setResult(null);
    setError("");
  };

  const toggleColumn = (index: number) => {
    setSelectedColumns((current) => current.includes(index)
      ? current.filter((item) => item !== index)
      : [...current, index].sort((a, b) => a - b));
    setResult(null);
  };

  const execute = async () => {
    if (!activeGroup || !activeGroup.rows.length) { setError("Choose a file or paste a dataset first."); return; }
    let chosen: number[];
    if (mode === 0) chosen = selectedColumns.slice(0, 1);
    else if (mode === 2) chosen = [xColumn, yColumn];
    else chosen = selectedColumns;
    const required = mode === 0 ? 1 : mode === 3 ? 2 : 2;
    if (chosen.length < required || new Set(chosen).size !== chosen.length) {
      setError(mode === 0 ? "Select one numeric column for a sequence." : mode === 2 ? "Choose different X and Y columns." : "Select the required numeric columns.");
      return;
    }
    const cleanRows = activeGroup.rows
      .filter((row) => chosen.every((column) => Number.isFinite(row[column])))
      .map((row) => chosen.map((column) => row[column]));
    if (mode === 1 && chosen.length < 2) { setError("Multivariate analysis needs at least two selected variables."); return; }
    if ((mode === 4 || mode === 5 || mode === 6) && chosen.length < 3) { setError("This 3D coordinate mode needs three numeric columns."); return; }
    if (cleanRows.length < (mode === 1 ? 8 : 3)) {
      setError(mode === 1 ? "Multivariate analysis needs at least 8 complete rows. Choose columns with fewer missing values." : "At least three complete data rows are required.");
      return;
    }
    const payloadText = cleanRows.map((row) => row.join(" ")).join("\\n");
    setLoading(true);
    setError("");
    try {
      const { data } = await axios.post(`${API}/file`, {
        text: payloadText, mode, variables: chosen.length, angle_unit: angleUnit, horizon, tolerance,
      }, { timeout: 180000 });
      if (data?.ok === false || data?.error) throw new Error(data?.error || "Native analyser rejected the dataset.");
      setResult(data);
    } catch (requestError: any) {
      const detail = requestError.response?.data?.detail;
      setError(Array.isArray(detail) ? detail.map((item: any) => item.msg || "Invalid input").join("; ") : detail || requestError.message || "File analysis failed.");
    } finally { setLoading(false); }
  };

  return (
    <PageShell title="Load & Group Dataset" subtitle="Import CSV or text files, detect table headings and sections, choose numeric columns, and send only the selected clean series to the existing native analyser." onBack={() => go("home")}>
      <div className="grid gap-5 xl:grid-cols-[410px_minmax(0,1fr)]">
        <div className={`${shellCard} space-y-4 p-5`}>
          <label className={`${secondaryButton} w-full cursor-pointer`}>
            <Upload size={16} /> {fileName || "Choose CSV / text file"}
            <input type="file" className="hidden" onChange={handleFile} accept=".txt,.csv,.tsv,.dat,.log" />
          </label>
          <Field label="File contents / paste data">
            <textarea value={text} onChange={(event) => handleTextChange(event.target.value)} rows={12} className={`${controlClass} mt-2 resize-y font-mono text-xs`} placeholder={"Time,Temperature,Pressure\\n0,21.5,101.2\\n1,22.1,101.4"} />
          </Field>
          <div className="rounded-xl border border-slate-200 p-3 dark:border-slate-700">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="text-sm font-semibold">Detected sections</div>
              <span className="text-xs text-slate-500">{groups.length} group(s)</span>
            </div>
            {groups.length ? (
              <div className="mt-2 space-y-2">
                <Field label="Active section">
                  <select value={groupIndex} onChange={(event) => selectGroup(Number(event.target.value))} className={`${controlClass} mt-1`}>
                    {groups.map((group, index) => <option key={group.id} value={index}>{group.title} · {group.rows.length} rows</option>)}
                  </select>
                </Field>
                {activeGroup && <div className="text-[11px] text-slate-500">Detected {activeGroup.headers.length} column(s). Rows with missing values are excluded only from the selected series.</div>}
              </div>
            ) : <p className="mt-2 text-xs text-slate-500">Choose a CSV/text file or paste content to detect numeric tables.</p>}
          </div>
          {activeGroup && (
            <div className="rounded-xl border border-slate-200 p-3 dark:border-slate-700">
              <div className="mb-2 text-sm font-semibold">Numeric columns</div>
              <div className="grid gap-2 sm:grid-cols-2">
                {columns.map((column) => (
                  <label key={column} className="flex min-w-0 items-center gap-2 rounded-lg bg-slate-50 px-3 py-2 text-xs dark:bg-slate-900">
                    <input type="checkbox" checked={selectedColumns.includes(column)} onChange={() => toggleColumn(column)} />
                    <span className="truncate">{activeGroup.headers[column]}</span>
                  </label>
                ))}
              </div>
              <p className="mt-2 text-[11px] text-slate-500">Sequence mode uses the first checked column. Multivariate mode uses all checked columns.</p>
              {mode === 2 && columns.length >= 2 && (
                <div className="mt-3 grid grid-cols-2 gap-2">
                  <Field label="X axis"><select value={xColumn} onChange={(event) => setXColumn(Number(event.target.value))} className={`${controlClass} mt-1`}>{columns.map((column) => <option key={column} value={column}>{activeGroup.headers[column]}</option>)}</select></Field>
                  <Field label="Y axis"><select value={yColumn} onChange={(event) => setYColumn(Number(event.target.value))} className={`${controlClass} mt-1`}>{columns.map((column) => <option key={column} value={column}>{activeGroup.headers[column]}</option>)}</select></Field>
                </div>
              )}
            </div>
          )}
          <Field label="Interpretation">
            <select value={mode} onChange={(event) => setMode(Number(event.target.value))} className={controlClass}>
              <option value={0}>1D sequence</option>
              <option value={1}>N-dimensional / multivariate</option>
              <option value={2}>2D Cartesian (X / Y)</option>
              <option value={3}>2D Polar (radius / angle)</option>
              <option value={4}>3D Cartesian (X / Y / Z)</option>
              <option value={5}>3D Cylindrical (radius / angle / height)</option>
              <option value={6}>3D Spherical (radius / azimuth / polar angle)</option>
            </select>
          </Field>
          {(mode === 3 || mode >= 5) && <Field label="Angle units"><select value={angleUnit} onChange={(event) => setAngleUnit(Number(event.target.value))} className={controlClass}><option value={0}>Radians</option><option value={1}>Degrees</option></select></Field>}
          <div className="grid grid-cols-2 gap-3">
            <Field label="Forecast horizon"><input type="number" min={1} max={1000} value={horizon} onChange={(event) => setHorizon(Math.max(1, Math.min(1000, Number(event.target.value) || 1)))} className={controlClass} /></Field>
            <Field label="Tolerance (%)"><input type="number" min={-1} max={100} step="0.1" value={tolerance} onChange={(event) => setTolerance(Number(event.target.value))} className={controlClass} /></Field>
          </div>
          {error && <div role="alert" className="rounded-xl border border-red-200 bg-red-50 p-3 text-sm text-red-700 dark:border-red-900 dark:bg-red-950/30 dark:text-red-300">{error}</div>}
          <button type="button" onClick={execute} disabled={loading || !activeGroup} className={`${primaryButton} w-full`}>{loading ? <RefreshCcw size={16} className="animate-spin" /> : <FileText size={16} />} Analyse selected data</button>
        </div>
        <div className="space-y-5">
          {activeGroup && (
            <div className={`${shellCard} overflow-hidden`}>
              <div className="border-b border-slate-200 px-5 py-4 dark:border-slate-700">
                <h3 className="font-bold">Preview · {activeGroup.title}</h3>
                <p className="mt-1 text-xs text-slate-500">Showing up to five parsed rows. Column labels are retained for selection.</p>
              </div>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-slate-200 text-xs dark:divide-slate-700">
                  <thead className="bg-slate-50 dark:bg-slate-900"><tr>{activeGroup.headers.map((header, index) => <th key={index} className="whitespace-nowrap px-3 py-2 text-left font-semibold text-slate-600 dark:text-slate-300">{header}</th>)}</tr></thead>
                  <tbody>{activeGroup.rows.slice(0, 5).map((row, rowIndex) => <tr key={rowIndex} className="odd:bg-white even:bg-slate-50 dark:odd:bg-slate-800 dark:even:bg-slate-900">{activeGroup.headers.map((_, column) => <td key={column} className="whitespace-nowrap px-3 py-2 font-mono">{Number.isFinite(row[column]) ? row[column] : "—"}</td>)}</tr>)}</tbody>
                </table>
              </div>
            </div>
          )}
          {result ? (
            mode === 1 ? (
              <div className="space-y-5">
                <div className={`${shellCard} p-5`}>
                  <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                    <div className="rounded-xl bg-slate-50 p-3 dark:bg-slate-900/60"><div className="text-[10px] font-bold uppercase text-slate-400">Rows</div><div className="mt-1 font-mono text-sm font-semibold">{result.analysis?.rows ?? result.rows?.length ?? "—"}</div></div>
                    <div className="rounded-xl bg-slate-50 p-3 dark:bg-slate-900/60"><div className="text-[10px] font-bold uppercase text-slate-400">Variables</div><div className="mt-1 font-mono text-sm font-semibold">{result.analysis?.variables ?? selectedColumns.length}</div></div>
                    <div className="rounded-xl bg-slate-50 p-3 dark:bg-slate-900/60"><div className="text-[10px] font-bold uppercase text-slate-400">Forecast</div><div className="mt-1 text-sm font-semibold">{result.analysis?.fullForecastAvailable ? "Available" : "Not available"}</div></div>
                    <div className="rounded-xl bg-slate-50 p-3 dark:bg-slate-900/60"><div className="text-[10px] font-bold uppercase text-slate-400">Tolerance</div><div className="mt-1 font-mono text-sm font-semibold">{safeNumber(result.analysis?.tolerancePct, 3)}%</div></div>
                  </div>
                </div>
                <div className={`${shellCard} overflow-hidden`}>
                  <div className="border-b border-slate-200 px-5 py-4 dark:border-slate-700"><h3 className="font-bold">Loaded multivariate trajectories</h3></div>
                  <div className="h-[520px] p-2">
                    <Plot data={Array.from({ length: selectedColumns.length }).map((_, j) => ({ x: (result.rows ?? []).map((_: number[], i: number) => i + 1), y: (result.rows ?? []).map((row: number[]) => row[j]), mode: "lines+markers", type: "scatter", name: activeGroup?.headers[selectedColumns[j]] ?? `Variable ${j + 1}` })).concat((result.analysis?.predictions ?? []).length ? Array.from({ length: selectedColumns.length }).map((_, j) => ({ x: (result.analysis.predictions ?? []).map((_: number[], i: number) => (result.rows ?? []).length + i + 1), y: (result.analysis.predictions ?? []).map((row: number[]) => row[j]), mode: "lines+markers", type: "scatter", name: `Forecast ${activeGroup?.headers[selectedColumns[j]] ?? j + 1}`, line: { dash: "dot" } })) : [])} layout={{ autosize: true, margin: { l: 60, r: 25, t: 30, b: 55 }, xaxis: { title: "row", gridcolor: dark ? "#334155" : "#e2e8f0" }, yaxis: { title: "value", gridcolor: dark ? "#334155" : "#e2e8f0" }, legend: { orientation: "h", y: 1.08 }, paper_bgcolor: dark ? "#1e293b" : "white", plot_bgcolor: dark ? "#1e293b" : "white", font: { color: dark ? "#e2e8f0" : "#1e293b" } }} style={{ width: "100%", height: "100%" }} useResizeHandler config={{ responsive: true, displaylogo: false }} />
                  </div>
                </div>
              </div>
            ) : (
              <div className="space-y-5"><AnalysisSummary result={result} /><SequencePlot result={result} title="Loaded-data analysis" dark={dark} /></div>
            )
          ) : <div className={`${shellCard} flex min-h-[300px] items-center justify-center p-8 text-center text-slate-400 dark:text-slate-500`}><div><Database size={40} className="mx-auto mb-3 opacity-40" /><div className="font-semibold">Import and select a dataset to begin</div><p className="mt-1 text-sm">Choose a detected section and numeric columns, then analyse them.</p></div></div>}
        </div>
      </div>
    </PageShell>
  );
}

function CryptoScreen({ go }: { go: (screen: string) => void }) {
  const [operation, setOperation] = useState('analysis');
  const [lines, setLines] = useState<string[]>(['']);
  const [result, setResult] = useState<AnyRecord | null>(null);
  const [loading, setLoading] = useState(false);
  const [aes, setAes] = useState({ key: '', iv: '', message: '', tag: '' });
  const opMeta: AnyRecord = {
    analysis: ['Auto analysis', ['Input']],
    caesar: ['Caesar / ROT + frequency', ['Text', 'Shift']],
    vigenere: ['Vigenere decrypt + IC', ['Ciphertext', 'Key']],
    xor: ['XOR encode / decode', ['Text', 'Key']],
    base64_encode: ['Base64 encode', ['Text']],
    base64_decode: ['Base64 decode', ['Base64']],
    hex_encode: ['Hex encode', ['Text']],
    hex_decode: ['Hex decode', ['Hex']],
    morse_encode: ['Morse encode', ['Text']],
    morse_decode: ['Morse decode', ['Morse']],
    atbash: ['Atbash', ['Text']],
    affine: ['Affine cipher', ['Text', 'a', 'b', 'Decrypt 0/1']],
    rail_fence: ['Rail Fence', ['Text', 'Rails', 'Decrypt 0/1']],
    sha256: ['SHA-256', ['Message']],
    hmac: ['HMAC-SHA256', ['Key', 'Message']],
    pbkdf2: ['PBKDF2-HMAC-SHA256', ['Password', 'Salt text (blank = generated)', 'Iterations', 'Bytes']],
    random_key: ['Random key bytes', ['Bytes 8-64']],
    theory: ['Cryptography theory', []],
    aes_encrypt: ['AES-256-GCM encrypt', ['Key hex', '12-byte IV hex', 'Plaintext']],
    aes_decrypt: ['AES-256-GCM decrypt', ['Key hex', '12-byte IV hex', 'Ciphertext hex', 'Tag hex']],
  };

  const setOperationState = (next: string) => {
    setOperation(next);
    const fields = opMeta[next]?.[1] ?? [];
    setLines(fields.map(() => ''));
  };

  const execute = async () => {
    setLoading(true);
    try {
      let op = operation;
      let payload = lines;
      if (operation.startsWith('aes_')) {
        payload = operation === 'aes_encrypt'
          ? [aes.key, aes.iv, aes.message]
          : [aes.key, aes.iv, aes.message, aes.tag];
      }
      const { data } = await axios.post(`${API}/cipher`, { operation: op, lines: payload });
      setResult(data);
    } catch (error: any) {
      alert(error.response?.data?.detail ?? 'Crypto operation failed.');
    } finally { setLoading(false); }
  };

  const fields = opMeta[operation]?.[1] ?? [];

  return (
    <PageShell title="Crypto / Encoding Lab" subtitle="Every operation below maps to the corresponding native C++ implementation. Classical ciphers, encoding utilities, SHA/HMAC/PBKDF2, secure random key bytes and the optional OpenSSL AES-256-GCM path are all kept separate instead of being hidden behind a generic text box." onBack={() => go('home')}>
      <div className="grid gap-5 xl:grid-cols-[330px_1fr]">
        <div className={`${shellCard} p-4`}>
          <div className="mb-3 flex items-center gap-2 font-bold"><LockKeyhole size={18} className="text-blue-600 dark:text-blue-400" /> Operations</div>
          <div className="space-y-1.5">
            {Object.entries(opMeta).map(([id, meta]) => (
              <button key={id} type="button" onClick={() => setOperationState(id)} className={`w-full rounded-xl px-3 py-2 text-left text-sm transition ${operation === id ? 'bg-blue-600 font-semibold text-white' : 'text-slate-700 hover:bg-slate-100 dark:text-slate-200 dark:hover:bg-slate-700'}`}>
                {meta[0]}
              </button>
            ))}
          </div>
        </div>
        <div className="space-y-5">
          <div className={`${shellCard} p-5`}>
            <div className="flex items-center gap-2"><KeyRound size={18} className="text-blue-600 dark:text-blue-400" /><h3 className="font-bold">{opMeta[operation]?.[0]}</h3></div>
            {operation === 'aes_encrypt' || operation === 'aes_decrypt' ? (
              <div className="mt-4 grid gap-3">
                <InfoStrip>AES-256-GCM is shown because your C++ engine supports it when built with <code>ENGINE_USE_OPENSSL</code>. The UI does not add a JavaScript crypto replacement.</InfoStrip>
                <div className="grid gap-3 md:grid-cols-2"><Field label="Key hex"><input value={aes.key} onChange={(e) => setAes({ ...aes, key: e.target.value })} className={controlClass} /></Field><Field label="12-byte IV hex"><input value={aes.iv} onChange={(e) => setAes({ ...aes, iv: e.target.value })} className={controlClass} /></Field></div>
                <Field label={operation === 'aes_encrypt' ? 'Plaintext' : 'Ciphertext hex'}><textarea value={aes.message} onChange={(e) => setAes({ ...aes, message: e.target.value })} rows={5} className={`${controlClass} font-mono text-xs`} /></Field>
                {operation === 'aes_decrypt' && <Field label="Tag hex"><input value={aes.tag} onChange={(e) => setAes({ ...aes, tag: e.target.value })} className={controlClass} /></Field>}
              </div>
            ) : fields.length ? (
              <div className="mt-4 grid gap-3 md:grid-cols-2">{fields.map((field: string, i: number) => <Field key={field} label={field}><input value={lines[i] ?? ''} onChange={(e) => setLines(lines.map((v, ix) => ix === i ? e.target.value : v))} className={controlClass} /></Field>)}</div>
            ) : <div className="mt-4 text-sm text-slate-600 dark:text-slate-300">This tool displays the native C++ educational explanation without adding another implementation.</div>}
            <button type="button" onClick={execute} disabled={loading} className={`${primaryButton} mt-5`}>{loading ? <RefreshCcw size={16} className="animate-spin" /> : <Play size={16} />} Execute native operation</button>
          </div>
          {result && <div className={`${shellCard} overflow-hidden`}><div className="border-b border-slate-200 px-5 py-4 font-bold dark:border-slate-700">Result</div><div className="p-5">{result.text ? <pre className="max-h-[520px] overflow-auto whitespace-pre-wrap rounded-xl bg-slate-950 p-4 font-mono text-xs leading-5 text-slate-100">{result.text}</pre> : <pre className="overflow-auto rounded-xl bg-slate-50 p-4 font-mono text-xs dark:bg-slate-900">{JSON.stringify(result, null, 2)}</pre>}</div></div>}
        </div>
      </div>
    </PageShell>
  );
}

function StatsScreen({ go }: { go: (screen: string) => void }) {
  const [data, setData] = useState('10 15 23 35 53 80 121 183');
  const [result, setResult] = useState<AnyRecord | null>(null);
  const [loading, setLoading] = useState(false);
  const execute = async () => {
    setLoading(true);
    try { const { data: payload } = await axios.post(`${API}/stats`, { data }); setResult(payload); }
    catch (error: any) { alert(error.response?.data?.detail ?? 'Statistics failed.'); }
    finally { setLoading(false); }
  };
  return <PageShell title="Statistical Analysis Only" subtitle="Descriptive statistics and distribution/serial diagnostics already implemented by the native engine." onBack={() => go('home')}>
    <div className="grid gap-5 xl:grid-cols-[380px_1fr]">
      <div className={`${shellCard} p-5`}><Field label="Observations"><textarea value={data} onChange={(e) => setData(e.target.value)} rows={12} className={`${controlClass} font-mono text-xs`} /></Field><button type="button" onClick={execute} disabled={loading} className={`${primaryButton} mt-4 w-full`}>{loading ? <RefreshCcw size={16} className="animate-spin" /> : <BarChart3 size={16} />} Run statistics</button></div>
      {result ? <div className="space-y-5"><div className={`${shellCard} p-5`}><div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">{Object.entries(result.stats ?? {}).filter(([key]) => key !== 'n').map(([key, value]) => <div key={key} className="rounded-xl bg-slate-50 p-3 dark:bg-slate-900/60"><div className="text-[10px] font-bold uppercase tracking-wide text-slate-400 dark:text-slate-500">{key}</div><div className="mt-1 font-mono text-sm font-semibold">{safeNumber(value, 4)}</div></div>)}</div></div><div className={`${shellCard} overflow-hidden`}><div className="border-b border-slate-200 px-5 py-4 font-bold dark:border-slate-700">Distribution view</div><div className="h-[440px] p-2"><Plot data={[{ x: result.values ?? [], type: 'histogram', name: 'Observations', nbinsx: Math.min(30, Math.max(8, Math.round(Math.sqrt((result.values ?? []).length || 1)))) }]} layout={{ autosize: true, margin: { l: 60, r: 25, t: 30, b: 55 }, xaxis: { title: 'value' }, yaxis: { title: 'count' } }} style={{ width: '100%', height: '100%' }} useResizeHandler config={{ responsive: true, displaylogo: false }} /></div></div></div> : <div className={`${shellCard} flex min-h-[460px] items-center justify-center p-8 text-center text-slate-400 dark:text-slate-500`}><div><Sigma size={40} className="mx-auto mb-3 opacity-40" /><div className="font-semibold">Statistics will appear here</div></div></div>}
    </div>
  </PageShell>;
}

function RandomScreen({ go }: { go: (screen: string) => void }) {
  const [mode, setMode] = useState<'analyze' | 'generate'>('analyze');
  const [data, setData] = useState('1 4 7 10 13 16 19 22 25 28');
  const [generator, setGenerator] = useState(1);
  const [seed, setSeed] = useState(12345);
  const [count, setCount] = useState(100);
  const [modulus, setModulus] = useState(2147483647);
  const [multiplier, setMultiplier] = useState(48271);
  const [increment, setIncrement] = useState(0);
  const [result, setResult] = useState<AnyRecord | null>(null);
  const [loading, setLoading] = useState(false);
  const execute = async () => {
    setLoading(true);
    try {
      if (mode === 'analyze') {
        const { data: payload } = await axios.post(`${API}/random/analyze`, { data });
        setResult(payload);
      } else {
        const { data: payload } = await axios.post(`${API}/random/generate`, { generator, seed, count, modulus: generator === 2 ? modulus : undefined, multiplier: generator === 2 ? multiplier : undefined, increment: generator === 2 ? increment : undefined });
        setResult(payload);
      }
    } catch (error: any) { alert(error.response?.data?.detail ?? 'PRNG operation failed.'); }
    finally { setLoading(false); }
  };
  return <PageShell title="Random / PRNG Tools" subtitle="The C++ engine's LCG and xorshift detection, diagnostic statistics and native generator implementations are exposed here." onBack={() => go('home')}>
    <div className="grid gap-5 xl:grid-cols-[380px_1fr]">
      <div className={`${shellCard} p-5`}>
        <div className="grid grid-cols-2 gap-2"><button type="button" onClick={() => setMode('analyze')} className={mode === 'analyze' ? primaryButton : secondaryButton}>Analyze</button><button type="button" onClick={() => setMode('generate')} className={mode === 'generate' ? primaryButton : secondaryButton}>Generate</button></div>
        {mode === 'analyze' ? <Field label="Series"><textarea value={data} onChange={(e) => setData(e.target.value)} rows={12} className={`${controlClass} mt-4 font-mono text-xs`} /></Field> : <div className="mt-4 space-y-3"><Field label="Generator"><select value={generator} onChange={(e) => setGenerator(Number(e.target.value))} className={controlClass}><option value={1}>std::mt19937 integer</option><option value={2}>Linear congruential</option><option value={3}>xorshift32</option><option value={4}>splitmix64</option></select></Field><div className="grid grid-cols-2 gap-3"><Field label="Seed"><input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} className={controlClass} /></Field><Field label="Count"><input type="number" min={1} max={1000000} value={count} onChange={(e) => setCount(Number(e.target.value) || 1)} className={controlClass} /></Field></div>{generator === 2 && <div className="grid grid-cols-3 gap-2"><Field label="m"><input type="number" value={modulus} onChange={(e) => setModulus(Number(e.target.value))} className={controlClass} /></Field><Field label="a"><input type="number" value={multiplier} onChange={(e) => setMultiplier(Number(e.target.value))} className={controlClass} /></Field><Field label="c"><input type="number" value={increment} onChange={(e) => setIncrement(Number(e.target.value))} className={controlClass} /></Field></div>}</div>}
        <button type="button" onClick={execute} disabled={loading} className={`${primaryButton} mt-4 w-full`}>{loading ? <RefreshCcw size={16} className="animate-spin" /> : <Zap size={16} />} Run native PRNG tool</button>
      </div>
      <div className="space-y-5">{result ? <>{result.stats && <AnalysisSummary result={{ stats: result.stats, analysis: {} }} />} {result.values?.length ? <div className={`${shellCard} overflow-hidden`}><div className="border-b border-slate-200 px-5 py-4 font-bold dark:border-slate-700">Generated series</div><div className="h-[440px] p-2"><Plot data={[{ x: result.values.map((_: number, i: number) => i + 1), y: result.values, mode: 'lines', type: 'scatter', name: 'Generated' }]} layout={{ autosize: true, margin: { l: 60, r: 25, t: 30, b: 55 }, xaxis: { title: 'index' }, yaxis: { title: 'value' } }} style={{ width: '100%', height: '100%' }} useResizeHandler config={{ responsive: true, displaylogo: false }} /></div></div> : <div className={`${shellCard} p-5`}><pre className="whitespace-pre-wrap font-mono text-xs leading-5 text-slate-700 dark:text-slate-300">{result.text}</pre></div>}</> : <div className={`${shellCard} flex min-h-[460px] items-center justify-center p-8 text-center text-slate-400 dark:text-slate-500`}><div><Binary size={40} className="mx-auto mb-3 opacity-40" /><div className="font-semibold">PRNG result will appear here</div></div></div>}</div>
    </div>
  </PageShell>;
}

function SelfTestScreen({ go }: { go: (screen: string) => void }) {
  const [result, setResult] = useState('');
  const [loading, setLoading] = useState(false);
  const execute = async () => { setLoading(true); try { const { data } = await axios.post(`${API}/selftest`); setResult(data.text ?? ''); } catch (error: any) { alert(error.response?.data?.detail ?? 'Self-test failed.'); } finally { setLoading(false); } };
  return <PageShell title="Engine Self-Test / Benchmark" subtitle="Runs the existing C++ self-test suite, including cryptographic references, model recognition, root diagnostics, zero-order hold, cube moves and generic puzzle search." onBack={() => go('home')}>
    <div className={`${shellCard} p-5`}><button type="button" onClick={execute} disabled={loading} className={primaryButton}>{loading ? <RefreshCcw size={16} className="animate-spin" /> : <TestTube2 size={16} />} Run self-test suite</button><div className="mt-4 rounded-xl bg-slate-950 p-5"><pre className="max-h-[680px] overflow-auto whitespace-pre-wrap font-mono text-xs leading-5 text-slate-100">{result || 'No self-test run yet.'}</pre></div></div>
  </PageShell>;
}

function PuzzleScreen({ go }: { go: (screen: string) => void }) {
  const [mode, setMode] = useState<'cube' | 'generic'>('cube');
  const [size, setSize] = useState(3);
  const [scramble, setScramble] = useState("R U R' U'");
  const [depth, setDepth] = useState(8);
  const [nodeLimit, setNodeLimit] = useState(3000000);
  const [stateSize, setStateSize] = useState(4);
  const [start, setStart] = useState('1 0 3 2');
  const [goal, setGoal] = useState('0 1 2 3');
  const [moves, setMoves] = useState('swap01|0|1 0 2 3\nswap23|1|0 1 3 2');
  const [result, setResult] = useState<AnyRecord | null>(null);
  const [loading, setLoading] = useState(false);

  const execute = async () => {
    setLoading(true);
    try {
      if (mode === 'cube') {
        const { data } = await axios.post(`${API}/puzzle/cube`, { size, scramble, depth, node_limit: nodeLimit });
        setResult(data);
      } else {
        const parsedMoves = moves.split(/\r?\n/).filter(Boolean).map((line) => {
          const [name, group, perm] = line.split('|');
          return { name, group: Number(group), perm: perm.trim().split(/[ ,]+/).map(Number) };
        });
        const { data } = await axios.post(`${API}/puzzle/generic`, { state_size: stateSize, start: start.trim().split(/[ ,]+/).map(Number), goal: goal.trim().split(/[ ,]+/).map(Number), depth, node_limit: nodeLimit, moves: parsedMoves });
        setResult(data);
      }
    } catch (error: any) { alert(error.response?.data?.detail ?? 'Puzzle solver failed.'); }
    finally { setLoading(false); }
  };

  return <PageShell title="Generic Puzzle / Cube Graph Solver" subtitle="The original IDA* solver and cube move machinery are invoked directly. The UI only supplies the state, moves, scramble and search limits." onBack={() => go('home')}>
    <div className="grid gap-5 xl:grid-cols-[390px_1fr]">
      <div className={`${shellCard} p-5`}>
        <div className="grid grid-cols-2 gap-2"><button type="button" onClick={() => setMode('cube')} className={mode === 'cube' ? primaryButton : secondaryButton}>Cube</button><button type="button" onClick={() => setMode('generic')} className={mode === 'generic' ? primaryButton : secondaryButton}>Generic state</button></div>
        <div className="mt-4 space-y-3">
          {mode === 'cube' ? <><Field label="Cube size"><select value={size} onChange={(e) => setSize(Number(e.target.value))} className={controlClass}><option value={2}>2×2</option><option value={3}>3×3</option><option value={4}>4×4</option></select></Field><Field label="Scramble"><input value={scramble} onChange={(e) => setScramble(e.target.value)} className={`${controlClass} font-mono`} /></Field></> : <><Field label="State size"><input type="number" min={2} max={128} value={stateSize} onChange={(e) => setStateSize(Number(e.target.value) || 2)} className={controlClass} /></Field><Field label="Start state"><input value={start} onChange={(e) => setStart(e.target.value)} className={`${controlClass} font-mono`} /></Field><Field label="Goal state"><input value={goal} onChange={(e) => setGoal(e.target.value)} className={`${controlClass} font-mono`} /></Field><Field label="Moves"><textarea value={moves} onChange={(e) => setMoves(e.target.value)} rows={8} className={`${controlClass} resize-y font-mono text-xs`} /></Field></>}
          <div className="grid grid-cols-2 gap-3"><Field label="Max depth"><input type="number" min={0} max={24} value={depth} onChange={(e) => setDepth(Number(e.target.value) || 0)} className={controlClass} /></Field><Field label="Node limit"><input type="number" min={1} max={10000000} value={nodeLimit} onChange={(e) => setNodeLimit(Number(e.target.value) || 1)} className={controlClass} /></Field></div>
          <button type="button" onClick={execute} disabled={loading} className={`${primaryButton} w-full`}>{loading ? <RefreshCcw size={16} className="animate-spin" /> : <Workflow size={16} />} Solve with native IDA*</button>
        </div>
      </div>
      <div className={`${shellCard} p-5`}>{result ? <div className="space-y-4"><div className={`rounded-xl p-4 ${result.solved ? 'bg-emerald-50 text-emerald-900 dark:bg-emerald-950/40 dark:text-emerald-200' : 'bg-amber-50 text-amber-900 dark:bg-amber-950/40 dark:text-amber-200'}`}><div className="flex items-center gap-2 font-bold">{result.solved ? <CheckCircle2 size={18} /> : <CircleHelp size={18} />}{result.solved ? 'Solved' : 'No solution within the requested search bound'}</div><div className="mt-2 text-sm">Nodes explored: <span className="font-mono">{result.nodes}</span></div></div><div><div className="text-xs font-bold uppercase tracking-wide text-slate-400 dark:text-slate-500">Solution</div><div className="mt-2 flex flex-wrap gap-2">{(result.solution ?? []).map((move: string, i: number) => <span key={`${move}-${i}`} className="rounded-lg bg-slate-100 px-3 py-2 font-mono text-sm dark:bg-slate-700">{move}</span>)}</div></div></div> : <div className="flex min-h-[460px] items-center justify-center text-center text-slate-400 dark:text-slate-500"><div><Boxes size={42} className="mx-auto mb-3 opacity-40" /><div className="font-semibold">Puzzle result will appear here</div></div></div>}</div>
    </div>
  </PageShell>;
}

function HomeScreen({ go }: { go: (screen: string) => void }) {
  const modes = [
    { id: 'sequence', number: '1', title: 'Single 1D Sequence', desc: 'Exact recurrence detection, model selection, validation and forecasting.', icon: TrendingUp },
    { id: 'multivariate', number: '2', title: 'N-Dimensional / Multivariate', desc: 'Joint lag models and dependency links across variables.', icon: Table2 },
    { id: 'coordinates', number: '3', title: 'Manual Coordinates', desc: 'Cartesian, polar, cylindrical and spherical point analysis.', icon: Grid3X3 },
    { id: 'file', number: '4', title: 'Load Dataset / Sequence', desc: 'Interpret uploaded text/CSV-style datasets using the native parser.', icon: FileText },
    { id: 'crypto', number: '5', title: 'Crypto / Encoding', desc: 'Classical ciphers, encoders, hashes, KDFs and optional AES-GCM.', icon: LockKeyhole },
    { id: 'stats', number: '6', title: 'Statistical Analysis', desc: 'Distribution, serial-correlation and descriptive diagnostics.', icon: BarChart3 },
    { id: 'random', number: '7', title: 'Random / PRNG', desc: 'Analyze or generate native PRNG/LCG/xorshift/splitmix sequences.', icon: Binary },
    { id: 'curve', number: '8', title: 'Graph / Curve / Model Lab', desc: 'Polynomial, Fourier, damped-sinusoid, root-guided, symbolic and combined family fitting.', icon: LineChart },
    { id: 'selftest', number: '9', title: 'Self-Test / Benchmark', desc: 'Run the engine’s built-in correctness checks.', icon: TestTube2 },
  ];

  return <div className="space-y-6 text-slate-900 dark:text-slate-100">
    <div className="overflow-hidden rounded-3xl bg-slate-900 p-7 text-white shadow-lg dark:bg-slate-950">
      <div className="flex flex-col gap-6 xl:flex-row xl:items-end xl:justify-between">
        <div>
          <div className="text-xs font-bold uppercase tracking-[0.2em] text-blue-300">Engineering OS · Analyser</div>
          <h1 className="mt-2 text-3xl font-bold tracking-tight">Advanced Multivariate Sequence / Curve / Crypto Engine v6.0</h1>
          <p className="mt-3 max-w-4xl text-sm leading-6 text-slate-300">One analyser section, nine native engine capabilities. Choose a method, move to its dedicated workspace, run the C++ implementation, and keep the rest of the application available through the existing ENG-OS navigation.</p>
        </div>
        <div className="grid grid-cols-3 gap-2 text-center text-xs"><div className="rounded-xl bg-white/10 px-4 py-3"><div className="font-mono text-lg font-bold">9</div><div className="text-slate-300">engine modes</div></div><div className="rounded-xl bg-white/10 px-4 py-3"><div className="font-mono text-lg font-bold">30+</div><div className="text-slate-300">curve families</div></div><div className="rounded-xl bg-white/10 px-4 py-3"><div className="font-mono text-lg font-bold">C++</div><div className="text-slate-300">source of truth</div></div></div>
      </div>
    </div>

    <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
      {modes.map(({ id, number, title, desc, icon: Icon }) => (
        <button key={id} type="button" onClick={() => go(id)} className={`${shellCard} group p-5 text-left transition hover:-translate-y-0.5 hover:border-blue-200 hover:shadow-md dark:hover:border-blue-800`}>
          <div className="flex items-start justify-between gap-4"><div className="flex h-11 w-11 items-center justify-center rounded-xl bg-blue-50 text-blue-600 dark:bg-blue-950/60 dark:text-blue-300"><Icon size={22} /></div><span className="font-mono text-sm text-slate-300 dark:text-slate-600">{number.padStart(2, '0')}</span></div>
          <div className="mt-4 flex items-center justify-between gap-3"><div className="font-bold">{title}</div><ChevronRight size={17} className="text-slate-300 transition group-hover:translate-x-0.5 group-hover:text-blue-500 dark:text-slate-600" /></div>
          <p className="mt-2 text-sm leading-6 text-slate-500 dark:text-slate-400">{desc}</p>
        </button>
      ))}
    </div>

    <div className={`${shellCard} p-5`}>
      <div className="flex items-start gap-3"><TerminalSquare size={19} className="mt-0.5 text-blue-600 dark:text-blue-400" /><div><div className="font-bold">Architecture note</div><p className="mt-1 text-sm leading-6 text-slate-500 dark:text-slate-400">The browser owns navigation and visualization. FastAPI is transport and process control. The mathematical and algorithmic behavior remains in the existing C++ code, including the native family generator and `fitByFamily()` path.</p></div></div>
    </div>
  </div>;
}

export default function SequenceLab() {
  const [screen, setScreen] = useState('home');
  const [theme, setTheme] = useState<'light' | 'dark'>('light');
  const dark = theme === 'dark';

  const go = (next: string) => setScreen(next);
  const toggleTheme = () => setTheme((t) => (t === 'light' ? 'dark' : 'light'));

  let body: ReactNode;
  switch (screen) {
    case 'sequence': body = <SequenceScreen go={go} dark={dark} />; break;
    case 'multivariate': body = <MultivariateScreen go={go} dark={dark} />; break;
    case 'coordinates': body = <CoordinatesScreen go={go} dark={dark} />; break;
    case 'file': body = <FileScreen go={go} dark={dark} />; break;
    case 'crypto': body = <CryptoScreen go={go} />; break;
    case 'stats': body = <StatsScreen go={go} />; break;
    case 'random': body = <RandomScreen go={go} />; break;
    case 'curve': body = <CurveScreen go={go} dark={dark} />; break;
    case 'selftest': body = <SelfTestScreen go={go} />; break;
    case 'puzzle': body = <PuzzleScreen go={go} />; break;
    default: body = <HomeScreen go={go} />;
  }

  return (
    <div className={dark ? 'dark' : ''}>
      <div className="min-h-full bg-slate-50 px-1 pb-10 dark:bg-slate-900">
        <div className="flex justify-end pb-2 pt-1">
          <button
            type="button"
            onClick={toggleTheme}
            title="Toggle light / dark theme"
            className="inline-flex items-center gap-2 rounded-xl border border-slate-300 bg-white px-3 py-2 text-xs font-semibold text-slate-700 shadow-sm transition hover:bg-slate-50 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-200 dark:hover:bg-slate-700"
          >
            {dark ? <Sun size={15} /> : <Moon size={15} />}
            {dark ? 'Light mode' : 'Dark mode'}
          </button>
        </div>
        {body}
      </div>
    </div>
  );
}