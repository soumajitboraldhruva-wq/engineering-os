import { useRef, useState } from 'react';
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

// The Cube Solver (puzzle tile) is hidden for now; set VITE_SHOW_CUBE_SOLVER=true to bring it back.
const SHOW_CUBE_SOLVER = import.meta.env.VITE_SHOW_CUBE_SOLVER === 'true';

const API = (import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000').replace(/\/+$/, '') + '/api/analyser';

const MODEL_FAMILIES = [
  ['mean', 'Mean baseline'],
  ['linear', 'Linear'],
  ['poly', 'Polynomial'],
  ['chebyshev', 'Chebyshev polynomial'],
  ['taylor', 'Taylor / Newton'],
  ['exp', 'Exponential'],
  ['exp_offset', 'Shifted exponential (c + a·e^bx)'],
  ['log', 'Logarithmic'],
  ['power', 'Power law'],
  ['rational', 'Rational'],
  ['sin', 'Sinusoid'],
  ['fourier', 'Fourier / harmonics'],
  ['damped_sine', 'Damped sinusoid (RLC transient)'],
  ['multitone', 'Multi-tone sinusoids'],
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

// Which of the three manual inputs each family actually uses (and what they mean).
// A frequency / width / penalty left at 0 is estimated automatically by the engine.
interface FamilyParamSpec { degree?: string; param?: string; param2?: string; defaultDegree?: number; hint?: string }
const FAMILY_PARAMS: Record<string, FamilyParamSpec> = {
  poly: { degree: 'Degree', defaultDegree: 3 },
  chebyshev: { degree: 'Degree', defaultDegree: 6 },
  taylor: { degree: 'Order', defaultDegree: 3 },
  rational: { degree: 'Order (1 or 2)', defaultDegree: 1 },
  sin: { param: 'Angular frequency ω', hint: 'ω = 2π / period. Leave 0 to estimate it from the data.' },
  fourier: { degree: 'Harmonics', param: 'Period', defaultDegree: 2, hint: 'Period in x units. Leave 0 to estimate it.' },
  damped_sine: { param: 'Angular frequency ω', param2: 'Decay rate α', hint: 'Leave ω at 0 to estimate it; α = 0 means no damping.' },
  multitone: { degree: 'Tones (2–4)', defaultDegree: 2 },
  chirp: { param: 'Phase w₁', param2: 'Phase w₂', hint: 'phase = w₁·x + w₂·x²' },
  ar: { degree: 'Order', param: 'Ridge λ', defaultDegree: 2 },
  differenced_ar: { degree: 'Order', param: 'Ridge λ', defaultDegree: 2 },
  prony: { degree: 'Order', defaultDegree: 2 },
  segmented: { degree: 'Breakpoints (1–3)', defaultDegree: 1 },
  piecewise_adaptive: { degree: 'Local degree (1–2)', defaultDegree: 2 },
  root_poly: { degree: 'Polynomial degree', param: 'Roots to use (1–3)', defaultDegree: 2 },
  seasonal: { degree: 'Trend degree', param: 'Period (samples)', defaultDegree: 1 },
  periodic_delta: { param: 'Period (samples)' },
  osc_envelope: { param: 'Period' },
  rbf: { param: 'Kernel width σ', hint: 'Leave 0 to choose a width from the x-range.' },
  ridge_basis: { param: 'Penalty λ', hint: 'Leave 0 for the default 0.01.' },
  lasso_basis: { param: 'Penalty λ', hint: 'Leave 0 for the default 0.01.' },
  elastic_basis: { param: 'Penalty λ', hint: 'Leave 0 for the default 0.01.' },
  huber_basis: { param: 'Penalty λ', hint: 'Leave 0 for the default 0.01.' },
  symbolic: { param: 'Penalty λ', hint: 'Leave 0 for the default 0.01.' },
};

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
  const ranked = families.map((item, rank) => ({ item, rank: rank + 1 }));
  const filtered = ranked.filter(({ item }) => {
    const q = filter.trim().toLowerCase();
    if (!q) return true;
    return `${item.family} ${item.model?.name} ${item.model?.category}`.toLowerCase().includes(q);
  });
  const aicc = (v: unknown) => (typeof v === 'number' && Number.isFinite(v) ? v.toFixed(1) : '—');

  return (
    <div className={`${shellCard} overflow-hidden`}>
      <div className="border-b border-slate-200 px-5 py-4 dark:border-slate-700">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="font-bold">Curve-family candidates</h3>
            <p className="mt-1 max-w-2xl text-xs leading-5 text-slate-500 dark:text-slate-400">
              Every family is fitted to the full data and ranked by <strong>AICc</strong> (fit quality penalised by the number of parameters, so a
              high-degree curve cannot win just by passing through every point). Models within ΔAICc &lt; 2 of the best are practically tied.
              Recursive models (AR, Holt, Prony) are scored on one-step-ahead error. The Analyze button additionally uses rolling-origin validation.
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
              <th className="px-3 py-3">#</th>
              <th className="px-3 py-3">Family</th>
              <th className="px-3 py-3">Model</th>
              <th className="px-3 py-3">AICc</th>
              <th className="px-3 py-3">ΔAICc</th>
              <th className="px-3 py-3">RMSE</th>
              <th className="px-3 py-3">R²</th>
              <th className="px-3 py-3"></th>
            </tr>
          </thead>
          <tbody>
            {filtered.map(({ item, rank }) => (
              <tr key={`${item.family}-${item.model?.name}-${rank}`} className={`border-t border-slate-100 hover:bg-slate-50 dark:border-slate-700 dark:hover:bg-slate-700/40 ${item.overfit ? 'opacity-50' : ''}`}>
                <td className="px-3 py-3 font-mono text-slate-400">{rank}</td>
                <td className="px-3 py-3 font-semibold text-slate-700 dark:text-slate-200">{familyLabel(item.family)}</td>
                <td className="max-w-[340px] px-3 py-3">
                  <div className="truncate font-medium text-slate-800 dark:text-slate-100">{item.model?.name}{rank === 1 && <span className="ml-2 rounded-full bg-emerald-100 px-2 py-0.5 text-[10px] font-bold text-emerald-700">best</span>}{item.overfit && <span className="ml-2 rounded-full bg-amber-100 px-2 py-0.5 text-[10px] font-bold text-amber-700">too few points</span>}</div>
                  <div className="mt-0.5 truncate font-mono text-[11px] text-slate-400 dark:text-slate-500">{item.model?.formula}</div>
                </td>
                <td className="px-3 py-3 font-mono">{aicc(item.aicc)}</td>
                <td className="px-3 py-3 font-mono">{aicc(item.deltaAicc)}</td>
                <td className="px-3 py-3 font-mono">{safeNumber(item.model?.metrics?.rmse, 5)}</td>
                <td className="px-3 py-3 font-mono">{safeNumber(item.model?.metrics?.r2, 5)}</td>
                <td className="px-3 py-3 text-right">
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
                <select value={family} onChange={(e) => { const f = e.target.value; setFamily(f); setParam(0); setParam2(0); setDegree(FAMILY_PARAMS[f]?.defaultDegree ?? 1); }} className={controlClass}>
                  {MODEL_FAMILIES.map(([id, label]) => <option key={id} value={id}>{label}</option>)}
                </select>
              </Field>
              {(() => {
                const spec = FAMILY_PARAMS[family] ?? {};
                const any = spec.degree || spec.param || spec.param2;
                return any ? (
                  <>
                    <div className="grid grid-cols-3 gap-2">
                      {spec.degree && <Field label={spec.degree}><input type="number" min={1} max={18} value={degree} onChange={(e) => setDegree(Number(e.target.value) || 1)} className={controlClass} /></Field>}
                      {spec.param && <Field label={spec.param} hint="0 = auto"><input type="number" step="0.001" value={param} onChange={(e) => setParam(Number(e.target.value))} className={controlClass} /></Field>}
                      {spec.param2 && <Field label={spec.param2}><input type="number" step="0.001" value={param2} onChange={(e) => setParam2(Number(e.target.value))} className={controlClass} /></Field>}
                    </div>
                    {spec.hint && <p className="-mt-1 text-[11px] leading-4 text-slate-400 dark:text-slate-500">{spec.hint}</p>}
                  </>
                ) : <p className="text-[11px] text-slate-400 dark:text-slate-500">This family has no adjustable parameters.</p>;
              })()}
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

/* ------------------------------------------------------------------------------------------
 * Smart data import
 *
 * 1. Upload a file (CSV / TSV / TXT / Excel / ODS ...) or paste text -> /inspect-* parses it into
 *    one or more TABLES of named columns (headers, units, dates, decimal commas, multi-block
 *    files, sheets ... all detected server-side).
 * 2. Columns are grouped by heading (families such as Temp_1/Temp_2, shared units, X/Y pairs).
 *    The user assigns roles: X axis / Series / Skip.
 * 3. Run: curve (each series vs X), sequence (each series alone), pairs (X1/Y1, X2/Y2 ...)
 *    or multivariate (all series jointly). The numbers are sent to the C++ engine endpoints.
 * ------------------------------------------------------------------------------------------ */
type ImportRole = 'x' | 'series' | 'ignore';
type ImportMode = 'curve' | 'sequence' | 'multivariate' | 'pairs';

interface ImportColumn {
  index: number;
  name: string;
  label: string;
  unit: string;
  kind: 'numeric' | 'datetime' | 'text';
  family: string;
  values?: (number | null)[];
  min?: number;
  max?: number;
  mean?: number;
  missing?: number;
  numericCount?: number;
  monotonic?: string;
  step?: number;
  uniformStep?: boolean;
  suggestedRole: ImportRole;
  samples?: string[];
  origin?: string;
}

interface ImportGroup { key: string; label: string; kind: 'family' | 'unit' | 'pair'; columns: number[] }

interface ImportTable {
  name: string;
  source: string;
  headerDetected: boolean;
  unitRowDetected: boolean;
  decimalComma: boolean;
  rowCount: number;
  columns: ImportColumn[];
  groups: ImportGroup[];
  pairs: { x: number; y: number; label: string }[];
  suggestedX: number | null;
  suggestedAnalysis: ImportMode;
  preview: string[][];
  notes: string[];
}

interface ImportInspection { filename: string; format: string; tables: ImportTable[]; warnings: string[] }
interface ImportResult { label: string; kind: ImportMode; payload?: AnyRecord; error?: string; note?: string }

const IMPORT_MAX_POINTS = 2000;
const IMPORT_ACCEPT = '.csv,.tsv,.txt,.dat,.log,.prn,.json,.xlsx,.xlsm,.xls,.xlx,.xlsb,.ods';

const fmtStat = (v: unknown) => {
  const n = Number(v);
  if (!Number.isFinite(n)) return '—';
  const a = Math.abs(n);
  return a !== 0 && (a >= 1e6 || a < 1e-3) ? n.toExponential(3) : String(Number(n.toPrecision(5)));
};

function interpolateGaps(values: (number | null)[]): (number | null)[] {
  const out = [...values];
  for (let i = 0; i < out.length; i += 1) {
    if (out[i] !== null) continue;
    let lo = i - 1;
    while (lo >= 0 && out[lo] === null) lo -= 1;
    let hi = i + 1;
    while (hi < out.length && out[hi] === null) hi += 1;
    if (lo >= 0 && hi < out.length) {
      const t = (i - lo) / (hi - lo);
      out[i] = (out[lo] as number) * (1 - t) + (out[hi] as number) * t;
    } else if (lo >= 0) out[i] = out[lo];
    else if (hi < out.length) out[i] = out[hi];
  }
  return out;
}

/** Keeps at most `limit` rows, evenly spaced; returns the stride used. */
function thin<T>(rows: T[], limit: number): { rows: T[]; stride: number } {
  if (rows.length <= limit) return { rows, stride: 1 };
  const stride = Math.ceil(rows.length / limit);
  return { rows: rows.filter((_, i) => i % stride === 0), stride };
}

function FileScreen({ go, dark }: { go: (screen: string) => void; dark: boolean }) {
  const [inspection, setInspection] = useState<ImportInspection | null>(null);
  const [tableIndex, setTableIndex] = useState(0);
  const [roles, setRoles] = useState<Record<number, ImportRole>>({});
  const [mode, setMode] = useState<ImportMode>('curve');
  const [horizon, setHorizon] = useState(4);
  const [tolerance, setTolerance] = useState(-1);
  const [stepText, setStepText] = useState('');
  const [missing, setMissing] = useState<'drop' | 'interpolate'>('drop');
  const [pasteOpen, setPasteOpen] = useState(false);
  const [pasteText, setPasteText] = useState('');
  const [dragging, setDragging] = useState(false);
  const [inspecting, setInspecting] = useState(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<ImportResult[]>([]);
  const [activeResult, setActiveResult] = useState(0);
  const fileInput = useRef<HTMLInputElement | null>(null);

  const table: ImportTable | null = inspection ? inspection.tables[Math.min(tableIndex, inspection.tables.length - 1)] : null;

  const adoptTable = (t: ImportTable) => {
    const next: Record<number, ImportRole> = {};
    t.columns.forEach((c) => { next[c.index] = c.suggestedRole; });
    setRoles(next);
    setMode(t.suggestedAnalysis);
    const xCol = t.suggestedX !== null ? t.columns[t.suggestedX] : undefined;
    setStepText(xCol?.step && xCol.step > 0 ? String(Number(xCol.step.toPrecision(6))) : '');
    setResults([]);
    setActiveResult(0);
  };

  const accept = (data: ImportInspection) => {
    setInspection(data);
    setTableIndex(0);
    adoptTable(data.tables[0]);
    setError(null);
  };

  const inspectFile = async (file: File) => {
    setInspecting(true);
    setError(null);
    try {
      const form = new FormData();
      form.append('file', file);
      const { data } = await axios.post(`${API}/inspect-file`, form);
      accept(data);
    } catch (e: any) {
      setError(e.response?.data?.detail ?? e.message ?? 'Could not read this file.');
    } finally {
      setInspecting(false);
    }
  };

  const inspectPasted = async () => {
    if (!pasteText.trim()) return;
    setInspecting(true);
    setError(null);
    try {
      const { data } = await axios.post(`${API}/inspect-text`, { text: pasteText, filename: 'pasted data' });
      accept(data);
      setPasteOpen(false);
    } catch (e: any) {
      setError(e.response?.data?.detail ?? e.message ?? 'Could not read this text.');
    } finally {
      setInspecting(false);
    }
  };

  const onDrop = (event: React.DragEvent) => {
    event.preventDefault();
    setDragging(false);
    const file = event.dataTransfer.files?.[0];
    if (file) void inspectFile(file);
  };

  const onPick = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) void inspectFile(file);
    event.target.value = '';
  };

  const switchTable = (i: number) => {
    if (!inspection) return;
    setTableIndex(i);
    adoptTable(inspection.tables[i]);
  };

  const setRole = (index: number, role: ImportRole) => {
    setRoles((prev) => {
      const next = { ...prev, [index]: role };
      if (role === 'x') Object.keys(next).forEach((k) => { if (Number(k) !== index && next[Number(k)] === 'x') next[Number(k)] = 'series'; });
      return next;
    });
  };

  const applyGroup = (g: ImportGroup) => {
    if (!table) return;
    if (g.kind === 'pair') {
      setRoles((prev) => {
        const next = { ...prev };
        table.columns.forEach((c) => { if (c.kind !== 'text') next[c.index] = 'ignore'; });
        next[g.columns[0]] = 'x';
        next[g.columns[1]] = 'series';
        return next;
      });
      setMode('curve');
      return;
    }
    setRoles((prev) => {
      const next = { ...prev };
      table.columns.forEach((c) => { if (c.kind !== 'text' && next[c.index] !== 'x') next[c.index] = g.columns.includes(c.index) ? 'series' : 'ignore'; });
      return next;
    });
  };

  const numericCols = table ? table.columns.filter((c) => c.kind !== 'text') : [];
  const xCol = table ? table.columns.find((c) => roles[c.index] === 'x') : undefined;
  const seriesCols = table ? table.columns.filter((c) => roles[c.index] === 'series' && c.kind !== 'text') : [];

  const modeProblem = (m: ImportMode): string | null => {
    if (!table) return 'Load a dataset first.';
    if (m === 'curve' && !xCol) return 'Mark one column as X.';
    if (m === 'curve' && !seriesCols.length) return 'Mark at least one Series column.';
    if (m === 'sequence' && !seriesCols.length) return 'Mark at least one Series column.';
    if (m === 'multivariate' && seriesCols.length < 2) return 'Needs at least two Series columns.';
    if (m === 'pairs' && table.pairs.length < 2) return 'No X/Y column pairs were detected.';
    return null;
  };

  const run = async () => {
    if (!table) return;
    const problem = modeProblem(mode);
    if (problem) { setError(problem); return; }
    setRunning(true);
    setError(null);
    setResults([]);
    setActiveResult(0);
    const out: ImportResult[] = [];
    const prep = (vals: (number | null)[] | undefined) => (missing === 'interpolate' ? interpolateGaps(vals ?? []) : vals ?? []);
    const forecastStep = Number(stepText) > 0 ? Number(stepText) : undefined;

    const runCurve = async (xc: ImportColumn, yc: ImportColumn, label: string) => {
      const xs = xc.values ?? [];
      const ys = prep(yc.values);
      let rows: [number, number][] = [];
      for (let i = 0; i < Math.min(xs.length, ys.length); i += 1) {
        const xv = xs[i];
        const yv = ys[i];
        if (xv !== null && yv !== null) rows.push([xv, yv]);
      }
      const dropped = Math.min(xs.length, ys.length) - rows.length;
      rows.sort((a, b) => a[0] - b[0]);
      rows = rows.filter((r, i) => i === 0 || r[0] !== rows[i - 1][0]); // engine needs strictly increasing x
      const t = thin(rows, IMPORT_MAX_POINTS);
      const notes: string[] = [];
      if (dropped > 0) notes.push(`${dropped} row(s) with missing values skipped`);
      if (t.stride > 1) notes.push(`every ${t.stride}th row used (limit ${IMPORT_MAX_POINTS} points)`);
      if (t.rows.length < 5) { out.push({ label, kind: 'curve', error: 'Fewer than 5 usable (x, y) rows.' }); return; }
      let step = forecastStep;
      if (!step) {
        const diffs = t.rows.slice(1).map((r, i) => r[0] - t.rows[i][0]).sort((a, b) => a - b);
        step = diffs[Math.floor(diffs.length / 2)] || 1;
      }
      try {
        const { data } = await axios.post(`${API}/curve`, {
          pairs: t.rows.map((r) => `${r[0]} ${r[1]}`).join('\n'), horizon, tolerance, step,
        });
        out.push({ label, kind: 'curve', payload: data, note: notes.join(' · ') });
      } catch (e: any) {
        out.push({ label, kind: 'curve', error: e.response?.data?.detail ?? e.message });
      }
    };

    try {
      if (mode === 'curve' && xCol) {
        for (const yc of seriesCols) await runCurve(xCol, yc, `${yc.name} vs ${xCol.name}`);
      } else if (mode === 'pairs') {
        for (const p of table.pairs) {
          const xc = table.columns[p.x];
          const yc = table.columns[p.y];
          if (roles[yc.index] !== 'ignore') await runCurve(xc, yc, p.label);
        }
      } else if (mode === 'sequence') {
        for (const yc of seriesCols) {
          const vals = prep(yc.values).filter((v): v is number => v !== null);
          const t = thin(vals, IMPORT_MAX_POINTS);
          if (t.rows.length < 3) { out.push({ label: yc.name, kind: 'sequence', error: 'Fewer than 3 numeric values.' }); continue; }
          try {
            const { data } = await axios.post(`${API}/analyze`, { data: t.rows.join('\n'), horizon, tolerance });
            out.push({ label: yc.name, kind: 'sequence', payload: data, note: t.stride > 1 ? `every ${t.stride}th value used` : undefined });
          } catch (e: any) {
            out.push({ label: yc.name, kind: 'sequence', error: e.response?.data?.detail ?? e.message });
          }
        }
      } else if (mode === 'multivariate') {
        const cols = seriesCols.map((c) => prep(c.values));
        const n = Math.min(...cols.map((c) => c.length));
        let rows: number[][] = [];
        for (let i = 0; i < n; i += 1) {
          const row = cols.map((c) => c[i]);
          if (row.every((v) => v !== null)) rows.push(row as number[]);
        }
        const dropped = n - rows.length;
        const t = thin(rows, 600);
        rows = t.rows;
        if (rows.length < 8) {
          out.push({ label: 'Multivariate', kind: 'multivariate', error: `Only ${rows.length} complete rows - at least 8 are required.` });
        } else {
          try {
            const { data } = await axios.post(`${API}/multivariate`, {
              rows: rows.map((r) => r.join(' ')).join('\n'), variables: seriesCols.length, horizon, tolerance,
            });
            out.push({
              label: `Multivariate (${seriesCols.map((c) => c.name).join(', ')})`, kind: 'multivariate', payload: { ...data, names: seriesCols.map((c) => c.name) },
              note: [dropped > 0 ? `${dropped} incomplete row(s) skipped` : '', t.stride > 1 ? `every ${t.stride}th row used` : ''].filter(Boolean).join(' · '),
            });
          } catch (e: any) {
            out.push({ label: 'Multivariate', kind: 'multivariate', error: e.response?.data?.detail ?? e.message });
          }
        }
      }
      setResults(out);
    } finally {
      setRunning(false);
    }
  };

  const modeOptions: { id: ImportMode; title: string; desc: string }[] = [
    { id: 'curve', title: 'Curve', desc: 'Each Series column against the X column' },
    { id: 'sequence', title: 'Sequence', desc: 'Each Series column on its own' },
    { id: 'multivariate', title: 'Multivariate', desc: 'All Series columns together (joint model)' },
    ...(table && table.pairs.length >= 2 ? [{ id: 'pairs' as ImportMode, title: 'X/Y pairs', desc: 'X1↔Y1, X2↔Y2 … each analysed' }] : []),
  ];

  const current = results[Math.min(activeResult, Math.max(0, results.length - 1))];

  return (
    <PageShell
      title="Import dataset"
      subtitle="Drop a CSV, TSV, TXT or Excel file (or paste data). Headers, units, dates, decimal commas, multiple tables and sheets are detected automatically, and columns are grouped by heading so you can choose exactly what to analyse."
      onBack={() => go('home')}
    >
      <div className="grid gap-5 xl:grid-cols-[400px_1fr]">
        {/* ---------------- left: source + options ---------------- */}
        <div className="space-y-5">
          <div className={`${shellCard} p-5`}>
            <div className="mb-3 text-xs font-bold uppercase tracking-wide text-slate-500 dark:text-slate-400">1 · Source</div>
            <div
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
              onClick={() => fileInput.current?.click()}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') fileInput.current?.click(); }}
              className={`flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed px-4 py-8 text-center transition ${
                dragging ? 'border-blue-500 bg-blue-50 dark:bg-blue-950/30' : 'border-slate-300 hover:border-blue-400 hover:bg-slate-50 dark:border-slate-600 dark:hover:bg-slate-700/40'
              }`}
            >
              {inspecting ? <RefreshCcw size={26} className="animate-spin text-blue-600" /> : <Upload size={26} className="text-blue-600 dark:text-blue-400" />}
              <div className="mt-2 text-sm font-semibold">{inspecting ? 'Reading file…' : 'Drop a file here or click to browse'}</div>
              <div className="mt-1 text-xs text-slate-500 dark:text-slate-400">CSV · TSV · TXT · XLSX · XLS · ODS</div>
              <input ref={fileInput} type="file" className="hidden" accept={IMPORT_ACCEPT} onChange={onPick} />
            </div>
            <button type="button" onClick={() => setPasteOpen((o) => !o)} className="mt-3 text-xs font-semibold text-blue-600 hover:text-blue-800 dark:text-blue-400">
              {pasteOpen ? '▾ Hide paste box' : '▸ …or paste data'}
            </button>
            {pasteOpen && (
              <div className="mt-2">
                <textarea value={pasteText} onChange={(e) => setPasteText(e.target.value)} rows={8} spellCheck={false} placeholder={'time,temp,pressure\n0,20.1,101\n1,20.5,102\n…'} className={`${controlClass} font-mono text-xs`} />
                <button type="button" onClick={() => void inspectPasted()} disabled={inspecting || !pasteText.trim()} className={`${primaryButton} mt-2 w-full`}>
                  <Table2 size={16} /> Detect columns
                </button>
              </div>
            )}
            {inspection && (
              <div className="mt-4 rounded-xl bg-slate-50 p-3 text-xs leading-5 text-slate-600 dark:bg-slate-900/60 dark:text-slate-300">
                <div className="font-semibold text-slate-800 dark:text-slate-100">{inspection.filename}</div>
                <div>{inspection.tables.length} table{inspection.tables.length === 1 ? '' : 's'} found · format: {inspection.format}</div>
                {inspection.warnings.map((w) => <div key={w} className="text-amber-700 dark:text-amber-300">⚠ {w}</div>)}
              </div>
            )}
          </div>

          {table && (
            <div className={`${shellCard} p-5`}>
              <div className="mb-3 text-xs font-bold uppercase tracking-wide text-slate-500 dark:text-slate-400">3 · Analysis</div>
              <div className="space-y-2">
                {modeOptions.map((m) => {
                  const problem = modeProblem(m.id);
                  return (
                    <button
                      key={m.id}
                      type="button"
                      onClick={() => setMode(m.id)}
                      className={`w-full rounded-xl border p-3 text-left transition ${
                        mode === m.id ? 'border-blue-600 bg-blue-50 ring-1 ring-blue-600 dark:bg-blue-950/30' : 'border-slate-200 hover:bg-slate-50 dark:border-slate-600 dark:hover:bg-slate-700/40'
                      }`}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-sm font-semibold">{m.title}{table.suggestedAnalysis === m.id && <span className="ml-2 rounded-full bg-emerald-100 px-2 py-0.5 text-[10px] font-bold text-emerald-700">suggested</span>}</span>
                      </div>
                      <div className="text-[11px] text-slate-500 dark:text-slate-400">{m.desc}</div>
                      {mode === m.id && problem && <div className="mt-1 text-[11px] font-semibold text-amber-700 dark:text-amber-300">{problem}</div>}
                    </button>
                  );
                })}
              </div>
              <div className="mt-4 grid grid-cols-2 gap-3">
                <Field label="Forecast horizon"><input type="number" min={1} max={1000} value={horizon} onChange={(e) => setHorizon(Number(e.target.value) || 1)} className={controlClass} /></Field>
                <Field label="Tolerance %" hint="-1 = auto"><input type="number" step="0.1" min={-1} max={100} value={tolerance} onChange={(e) => setTolerance(Number(e.target.value))} className={controlClass} /></Field>
                {(mode === 'curve' || mode === 'pairs') && (
                  <Field label="X step" hint="blank = detected"><input value={stepText} onChange={(e) => setStepText(e.target.value)} placeholder="auto" className={controlClass} /></Field>
                )}
                <Field label="Missing values">
                  <select value={missing} onChange={(e) => setMissing(e.target.value as 'drop' | 'interpolate')} className={controlClass}>
                    <option value="drop">Skip those rows</option>
                    <option value="interpolate">Interpolate gaps</option>
                  </select>
                </Field>
              </div>
              <button type="button" onClick={() => void run()} disabled={running} className={`${primaryButton} mt-4 w-full`}>
                {running ? <RefreshCcw size={16} className="animate-spin" /> : <Play size={16} />}
                {running ? 'Running native engine…' : 'Analyze'}
              </button>
            </div>
          )}
          {error && <div className="rounded-xl border border-rose-200 bg-rose-50 p-3 text-sm leading-6 text-rose-800 dark:border-rose-900/50 dark:bg-rose-950/40 dark:text-rose-200">{error}</div>}
        </div>

        {/* ---------------- right: columns + results ---------------- */}
        <div className="space-y-5">
          {!table ? (
            <div className={`${shellCard} flex min-h-[420px] items-center justify-center p-8 text-center text-slate-400 dark:text-slate-500`}>
              <div>
                <Database size={42} className="mx-auto mb-3 opacity-40" />
                <div className="font-semibold">Load a dataset to begin</div>
                <p className="mx-auto mt-1 max-w-md text-sm">Any layout works: with or without a header row, comma / tab / semicolon / space separated, one column or many, several sheets or blocks.</p>
              </div>
            </div>
          ) : (
            <>
              <div className={`${shellCard} overflow-hidden`}>
                <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 px-5 py-4 dark:border-slate-700">
                  <div>
                    <h3 className="font-bold">2 · Columns <span className="font-normal text-slate-400">— {table.rowCount.toLocaleString()} rows</span></h3>
                    <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
                      {table.headerDetected ? 'Header row detected' : 'No header row — columns were numbered'}
                      {table.unitRowDetected ? ' · units row detected' : ''}{table.decimalComma ? ' · decimal commas converted' : ''}
                    </p>
                  </div>
                  {inspection && inspection.tables.length > 1 && (
                    <div className="flex flex-wrap gap-1.5">
                      {inspection.tables.map((t, i) => (
                        <button key={`${t.name}-${i}`} type="button" onClick={() => switchTable(i)} className={`rounded-full px-3 py-1 text-xs font-semibold transition ${i === tableIndex ? 'bg-blue-600 text-white' : 'bg-slate-100 text-slate-600 hover:bg-slate-200 dark:bg-slate-700 dark:text-slate-200'}`}>{t.name}</button>
                      ))}
                    </div>
                  )}
                </div>
                {table.notes.map((n) => <div key={n} className="border-b border-slate-100 bg-amber-50 px-5 py-2 text-xs text-amber-800 dark:border-slate-700 dark:bg-amber-950/30 dark:text-amber-200">{n}</div>)}

                {table.groups.length > 0 && (
                  <div className="flex flex-wrap items-center gap-2 border-b border-slate-100 px-5 py-3 dark:border-slate-700">
                    <span className="text-[11px] font-bold uppercase tracking-wide text-slate-400">Groups</span>
                    {table.groups.map((g) => (
                      <button key={g.key} type="button" onClick={() => applyGroup(g)} title={g.columns.map((i) => table.columns[i].name).join(', ')} className="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-semibold text-slate-600 transition hover:border-blue-400 hover:text-blue-700 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-200">
                        {g.label} <span className="text-slate-400">×{g.columns.length}</span>
                      </button>
                    ))}
                    <button type="button" onClick={() => setRoles((prev) => { const next = { ...prev }; table.columns.forEach((c) => { if (c.kind !== 'text' && next[c.index] !== 'x') next[c.index] = 'series'; }); return next; })} className="ml-auto text-xs font-semibold text-blue-600 hover:text-blue-800 dark:text-blue-400">All numeric → Series</button>
                  </div>
                )}

                <div className="max-h-[420px] overflow-auto">
                  <table className="w-full text-left text-xs">
                    <thead className="sticky top-0 bg-slate-100 text-[10px] uppercase tracking-wide text-slate-500 dark:bg-slate-900 dark:text-slate-400">
                      <tr><th className="px-4 py-3">Use as</th><th className="px-4 py-3">Heading</th><th className="px-4 py-3">Type</th><th className="px-4 py-3">Range</th><th className="px-4 py-3">Mean</th><th className="px-4 py-3">Gaps</th></tr>
                    </thead>
                    <tbody>
                      {table.columns.map((c) => (
                        <tr key={c.index} className={`border-t border-slate-100 dark:border-slate-700 ${roles[c.index] === 'ignore' || c.kind === 'text' ? 'opacity-55' : ''}`}>
                          <td className="px-4 py-2.5">
                            {c.kind === 'text' ? <span className="text-slate-400">text</span> : (
                              <div className="inline-flex overflow-hidden rounded-lg border border-slate-300 dark:border-slate-600">
                                {(['x', 'series', 'ignore'] as ImportRole[]).map((r) => (
                                  <button key={r} type="button" onClick={() => setRole(c.index, r)} className={`px-2.5 py-1 text-[11px] font-semibold transition ${roles[c.index] === r ? (r === 'x' ? 'bg-violet-600 text-white' : r === 'series' ? 'bg-blue-600 text-white' : 'bg-slate-500 text-white') : 'bg-white text-slate-500 hover:bg-slate-50 dark:bg-slate-800 dark:text-slate-300'}`}>
                                    {r === 'x' ? 'X' : r === 'series' ? 'Series' : 'Skip'}
                                  </button>
                                ))}
                              </div>
                            )}
                          </td>
                          <td className="px-4 py-2.5"><div className="font-semibold">{c.label || c.name}</div>{c.unit && <div className="text-[11px] text-slate-400">{c.unit}</div>}</td>
                          <td className="px-4 py-2.5"><span className="rounded-full bg-slate-100 px-2 py-0.5 text-[10px] font-semibold text-slate-600 dark:bg-slate-700 dark:text-slate-200">{c.kind === 'datetime' ? 'date/time' : c.kind}</span>{c.monotonic === 'increasing' && <span className="ml-1 text-[10px] text-emerald-600">↗ sorted</span>}</td>
                          <td className="px-4 py-2.5 font-mono">{c.kind === 'text' ? <span className="text-slate-400">{(c.samples ?? []).slice(0, 2).join(', ')}</span> : `${fmtStat(c.min)} … ${fmtStat(c.max)}`}</td>
                          <td className="px-4 py-2.5 font-mono">{c.kind === 'text' ? '—' : fmtStat(c.mean)}</td>
                          <td className="px-4 py-2.5 font-mono">{c.missing ? <span className="text-amber-600">{c.missing}</span> : '0'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="border-t border-slate-100 px-5 py-3 text-xs text-slate-500 dark:border-slate-700 dark:text-slate-400">
                  {numericCols.length} numeric column{numericCols.length === 1 ? '' : 's'} · X: <strong>{xCol?.name ?? 'none'}</strong> · Series: <strong>{seriesCols.length ? seriesCols.map((c) => c.name).join(', ') : 'none'}</strong>
                </div>
              </div>

              {table.preview.length > 0 && (
                <details className={`${shellCard} overflow-hidden`}>
                  <summary className="cursor-pointer px-5 py-3 text-sm font-semibold">Raw preview (first rows)</summary>
                  <div className="max-h-56 overflow-auto border-t border-slate-100 dark:border-slate-700">
                    <table className="w-full text-left font-mono text-[11px]"><tbody>
                      {table.preview.map((r, i) => <tr key={i} className="border-b border-slate-50 dark:border-slate-700">{r.map((c, j) => <td key={j} className="px-3 py-1.5">{c}</td>)}</tr>)}
                    </tbody></table>
                  </div>
                </details>
              )}
            </>
          )}

          {results.length > 0 && (
            <div className="space-y-4">
              {results.length > 1 && (
                <div className="flex flex-wrap gap-1.5">
                  {results.map((r, i) => (
                    <button key={r.label + i} type="button" onClick={() => setActiveResult(i)} className={`rounded-full px-3.5 py-1.5 text-xs font-semibold transition ${i === activeResult ? 'bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900' : 'bg-white text-slate-600 shadow-sm hover:bg-slate-50 dark:bg-slate-800 dark:text-slate-200'} ${r.error ? 'ring-1 ring-rose-400' : ''}`}>{r.label}</button>
                  ))}
                </div>
              )}
              {current && (
                <div className="space-y-5">
                  <div className="flex flex-wrap items-baseline gap-3">
                    <h3 className="text-lg font-bold">{current.label}</h3>
                    {current.note && <span className="text-xs text-slate-500 dark:text-slate-400">{current.note}</span>}
                  </div>
                  {current.error && <div className="rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-800 dark:border-rose-900/50 dark:bg-rose-950/40 dark:text-rose-200">{current.error}</div>}
                  {current.payload && current.kind !== 'multivariate' && (
                    <><AnalysisSummary result={current.payload} /><SequencePlot result={current.payload} title={current.label} dark={dark} /></>
                  )}
                  {current.payload && current.kind === 'multivariate' && <ImportMultivariateResult payload={current.payload} dark={dark} />}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </PageShell>
  );
}

function ImportMultivariateResult({ payload, dark }: { payload: AnyRecord; dark: boolean }) {
  const rows: number[][] = payload.rows ?? [];
  const analysis = payload.analysis ?? {};
  const names: string[] = payload.names ?? [];
  const predictions: number[][] = analysis.predictions ?? [];
  const k = rows[0]?.length ?? 0;
  const grid = dark ? '#334155' : '#e2e8f0';
  const traces = [
    ...Array.from({ length: k }).map((_, j) => ({
      x: rows.map((_r, i) => i + 1), y: rows.map((r) => r[j]), mode: 'lines', type: 'scatter', name: names[j] ?? `Variable ${j + 1}`,
    })),
    ...(predictions.length ? Array.from({ length: k }).map((_, j) => ({
      x: [rows.length, ...predictions.map((_p, i) => rows.length + i + 1)],
      y: [rows[rows.length - 1]?.[j], ...predictions.map((p) => p[j])],
      mode: 'lines+markers', type: 'scatter', name: `${names[j] ?? `Variable ${j + 1}`} forecast`, line: { dash: 'dot' },
    })) : []),
  ];
  return (
    <div className="space-y-5">
      <div className={`${shellCard} p-5`}>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <div className="rounded-xl bg-slate-50 p-3 dark:bg-slate-900/60"><div className="text-[10px] font-bold uppercase text-slate-400">Rows</div><div className="mt-1 font-mono text-sm font-semibold">{analysis.rows ?? rows.length}</div></div>
          <div className="rounded-xl bg-slate-50 p-3 dark:bg-slate-900/60"><div className="text-[10px] font-bold uppercase text-slate-400">Variables</div><div className="mt-1 font-mono text-sm font-semibold">{analysis.variables ?? k}</div></div>
          <div className="rounded-xl bg-slate-50 p-3 dark:bg-slate-900/60"><div className="text-[10px] font-bold uppercase text-slate-400">Forecast</div><div className="mt-1 text-sm font-semibold">{analysis.fullForecastAvailable ? 'Available' : 'Not available'}</div></div>
          <div className="rounded-xl bg-slate-50 p-3 dark:bg-slate-900/60"><div className="text-[10px] font-bold uppercase text-slate-400">Tolerance</div><div className="mt-1 font-mono text-sm font-semibold">{safeNumber(analysis.tolerancePct, 3)}%</div></div>
        </div>
        {analysis.failureReason && <p className="mt-3 text-sm text-amber-700 dark:text-amber-300">{analysis.failureReason}</p>}
        {analysis.links?.[0]?.formula && <p className="mt-3 break-words font-mono text-xs text-slate-600 dark:text-slate-300">{analysis.links[0].formula}</p>}
      </div>
      <div className={`${shellCard} overflow-hidden`}>
        <div className="border-b border-slate-200 px-5 py-4 font-bold dark:border-slate-700">Trajectories and forecast</div>
        <div className="h-[480px] p-2">
          <Plot data={traces} layout={{ autosize: true, margin: { l: 60, r: 25, t: 30, b: 55 }, xaxis: { title: 'row', gridcolor: grid }, yaxis: { title: 'value', gridcolor: grid }, legend: { orientation: 'h', y: 1.1 }, paper_bgcolor: dark ? '#1e293b' : 'white', plot_bgcolor: dark ? '#1e293b' : 'white', font: { color: dark ? '#e2e8f0' : '#1e293b' } }} style={{ width: '100%', height: '100%' }} useResizeHandler config={{ responsive: true, displaylogo: false }} />
        </div>
      </div>
      {predictions.length > 0 && (
        <div className={`${shellCard} overflow-hidden`}>
          <div className="border-b border-slate-200 px-5 py-4 font-bold dark:border-slate-700">Predicted rows</div>
          <div className="max-h-64 overflow-auto"><table className="w-full text-left text-xs">
            <thead className="sticky top-0 bg-slate-100 text-[10px] uppercase text-slate-500 dark:bg-slate-900"><tr><th className="px-4 py-2">Step</th>{Array.from({ length: k }).map((_, j) => <th key={j} className="px-4 py-2">{names[j] ?? `Variable ${j + 1}`}</th>)}</tr></thead>
            <tbody>{predictions.map((p, i) => <tr key={i} className="border-t border-slate-100 dark:border-slate-700"><td className="px-4 py-2 font-semibold">+{i + 1}</td>{p.map((v, j) => <td key={j} className="px-4 py-2 font-mono">{safeNumber(v, 5)}</td>)}</tr>)}</tbody>
          </table></div>
        </div>
      )}
    </div>
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
    { id: 'file', number: '4', title: 'Import Dataset', desc: 'CSV, TSV, TXT or Excel files with headers, units and dates; columns grouped by heading, then analysed.', icon: FileText },
    { id: 'crypto', number: '5', title: 'Crypto / Encoding', desc: 'Classical ciphers, encoders, hashes, KDFs and optional AES-GCM.', icon: LockKeyhole },
    { id: 'stats', number: '6', title: 'Statistical Analysis', desc: 'Distribution, serial-correlation and descriptive diagnostics.', icon: BarChart3 },
    { id: 'random', number: '7', title: 'Random / PRNG', desc: 'Analyze or generate native PRNG/LCG/xorshift/splitmix sequences.', icon: Binary },
    { id: 'curve', number: '8', title: 'Graph / Curve / Model Lab', desc: 'Polynomial, Fourier, damped-sinusoid, root-guided, symbolic and combined family fitting.', icon: LineChart },
    { id: 'selftest', number: '9', title: 'Self-Test / Benchmark', desc: 'Run the engine’s built-in correctness checks.', icon: TestTube2 },
    ...(SHOW_CUBE_SOLVER ? [{ id: 'puzzle', number: '10', title: 'Puzzle / Cube Solver', desc: 'Generic permutation search and NxNxN cube moves.', icon: Boxes }] : []),
  ];

  return <div className="space-y-6 text-slate-900 dark:text-slate-100">
    <div className="overflow-hidden rounded-3xl bg-slate-900 p-7 text-white shadow-lg dark:bg-slate-950">
      <div className="flex flex-col gap-6 xl:flex-row xl:items-end xl:justify-between">
        <div>
          <div className="text-xs font-bold uppercase tracking-[0.2em] text-blue-300">Engineering OS · Analyser</div>
          <h1 className="mt-2 text-3xl font-bold tracking-tight">Advanced Multivariate Sequence / Curve / Crypto Engine v6.0</h1>
          <p className="mt-3 max-w-4xl text-sm leading-6 text-slate-300">One analyser section, ten native engine capabilities. Choose a method, move to its dedicated workspace, run the C++ implementation, and keep the rest of the application available through the existing ENG-OS navigation.</p>
        </div>
        <div className="grid grid-cols-3 gap-2 text-center text-xs"><div className="rounded-xl bg-white/10 px-4 py-3"><div className="font-mono text-lg font-bold">{modes.length}</div><div className="text-slate-300">engine modes</div></div><div className="rounded-xl bg-white/10 px-4 py-3"><div className="font-mono text-lg font-bold">30+</div><div className="text-slate-300">curve families</div></div><div className="rounded-xl bg-white/10 px-4 py-3"><div className="font-mono text-lg font-bold">C++</div><div className="text-slate-300">source of truth</div></div></div>
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
    case 'puzzle': body = SHOW_CUBE_SOLVER ? <PuzzleScreen go={go} /> : <HomeScreen go={go} />; break;
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