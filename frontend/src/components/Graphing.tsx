import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ClipboardEvent,
  type CSSProperties,
  type PointerEvent,
  type WheelEvent as ReactWheelEvent,
  type ReactNode,
} from 'react';
import axios from 'axios';
import PlotComponent from 'react-plotly.js';
import 'katex/dist/katex.min.css';
import { BlockMath } from 'react-katex';
import {
  Plus,
  Trash2,
  Copy,
  CheckCircle,
  Eye,
  EyeOff,
  Moon,
  Sun,
  Terminal,
  Info,
  Play,
  Pause,
  Maximize2,
  Link2,
  WifiOff,
  Crosshair,
  Box,
  Rotate3d,
  X,
  Home,
  LayoutGrid,
  Settings2,
  Grip,
  RotateCcw,
  Grid3X3,
  ChevronRight,
  MousePointer2,
  Eraser,
  Search,
  Target,
  Command,
} from 'lucide-react';

const Plot = (PlotComponent as any).default || PlotComponent;
const API_URL = ((import.meta as any)?.env?.VITE_API_URL || 'http://127.0.0.1:8000') + '/api/graph/plot';

// High-contrast palette: no black curve, and colors remain distinct on both themes.
const COLORS = [
  '#ef4444', '#3b82f6', '#22c55e', '#a855f7', '#eab308', '#f97316',
  '#06b6d4', '#ec4899', '#84cc16', '#6366f1', '#14b8a6', '#f43f5e',
  '#8b5cf6', '#0ea5e9', '#10b981', '#e11d48', '#ca8a04', '#7c3aed',
  '#0891b2', '#c026d3', '#16a34a', '#2563eb', '#9333ea', '#ea580c',
];

const COMMAND_CATALOG = [
  { category: 'Intersections', title: 'Intersection of two curves', command: 'intersection(A, B)', help: 'Find all real intersection points of two named expressions.' },
  { category: 'Intersections', title: 'All pairwise intersections', command: 'intersectall()', help: 'Check every visible 2D pair.' },
  { category: 'Geometry', title: 'Axis intercepts', command: 'intercepts(A)', help: 'Find physical axis intercepts.' },
  { category: 'Geometry', title: 'Distance', command: 'distance(A, B)', help: 'Find exact line distance when possible, otherwise a bounded numerical minimum.' },
  { category: 'Regions', title: 'Area under curve', command: 'area_under(A, a, b)', help: 'Choose limits explicitly, or use Pick on graph for current bounds.' },
  { category: 'Regions', title: 'Area between curves', command: 'area_between(A, B, a, b)', help: 'Area between two compatible explicit functions.' },
  { category: 'Regions', title: 'Bounded region area', command: 'area_region(A, B)', help: 'Automatically use intersections to form bounded regions.' },
  { category: 'Regions', title: 'Closed curve area', command: 'area_closed(A)', help: 'Estimate or calculate a closed implicit/polar boundary.' },
  { category: 'Polar', title: 'Polar area', command: 'polar_area(A, θ1, θ2)', help: 'Area of r=f(θ).' },
  { category: 'Polar', title: 'Polar arc length', command: 'polar_arc_length(A, θ1, θ2)', help: 'Arc length of a polar curve.' },
  { category: 'Geometry', title: 'Arc length', command: 'arc_length(A, a, b)', help: 'Arc length of y=f(x).' },
  { category: 'Geometry', title: 'Locus fit', command: 'locus(A, B)', help: 'Approximate a fitted locus through intersection points.' },
  { category: 'Conversion', title: 'Coordinate conversion', command: 'convert(A)', help: 'Show the equivalent forms across the supported coordinate systems.' },
  { category: '3D Geometry', title: 'Volume under surface', command: 'volume_under(S)', help: 'Use the current 3D x/y bounds, or supply explicit limits.' },
  { category: '3D Geometry', title: 'Surface area', command: 'surface_area(S)', help: 'Use the current 3D x/y bounds, or supply explicit limits.' },
  { category: '3D Geometry', title: 'Volume between 2 surfaces/planes', command: 'volume_between(A, B)', help: 'Generalizes volume_under to any two surfaces, or a surface vs. a plane like z=0. Exact grid double-integral.' },
  { category: '3D Geometry', title: 'Volume of intersection (2-3 solids)', command: 'volume_intersect(A, B)', help: 'Volume common to 2-3 arbitrary implicit objects or planes (e.g. z=0, x=3) via Monte Carlo sampling — no explicit z=f(x,y) required.' },
];

const TRACE_COLOR = '#ff2d95';
const COMMAND_COLOR = '#a855f7';

 type Quality = 'low' | 'medium' | 'high';
 type PlotMode = 'cartesian2d' | 'polar2d' | 'cartesian3d' | 'cylindrical3d' | 'spherical3d' | 'complex';
 type ComplexView = 'magnitude' | 'real' | 'imag' | 'phase' | 'surface';
 type AngleUnit = 'degrees' | 'radians';
 type PanelId = 'expressions' | 'setup' | 'analysis';
 type Page = 'home' | 'graph';
 type CameraView = 'iso' | 'top' | 'front' | 'right' | 'back' | 'left' | 'bottom';
type PickMode = 'intersection' | 'area-under' | 'area-between' | 'area-region' | 'area-closed' | 'distance' | 'arc-length' | 'polar-area' | 'polar-arc-length' | 'volume' | 'surface-area' | null;

interface ExprState {
  id: number;
  name: string;
  text: string;
  visible: boolean;
}

interface TracePoint {
  owner: number;
  name: string;
  x?: number;
  y?: number;
  z?: number;
}

interface AxisConfig {
  x_min: number;
  x_max: number;
  y_min: number;
  y_max: number;
  z_min: number;
  z_max: number;
  theta_min: number;
  theta_max: number;
  r_max: number;
  phi_min: number;
  phi_max: number;
  re_min: number;
  re_max: number;
  im_min: number;
  im_max: number;
  x_scale: 'linear' | 'log';
  y_scale: 'linear' | 'log';
  x_label: string;
  y_label: string;
  z_label: string;
  show_grid: boolean;
  show_zero: boolean;
  equal_aspect: boolean;
  theta_direction: 'counterclockwise' | 'clockwise';
  theta_rotation: number;
}

interface PanelPos {
  x: number;
  y: number;
  w: number;
}

interface WorkspacePanelProps {
  id: PanelId;
  title: string;
  icon: ReactNode;
  open: boolean;
  position: PanelPos;
  dark: boolean;
  onClose: () => void;
  onDragStart: (id: PanelId, e: PointerEvent<HTMLDivElement>) => void;
  children: ReactNode;
}

const DEFAULT_AXIS: AxisConfig = {
  x_min: -20,
  x_max: 20,
  y_min: -20,
  y_max: 20,
  z_min: -20,
  z_max: 20,
  theta_min: 0,
  theta_max: 360,
  r_max: 20,
  phi_min: 0,
  phi_max: 180,
  re_min: -8,
  re_max: 8,
  im_min: -8,
  im_max: 8,
  x_scale: 'linear',
  y_scale: 'linear',
  x_label: 'X',
  y_label: 'Y',
  z_label: 'Z',
  show_grid: true,
  show_zero: true,
  equal_aspect: true,
  theta_direction: 'counterclockwise',
  theta_rotation: 0,
};

const VIEW_CAMERAS: Record<CameraView, any> = {
  iso: { eye: { x: 1.55, y: 1.55, z: 1.55 }, up: { x: 0, y: 0, z: 1 } },
  top: { eye: { x: 0, y: 0, z: 2.4 }, up: { x: 0, y: 1, z: 0 } },
  front: { eye: { x: 0, y: -2.4, z: 0 }, up: { x: 0, y: 0, z: 1 } },
  right: { eye: { x: 2.4, y: 0, z: 0 }, up: { x: 0, y: 0, z: 1 } },
  back: { eye: { x: 0, y: 2.4, z: 0 }, up: { x: 0, y: 0, z: 1 } },
  left: { eye: { x: -2.4, y: 0, z: 0 }, up: { x: 0, y: 0, z: 1 } },
  bottom: { eye: { x: 0, y: 0, z: -2.4 }, up: { x: 0, y: -1, z: 0 } },
};

const INITIAL_POSITIONS: Record<PanelId, PanelPos> = {
  expressions: { x: 18, y: 18, w: 410 },
  setup: { x: 448, y: 18, w: 360 },
  analysis: { x: 18, y: 510, w: 410 },
};

function WorkspacePanel({
  id,
  title,
  icon,
  open,
  position,
  dark,
  onClose,
  onDragStart,
  children,
}: WorkspacePanelProps) {
  if (!open) return null;

  const card = dark
    ? 'bg-slate-900/96 border-slate-700 text-slate-100'
    : 'bg-white/96 border-slate-200 text-slate-800';

  return (
    <div
      className={`absolute z-30 rounded-2xl border shadow-2xl backdrop-blur-xl overflow-hidden ${card}`}
      style={{
        left: position.x,
        top: position.y,
        width: position.w,
        maxWidth: 'calc(100% - 8px)',
        maxHeight: 'calc(100% - 20px)',
      } as CSSProperties}
    >
      <div
        onPointerDown={(e) => onDragStart(id, e)}
        className={`h-11 px-3 flex items-center gap-2 border-b select-none cursor-grab active:cursor-grabbing ${
          dark ? 'border-slate-700 bg-slate-900/95' : 'border-slate-200 bg-white/95'
        }`}
      >
        <Grip size={15} className="text-slate-500" />
        <span className="text-blue-500">{icon}</span>
        <span className="font-bold text-sm flex-1">{title}</span>
        <button
          type="button"
          onClick={onClose}
          className="p-1 rounded hover:bg-slate-500/15 text-slate-400"
          aria-label={`Hide ${title}`}
        >
          <X size={15} />
        </button>
      </div>
      <div className="p-3 overflow-y-auto max-h-[calc(100vh-180px)]">{children}</div>
    </div>
  );
}

function NumberField({
  label,
  value,
  onChange,
  fieldClass,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  fieldClass: string;
}) {
  return (
    <label className="text-[10px] text-slate-500">
      {label}
      <input
        type="number"
        value={Number.isFinite(value) ? value : ''}
        onChange={(e) => onChange(e.target.value === '' ? 0 : Number(e.target.value))}
        className={`mt-1 w-full rounded-lg border px-2 py-1.5 text-xs bg-transparent outline-none ${fieldClass}`}
      />
    </label>
  );
}

function SelectField({
  label,
  value,
  onChange,
  options,
  fieldClass,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: string[];
  fieldClass: string;
}) {
  return (
    <label className="text-[10px] text-slate-500">
      {label}
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className={`mt-1 w-full rounded-lg border px-2 py-1.5 text-xs outline-none ${fieldClass}`}
      >
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </label>
  );
}

export default function GraphingModule() {
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [page, setPage] = useState<Page>('graph');
  const [expressions, setExpressions] = useState<ExprState[]>([
    { id: 1, name: 'Circle', text: 'x^2 + y^2 = 25', visible: true },
    { id: 2, name: 'Line', text: 'y = x - 1', visible: true },
  ]);
  const [plotMode, setPlotMode] = useState<PlotMode>('cartesian2d');
  const [complexView, setComplexView] = useState<ComplexView>('magnitude');
  const [angleUnit, setAngleUnit] = useState<AngleUnit>('degrees');
  const [quality, setQuality] = useState<Quality>('high');
  const [axis, setAxis] = useState<AxisConfig>(DEFAULT_AXIS);

  const [plotData, setPlotData] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [connError, setConnError] = useState(false);
  const [plotError, setPlotError] = useState('');
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);

  const [hoveredId, setHoveredId] = useState<number | null>(null);
  const [traceEnabled, setTraceEnabled] = useState(true);
  const [tracePoint, setTracePoint] = useState<TracePoint | null>(null);

  const [tVal, setTVal] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playSpeed, setPlaySpeed] = useState(1);

  const [camera, setCamera] = useState<any>(VIEW_CAMERAS.iso);
  const [viewRevision, setViewRevision] = useState('init');
  const [mouseZoom, setMouseZoom] = useState(1);
  const [plotView, setPlotView] = useState<any>({});
  const [pickMode, setPickMode] = useState<PickMode>(null);
  const [pickedOwners, setPickedOwners] = useState<number[]>([]);
  const [commandSearch, setCommandSearch] = useState('');
  const [viewCubeOpen, setViewCubeOpen] = useState(true);

  const [panels, setPanels] = useState<Record<PanelId, boolean>>({
    expressions: true,
    setup: true,
    analysis: true,
  });
  const [positions, setPositions] = useState<Record<PanelId, PanelPos>>(INITIAL_POSITIONS);

  const [cmdInput, setCmdInput] = useState('');
  const [commandHistory, setCommandHistory] = useState<string[]>([]);
  const [commandResults, setCommandResults] = useState<string[]>([]);
  const [commandPlots, setCommandPlots] = useState<any[]>([]);
  const [showCmdHelp, setShowCmdHelp] = useState(false);
  const [showCommandConsole, setShowCommandConsole] = useState(false);
  const [openConversions, setOpenConversions] = useState<Record<number, boolean>>({});
  const [conversionResults, setConversionResults] = useState<Record<number, string[]>>({});
  const [conversionLoading, setConversionLoading] = useState<Record<number, boolean>>({});
  const [conversionErrors, setConversionErrors] = useState<Record<number, string>>({});
  const [selectedIds, setSelectedIds] = useState<number[]>([]);
  const [commandBusy, setCommandBusy] = useState(false);
  const [commandError, setCommandError] = useState('');

  const [permalinkCopied, setPermalinkCopied] = useState(false);
  const [workspaceTick, setWorkspaceTick] = useState(0);

  const inputRefs = useRef<Record<number, HTMLInputElement | null>>({});
  const traceOwnerRef = useRef<number[]>([]);
  const didLoadPermalink = useRef(false);
  const workspaceRef = useRef<HTMLDivElement | null>(null);
  const dragCleanupRef = useRef<(() => void) | null>(null);
  const plotRequestRef = useRef<AbortController | null>(null);
  const commandRequestRef = useRef<AbortController | null>(null);
  const userViewportTouchedRef = useRef(false);
  const cameraTweenRef = useRef<number | null>(null);

  const hasTime = expressions.some((e) => /\bt\b/.test(e.text));
  const complexSurface = plotMode === 'complex' && complexView === 'surface';
  const is3D =
    plotMode === 'cartesian3d' ||
    plotMode === 'cylindrical3d' ||
    plotMode === 'spherical3d' ||
    complexSurface;
  const hasAnyContent = expressions.some((e) => e.text.trim() !== '');

  const plotModeLabel: Record<PlotMode, string> = {
    cartesian2d: 'Cartesian 2D',
    polar2d: 'Polar 2D',
    cartesian3d: 'Cartesian 3D',
    cylindrical3d: 'Cylindrical 3D',
    spherical3d: 'Spherical 3D',
    complex: 'Complex plane',
  };

  useEffect(() => {
    if (didLoadPermalink.current) return;
    didLoadPermalink.current = true;
    const params = new URLSearchParams(window.location.search);
    const state = params.get('state');
    if (!state) return;
    try {
      const payload = JSON.parse(decodeURIComponent(atob(state)));
      if (Array.isArray(payload.e) && payload.e.length) {
        setExpressions(
          payload.e.map((tuple: [string, string, boolean], index: number) => ({
            id: Date.now() + index,
            name: tuple[0],
            text: tuple[1],
            visible: tuple[2],
          }))
        );
      }
      if (typeof payload.t === 'number') setTVal(payload.t);
      if (payload.plotMode) setPlotMode(payload.plotMode);
      if (payload.complexView) setComplexView(payload.complexView);
      if (payload.angleUnit) setAngleUnit(payload.angleUnit);
      if (payload.axis) setAxis((current) => ({ ...current, ...payload.axis }));
    } catch {
      // Ignore malformed shared state.
    }
  }, []);

  useEffect(() => {
    if (!isPlaying) return;
    const id = window.setInterval(() => {
      setTVal((previous) => {
        const next = previous + 0.1 * playSpeed;
        return next > 10 ? 0 : Number(next.toFixed(2));
      });
    }, 120);
    return () => window.clearInterval(id);
  }, [isPlaying, playSpeed]);

  // Plot-only requests never carry the analysis command history. This prevents
  // normal typing/mode changes from re-executing old commands.
  useEffect(() => {
    plotRequestRef.current?.abort();
    const controller = new AbortController();
    plotRequestRef.current = controller;

    const timeoutId = window.setTimeout(() => {
      const texts = expressions.map((e) => e.text);
      if (!texts.some((text) => text.trim())) {
        setPlotData(null);
        setPlotError('');
        setIsLoading(false);
        return;
      }

      setIsLoading(true);
      setConnError(false);
      axios
        .post(
          API_URL,
          {
            expressions: texts,
            expression_names: expressions.map((e) => e.name),
            t_val: tVal,
            commands: [],
            include_plots: true,
            quality,
            coordinate_system: plotMode,
            complex_view: complexView,
            angle_unit: angleUnit,
            axis,
          },
          { signal: controller.signal as any }
        )
        .then((response) => {
          setPlotData(response.data);
          setCommandPlots([]);
          setCommandResults([]);
          setCommandError('');
          setPlotError(response.data?.error ? String(response.data.error) : '');
          setConnError(false);
        })
        .catch((error) => {
          if (axios.isCancel(error) || error?.code === 'ERR_CANCELED') return;
          setConnError(true);
          setPlotError(error?.response?.data?.detail || error?.message || 'Plot request failed.');
        })
        .finally(() => setIsLoading(false));
    }, hasTime ? 140 : 260);

    return () => {
      window.clearTimeout(timeoutId);
      controller.abort();
    };
  }, [expressions, tVal, quality, plotMode, complexView, angleUnit, axis, hasTime]);

  useEffect(() => {
    return () => {
      plotRequestRef.current?.abort();
      commandRequestRef.current?.abort();
      dragCleanupRef.current?.();
      if (cameraTweenRef.current) window.cancelAnimationFrame(cameraTweenRef.current);
    };
  }, []);

  const clearTransientGraphState = useCallback(() => {
    setTracePoint(null);
    setHoveredId(null);
    setPlotData(null);
    setCommandPlots([]);
    setCommandResults([]);
    setCommandHistory([]);
    setCommandError('');
    setPlotError('');
    setWorkspaceTick((value) => value + 1);
  }, []);

  const changePlotMode = useCallback(
    (mode: PlotMode) => {
      if (mode === plotMode) return;
      commandRequestRef.current?.abort();
      clearTransientGraphState();
      setPlotMode(mode);
      setIsPlaying(false);
      setCamera(VIEW_CAMERAS.iso);
      setMouseZoom(1);
      setPlotView({});
      userViewportTouchedRef.current = false;
      setViewRevision(`mode-${mode}-${Date.now()}`);
    },
    [clearTransientGraphState, plotMode]
  );

  const updateExpr = useCallback(
    (index: number, field: keyof ExprState, value: string | boolean) => {
      setExpressions((previous) =>
        previous.map((entry, entryIndex) =>
          entryIndex === index ? { ...entry, [field]: value } : entry
        )
      );
      if (field === 'text' || field === 'name') {
        setTracePoint(null);
        setHoveredId(null);
        setCommandPlots([]);
        setCommandResults([]);
        setCommandError('');
        setPlotError('');
        setPlotView({});
        userViewportTouchedRef.current = false;
        setSelectedIds([]);
        const id = expressions[index]?.id;
        if (id !== undefined) {
          setConversionResults((current) => { const { [id]: _drop, ...rest } = current; return rest; });
          setConversionErrors((current) => { const { [id]: _drop, ...rest } = current; return rest; });
        }
      }
    },
    [expressions]
  );

  const updateAxis = useCallback((field: keyof AxisConfig, value: any) => {
    setAxis((current) => ({ ...current, [field]: value }));
    setTracePoint(null);
    setCommandPlots([]);
    setCommandResults([]);
    setPlotError('');
    setPlotView({});
    userViewportTouchedRef.current = false;
  }, []);

  const addExpression = useCallback(() => {
    const id = Date.now();
    setExpressions((previous) => [
      ...previous,
      { id, name: `Eq ${previous.length + 1}`, text: '', visible: true },
    ]);
    window.requestAnimationFrame(() => inputRefs.current[id]?.focus());
  }, []);

  const removeExpression = useCallback((index: number) => {
    setExpressions((previous) => previous.filter((_, entryIndex) => entryIndex !== index));
    setTracePoint(null);
    setHoveredId(null);
    setCommandPlots([]);
    setCommandResults([]);
    setCommandError('');
  }, []);

  const handlePaste = useCallback(
    (index: number, event: ClipboardEvent<HTMLInputElement>) => {
      event.preventDefault();
      let text = event.clipboardData
        .getData('text')
        .replace(/\$\$/g, '')
        .replace(/\$/g, '')
        .replace(/\\\[/g, '')
        .replace(/\\\]/g, '')
        .replace(/\\cos/g, 'cos')
        .replace(/\\sin/g, 'sin')
        .replace(/\\tan/g, 'tan')
        .replace(/\\theta/g, 'theta')
        .replace(/\\phi/g, 'phi')
        .replace(/\\rho/g, 'rho')
        .replace(/\\pi/g, 'pi')
        .replace(/\\left/g, '')
        .replace(/\\right/g, '');

      let previous = '';
      while (text !== previous) {
        previous = text;
        text = text.replace(/\\frac{([^{}]*)}{([^{}]*)}/g, '($1)/($2)');
      }

      updateExpr(index, 'text', `${expressions[index].text}${text.replace(/\\/g, '').trim()}`);
    },
    [expressions, updateExpr]
  );

  const copyToClipboard = useCallback((text: string, index: number) => {
    navigator.clipboard.writeText(text);
    setCopiedIndex(index);
    window.setTimeout(() => setCopiedIndex(null), 1800);
  }, []);

  const copyPermalink = useCallback(() => {
    const payload = {
      e: expressions.map((entry) => [entry.name, entry.text, entry.visible]),
      t: tVal,
      plotMode,
      complexView,
      angleUnit,
      axis,
    };
    const encoded = btoa(encodeURIComponent(JSON.stringify(payload)));
    const url = `${window.location.origin}${window.location.pathname}?state=${encoded}`;
    navigator.clipboard.writeText(url);
    setPermalinkCopied(true);
    window.setTimeout(() => setPermalinkCopied(false), 1800);
  }, [expressions, tVal, plotMode, complexView, angleUnit, axis]);

  const recenterView = useCallback(() => {
    setMouseZoom(1);
    setPlotView({});
    userViewportTouchedRef.current = false;
    setViewRevision(`fit-${Date.now()}`);
  }, []);

  // Smoothly tween the camera eye position between presets instead of snapping
  // instantly. This is a small, honest step toward a livelier 3D feel — a full
  // choreographed animation engine (multi-object reveals, synced camera moves)
  // is a separate, much larger undertaking and isn't attempted here.
  const easeInOutCubic = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

  const tweenCameraTo = useCallback((target: any, durationMs = 480) => {
    if (cameraTweenRef.current) window.cancelAnimationFrame(cameraTweenRef.current);
    const start = camera?.eye || VIEW_CAMERAS.iso.eye;
    const startUp = camera?.up || VIEW_CAMERAS.iso.up;
    const startTime = performance.now();

    const step = (now: number) => {
      const t = Math.min(1, (now - startTime) / durationMs);
      const eased = easeInOutCubic(t);
      const eye = {
        x: start.x + (target.eye.x - start.x) * eased,
        y: start.y + (target.eye.y - start.y) * eased,
        z: start.z + (target.eye.z - start.z) * eased,
      };
      const up = {
        x: startUp.x + (target.up.x - startUp.x) * eased,
        y: startUp.y + (target.up.y - startUp.y) * eased,
        z: startUp.z + (target.up.z - startUp.z) * eased,
      };
      const frameCamera = { eye, up };
      setCamera(frameCamera);
      setPlotView((current: any) => ({ ...current, camera: frameCamera }));
      if (t < 1) {
        cameraTweenRef.current = window.requestAnimationFrame(step);
      } else {
        cameraTweenRef.current = null;
        setViewRevision(`cam-settled-${Date.now()}`);
      }
    };
    cameraTweenRef.current = window.requestAnimationFrame(step);
  }, [camera]);

  const setView = useCallback((view: CameraView) => {
    userViewportTouchedRef.current = true;
    tweenCameraTo(VIEW_CAMERAS[view]);
  }, [tweenCameraTo]);

  const resetLayout = useCallback(() => setPositions(INITIAL_POSITIONS), []);

  const startPanelDrag = useCallback(
    (id: PanelId, event: PointerEvent<HTMLDivElement>) => {
      if ((event.target as HTMLElement).closest('button,input,select,textarea')) return;
      const bounds = workspaceRef.current?.getBoundingClientRect();
      if (!bounds) return;

      const startPosition = positions[id];
      const offsetX = event.clientX - bounds.left - startPosition.x;
      const offsetY = event.clientY - bounds.top - startPosition.y;
      document.body.style.userSelect = 'none';

      const move = (moveEvent: globalThis.PointerEvent) => {
        const maxX = Math.max(4, bounds.width - startPosition.w - 6);
        const maxY = Math.max(4, bounds.height - 60);
        setPositions((current) => ({
          ...current,
          [id]: {
            ...current[id],
            x: Math.min(maxX, Math.max(4, moveEvent.clientX - bounds.left - offsetX)),
            y: Math.min(maxY, Math.max(4, moveEvent.clientY - bounds.top - offsetY)),
          },
        }));
      };

      const up = () => {
        document.removeEventListener('pointermove', move);
        document.removeEventListener('pointerup', up);
        document.body.style.userSelect = '';
        dragCleanupRef.current = null;
      };

      dragCleanupRef.current = up;
      document.addEventListener('pointermove', move);
      document.addEventListener('pointerup', up);
    },
    [positions]
  );

  const runAnalysisCommand = useCallback(() => {
    const command = cmdInput.trim();
    if (!command) return;

    commandRequestRef.current?.abort();
    const controller = new AbortController();
    commandRequestRef.current = controller;

    setCommandBusy(true);
    setCommandError('');
    setTracePoint(null);
    setHoveredId(null);

    setCommandHistory((previous) => [...previous.slice(-19), command]);
    setCmdInput('');

    axios
      .post(
        API_URL,
        {
          expressions: expressions.map((entry) => entry.text),
          expression_names: expressions.map((entry) => entry.name),
          t_val: tVal,
          commands: [command],
          include_plots: false,
          quality,
          coordinate_system: plotMode,
          complex_view: complexView,
          angle_unit: angleUnit,
          axis,
        },
        { signal: controller.signal as any }
      )
      .then((response) => {
        setCommandResults(response.data.command_results || []);
        setCommandPlots(response.data.command_plots || []);
        const usedIndices: number[] = response.data.command_selection || [];
        setSelectedIds(usedIndices.map((i: number) => expressions[i]?.id).filter((id: number | undefined): id is number => id !== undefined));
        if (response.data.error) setCommandError(String(response.data.error));
      })
      .catch((error) => {
        if (axios.isCancel(error) || error?.code === 'ERR_CANCELED') return;
        setCommandError(error?.response?.data?.detail || error?.message || 'Analysis request failed.');
      })
      .finally(() => setCommandBusy(false));
  }, [cmdInput, expressions, tVal, quality, plotMode, complexView, angleUnit, axis]);

  const runQuickCommand = useCallback((command: string) => {
    if (!command) return;
    commandRequestRef.current?.abort();
    const controller = new AbortController();
    commandRequestRef.current = controller;
    setCommandBusy(true);
    setCommandError('');
    setTracePoint(null);
    setHoveredId(null);
    setCommandHistory((previous) => [...previous.slice(-19), command]);

    axios.post(API_URL, {
      expressions: expressions.map((entry) => entry.text),
      expression_names: expressions.map((entry) => entry.name),
      t_val: tVal,
      commands: [command],
      include_plots: false,
      quality,
      coordinate_system: plotMode,
      complex_view: complexView,
      angle_unit: angleUnit,
      axis,
    }, { signal: controller.signal as any })
      .then((response) => {
        setCommandResults(response.data.command_results || []);
        setCommandPlots(response.data.command_plots || []);
        const usedIndices: number[] = response.data.command_selection || [];
        setSelectedIds(usedIndices.map((i: number) => expressions[i]?.id).filter((id: number | undefined): id is number => id !== undefined));
        if (response.data.error) setCommandError(String(response.data.error));
      })
      .catch((error) => {
        if (axios.isCancel(error) || error?.code === 'ERR_CANCELED') return;
        setCommandError(error?.response?.data?.detail || error?.message || 'Analysis request failed.');
      })
      .finally(() => setCommandBusy(false));
  }, [angleUnit, axis, complexView, expressions, plotMode, quality, tVal]);

  const clearAnalysis = useCallback(() => {
    commandRequestRef.current?.abort();
    setCommandHistory([]);
    setCommandResults([]);
    setCommandPlots([]);
    setCommandError('');
    setCmdInput('');
    setCommandBusy(false);
    setSelectedIds([]);
  }, []);

  // Coordinate-form conversion is opt-in: it used to run automatically (and
  // expensively) for every curve on every plot request. Now it's fetched once,
  // on demand, the first time a person opens "show changed coordinates" for a
  // given expression, and cached client-side after that (the backend caches it
  // too, so re-opening after a page reload is still fast).
  const fetchConversion = useCallback((expression: ExprState) => {
    if (conversionResults[expression.id] || conversionLoading[expression.id]) return;
    setConversionLoading((current) => ({ ...current, [expression.id]: true }));
    setConversionErrors((current) => ({ ...current, [expression.id]: '' }));
    axios.post(API_URL, {
      expressions: expressions.map((entry) => entry.text),
      expression_names: expressions.map((entry) => entry.name),
      t_val: tVal,
      commands: [`convert(${expression.name})`],
      include_plots: false,
      quality,
      coordinate_system: plotMode,
      complex_view: complexView,
      angle_unit: angleUnit,
      axis,
    })
      .then((response) => {
        if (response.data.error) {
          setConversionErrors((current) => ({ ...current, [expression.id]: String(response.data.error) }));
        }
        setConversionResults((current) => ({ ...current, [expression.id]: response.data.command_results || [] }));
      })
      .catch((error) => {
        setConversionErrors((current) => ({ ...current, [expression.id]: error?.response?.data?.detail || error?.message || 'Conversion request failed.' }));
      })
      .finally(() => setConversionLoading((current) => ({ ...current, [expression.id]: false })));
  }, [angleUnit, axis, complexView, conversionLoading, conversionResults, expressions, plotMode, quality, tVal]);

  const toggleConversion = useCallback((expression: ExprState) => {
    setOpenConversions((current) => {
      const next = !current[expression.id];
      if (next) fetchConversion(expression);
      return { ...current, [expression.id]: next };
    });
  }, [fetchConversion]);

  const baseCommandPlots = commandPlots;
  const coordinate3D =
    plotMode === 'cartesian3d'
      ? [axis.x_label, axis.y_label, axis.z_label]
      : plotMode === 'cylindrical3d'
      ? [`θ (${angleUnit === 'degrees' ? '°' : 'rad'})`, 'R', 'Z']
      : plotMode === 'spherical3d'
      ? [`θ (${angleUnit === 'degrees' ? '°' : 'rad'})`, `φ (${angleUnit === 'degrees' ? '°' : 'rad'})`, 'ρ']
      : ['Re(z)', 'Im(z)', '|f(z)|'];

  const polarPoint = useCallback(
    (x: number, y: number) => {
      const r = Math.hypot(x, y);
      let theta = Math.atan2(y, x);
      if (theta < 0) theta += 2 * Math.PI;
      return {
        theta: angleUnit === 'degrees' ? (theta * 180) / Math.PI : theta,
        r,
      };
    },
    [angleUnit]
  );

  const transform3DPoint = useCallback(
    (point: any) => {
      const x = Number(point.x ?? 0);
      const y = Number(point.y ?? 0);
      const z = Number(point.z ?? 0);

      if (plotMode === 'cylindrical3d') {
        const polar = polarPoint(x, y);
        return { x: polar.theta, y: polar.r, z };
      }

      if (plotMode === 'spherical3d') {
        const rho = Math.hypot(x, y, z);
        let theta = Math.atan2(y, x);
        if (theta < 0) theta += 2 * Math.PI;
        const phi = rho === 0 ? 0 : Math.acos(Math.max(-1, Math.min(1, z / rho)));
        return {
          x: angleUnit === 'degrees' ? (theta * 180) / Math.PI : theta,
          y: angleUnit === 'degrees' ? (phi * 180) / Math.PI : phi,
          z: rho,
        };
      }

      return { x, y, z };
    },
    [angleUnit, plotMode, polarPoint]
  );

  const expressionColors = useMemo(() => {
    const used = new Set<number>();
    const map: Record<number, string> = {};
    expressions.forEach((entry, index) => {
      let seed = 0;
      for (let i = 0; i < entry.name.length; i += 1) seed = (seed * 31 + entry.name.charCodeAt(i)) >>> 0;
      seed = (seed + entry.id + index * 13) % COLORS.length;
      let chosen = seed;
      for (let step = 0; step < COLORS.length; step += 1) {
        const candidate = (seed + step) % COLORS.length;
        if (!used.has(candidate) || step === COLORS.length - 1) { chosen = candidate; break; }
      }
      used.add(chosen);
      map[entry.id] = COLORS[chosen];
    });
    return map;
  }, [expressions]);

  const traces = useMemo(() => {
    const nextTraces: any[] = [];
    const owners: number[] = [];

    (plotData?.plots || []).forEach((plot: any, index: number) => {
      const expression = expressions[index];
      if (!expression || !expression.visible || plot.empty || plot.error) return;

      const color = expressionColors[expression.id] || COLORS[index % COLORS.length];
      const active = hoveredId === expression.id;
      const dimmed = hoveredId !== null && !active;
      const width = active ? 5 : 3.2;
      // Selected = used by the last-run analysis command, OR currently clicked
      // during an active "pick on graph" flow — both get the same halo treatment
      // so it's visually obvious what's about to be (or was just) computed on.
      const selected = selectedIds.includes(expression.id) || (!!pickMode && pickedOwners.includes(index));
      const haloWidth = width + 6;
      const pushHalo = (haloTrace: any) => {
        if (!selected) return;
        nextTraces.push({ ...haloTrace, showlegend: false, hoverinfo: 'skip', name: `${expression.name} (selected)` });
        owners.push(-1);
      };

      if (plot.type === 'complex_geometry2d') {
        pushHalo({ x: plot.x, y: plot.y, z: plot.z, type: 'contour', contours: { start: 0, end: 0, size: 1, coloring: 'lines' }, line: { color: '#fbbf24', width: haloWidth }, opacity: 0.55, showscale: false });
        nextTraces.push({
          x: plot.x,
          y: plot.y,
          z: plot.z,
          type: 'contour',
          contours: { start: 0, end: 0, size: 1, coloring: 'lines' },
          line: { color, width },
          opacity: dimmed ? 0.18 : 1,
          showscale: false,
          name: expression.name,
          hovertemplate: `${expression.name}<br>Re(z)=%{x:.5g}<br>Im(z)=%{y:.5g}<extra></extra>`,
        });
        owners.push(index);
        return;
      }

      if (plot.type === 'complex_heatmap') {
        nextTraces.push({
          x: plot.x,
          y: plot.y,
          z: plot.z,
          type: 'heatmap',
          zsmooth: 'best',
          colorscale:
            complexView === 'phase'
              ? 'HSV'
              : [
                  [0, '#172554'],
                  [0.22, '#2563eb'],
                  [0.5, '#22d3ee'],
                  [0.72, '#facc15'],
                  [1, '#ef4444'],
                ],
          colorbar: { title: complexView },
          name: expression.name,
          opacity: dimmed ? 0.45 : 1,
          hovertemplate: `${expression.name}<br>Re(z)=%{x:.5g}<br>Im(z)=%{y:.5g}<br>${complexView}=%{z:.6g}<extra></extra>`,
        });
        owners.push(index);
        return;
      }

      if (plot.type === 'polar2d') {
        pushHalo({ r: plot.r, theta: plot.theta, type: 'scatterpolar', mode: 'lines', line: { color: '#fbbf24', width: haloWidth }, opacity: 0.55 });
        nextTraces.push({
          r: plot.r,
          theta: plot.theta,
          type: 'scatterpolar',
          mode: 'lines',
          line: { color, width },
          opacity: dimmed ? 0.18 : 1,
          name: expression.name,
          hovertemplate: `${expression.name}<br>r=%{r:.5g}<br>θ=%{theta:.5g}<extra></extra>`,
        });
        owners.push(index);
        return;
      }

      if (plot.type === 'surface3d') {
        nextTraces.push({
          x: plot.x,
          y: plot.y,
          z: plot.z,
          type: 'surface',
          colorscale: selected ? [[0, '#fbbf24'], [1, '#fde68a']] : 'Viridis',
          showscale: false,
          opacity: selected ? 1 : (dimmed ? 0.16 : 0.96),
          lighting: { ambient: 0.55, diffuse: 0.78, specular: 0.28, roughness: 0.42, fresnel: 0.08 },
          lightposition: { x: 1.5, y: -1.2, z: 2.8 },
          name: expression.name,
          hovertemplate: `${expression.name}<br>${coordinate3D[0]}=%{x:.5g}<br>${coordinate3D[1]}=%{y:.5g}<br>${coordinate3D[2]}=%{z:.5g}<extra></extra>`,
        });
        owners.push(index);
        return;
      }

      if (plot.type === 'isosurface3d') {
        nextTraces.push({
          x: plot.x,
          y: plot.y,
          z: plot.z,
          value: plot.val,
          type: 'isosurface',
          isomin: 0,
          isomax: 0,
          surface: { show: true },
          colorscale: selected ? [[0, '#fbbf24'], [1, '#fde68a']] : 'Viridis',
          showscale: false,
          opacity: selected ? 0.9 : (dimmed ? 0.10 : 0.72),
          name: expression.name,
          hovertemplate: `${expression.name}<br>${coordinate3D[0]}=%{x:.5g}<br>${coordinate3D[1]}=%{y:.5g}<br>${coordinate3D[2]}=%{z:.5g}<extra></extra>`,
        });
        owners.push(index);
        return;
      }

      if (plot.type === 'line3d') {
        pushHalo({ x: plot.x, y: plot.y, z: plot.z, type: 'scatter3d', mode: 'lines', line: { color: '#fbbf24', width: haloWidth }, opacity: 0.55 });
        nextTraces.push({
          x: plot.x,
          y: plot.y,
          z: plot.z,
          type: 'scatter3d',
          mode: 'lines',
          line: { color, width },
          opacity: dimmed ? 0.18 : 1,
          name: expression.name,
          hovertemplate: `${expression.name}<br>${coordinate3D[0]}=%{x:.5g}<br>${coordinate3D[1]}=%{y:.5g}<br>${coordinate3D[2]}=%{z:.5g}<extra></extra>`,
        });
        owners.push(index);
        return;
      }

      if (plot.type === 'contour2d') {
        pushHalo({ x: plot.x, y: plot.y, z: plot.z, type: 'contour', contours: { start: 0, end: 0, size: 1, coloring: 'lines' }, line: { color: '#fbbf24', width: haloWidth }, opacity: 0.55, showscale: false });
        nextTraces.push({
          x: plot.x,
          y: plot.y,
          z: plot.z,
          type: 'contour',
          contours: { start: 0, end: 0, size: 1, coloring: 'lines' },
          line: { color, width },
          opacity: dimmed ? 0.18 : 1,
          showscale: false,
          name: expression.name,
          hovertemplate: `${expression.name}<br>x=%{x:.5g}<br>y=%{y:.5g}<extra></extra>`,
        });
        owners.push(index);
        return;
      }

      pushHalo({ x: plot.x, y: plot.y, type: 'scatter', mode: 'lines', line: { color: '#fbbf24', width: haloWidth, shape: 'linear' }, opacity: 0.55 });
      nextTraces.push({
        x: plot.x,
        y: plot.y,
        type: 'scatter',
        mode: 'lines',
        line: { color, width, shape: 'linear' },
        opacity: dimmed ? 0.18 : 1,
        name: expression.name,
        hovertemplate: `${expression.name}<br>x=%{x:.5g}<br>y=%{y:.5g}<extra></extra>`,
      });
      owners.push(index);
    });

    const specialPoints = (plotData?.special_points || []).filter(
      (point: any) => expressions[point.owner]?.visible
    );

    if (specialPoints.length && traceEnabled) {
      if (plotMode === 'polar2d') {
        const converted = specialPoints.map((point: any) => polarPoint(Number(point.x), Number(point.y)));
        nextTraces.push({
          r: converted.map((point) => point.r),
          theta: converted.map((point) => point.theta),
          type: 'scatterpolar',
          mode: 'markers+text',
          marker: { size: 9, color: '#fff', line: { width: 2.4, color: '#0f172a' } },
          text: specialPoints.map((point: any) => point.label),
          textposition: 'top center',
          name: 'Axis intercepts',
          showlegend: false,
          hovertemplate: specialPoints.map(
            (point: any, index: number) =>
              `${point.label}<br>r=${converted[index].r.toPrecision(6)}<br>θ=${converted[index].theta.toPrecision(6)}<extra></extra>`
          ),
        });
        owners.push(-1);
      } else if (is3D) {
        const converted = specialPoints.map(transform3DPoint);
        nextTraces.push({
          x: converted.map((point: any) => point.x),
          y: converted.map((point: any) => point.y),
          z: converted.map((point: any) => point.z),
          type: 'scatter3d',
          mode: 'markers+text',
          marker: { size: 6, color: '#fff', line: { width: 2, color: '#0f172a' } },
          text: specialPoints.map((point: any) => point.label),
          textposition: 'top center',
          name: 'Axis intercepts',
          showlegend: false,
        });
        owners.push(-1);
      } else if (plotMode !== 'complex') {
        nextTraces.push({
          x: specialPoints.map((point: any) => point.x),
          y: specialPoints.map((point: any) => point.y),
          type: 'scatter',
          mode: 'markers+text',
          marker: { size: 9, color: '#fff', line: { width: 2.4, color: '#0f172a' } },
          text: specialPoints.map((point: any) => point.label),
          textposition: 'top center',
          name: 'Axis intercepts',
          showlegend: false,
        });
        owners.push(-1);
      }
    }

    baseCommandPlots.forEach((commandPlot: any) => {
      if (plotMode === 'polar2d') {
        const converted = (commandPlot.x || []).map((x: number, index: number) =>
          polarPoint(Number(x), Number(commandPlot.y?.[index] ?? 0))
        );
        nextTraces.push({
          r: converted.map((point) => point.r),
          theta: converted.map((point) => point.theta),
          type: 'scatterpolar',
          mode: commandPlot.mode || 'markers',
          marker: commandPlot.marker,
          line: commandPlot.line,
          fill: commandPlot.fill,
          fillcolor: commandPlot.fillcolor,
          text: commandPlot.text,
          textposition: 'top center',
          name: commandPlot.name || 'Analysis result',
          showlegend: true,
        });
        owners.push(-1);
        return;
      }

      if (is3D) {
        const converted = (commandPlot.x || []).map((x: number, index: number) =>
          transform3DPoint({ x, y: commandPlot.y?.[index], z: commandPlot.z?.[index] })
        );
        nextTraces.push({
          x: converted.map((point: any) => point.x),
          y: converted.map((point: any) => point.y),
          z: converted.map((point: any) => point.z),
          type: 'scatter3d',
          mode: commandPlot.mode || 'markers',
          marker: commandPlot.marker,
          line: commandPlot.line,
          fill: commandPlot.fill,
          fillcolor: commandPlot.fillcolor,
          text: commandPlot.text,
          textposition: 'top center',
          name: commandPlot.name || 'Analysis result',
          showlegend: true,
        });
        owners.push(-1);
        return;
      }

      nextTraces.push({
        x: commandPlot.x,
        y: commandPlot.y,
        type: 'scatter',
        mode: commandPlot.mode || 'markers',
        marker: commandPlot.marker,
        line: commandPlot.line,
        fill: commandPlot.fill,
        fillcolor: commandPlot.fillcolor,
        text: commandPlot.text,
        textposition: 'top center',
        name: commandPlot.name || 'Analysis result',
        showlegend: true,
      });
      owners.push(-1);
    });

    traceOwnerRef.current = owners;

    if (traceEnabled && tracePoint) {
      if (plotMode === 'polar2d' && tracePoint.x !== undefined && tracePoint.y !== undefined) {
        nextTraces.push({
          r: [tracePoint.y],
          theta: [tracePoint.x],
          type: 'scatterpolar',
          mode: 'markers',
          marker: { size: 12, color: '#fff', line: { width: 3, color: TRACE_COLOR }, symbol: 'diamond' },
          hoverinfo: 'skip',
          showlegend: false,
          name: 'Trace',
        });
        nextTraces.push({
          r: [0, tracePoint.y],
          theta: [tracePoint.x, tracePoint.x],
          type: 'scatterpolar',
          mode: 'lines',
          line: { color: TRACE_COLOR, width: 2, dash: 'dot' },
          hoverinfo: 'skip',
          showlegend: false,
          name: 'Trace guide',
        });
      } else if (
        is3D &&
        tracePoint.x !== undefined &&
        tracePoint.y !== undefined &&
        tracePoint.z !== undefined
      ) {
        nextTraces.push({
          x: [tracePoint.x],
          y: [tracePoint.y],
          z: [tracePoint.z],
          type: 'scatter3d',
          mode: 'markers',
          marker: { size: 10, color: '#fff', line: { width: 3, color: TRACE_COLOR }, symbol: 'diamond' },
          hoverinfo: 'skip',
          showlegend: false,
          name: 'Trace',
        });
      } else if (tracePoint.x !== undefined && tracePoint.y !== undefined) {
        nextTraces.push({
          x: [tracePoint.x],
          y: [tracePoint.y],
          type: 'scatter',
          mode: 'markers',
          marker: { size: 11, color: '#fff', line: { width: 3, color: TRACE_COLOR }, symbol: 'diamond' },
          hoverinfo: 'skip',
          showlegend: false,
          name: 'Trace',
        });
        nextTraces.push({
          x: [tracePoint.x, tracePoint.x],
          y: [0, tracePoint.y],
          type: 'scatter',
          mode: 'lines',
          line: { color: TRACE_COLOR, width: 2, dash: 'dot' },
          hoverinfo: 'skip',
          showlegend: false,
        });
        nextTraces.push({
          x: [0, tracePoint.x],
          y: [tracePoint.y, tracePoint.y],
          type: 'scatter',
          mode: 'lines',
          line: { color: TRACE_COLOR, width: 2, dash: 'dot' },
          hoverinfo: 'skip',
          showlegend: false,
        });
      }
    }

    return nextTraces;
  }, [
    baseCommandPlots,
    complexView,
    expressions,
    hoveredId,
    is3D,
    plotData,
    plotMode,
    polarPoint,
    traceEnabled,
    tracePoint,
    transform3DPoint,
    coordinate3D,
    expressionColors,
    selectedIds,
    pickMode,
    pickedOwners,
  ]);

  const saveTraceFromPoint = useCallback(
    (point: any) => {
      if (!point) return;
      const curveNumber = point.curveNumber;
      const owner = typeof curveNumber === 'number' ? traceOwnerRef.current[curveNumber] : undefined;
      if (owner === undefined || owner < 0) return;
      const expression = expressions[owner];
      if (!expression) return;

      if (pickMode) {
        const needed = pickMode === 'intersection' || pickMode === 'area-between' || pickMode === 'area-region' || pickMode === 'distance' ? 2 : 1;
        const nextOwners = pickedOwners.includes(owner) ? pickedOwners : [...pickedOwners, owner];
        setPickedOwners(nextOwners.slice(0, needed));
        if (nextOwners.length >= needed) {
          const names = nextOwners.slice(0, needed).map((i) => expressions[i]?.name).filter(Boolean) as string[];
          let command = '';
          if (needed === 2) {
            if (pickMode === 'intersection') command = `intersection(${names[0]}, ${names[1]})`;
            if (pickMode === 'area-between') command = plotMode === 'polar2d' ? `area_between(${names[0]}, ${names[1]}, ${axis.theta_min}, ${axis.theta_max})` : `area_between(${names[0]}, ${names[1]}, ${axis.x_min}, ${axis.x_max})`;
            if (pickMode === 'area-region') command = `area_region(${names[0]}, ${names[1]})`;
            if (pickMode === 'distance') command = `distance(${names[0]}, ${names[1]})`;
          } else {
            if (pickMode === 'area-under') command = plotMode === 'polar2d' ? `polar_area(${names[0]}, ${axis.theta_min}, ${axis.theta_max})` : `area_under(${names[0]}, ${axis.x_min}, ${axis.x_max})`;
            if (pickMode === 'area-closed') command = `area_closed(${names[0]})`;
            if (pickMode === 'arc-length') command = `arc_length(${names[0]}, ${axis.x_min}, ${axis.x_max})`;
            if (pickMode === 'polar-area') command = `polar_area(${names[0]}, ${axis.theta_min}, ${axis.theta_max})`;
            if (pickMode === 'polar-arc-length') command = `polar_arc_length(${names[0]}, ${axis.theta_min}, ${axis.theta_max})`;
            if (pickMode === 'volume') command = `volume_under(${names[0]})`;
            if (pickMode === 'surface-area') command = `surface_area(${names[0]})`;
          }
          if (command) runQuickCommand(command);
          setPickMode(null);
          setPickedOwners([]);
        }
        return;
      }

      if (!traceEnabled) return;
      if (plotMode === 'polar2d' && point.r !== undefined && point.theta !== undefined) {
        setTracePoint({ owner, name: expression.name, x: Number(point.theta), y: Number(point.r) });
        return;
      }
      setTracePoint({ owner, name: expression.name, x: Number.isFinite(Number(point.x)) ? Number(point.x) : undefined, y: Number.isFinite(Number(point.y)) ? Number(point.y) : undefined, z: Number.isFinite(Number(point.z)) ? Number(point.z) : undefined });
    },
    [axis, expressions, pickMode, pickedOwners, plotMode, runQuickCommand, traceEnabled]
  );

  const chartBg = isDarkMode ? '#0b1220' : '#ffffff';
  const paperBg = isDarkMode ? '#0f172a' : '#f8fafc';
  const grid = isDarkMode ? '#334155' : '#dbe4ee';
  const zero = isDarkMode ? '#cbd5e1' : '#334155';
  const text = isDarkMode ? '#e5edf7' : '#0f172a';
  const muted = isDarkMode ? '#94a3b8' : '#64748b';

  const zoomRange = useCallback((lo: number, hi: number, scale: number) => {
    const center = (lo + hi) / 2;
    const half = Math.max((hi - lo) / 2, 1e-9) * scale;
    return [center - half, center + half];
  }, []);

  const layout = useMemo(() => {
    const next: any = {
      autosize: true,
      margin: { l: 64, r: 28, b: 58, t: 66 },
      paper_bgcolor: paperBg,
      plot_bgcolor: chartBg,
      font: { color: text, family: 'Inter,ui-sans-serif,system-ui,sans-serif', size: 13 },
      uirevision: `graphlab-view-${viewRevision}`,
      hovermode: 'closest',
      dragmode: 'pan',
      showlegend: true,
      legend: {
        orientation: 'h',
        yanchor: 'bottom',
        y: 1.02,
        xanchor: 'left',
        x: 0,
        bgcolor: isDarkMode ? 'rgba(15,23,42,.88)' : 'rgba(248,250,252,.94)',
        bordercolor: isDarkMode ? '#334155' : '#e2e8f0',
        borderwidth: 1,
        font: { color: text, size: 12 },
      },
      hoverlabel: {
        bgcolor: isDarkMode ? '#111827' : '#fff',
        bordercolor: grid,
        font: { color: text, size: 13 },
      },
    };

    if (plotMode === 'polar2d') {
      next.polar = {
        bgcolor: chartBg,
        radialaxis: {
          range: plotView.polarRadialRange || (plotData?.suggested_view?.r ? plotData.suggested_view.r : [0, (Math.abs(axis.r_max) || 20) * mouseZoom]),
          showgrid: axis.show_grid,
          gridcolor: grid,
          showline: true,
          linecolor: grid,
          zeroline: axis.show_zero,
          zerolinecolor: zero,
          tickfont: { color: muted },
          title: { text: 'r', font: { color: text } },
        },
        angularaxis: {
          showgrid: axis.show_grid,
          gridcolor: grid,
          showline: true,
          linecolor: grid,
          direction: axis.theta_direction,
          rotation: axis.theta_rotation,
          tickfont: { color: muted },
          title: { text: `θ (${angleUnit === 'degrees' ? '°' : 'rad'})`, font: { color: text } },
        },
      };
    } else if (is3D) {
      const sceneRanges: any =
        plotMode === 'cartesian3d'
          ? { x: plotData?.suggested_view?.x || zoomRange(axis.x_min, axis.x_max, mouseZoom), y: plotData?.suggested_view?.y || zoomRange(axis.y_min, axis.y_max, mouseZoom), z: plotData?.suggested_view?.z || zoomRange(axis.z_min, axis.z_max, mouseZoom) }
          : plotMode === 'cylindrical3d'
          ? { x: [axis.theta_min, axis.theta_max], y: plotData?.suggested_view?.y || [0, (Math.abs(axis.r_max) || 20) * mouseZoom], z: plotData?.suggested_view?.z || zoomRange(axis.z_min, axis.z_max, mouseZoom) }
          : plotMode === 'spherical3d'
          ? { x: [axis.theta_min, axis.theta_max], y: [axis.phi_min, axis.phi_max], z: plotData?.suggested_view?.z || [0, (Math.abs(axis.r_max) || 20) * mouseZoom] }
          : { x: zoomRange(axis.re_min, axis.re_max, mouseZoom), y: zoomRange(axis.im_min, axis.im_max, mouseZoom) };

      next.scene = {
        aspectmode: axis.equal_aspect ? 'data' : 'auto',
        camera: plotView.camera || camera,
        bgcolor: chartBg,
        dragmode: 'orbit',
        xaxis: {
          title: { text: coordinate3D[0], font: { color: text } },
          range: plotView.sceneXRange || sceneRanges.x,
          showbackground: true,
          backgroundcolor: isDarkMode ? '#101827' : '#f8fafc',
          showgrid: axis.show_grid,
          gridcolor: axis.show_grid ? grid : 'rgba(0,0,0,0)',
          zeroline: axis.show_zero,
          zerolinecolor: zero,
          tickfont: { color: muted },
        },
        yaxis: {
          title: { text: coordinate3D[1], font: { color: text } },
          range: plotView.sceneYRange || sceneRanges.y,
          showbackground: true,
          backgroundcolor: isDarkMode ? '#0f1a2b' : '#f1f5f9',
          showgrid: axis.show_grid,
          gridcolor: axis.show_grid ? grid : 'rgba(0,0,0,0)',
          zeroline: axis.show_zero,
          zerolinecolor: zero,
          tickfont: { color: muted },
        },
        zaxis: {
          title: { text: coordinate3D[2], font: { color: text } },
          range: plotView.sceneZRange || sceneRanges.z,
          showbackground: true,
          backgroundcolor: isDarkMode ? '#111c2e' : '#eaf0f6',
          showgrid: axis.show_grid,
          gridcolor: axis.show_grid ? grid : 'rgba(0,0,0,0)',
          zeroline: axis.show_zero,
          zerolinecolor: zero,
          tickfont: { color: muted },
        },
      };
    } else {
      next.xaxis = {
        title: { text: plotMode === 'complex' ? 'Re(z)' : axis.x_label, font: { color: text } },
        range: plotView.xRange || plotData?.suggested_view?.x || (plotMode === 'complex' ? zoomRange(axis.re_min, axis.re_max, mouseZoom) : zoomRange(axis.x_min, axis.x_max, mouseZoom)),
        type: plotMode === 'complex' ? 'linear' : axis.x_scale,
        showgrid: axis.show_grid,
        gridcolor: grid,
        showline: true,
        linecolor: grid,
        zeroline: axis.show_zero,
        zerolinecolor: zero,
        tickfont: { color: muted },
      };
      next.yaxis = {
        title: { text: plotMode === 'complex' ? 'Im(z)' : axis.y_label, font: { color: text } },
        range: plotView.yRange || plotData?.suggested_view?.y || (plotMode === 'complex' ? zoomRange(axis.im_min, axis.im_max, mouseZoom) : zoomRange(axis.y_min, axis.y_max, mouseZoom)),
        type: plotMode === 'complex' ? 'linear' : axis.y_scale,
        showgrid: axis.show_grid,
        gridcolor: grid,
        showline: true,
        linecolor: grid,
        zeroline: axis.show_zero,
        zerolinecolor: zero,
        scaleanchor: axis.equal_aspect ? 'x' : undefined,
        scaleratio: 1,
        tickfont: { color: muted },
      };
    }

    return next;
  }, [
    angleUnit,
    axis,
    camera,
    chartBg,
    coordinate3D,
    grid,
    is3D,
    isDarkMode,
    muted,
    paperBg,
    plotMode,
    text,
    viewRevision,
    workspaceTick,
    plotView,
    plotData,
    mouseZoom,
    zoomRange,
    zero,
  ]);

  const activeTraceText = tracePoint
    ? [
        tracePoint.name,
        tracePoint.x !== undefined
          ? `${plotMode === 'polar2d' ? 'θ' : is3D ? coordinate3D[0] : 'x'} = ${tracePoint.x.toPrecision(6)}`
          : null,
        tracePoint.y !== undefined
          ? `${plotMode === 'polar2d' ? 'r' : is3D ? coordinate3D[1] : 'y'} = ${tracePoint.y.toPrecision(6)}`
          : null,
        tracePoint.z !== undefined
          ? `${is3D ? coordinate3D[2] : 'z'} = ${tracePoint.z.toPrecision(6)}`
          : null,
      ]
        .filter(Boolean)
        .join('   ')
    : '';

  const shell = isDarkMode ? 'bg-slate-950 text-white' : 'bg-slate-100 text-slate-800';
  const card = isDarkMode
    ? 'bg-slate-900/96 border-slate-700 text-slate-100'
    : 'bg-white/96 border-slate-200 text-slate-800';
  const subcard = isDarkMode ? 'bg-slate-800/90 border-slate-700' : 'bg-slate-50 border-slate-200';
  const field = isDarkMode
    ? 'bg-slate-950 border-slate-700 text-white'
    : 'bg-white border-slate-200 text-slate-800';
  const filteredCommands = useMemo(() => {
    const query = commandSearch.trim().toLowerCase();
    if (!query) return COMMAND_CATALOG;
    return COMMAND_CATALOG.filter((item) => `${item.category} ${item.title} ${item.command} ${item.help}`.toLowerCase().includes(query));
  }, [commandSearch]);

  const startPick = useCallback((mode: PickMode) => {
    setPickedOwners([]);
    setPickMode(mode);
    setTracePoint(null);
    setCommandError('');
  }, []);

  const katexFallback = () => <span className="text-xs text-slate-400 italic">Preview unavailable</span>;

  return (
    <div className={`h-screen w-full overflow-hidden ${shell}`}>
      <header
        className={`h-14 flex items-center gap-2 px-4 border-b relative z-50 ${
          isDarkMode ? 'bg-slate-950/98 border-slate-800' : 'bg-white/98 border-slate-200'
        } backdrop-blur-xl`}
      >
        <button
          type="button"
          onClick={() => setPage('home')}
          className="font-black tracking-tight text-lg mr-4 hover:text-blue-500 transition"
          title="Home"
        >
          Graph<span className="text-blue-500">Lab</span>
        </button>

        <button
          type="button"
          onClick={() => setPage('home')}
          className={`px-3 py-2 rounded-lg text-sm font-semibold flex items-center gap-1.5 ${
            page === 'home' ? 'bg-blue-600 text-white' : 'hover:bg-slate-500/10'
          }`}
        >
          <Home size={15} /> Home
        </button>

        <button
          type="button"
          onClick={() => setPage('graph')}
          className={`px-3 py-2 rounded-lg text-sm font-semibold flex items-center gap-1.5 ${
            page === 'graph' ? 'bg-blue-600 text-white' : 'hover:bg-slate-500/10'
          }`}
        >
          <Grid3X3 size={15} /> Workspace
        </button>

        <div className="flex-1" />

        <div className={`hidden md:flex items-center gap-2 text-xs px-3 py-1.5 rounded-lg border ${subcard}`}>
          <span className={`w-2 h-2 rounded-full ${isLoading ? 'bg-amber-400 animate-pulse' : connError ? 'bg-red-500' : 'bg-emerald-500'}`} />
          {isLoading ? 'Calculating' : connError ? 'Server error' : 'Ready'}
        </div>

        <button
          type="button"
          onClick={() => setIsDarkMode((value) => !value)}
          className="p-2 rounded-lg hover:bg-slate-500/10"
          title="Toggle theme"
        >
          {isDarkMode ? <Sun size={19} className="text-yellow-400" /> : <Moon size={19} className="text-slate-600" />}
        </button>

        <button
          type="button"
          onClick={copyPermalink}
          className="p-2 rounded-lg hover:bg-slate-500/10"
          title="Copy graph link"
        >
          {permalinkCopied ? <CheckCircle size={18} className="text-emerald-500" /> : <Link2 size={18} />}
        </button>
      </header>

      <div ref={workspaceRef} className="relative h-[calc(100vh-56px)] overflow-hidden">
        <div className="absolute inset-0 z-0">
          {connError && (
            <div className="absolute top-4 left-4 z-20 flex items-center gap-2 bg-red-600/90 text-white text-xs px-3 py-2 rounded-lg shadow">
              <WifiOff size={14} /> Can't reach the calculation server
            </div>
          )}
          {plotError && !connError && (
            <div className="absolute top-4 left-4 z-20 max-w-[min(520px,calc(100%-32px))] bg-amber-500/95 text-white text-xs px-3 py-2 rounded-lg shadow">
              {plotError}
            </div>
          )}

          <div className="absolute top-14 right-4 z-20 flex flex-col items-end gap-2">
            <div className="flex gap-2 items-start">
              <div className={`flex items-center gap-1 p-1 rounded-xl border shadow-lg backdrop-blur ${card}`}>
                <button
                  type="button"
                  onClick={() => setPanels((current) => ({ ...current, expressions: !current.expressions }))}
                  className={`px-2 py-1 text-[10px] font-bold rounded ${panels.expressions ? 'bg-blue-600 text-white' : 'hover:bg-slate-500/10'}`}
                >
                  Expressions
                </button>
                <button
                  type="button"
                  onClick={() => setPanels((current) => ({ ...current, setup: !current.setup }))}
                  className={`px-2 py-1 text-[10px] font-bold rounded ${panels.setup ? 'bg-blue-600 text-white' : 'hover:bg-slate-500/10'}`}
                >
                  Setup
                </button>
                <button
                  type="button"
                  onClick={() => setPanels((current) => ({ ...current, analysis: !current.analysis }))}
                  className={`px-2 py-1 text-[10px] font-bold rounded ${panels.analysis ? 'bg-blue-600 text-white' : 'hover:bg-slate-500/10'}`}
                >
                  Analysis
                </button>
                <button
                  type="button"
                  onClick={resetLayout}
                  className="p-1.5 rounded hover:bg-slate-500/10 text-slate-500"
                  title="Reset panel layout"
                >
                  <RotateCcw size={14} />
                </button>
              </div>

              {is3D && (
                <button
                  type="button"
                  onClick={() => setViewCubeOpen((value) => !value)}
                  className={`p-2 rounded-xl border shadow-lg backdrop-blur ${card}`}
                  title="Toggle view cube"
                >
                  <Box size={16} />
                </button>
              )}
            </div>

            {is3D && viewCubeOpen && (
              <div className={`w-44 p-2 rounded-2xl border shadow-xl backdrop-blur ${card}`}>
                <div className="text-[9px] uppercase tracking-wider font-black text-slate-500 mb-1.5">3D View Cube</div>
                <div className="grid grid-cols-3 gap-1">
                  {(['iso', 'top', 'front', 'right', 'back', 'left', 'bottom'] as CameraView[]).map((view) => (
                    <button
                      type="button"
                      key={view}
                      onClick={() => setView(view)}
                      className="px-1.5 py-1.5 rounded-lg text-[9px] font-black bg-blue-500/10 text-blue-500 hover:bg-blue-600 hover:text-white"
                    >
                      {view.toUpperCase()}
                    </button>
                  ))}
                </div>
                <button
                  type="button"
                  onClick={recenterView}
                  className="mt-1.5 w-full px-2 py-1.5 rounded-lg text-[9px] font-black hover:bg-slate-500/10 flex items-center justify-center gap-1"
                >
                  <Maximize2 size={12} /> Fit
                </button>
              </div>
            )}
          </div>

          {tracePoint && (
            <div className={`absolute bottom-12 left-4 z-20 px-3 py-2 rounded-xl border shadow-lg text-[11px] backdrop-blur-md ${card}`}>
              <div className="font-bold text-pink-500 flex items-center gap-1">
                <Crosshair size={12} /> Trace point
              </div>
              <div className="font-mono whitespace-nowrap mt-0.5">{activeTraceText}</div>
              <button
                type="button"
                onClick={() => setTracePoint(null)}
                className="mt-1 text-slate-400 hover:text-red-500 text-[10px]"
              >
                Clear trace
              </button>
            </div>
          )}

          <div className={`absolute bottom-3 left-1/2 -translate-x-1/2 z-20 flex items-center gap-1 p-1 rounded-xl border shadow-lg backdrop-blur ${card}`}>
            <button
              type="button"
              onClick={() => setTraceEnabled((enabled) => {
                const next = !enabled;
                if (!next) setTracePoint(null);
                return next;
              })}
              className={`px-2.5 py-1.5 rounded-lg text-[10px] font-black flex items-center gap-1 ${traceEnabled ? 'bg-pink-500 text-white' : 'hover:bg-slate-500/10 text-slate-500'}`}
            >
              <MousePointer2 size={12} /> Trace
            </button>
            <button
              type="button"
              onClick={() => setTracePoint(null)}
              className="px-2.5 py-1.5 rounded-lg text-[10px] font-black hover:bg-slate-500/10 text-slate-500 flex items-center gap-1"
            >
              <Eraser size={12} /> Clear
            </button>
            <button
              type="button"
              onClick={recenterView}
              className="p-1.5 rounded-lg hover:bg-slate-500/10 text-slate-500"
              title="Fit graph"
            >
              <Maximize2 size={14} />
            </button>
          </div>

          <div className="absolute bottom-3 left-[calc(50%+170px)] z-20 hidden lg:block px-2.5 py-1.5 rounded-lg border text-[9px] font-semibold shadow backdrop-blur-md bg-white/80 dark:bg-slate-900/80 text-slate-500">Wheel: zoom · drag: pan · click: trace</div>

          <div className="absolute bottom-3 right-3 z-20 px-2.5 py-1.5 rounded-lg border text-[10px] shadow backdrop-blur-md bg-white/80 dark:bg-slate-900/80">
            <span className="font-bold">{plotModeLabel[plotMode]}</span>
            {plotData?.special_points?.length ? ` · ${plotData.special_points.length} axis points` : ''}
            {commandPlots.length ? ` · ${commandPlots.length} analysis traces` : ''}
          </div>

          {!hasAnyContent && (
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-10">
              <div className={`text-center ${isDarkMode ? 'text-slate-600' : 'text-slate-300'}`}>
                <p className="text-xl font-semibold">Nothing plotted yet</p>
                <p className="text-sm">Open Expressions and enter an equation.</p>
              </div>
            </div>
          )}

          <div className="absolute inset-0">
          <Plot
            data={traces}
            layout={layout}
            useResizeHandler
            style={{ width: '100%', height: '100%' }}
            config={{
              responsive: true,
              scrollZoom: true,
              displaylogo: false,
              modeBarButtonsToRemove: ['lasso2d', 'select2d'],
            }}
            onClick={(event: any) => saveTraceFromPoint(event?.points?.[0])}
            onRelayout={(event: any) => {
              userViewportTouchedRef.current = true;
              setPlotView((current: any) => {
                const next = { ...current };
                if (event?.['xaxis.range[0]'] !== undefined && event?.['xaxis.range[1]'] !== undefined) next.xRange = [event['xaxis.range[0]'], event['xaxis.range[1]']];
                if (event?.['yaxis.range[0]'] !== undefined && event?.['yaxis.range[1]'] !== undefined) next.yRange = [event['yaxis.range[0]'], event['yaxis.range[1]']];
                if (event?.['polar.radialaxis.range[0]'] !== undefined && event?.['polar.radialaxis.range[1]'] !== undefined) next.polarRadialRange = [event['polar.radialaxis.range[0]'], event['polar.radialaxis.range[1]']];
                if (event?.['scene.camera']) next.camera = event['scene.camera'];
                if (event?.['scene.xaxis.range[0]'] !== undefined && event?.['scene.xaxis.range[1]'] !== undefined) next.sceneXRange = [event['scene.xaxis.range[0]'], event['scene.xaxis.range[1]']];
                if (event?.['scene.yaxis.range[0]'] !== undefined && event?.['scene.yaxis.range[1]'] !== undefined) next.sceneYRange = [event['scene.yaxis.range[0]'], event['scene.yaxis.range[1]']];
                if (event?.['scene.zaxis.range[0]'] !== undefined && event?.['scene.zaxis.range[1]'] !== undefined) next.sceneZRange = [event['scene.zaxis.range[0]'], event['scene.zaxis.range[1]']];
                return next;
              });
            }}
            onHover={(event: any) => {
              const curveNumber = event?.points?.[0]?.curveNumber;
              if (curveNumber === undefined) return;
              const owner = traceOwnerRef.current[curveNumber];
              if (owner !== undefined && owner >= 0) setHoveredId(expressions[owner]?.id ?? null);
            }}
            onUnhover={() => setHoveredId(null)}
          />
          </div>
        </div>

        <WorkspacePanel
          id="expressions"
          title="Expressions"
          icon={<Grid3X3 size={16} />}
          open={panels.expressions}
          position={positions.expressions}
          dark={isDarkMode}
          onClose={() => setPanels((current) => ({ ...current, expressions: false }))}
          onDragStart={startPanelDrag}
        >
          <div className="space-y-2">
            {hasTime && (
              <div className={`p-3 rounded-xl border ${subcard}`}>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-[10px] uppercase font-bold text-blue-500">Simulation t = {tVal}</span>
                  <div className="flex gap-1 items-center">
                    <button type="button" onClick={() => setIsPlaying((value) => !value)} className="p-1 rounded hover:bg-slate-500/15">
                      {isPlaying ? <Pause size={14} /> : <Play size={14} />}
                    </button>
                    <select value={playSpeed} onChange={(e) => setPlaySpeed(Number(e.target.value))} className={`text-[10px] rounded px-1 py-0.5 ${field}`}>
                      <option value={0.5}>0.5×</option>
                      <option value={1}>1×</option>
                      <option value={2}>2×</option>
                      <option value={4}>4×</option>
                    </select>
                  </div>
                </div>
                <input
                  type="range"
                  min="0"
                  max="10"
                  step="0.1"
                  value={tVal}
                  onChange={(e) => {
                    setIsPlaying(false);
                    setTVal(Number(e.target.value));
                  }}
                  className="w-full"
                />
              </div>
            )}

            {expressions.map((expression, index) => {
              const color = expressionColors[expression.id] || COLORS[index % COLORS.length];
              const active = hoveredId === expression.id;
              return (
                <div
                  key={expression.id}
                  onMouseEnter={() => setHoveredId(expression.id)}
                  onMouseLeave={() => setHoveredId(null)}
                  className={`p-3 rounded-xl border transition ${subcard} ${active ? 'ring-2 ring-blue-400/40' : ''}`}
                  style={{ borderLeftWidth: 5, borderLeftColor: color }}
                >
                  <div className="flex items-center gap-2 mb-2">
                    <button
                      type="button"
                      onClick={() => updateExpr(index, 'visible', !expression.visible)}
                      title={expression.visible ? 'Hide expression' : 'Show expression'}
                    >
                      {expression.visible ? <Eye size={17} color={color} /> : <EyeOff size={17} className="text-slate-500" />}
                    </button>
                    <input
                      value={expression.name}
                      onChange={(e) => updateExpr(index, 'name', e.target.value)}
                      className="w-28 bg-transparent outline-none font-bold text-xs"
                      style={{ color }}
                      aria-label="Equation name"
                    />
                    <button
                      type="button"
                      className="ml-auto text-slate-400 hover:text-red-500"
                      onClick={() => removeExpression(index)}
                      title="Delete expression"
                    >
                      <Trash2 size={16} />
                    </button>
                  </div>

                  <input
                    ref={(element) => {
                      inputRefs.current[expression.id] = element;
                    }}
                    value={expression.text}
                    onChange={(e) => updateExpr(index, 'text', e.target.value)}
                    onPaste={(e) => handlePaste(index, e)}
                    placeholder={plotMode === 'complex' ? 'e.g. exp(z), sin(z), z^2' : 'e.g. y = sin(x)'}
                    className={`w-full p-2 rounded-lg border outline-none font-mono text-xs ${field} ${!expression.visible ? 'opacity-50' : ''}`}
                    autoComplete="off"
                    spellCheck={false}
                  />

                  {plotData?.plots?.[index] && !plotData.plots[index].empty && (
                    <div className="mt-2 pt-2 border-t border-slate-500/10 flex items-center">
                      <div className="text-sm overflow-x-auto flex-1">
                        <BlockMath
                          math={plotData.plots[index].latex_eq || ''}
                          errorColor="#ef4444"
                          renderError={katexFallback}
                        />
                      </div>
                      <button
                        type="button"
                        onClick={() => copyToClipboard(plotData.plots[index].latex_eq || '', index)}
                        className="p-1 text-slate-400 hover:text-blue-500"
                        title="Copy equation"
                      >
                        {copiedIndex === index ? <CheckCircle size={15} className="text-emerald-500" /> : <Copy size={15} />}
                      </button>
                    </div>
                  )}

                  {plotData?.plots?.[index]?.error && (
                    <div className="text-red-500 text-[11px] font-bold mt-2">{plotData.plots[index].error}</div>
                  )}
                </div>
              );
            })}

            <button type="button" onClick={addExpression} className="w-full py-2 rounded-xl bg-blue-600 text-white text-xs font-bold hover:bg-blue-700">
              <Plus size={16} className="inline mr-1" /> Add expression
            </button>
          </div>
        </WorkspacePanel>

        <WorkspacePanel
          id="setup"
          title="Plot setup & axes"
          icon={<Settings2 size={16} />}
          open={panels.setup}
          position={positions.setup}
          dark={isDarkMode}
          onClose={() => setPanels((current) => ({ ...current, setup: false }))}
          onDragStart={startPanelDrag}
        >
          <div className="space-y-3">
            <div>
              <div className="text-[10px] uppercase font-bold tracking-wider text-blue-500 mb-1.5">Coordinate system</div>
              <div className="grid grid-cols-2 gap-1.5">
                {(['cartesian2d', 'polar2d', 'cartesian3d', 'cylindrical3d', 'spherical3d', 'complex'] as PlotMode[]).map((mode) => (
                  <button
                    type="button"
                    key={mode}
                    onClick={() => changePlotMode(mode)}
                    className={`px-2 py-2 rounded-lg border text-[10px] font-bold text-left ${
                      plotMode === mode
                        ? 'bg-blue-600 text-white border-blue-500'
                        : isDarkMode
                        ? 'bg-slate-800 border-slate-700 text-slate-300'
                        : 'bg-white border-slate-200 text-slate-600'
                    }`}
                  >
                    {plotModeLabel[mode]}
                  </button>
                ))}
              </div>
            </div>

            {(plotMode === 'cartesian2d' || plotMode === 'polar2d') && expressions.some((e) => e.text.trim()) && (
              <div className={`p-2.5 rounded-xl border ${subcard}`}>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[10px] font-bold text-slate-500 uppercase">2D interchange</span>
                  <button
                    type="button"
                    onClick={() => changePlotMode(plotMode === 'polar2d' ? 'cartesian2d' : 'polar2d')}
                    className="text-[10px] font-bold text-blue-500"
                  >
                    Cartesian ⇄ Polar
                  </button>
                </div>
                <div className="text-[10px] text-slate-500 leading-relaxed">
                  Polar mode interprets Cartesian equations through <b>x = r cosθ</b> and <b>y = r sinθ</b>. Switching modes clears the previous trace and analysis overlay before recalculating.
                </div>
              </div>
            )}

            {(plotMode === 'cartesian3d' || plotMode === 'cylindrical3d' || plotMode === 'spherical3d') && (
              <div className={`p-2.5 rounded-xl border ${subcard}`}>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[10px] font-bold text-slate-500 uppercase">3D interchange</span>
                  <div className="flex gap-2">
                    <button
                      type="button"
                      onClick={() => changePlotMode(plotMode === 'cartesian3d' ? 'cylindrical3d' : plotMode === 'cylindrical3d' ? 'spherical3d' : 'cartesian3d')}
                      className="text-[10px] font-bold text-blue-500"
                    >
                      Cycle coordinates
                    </button>
                  </div>
                </div>
                <div className="text-[10px] text-slate-500 leading-relaxed">
                  Use Cartesian <b>x,y,z</b>, cylindrical <b>r,θ,z</b>, or spherical <b>ρ,θ,φ</b>. The selected system controls the displayed axes and range interpretation.
                </div>
              </div>
            )}

            {plotMode === 'complex' && (
              <div>
                <div className="text-[10px] uppercase font-bold tracking-wider text-blue-500 mb-1.5">Complex view</div>
                <select value={complexView} onChange={(e) => { setTracePoint(null); setComplexView(e.target.value as ComplexView); }} className={`w-full rounded-lg border px-2 py-2 text-xs ${field}`}>
                  <option value="magnitude">Magnitude |f(z)|</option>
                  <option value="real">Real part Re(f(z))</option>
                  <option value="imag">Imaginary part Im(f(z))</option>
                  <option value="phase">Phase arg(f(z))</option>
                  <option value="surface">3D magnitude surface</option>
                </select>
              </div>
            )}

            {(plotMode === 'polar2d' || plotMode === 'cylindrical3d' || plotMode === 'spherical3d') && (
              <div className="grid grid-cols-2 gap-2">
                <label className="text-[10px] font-semibold text-slate-500">
                  Angle unit
                  <select value={angleUnit} onChange={(e) => { setTracePoint(null); setAngleUnit(e.target.value as AngleUnit); }} className={`mt-1 w-full rounded-lg border px-2 py-1.5 text-xs ${field}`}>
                    <option value="degrees">Degrees</option>
                    <option value="radians">Radians</option>
                  </select>
                </label>
                <label className="text-[10px] font-semibold text-slate-500">
                  Direction
                  <select value={axis.theta_direction} onChange={(e) => updateAxis('theta_direction', e.target.value)} className={`mt-1 w-full rounded-lg border px-2 py-1.5 text-xs ${field}`}>
                    <option value="counterclockwise">CCW</option>
                    <option value="clockwise">CW</option>
                  </select>
                </label>
              </div>
            )}

            {plotMode === 'polar2d' && (
              <NumberField label="θ rotation" value={axis.theta_rotation} onChange={(value) => updateAxis('theta_rotation', value)} fieldClass={field} />
            )}

            {plotMode === 'cartesian2d' && (
              <div className="grid grid-cols-2 gap-2">
                <NumberField label="X min" value={axis.x_min} onChange={(value) => updateAxis('x_min', value)} fieldClass={field} />
                <NumberField label="X max" value={axis.x_max} onChange={(value) => updateAxis('x_max', value)} fieldClass={field} />
                <NumberField label="Y min" value={axis.y_min} onChange={(value) => updateAxis('y_min', value)} fieldClass={field} />
                <NumberField label="Y max" value={axis.y_max} onChange={(value) => updateAxis('y_max', value)} fieldClass={field} />
              </div>
            )}

            {(plotMode === 'polar2d' || plotMode === 'cylindrical3d' || plotMode === 'spherical3d') && (
              <div className="grid grid-cols-2 gap-2">
                <NumberField label="θ min" value={axis.theta_min} onChange={(value) => updateAxis('theta_min', value)} fieldClass={field} />
                <NumberField label="θ max" value={axis.theta_max} onChange={(value) => updateAxis('theta_max', value)} fieldClass={field} />
                <NumberField label={plotMode === 'spherical3d' ? 'ρ max' : 'R max'} value={axis.r_max} onChange={(value) => updateAxis('r_max', value)} fieldClass={field} />
                {plotMode === 'spherical3d' && (
                  <>
                    <NumberField label="φ min" value={axis.phi_min} onChange={(value) => updateAxis('phi_min', value)} fieldClass={field} />
                    <NumberField label="φ max" value={axis.phi_max} onChange={(value) => updateAxis('phi_max', value)} fieldClass={field} />
                  </>
                )}
                {(plotMode === 'cylindrical3d' || plotMode === 'spherical3d') && (
                  <>
                    <NumberField label="Z min" value={axis.z_min} onChange={(value) => updateAxis('z_min', value)} fieldClass={field} />
                    <NumberField label="Z max" value={axis.z_max} onChange={(value) => updateAxis('z_max', value)} fieldClass={field} />
                  </>
                )}
              </div>
            )}

            {plotMode === 'cartesian3d' && (
              <div className="grid grid-cols-2 gap-2">
                <NumberField label="X min" value={axis.x_min} onChange={(value) => updateAxis('x_min', value)} fieldClass={field} />
                <NumberField label="X max" value={axis.x_max} onChange={(value) => updateAxis('x_max', value)} fieldClass={field} />
                <NumberField label="Y min" value={axis.y_min} onChange={(value) => updateAxis('y_min', value)} fieldClass={field} />
                <NumberField label="Y max" value={axis.y_max} onChange={(value) => updateAxis('y_max', value)} fieldClass={field} />
                <NumberField label="Z min" value={axis.z_min} onChange={(value) => updateAxis('z_min', value)} fieldClass={field} />
                <NumberField label="Z max" value={axis.z_max} onChange={(value) => updateAxis('z_max', value)} fieldClass={field} />
              </div>
            )}

            {plotMode === 'complex' && (
              <div className="grid grid-cols-2 gap-2">
                <NumberField label="Re min" value={axis.re_min} onChange={(value) => updateAxis('re_min', value)} fieldClass={field} />
                <NumberField label="Re max" value={axis.re_max} onChange={(value) => updateAxis('re_max', value)} fieldClass={field} />
                <NumberField label="Im min" value={axis.im_min} onChange={(value) => updateAxis('im_min', value)} fieldClass={field} />
                <NumberField label="Im max" value={axis.im_max} onChange={(value) => updateAxis('im_max', value)} fieldClass={field} />
              </div>
            )}

            <div className={`p-2.5 rounded-xl border ${subcard}`}>
              <div className="text-[10px] uppercase font-bold text-slate-500 mb-2">Axis appearance</div>
              <div className="grid grid-cols-2 gap-2">
                {plotMode !== 'polar2d' && !is3D && (
                  <>
                    <label className="text-[10px] text-slate-500">
                      X label
                      <input value={axis.x_label} onChange={(e) => updateAxis('x_label', e.target.value)} className={`mt-1 w-full rounded-lg border px-2 py-1.5 text-xs ${field}`} />
                    </label>
                    <label className="text-[10px] text-slate-500">
                      Y label
                      <input value={axis.y_label} onChange={(e) => updateAxis('y_label', e.target.value)} className={`mt-1 w-full rounded-lg border px-2 py-1.5 text-xs ${field}`} />
                    </label>
                  </>
                )}
                {is3D && plotMode !== 'complex' && (
                  <>
                    <label className="text-[10px] text-slate-500">
                      X label
                      <input value={axis.x_label} onChange={(e) => updateAxis('x_label', e.target.value)} className={`mt-1 w-full rounded-lg border px-2 py-1.5 text-xs ${field}`} />
                    </label>
                    <label className="text-[10px] text-slate-500">
                      Y label
                      <input value={axis.y_label} onChange={(e) => updateAxis('y_label', e.target.value)} className={`mt-1 w-full rounded-lg border px-2 py-1.5 text-xs ${field}`} />
                    </label>
                    <label className="text-[10px] text-slate-500">
                      Z label
                      <input value={axis.z_label} onChange={(e) => updateAxis('z_label', e.target.value)} className={`mt-1 w-full rounded-lg border px-2 py-1.5 text-xs ${field}`} />
                    </label>
                  </>
                )}
              </div>

              <div className="flex flex-wrap gap-2 mt-2">
                <label className="text-[10px] flex items-center gap-1">
                  <input type="checkbox" checked={axis.show_grid} onChange={(e) => updateAxis('show_grid', e.target.checked)} /> Grid
                </label>
                <label className="text-[10px] flex items-center gap-1">
                  <input type="checkbox" checked={axis.show_zero} onChange={(e) => updateAxis('show_zero', e.target.checked)} /> Zero axis
                </label>
                <label className="text-[10px] flex items-center gap-1">
                  <input type="checkbox" checked={axis.equal_aspect} onChange={(e) => updateAxis('equal_aspect', e.target.checked)} /> Equal aspect
                </label>
              </div>
            </div>

            {(plotMode === 'cartesian2d' || plotMode === 'complex') && (
              <div className="grid grid-cols-2 gap-2">
                <SelectField label="X scale" value={axis.x_scale} onChange={(value) => updateAxis('x_scale', value)} options={['linear', 'log']} fieldClass={field} />
                <SelectField label="Y scale" value={axis.y_scale} onChange={(value) => updateAxis('y_scale', value)} options={['linear', 'log']} fieldClass={field} />
              </div>
            )}

            <div>
              <div className="text-[10px] uppercase font-bold text-slate-500 mb-1">Rendering quality</div>
              <div className="flex gap-1">
                {(['low', 'medium', 'high'] as Quality[]).map((qualityValue) => (
                  <button
                    type="button"
                    key={qualityValue}
                    onClick={() => { setTracePoint(null); setQuality(qualityValue); }}
                    className={`flex-1 py-1.5 rounded-lg text-[10px] font-bold ${quality === qualityValue ? 'bg-blue-600 text-white' : isDarkMode ? 'bg-slate-800 text-slate-300' : 'bg-slate-100 text-slate-600'}`}
                  >
                    {qualityValue}
                  </button>
                ))}
              </div>
            </div>

            <button
              type="button"
              onClick={() => { setAxis(DEFAULT_AXIS); setTracePoint(null); setMouseZoom(1); setCamera(VIEW_CAMERAS.iso); setPlotView({}); userViewportTouchedRef.current = false; setViewRevision(`reset-${Date.now()}`); }}
              className="w-full py-1.5 rounded-lg border text-[10px] font-bold flex items-center justify-center gap-1 hover:bg-slate-500/10"
            >
              <RotateCcw size={13} /> Reset axes & view
            </button>
          </div>
        </WorkspacePanel>

        <WorkspacePanel
          id="analysis"
          title="Smart analysis"
          icon={<Terminal size={16} />}
          open={panels.analysis}
          position={positions.analysis}
          dark={isDarkMode}
          onClose={() => setPanels((current) => ({ ...current, analysis: false }))}
          onDragStart={startPanelDrag}
        >
          <div className="flex items-center gap-2 mb-2">
            <div className="text-[10px] text-slate-500 flex-1">
              The graph reads the equations and exposes the most relevant geometry actions automatically.
            </div>
            <button type="button" onClick={() => setShowCommandConsole((value) => !value)} className={`px-2 py-1 rounded-lg text-[9px] font-black ${showCommandConsole ? 'bg-blue-600 text-white' : 'hover:bg-slate-500/10 text-slate-500'}`}>Console</button>
            <button type="button" onClick={clearAnalysis} className="p-1 text-slate-400 hover:text-red-500" title="Clear analysis"><Trash2 size={15} /></button>
          </div>

          <div className="space-y-2">
            {(plotData?.suggestions || []).map((suggestion: any, index: number) => (
              <div key={`${suggestion.title}-${index}`} className={`p-2.5 rounded-xl border ${subcard}`}>
                <div className="text-[10px] uppercase tracking-wider font-black text-blue-500 mb-1">{suggestion.title}</div>
                <div className="flex flex-wrap gap-1.5">
                  {(suggestion.actions || []).map((action: any) => (
                    <button key={`${suggestion.title}-${action.label}`} type="button" disabled={!action.command} onClick={() => action.command && runQuickCommand(action.command)} className="px-2 py-1.5 rounded-lg text-[10px] font-bold bg-blue-500/10 text-blue-500 hover:bg-blue-600 hover:text-white disabled:opacity-60 disabled:cursor-default">{action.label}</button>
                  ))}
                </div>
              </div>
            ))}

            {(plotMode === 'cartesian2d' || plotMode === 'polar2d') && (
              <div className={`p-2.5 rounded-xl border ${subcard}`}>
                <div className="flex items-center justify-between mb-1">
                  <div className="text-[10px] uppercase tracking-wider font-black text-violet-500">Coordinate transformations</div>
                  <button type="button" onClick={() => changePlotMode(plotMode === 'polar2d' ? 'cartesian2d' : 'polar2d')} className="text-[9px] font-black text-blue-500 hover:underline">{plotMode === 'polar2d' ? 'View Cartesian' : 'View Polar'}</button>
                </div>
                <div className="text-[10px] text-slate-500">Every supported expression can be translated into the coordinate systems that can represent it. Forms are fetched once you open a curve below (not recomputed on every keystroke), and cached after that.</div>
                {expressions.filter((e) => e.text.trim()).map((expression) => {
                  const open = !!openConversions[expression.id];
                  const loading = !!conversionLoading[expression.id];
                  const results = conversionResults[expression.id];
                  const convError = conversionErrors[expression.id];
                  // convert() returns [selection caption, aligned forms block] —
                  // show just the forms block here; the caption is redundant with
                  // the expression name already shown on the toggle button.
                  const formsBlock = results && results.length ? results[results.length - 1] : null;
                  return (
                    <div key={`conversion-${expression.id}`} className="mt-2 rounded-lg border border-violet-500/20 p-2">
                      <button type="button" onClick={() => toggleConversion(expression)} className="w-full text-left text-[10px] font-bold">{expression.name} <span className="text-slate-400">· {open ? 'hide forms' : 'show changed coordinates'}</span></button>
                      {open && (
                        <div className="mt-2 space-y-2">
                          {loading && <div className="text-[10px] text-blue-500">Converting…</div>}
                          {!loading && convError && <div className="text-[10px] font-bold text-red-500">{convError}</div>}
                          {!loading && !convError && formsBlock && (
                            <div className="text-[11px] overflow-x-auto"><BlockMath math={formsBlock} errorColor="#ef4444" renderError={katexFallback} /></div>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}

            {(plotMode === 'cartesian2d' || plotMode === 'polar2d') && expressions.filter((e) => e.text.trim()).length >= 2 && (
              <div className={`p-2.5 rounded-xl border ${subcard}`}>
                <div className="text-[10px] uppercase tracking-wider font-black text-emerald-500 mb-1">Regions & intersections</div>
                <div className="text-[10px] text-slate-500 mb-2">The bounded-region button automatically uses the intersections of the selected pair instead of forcing you to type limits first.</div>
                <div className="flex flex-wrap gap-1.5">
                  {expressions.filter((e) => e.text.trim()).slice(0, 4).flatMap((a, i, arr) => arr.slice(i + 1).map((b) => (
                    <div key={`pair-${a.id}-${b.id}`} className="flex gap-1">
                      <button type="button" onClick={() => startPick('intersection')} className="px-2 py-1.5 rounded-lg text-[10px] font-bold bg-pink-500/10 text-pink-500 hover:bg-pink-500 hover:text-white">∩ {a.name} / {b.name}</button>
                      <button type="button" onClick={() => startPick('area-region')} className="px-2 py-1.5 rounded-lg text-[10px] font-bold bg-emerald-500/10 text-emerald-600 hover:bg-emerald-600 hover:text-white">Area</button>
                      <button type="button" onClick={() => startPick('distance')} className="px-2 py-1.5 rounded-lg text-[10px] font-bold bg-amber-500/10 text-amber-600 hover:bg-amber-600 hover:text-white">Distance</button>
                    </div>
                  )))}
                </div>
              </div>
            )}

            <div className={`p-2.5 rounded-xl border ${subcard}`}>
              <div className="text-[10px] uppercase tracking-wider font-black text-cyan-500 mb-1">Pick directly from the graph</div>
              <div className="text-[10px] text-slate-500 mb-2">Activate an operation, then click the curve/surface. For pair operations, click two objects.</div>
              <div className="flex flex-wrap gap-1.5">
                {(plotMode === 'polar2d' ? [
                  ['Polar area', 'polar-area'], ['Polar arc length', 'polar-arc-length'],
                ] : [
                  ['Area under', 'area-under'], ['Arc length', 'arc-length'], ['Closed area', 'area-closed'],
                ]).map(([label, mode]) => <button key={mode} type="button" onClick={() => startPick(mode as PickMode)} className="px-2 py-1.5 rounded-lg text-[10px] font-bold bg-cyan-500/10 text-cyan-600 hover:bg-cyan-600 hover:text-white">{label}</button>)}
                {is3D && <><button type="button" onClick={() => startPick('volume')} className="px-2 py-1.5 rounded-lg text-[10px] font-bold bg-indigo-500/10 text-indigo-600 hover:bg-indigo-600 hover:text-white">Volume</button><button type="button" onClick={() => startPick('surface-area')} className="px-2 py-1.5 rounded-lg text-[10px] font-bold bg-indigo-500/10 text-indigo-600 hover:bg-indigo-600 hover:text-white">Surface area</button></>}
                {expressions.filter((e) => e.text.trim()).length >= 2 && <><button type="button" onClick={() => startPick('area-between')} className="px-2 py-1.5 rounded-lg text-[10px] font-bold bg-emerald-500/10 text-emerald-600 hover:bg-emerald-600 hover:text-white">Area between</button></>}
              </div>
            </div>

            {showCommandConsole && (
              <div className={`p-2.5 rounded-xl border ${subcard}`}>
                <div className="flex items-center gap-2 mb-2">
                  <Command size={13} className="text-blue-500" />
                  <div className="text-[9px] uppercase font-black text-slate-500 flex-1">Command library</div>
                  <span className="text-[8px] text-slate-400">Click to copy</span>
                </div>
                <div className={`flex items-center gap-2 border rounded-lg px-2 py-1.5 mb-2 ${field}`}>
                  <Search size={13} className="text-slate-400" />
                  <input value={commandSearch} onChange={(e) => setCommandSearch(e.target.value)} placeholder="Search commands, geometry, area, 3D…" className="flex-1 bg-transparent outline-none text-[10px]" />
                </div>
                <div className="max-h-44 overflow-y-auto space-y-1 mb-2">
                  {filteredCommands.map((item) => (
                    <div key={item.command} className={`p-1.5 rounded-lg border ${subcard}`}>
                      <div className="flex items-center gap-2">
                        <button type="button" onClick={() => setCmdInput(item.command)} className="flex-1 min-w-0 text-left">
                          <div className="text-[9px] uppercase font-black text-blue-500">{item.category}</div>
                          <div className="text-[10px] font-bold truncate">{item.title}</div>
                          <div className="text-[9px] font-mono text-slate-500 truncate">{item.command}</div>
                        </button>
                        <button type="button" onClick={() => copyToClipboard(item.command, -1)} className="p-1 rounded hover:bg-slate-500/10 text-slate-400" title="Copy command"><Copy size={12} /></button>
                      </div>
                    </div>
                  ))}
                </div>
                <textarea value={cmdInput} onChange={(e) => setCmdInput(e.target.value)} onKeyDown={(event) => { if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') { event.preventDefault(); runAnalysisCommand(); } }} placeholder="Type a command or select one above…" className={`w-full border p-2.5 rounded-lg font-mono text-xs outline-none mb-2 min-h-16 resize-y ${field}`} autoComplete="off" spellCheck={false} />
                <div className="flex gap-1.5 flex-wrap">
                  <button type="button" onClick={runAnalysisCommand} className="px-3 py-1.5 rounded-lg bg-blue-600 text-white text-[10px] font-bold">Run</button>
                  <button type="button" onClick={() => setShowCmdHelp((value) => !value)} className="px-3 py-1.5 rounded-lg border text-[10px] font-bold">{showCmdHelp ? 'Hide tips' : 'Tips'}</button>
                </div>
                {showCmdHelp && <div className="mt-2 text-[9px] text-slate-500 leading-relaxed">Search the library, click a command to put it in the editor, or use Pick on graph for geometry operations. The command shell is optional; Calculus/ODE work remains in your separate section.</div>}
              </div>
            )}

            {pickMode && (
              <div className={`p-2 rounded-xl border border-blue-500/30 bg-blue-500/10 text-[10px] ${subcard}`}>
                <div className="flex items-center gap-2"><Target size={13} className="text-blue-500" /><strong>Pick on graph:</strong><span>{pickMode.replace('-', ' ')}</span></div>
                <div className="text-[9px] text-slate-500 mt-1">{pickedOwners.length ? `Selected: ${pickedOwners.map((i) => expressions[i]?.name).filter(Boolean).join(', ')}` : 'Click the required curve or surface directly.'}</div>
                <button type="button" onClick={() => { setPickMode(null); setPickedOwners([]); }} className="mt-1 text-[9px] font-bold text-red-500">Cancel pick</button>
              </div>
            )}
            {commandBusy && <div className="text-[10px] text-blue-500">Working…</div>}
            {commandError && <div className="text-[10px] font-bold text-red-500">{commandError}</div>}
            {commandResults.map((result, index) => (<div key={`${result}-${index}`} className={`p-2 rounded-lg border text-xs overflow-x-auto ${subcard}`}><BlockMath math={result} errorColor="#ef4444" renderError={katexFallback} /></div>))}
            {commandHistory.length > 0 && showCommandConsole && <div className={`p-2 rounded-lg border ${subcard}`}><div className="text-[9px] uppercase font-black text-slate-500 mb-1">Recent</div>{commandHistory.slice().reverse().slice(0, 6).map((command, index) => (<button key={`${command}-${index}`} type="button" onClick={() => setCmdInput(command)} className="block w-full text-left text-[9px] font-mono text-slate-500 truncate hover:text-blue-500">› {command}</button>))}</div>}
          </div>
        </WorkspacePanel>

        {page === 'home' && (
          <div className={`absolute inset-0 z-40 flex items-center justify-center backdrop-blur-sm ${isDarkMode ? 'bg-slate-950/82' : 'bg-white/82'}`}>
            <div className={`w-[min(820px,92vw)] rounded-3xl border shadow-2xl p-8 ${card}`}>
              <div className="flex items-center gap-3 mb-2">
                <div className="p-3 rounded-2xl bg-blue-600 text-white"><Grid3X3 size={24} /></div>
                <div>
                  <h1 className="text-3xl font-black">GraphLab</h1>
                  <p className="text-sm text-slate-500">Interactive engineering graphing workspace</p>
                </div>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-2 mt-6">
                {[
                  ['Cartesian', 'x / y axes'],
                  ['Polar', 'r / θ axes + conversion'],
                  ['3D', 'Cartesian / cylindrical / spherical'],
                  ['Complex', 'magnitude / real / imag / phase'],
                  ['Tracing', 'click a curve point to pin it'],
                  ['Analysis', 'named intersections / locus / calculus'],
                ].map(([title, description]) => (
                  <div key={title} className={`p-3 rounded-xl border ${subcard}`}>
                    <div className="font-bold text-sm">{title}</div>
                    <div className="text-[11px] text-slate-500 mt-1">{description}</div>
                  </div>
                ))}
              </div>
              <div className="flex items-center gap-2 mt-6">
                <button type="button" onClick={() => setPage('graph')} className="px-5 py-2.5 rounded-xl bg-blue-600 text-white font-bold text-sm hover:bg-blue-700 flex items-center gap-2">
                  Open workspace <ChevronRight size={17} />
                </button>
                <button type="button" onClick={() => setPanels({ expressions: true, setup: true, analysis: true })} className="px-4 py-2.5 rounded-xl border text-sm font-bold hover:bg-slate-500/10 flex items-center gap-2">
                  <LayoutGrid size={16} /> Show panels
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}