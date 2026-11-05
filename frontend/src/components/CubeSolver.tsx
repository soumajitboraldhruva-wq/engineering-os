// Cube Solver Component - 3D rotatable cube with scramble/solve/animate
import { useState, useRef, useEffect, useCallback } from 'react';

const FACE_NAMES: string[] = ['U', 'R', 'F', 'D', 'L', 'B'];
const FACE_COLORS: string[] = [
  '#FFFFFF', '#FF0000', '#00B400', '#FFFF00', '#FFA500', '#0000FF',
];

const SOLVER_ALGORITHMS = [
  { id: 'auto', name: 'Auto (best for size)', sizes: [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16] },
  { id: 'kociemba', name: 'Kociemba Two-Phase (optimal-ish)', sizes: [3] },
  { id: 'thistlethwaite', name: 'Thistlethwaite (4x4+)', sizes: [3,4,5,6,7] },
  { id: 'korf', name: 'Korf Optimal IDA* (3x3)', sizes: [3] },
  { id: 'native_ida', name: 'Native IDA* (all N via C++)', sizes: [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16] },
  { id: 'pocket', name: 'Pocket Cube (2x2 dedicated)', sizes: [2] },
];

function getApiBase(): string {
  return (import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000').replace(/\/+$/, '');
}

async function apiCall(path: string, options: RequestInit = {}): Promise<any> {
  const res = await fetch(`${getApiBase()}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Request failed' }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

function createSolvedState(size: number): number[] {
  const facelets: number[] = [];
  for (let f = 0; f < 6; f++) {
    for (let i = 0; i < size * size; i++) facelets.push(f);
  }
  return facelets;
}

function faceletsToNet(facelets: number[], size: number): Record<string, number[][]> {
  const net: Record<string, number[][]> = {};
  const faces = ['U', 'R', 'F', 'D', 'L', 'B'];
  faces.forEach((face, fi) => {
    const grid: number[][] = [];
    for (let r = 0; r < size; r++) {
      const row: number[] = [];
      for (let c = 0; c < size; c++) {
        row.push(facelets[fi * size * size + r * size + c]);
      }
      grid.push(row);
    }
    net[face] = grid;
  });
  return net;
}

function netToFacelets(net: Record<string, number[][]>, size: number): number[] {
  const faces = ['U', 'R', 'F', 'D', 'L', 'B'];
  const facelets: number[] = [];
  faces.forEach((face) => {
    for (let r = 0; r < size; r++) {
      for (let c = 0; c < size; c++) {
        facelets.push(net[face][r][c]);
      }
    }
  });
  return facelets;
}

function parseFaceletString(s: string): number[] {
  return s.split('').map(ch => parseInt(ch, 10));
}

export default function CubeSolver() {
  const [size, setSize] = useState(3);
  const [mode, setMode] = useState<'scramble' | 'manual' | 'solving' | 'animating'>('scramble');
  const [facelets, setFacelets] = useState<number[]>(() => createSolvedState(3));
  const [algorithm, setAlgorithm] = useState('auto');
  const [scrambleLength, setScrambleLength] = useState(20);
  const [scrambleMoves, setScrambleMoves] = useState<string[]>([]);
  const [solution, setSolution] = useState<{
    algorithm: string;
    moves: string[];
    move_count: number;
    solved: boolean;
    nodes: number;
    time_ms: number;
    states: string[];
    error?: string;
  } | null>(null);
  const [animating, setAnimating] = useState(false);
  const [animStep, setAnimStep] = useState(0);
  const [animSpeed, setAnimSpeed] = useState(400); // ms per move
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animFrameRef = useRef<number | null>(null);
  const animStartRef = useRef<number>(0);

  // Draw cube function - defined first to avoid TDZ
  const drawCube = useCallback((ctx: CanvasRenderingContext2D, width: number, height: number, cubeFacelets: number[]) => {
    ctx.clearRect(0, 0, width, height);
    const cx = width / 2;
    const cy = height / 2;
    const stickerSize = Math.min(width, height) / (size * 3.5);

    const net = faceletsToNet(cubeFacelets, size);
    // Cross layout: U at top, D at bottom, L-F-R-B in middle row
    const faces = ['U', 'L', 'F', 'R', 'B', 'D'];
    const positions = {
      U: { x: 1, y: 0 },
      L: { x: 0, y: 1 },
      F: { x: 1, y: 1 },
      R: { x: 2, y: 1 },
      B: { x: 3, y: 1 },
      D: { x: 1, y: 2 },
    };

    faces.forEach(face => {
      const pos = positions[face as keyof typeof positions];
      const startX = cx + (pos.x - 1.5) * size * stickerSize * 1.1;
      const startY = cy + (pos.y - 1) * size * stickerSize * 1.1;

      for (let r = 0; r < size; r++) {
        for (let c = 0; c < size; c++) {
          const colorIdx = net[face][r][c];
          ctx.fillStyle = FACE_COLORS[colorIdx];
          ctx.fillRect(
            startX + c * stickerSize * 1.02,
            startY + r * stickerSize * 1.02,
            stickerSize,
            stickerSize
          );
          ctx.strokeStyle = '#222';
          ctx.lineWidth = 1;
          ctx.strokeRect(
            startX + c * stickerSize * 1.02,
            startY + r * stickerSize * 1.02,
            stickerSize,
            stickerSize
          );
        }
      }
    });
  }, [size]);

  // Reset on size change
  useEffect(() => {
    setFacelets(createSolvedState(size));
    setScrambleMoves([]);
    setSolution(null);
    setError(null);
    setAnimating(false);
    setAnimStep(0);
  }, [size]);

  // Animation timer
  useEffect(() => {
    if (!animating) return;
    const start = performance.now();
    animStartRef.current = start;
    const animate = (now: number) => {
      const elapsed = now - start;
      const progress = Math.min(elapsed / animSpeed, 1);
      if (progress >= 1) {
        setAnimating(false);
        if (solution && animStep < solution.moves.length - 1) {
          setAnimStep(s => s + 1);
          setAnimating(true); // Continue to next step
        }
      } else {
        animFrameRef.current = requestAnimationFrame(animate);
      }
    };
    animFrameRef.current = requestAnimationFrame(animate);
    return () => { if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current); };
  }, [animating, animSpeed, animStep, solution]);

  // Render loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width;
      canvas.height = rect.height;
    };
    resize();
    window.addEventListener('resize', resize);

    const renderLoop = () => {
      let currentFacelets: number[];
      if (solution?.states?.[animStep]) {
        currentFacelets = parseFaceletString(solution.states[animStep]);
      } else {
        currentFacelets = facelets;
      }
      drawCube(ctx, canvas.width, canvas.height, currentFacelets);
      requestAnimationFrame(renderLoop);
    };
    renderLoop();

    return () => window.removeEventListener('resize', resize);
  }, [drawCube, animStep, solution, facelets]);

  const handleScramble = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiCall('/api/cube/scramble', {
        method: 'POST',
        body: JSON.stringify({ size, length: scrambleLength }),
      }) as { size: number; scramble: string; moves: string[]; facelets: string; state: number[] };
      setScrambleMoves(res.moves);
      setFacelets(res.state);
      setSolution(null);
      setMode('solving');
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const handleSolve = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiCall('/api/cube/solve', {
        method: 'POST',
        body: JSON.stringify({
          size,
          facelets: facelets.join(''),
          algorithm,
        }),
      }) as {
        algorithm: string;
        moves: string[];
        move_count: number;
        solved: boolean;
        nodes: number;
        time_ms: number;
        states: string[];
        error?: string;
      };
      if (!res.solved) {
        setError(res.error || 'Failed to solve');
        return;
      }
      setSolution(res);
      setMode('animating');
      setAnimStep(0);
      setAnimating(true);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const handlePlayPause = () => {
    if (animating) {
      setAnimating(false);
    } else if (solution && animStep < solution.moves.length - 1) {
      setAnimating(true);
    }
  };

  const handleNext = () => {
    if (solution && animStep < solution.moves.length - 1) {
      setAnimStep(s => s + 1);
    }
  };

  const handlePrev = () => {
    if (animStep > 0) {
      setAnimStep(s => s - 1);
    }
  };

  const handleReset = () => {
    setAnimating(false);
    setAnimStep(0);
    setFacelets(createSolvedState(size));
    setScrambleMoves([]);
    setSolution(null);
    setMode('scramble');
  };

  const handleStickerClick = (face: string, r: number, c: number) => {
    if (mode !== 'manual') return;
    const net = faceletsToNet(facelets, size);
    net[face][r][c] = (net[face][r][c] + 1) % 6;
    setFacelets(netToFacelets(net, size));
  };

  const progressPercent = solution && solution.moves.length > 0
    ? (animStep / Math.max(1, solution.moves.length - 1)) * 100
    : 0;

  return (
    <div className="flex h-screen bg-slate-50 dark:bg-slate-900">
      <aside className="w-72 shrink-0 bg-slate-950 text-white p-4 overflow-y-auto">
        <h2 className="text-lg font-bold mb-4">Cube Solver</h2>

        <div className="mb-4">
          <label className="block text-xs font-medium text-slate-400 mb-1">Cube Size</label>
          <select
            value={size}
            onChange={(e) => setSize(Number(e.target.value))}
            className="w-full rounded-xl bg-slate-800 border border-slate-700 px-3 py-2 text-sm text-white focus:ring-2 focus:ring-blue-500"
          >
            {[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16].map(s => (
              <option key={s} value={s}>{s}×{s}</option>
            ))}
          </select>
        </div>

        <div className="mb-4">
          <label className="block text-xs font-medium text-slate-400 mb-1">Algorithm</label>
          <select
            value={algorithm}
            onChange={(e) => setAlgorithm(e.target.value)}
            className="w-full rounded-xl bg-slate-800 border border-slate-700 px-3 py-2 text-sm text-white focus:ring-2 focus:ring-blue-500"
          >
            {SOLVER_ALGORITHMS
              .filter(a => a.sizes.includes(size))
              .map(a => <option key={a.id} value={a.id}>{a.name}</option>)}
          </select>
          <p className="mt-1 text-xs text-slate-500">
            {SOLVER_ALGORITHMS.find(a => a.id === algorithm)?.name}
          </p>
        </div>

        <div className="mb-4 flex gap-1">
          {(['scramble', 'manual', 'solving'] as const).map(m => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={`flex-1 px-3 py-2 rounded-xl text-sm font-medium transition ${
                mode === m
                  ? 'bg-blue-600 text-white'
                  : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
              }`}
            >
              {m.charAt(0).toUpperCase() + m.slice(1)}
            </button>
          ))}
        </div>

        {mode === 'scramble' && (
          <div className="space-y-3">
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1">
                Scramble Length: {scrambleLength}
              </label>
              <input
                type="range"
                min="1"
                max={Math.min(100, size * 10)}
                value={scrambleLength}
                onChange={(e) => setScrambleLength(Number(e.target.value))}
                className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-blue-500"
              />
            </div>
            <button
              onClick={handleScramble}
              disabled={loading}
              className="w-full rounded-xl bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-500 disabled:opacity-50"
            >
              {loading ? 'Scrambling...' : 'Generate Scramble'}
            </button>
            {scrambleMoves.length > 0 && (
              <div className="text-xs font-mono text-slate-400 max-h-24 overflow-auto">
                {scrambleMoves.join(' ')}
              </div>
            )}
            <button
              onClick={handleSolve}
              disabled={loading}
              className="w-full mt-2 rounded-xl bg-emerald-600 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-500 disabled:opacity-50"
            >
              Solve Scramble
            </button>
          </div>
        )}

        {mode === 'manual' && (
          <div className="space-y-3">
            <p className="text-xs text-slate-400">Click stickers to cycle colors</p>
            <div className="grid grid-cols-4 gap-1">
              {['U', 'L', 'F', 'R', 'B', 'D'].map(face => (
                <div key={face} className="col-span-1">
                  <div className="text-xs text-center text-slate-400 mb-1">{face}</div>
                  <div className={`grid grid-cols-${size} gap-1`}>
                    {Array.from({ length: size * size }, (_, i) => {
                      const r = Math.floor(i / size);
                      const c = i % size;
                      const net = faceletsToNet(facelets, size);
                      const colorIdx = net[face][r][c];
                      return (
                        <button
                          key={i}
                          onClick={() => handleStickerClick(face, r, c)}
                          className="aspect-square rounded border border-slate-600 transition"
                          style={{ backgroundColor: FACE_COLORS[colorIdx] }}
                        />
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>
            <button
              onClick={handleSolve}
              disabled={loading}
              className="w-full rounded-xl bg-emerald-600 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-500 disabled:opacity-50"
            >
              Solve Manual State
            </button>
          </div>
        )}

        {(mode === 'solving' || mode === 'animating') && solution && (
          <div className="space-y-3">
            <div className="flex gap-1">
              <button
                onClick={handlePrev}
                disabled={animStep === 0}
                className="flex-1 rounded-xl bg-slate-800 px-3 py-2 text-sm font-medium text-white disabled:opacity-30"
              >
                Prev
              </button>
              <button
                onClick={handlePlayPause}
                className="flex-1 rounded-xl bg-blue-600 px-3 py-2 text-sm font-medium text-white"
              >
                {animating ? 'Pause' : 'Play'}
              </button>
              <button
                onClick={handleNext}
                disabled={animStep >= solution.moves.length - 1}
                className="flex-1 rounded-xl bg-slate-800 px-3 py-2 text-sm font-medium text-white disabled:opacity-30"
              >
                Next
              </button>
            </div>
            <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-blue-500 transition-all duration-300"
                style={{ width: `${progressPercent}%` }}
              />
            </div>
            <div className="text-xs text-slate-400 text-center">
              Move {animStep + 1} / {solution.moves.length}
            </div>
            <div className="text-xs font-mono text-slate-400 max-h-24 overflow-auto">
              {solution.moves.map((m, i) => (
                <span
                  key={i}
                  className={`mx-1 px-1 rounded ${
                    i === animStep ? 'bg-blue-500 text-white' : 'text-slate-300'
                  }`}
                >
                  {m}
                </span>
              ))}
            </div>
            <div className="text-xs text-slate-500 text-center">
              {solution.algorithm} • {solution.move_count} moves • {solution.time_ms}ms
            </div>
            <div className="flex gap-2">
              <label className="flex-1 text-xs text-slate-400">
                Speed: {animSpeed}ms
                <input
                  type="range"
                  min="100"
                  max="2000"
                  step="50"
                  value={animSpeed}
                  onChange={(e) => setAnimSpeed(Number(e.target.value))}
                  className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-blue-500 mt-1"
                />
              </label>
            </div>
            <button
              onClick={handleReset}
              className="w-full rounded-xl bg-slate-700 px-4 py-2 text-sm font-medium text-slate-300 hover:bg-slate-600"
            >
              New Scramble
            </button>
          </div>
        )}

        {error && (
          <div className="mt-4 p-3 rounded-xl bg-red-900/30 border border-red-700 text-red-300 text-sm">
            {error}
          </div>
        )}
      </aside>

      <main className="flex-1 overflow-auto p-6 lg:p-8">
        <div className="mx-auto max-w-4xl">
          <div className="relative aspect-square max-w-full">
            <canvas
              ref={canvasRef}
              className="w-full h-full bg-slate-100 dark:bg-slate-900 rounded-2xl shadow-xl"
              style={{ touchAction: 'none' }}
            />
            {animating && (
              <div className="absolute top-4 left-4 bg-blue-600/90 text-white px-3 py-1 rounded-lg text-sm font-medium animate-pulse">
                Animating...
              </div>
            )}
          </div>
          <div className="mt-4 grid grid-cols-6 gap-2 text-center text-xs text-slate-500 dark:text-slate-400">
            {FACE_NAMES.map((name, i) => (
              <div key={name} className="flex items-center justify-center gap-1">
                <span
                  className="w-4 h-4 rounded border border-slate-300"
                  style={{ backgroundColor: FACE_COLORS[i] }}
                />
                {name}
              </div>
            ))}
          </div>
        </div>
      </main>
    </div>
  );
}