import { useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity,
  BookOpen,
  ChevronRight,
  CircleStop,
  Eraser,
  FlaskConical,
  Gauge,
  KeyRound,
  Play,
  RefreshCw,
  TerminalSquare,
} from 'lucide-react';

type EngineButton = {
  number: number;
  label: string;
  short: string;
  icon: any;
};

const MODES: EngineButton[] = [
  { number: 1, label: 'Single 1D Sequence', short: 'Sequence', icon: Activity },
  { number: 2, label: 'N-Dimensional / Multivariate Dataset', short: 'Multivariate', icon: Gauge },
  { number: 3, label: 'Manual Coordinates (2D/3D)', short: 'Coordinates', icon: Activity },
  { number: 4, label: 'Load Dataset / Sequence from File', short: 'File Mode', icon: BookOpen },
  { number: 5, label: 'Crypto / Encoding Lab', short: 'Crypto', icon: KeyRound },
  { number: 6, label: 'Statistical Analysis Only', short: 'Statistics', icon: FlaskConical },
  { number: 7, label: 'Random / PRNG Tools', short: 'PRNG', icon: Gauge },
  { number: 8, label: 'Graph / Curve Analysis', short: 'Graph / Curve', icon: Activity },
  { number: 9, label: 'Engine Self-Test / Benchmark', short: 'Self-Test', icon: FlaskConical },
  { number: 10, label: 'Generic Puzzle / Cube Graph Solver', short: 'Puzzle / Cube', icon: Gauge },
];

function backendWsUrl() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.hostname}:8000/api/analyser/ws`;
}

export default function NativeEngineConsole() {
  const [lines, setLines] = useState<string[]>([]);
  const [connected, setConnected] = useState(false);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const socketRef = useRef<WebSocket | null>(null);
  const terminalRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const outputText = useMemo(() => lines.join(''), [lines]);

  const append = (text: string) => {
    setLines((previous) => [...previous, text]);
  };

  const connect = () => {
    socketRef.current?.close();
    const socket = new WebSocket(backendWsUrl());
    socketRef.current = socket;

    socket.onopen = () => {
      setConnected(true);
      setBusy(false);
      append('\n[WEB CONSOLE] Connected to native analyser_bin.\n');
    };

    socket.onmessage = (event) => {
      append(event.data);
      setBusy(false);
    };

    socket.onerror = () => {
      append('\n[WEB CONSOLE] WebSocket error. Check that FastAPI is running.\n');
      setBusy(false);
    };

    socket.onclose = () => {
      setConnected(false);
      setBusy(false);
    };
  };

  useEffect(() => {
    connect();
    return () => socketRef.current?.close();
  }, []);

  useEffect(() => {
    const node = terminalRef.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [outputText]);

  const send = (value: string) => {
    const socket = socketRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      append('\n[WEB CONSOLE] Not connected.\n');
      return;
    }
    setBusy(true);
    socket.send(value.endsWith('\n') ? value : `${value}\n`);
    inputRef.current?.focus();
  };

  const sendInput = () => {
    const value = input.replace(/\r?\n/g, ' ').trim();
    if (!value) return;
    send(value);
    setInput('');
  };

  const stop = () => {
    const socket = socketRef.current;
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send('__ENGINE_EXIT__');
    }
  };

  const reset = () => {
    socketRef.current?.close();
    setLines([]);
    window.setTimeout(connect, 120);
  };

  const clear = () => setLines([]);

  return (
    <div className="h-full min-h-[calc(100vh-80px)] text-slate-100 flex flex-col gap-4">
      <header className="rounded-xl border border-slate-800 bg-slate-950 shadow-lg overflow-hidden">
        <div className="px-5 py-4 border-b border-slate-800 flex items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-3">
              <TerminalSquare className="text-cyan-400" size={22} />
              <h1 className="text-lg md:text-xl font-black tracking-wide">
                ADVANCED MULTIVARIATE SEQUENCE / CURVE / CRYPTO ENGINE
              </h1>
            </div>
            <p className="text-xs text-slate-400 mt-1 font-mono">
              v6.0 native console • unchanged C++ routines • interactive browser terminal
            </p>
          </div>
          <div className={`px-3 py-1.5 rounded-full text-xs font-mono border ${connected ? 'border-emerald-700 text-emerald-300 bg-emerald-950/50' : 'border-red-800 text-red-300 bg-red-950/40'}`}>
            {connected ? 'ENGINE ONLINE' : 'ENGINE OFFLINE'}
          </div>
        </div>

        <div className="px-5 py-4 font-mono text-xs md:text-sm leading-6 whitespace-pre-wrap text-slate-300">
          ========================================================
          {'\n'} ADVANCED MULTIVARIATE SEQUENCE / CURVE / CRYPTO ENGINE v6.0
          {'\n'}========================================================
          {'\n'}Core input methods:
          {'\n'}1) Single 1D Sequence
          {'\n'}2) N-Dimensional / Multivariate Dataset
          {'\n'}3) Manual Coordinates (2D/3D)
          {'\n'}4) Load Dataset / Sequence from File
          {'\n'}5) Crypto / encoding / classical + modern primitives
          {'\n'}
          {'\n'}Additional tools (original functionality retained):
          {'\n'}6) Statistical analysis only
          {'\n'}7) Random / PRNG analysis and generation
          {'\n'}8) Graph / curve analysis
          {'\n'}9) Engine self-test / benchmark
          {'\n'}10) Generic puzzle / cube graph solver
          {'\n'}0) Exit
          {'\n'}========================================================
        </div>
      </header>

      <section className="grid xl:grid-cols-[360px_1fr] gap-4 min-h-0 flex-1">
        <aside className="rounded-xl border border-slate-800 bg-slate-950 shadow-lg overflow-hidden flex flex-col min-h-[620px]">
          <div className="px-4 py-3 border-b border-slate-800 flex items-center justify-between">
            <span className="text-xs uppercase tracking-widest font-bold text-slate-400">Native shortcuts</span>
            <span className="text-[11px] text-slate-500">sends the real menu number</span>
          </div>

          <div className="p-3 overflow-y-auto space-y-1">
            {MODES.map((mode) => {
              const Icon = mode.icon;
              return (
                <button
                  key={mode.number}
                  type="button"
                  onClick={() => send(String(mode.number))}
                  className="w-full text-left p-3 rounded-lg border border-slate-800 hover:border-cyan-700 hover:bg-slate-900 transition group"
                >
                  <div className="flex items-center gap-3">
                    <span className="w-7 h-7 rounded-md bg-slate-900 border border-slate-800 flex items-center justify-center text-xs font-mono text-cyan-400">
                      {mode.number}
                    </span>
                    <Icon size={16} className="text-slate-500 group-hover:text-cyan-400" />
                    <div className="min-w-0 flex-1">
                      <div className="text-sm font-semibold text-slate-200 truncate">{mode.short}</div>
                      <div className="text-[11px] text-slate-500 truncate">{mode.label}</div>
                    </div>
                    <ChevronRight size={15} className="text-slate-700 group-hover:text-slate-400" />
                  </div>
                </button>
              );
            })}
          </div>

          <div className="mt-auto p-3 border-t border-slate-800 flex gap-2">
            <button type="button" onClick={reset} className="flex-1 py-2 rounded-lg bg-slate-900 hover:bg-slate-800 text-xs font-bold flex items-center justify-center gap-2">
              <RefreshCw size={14} /> Restart
            </button>
            <button type="button" onClick={stop} className="flex-1 py-2 rounded-lg bg-red-950/50 hover:bg-red-900/70 border border-red-900 text-xs font-bold flex items-center justify-center gap-2 text-red-200">
              <CircleStop size={14} /> Stop
            </button>
            <button type="button" onClick={clear} className="w-10 rounded-lg bg-slate-900 hover:bg-slate-800 flex items-center justify-center" title="Clear terminal">
              <Eraser size={14} />
            </button>
          </div>
        </aside>

        <div className="rounded-xl border border-slate-800 bg-[#020617] shadow-lg overflow-hidden flex flex-col min-h-[620px]">
          <div className="px-4 py-3 border-b border-slate-800 flex items-center justify-between">
            <div className="text-xs font-mono text-slate-400">native-analyser://console</div>
            <div className="flex items-center gap-2 text-[11px] text-slate-500">
              <span className="w-2 h-2 rounded-full bg-emerald-500" />
              stdin / stdout bridge
            </div>
          </div>

          <div ref={terminalRef} className="flex-1 overflow-y-auto p-5 font-mono text-[13px] leading-6 text-emerald-300 whitespace-pre-wrap selection:bg-cyan-900/50">
            {outputText || '[waiting for native engine output...]'}
          </div>

          <form
            onSubmit={(event) => {
              event.preventDefault();
              sendInput();
            }}
            className="border-t border-slate-800 p-3 bg-slate-950 flex items-center gap-2"
          >
            <span className="text-cyan-400 font-mono text-sm">&gt;</span>
            <input
              ref={inputRef}
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder={busy ? 'Waiting for engine…' : 'Type exactly what you would type in the C++ console'}
              className="flex-1 bg-transparent outline-none text-sm font-mono text-slate-100 placeholder:text-slate-600"
              disabled={!connected}
              autoComplete="off"
            />
            <button
              type="submit"
              disabled={!connected || !input.trim()}
              className="px-4 py-2 rounded-lg bg-cyan-600 hover:bg-cyan-500 disabled:bg-slate-800 disabled:text-slate-600 text-white font-bold text-xs flex items-center gap-2"
            >
              <Play size={13} /> Send
            </button>
          </form>
        </div>
      </section>
    </div>
  );
}
