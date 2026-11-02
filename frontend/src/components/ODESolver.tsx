import { useState } from 'react';
import axios from 'axios';
import PlotComponent from 'react-plotly.js';
import 'katex/dist/katex.min.css';
import { BlockMath } from 'react-katex';

const API_URL = (import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000').replace(/\/+$/, '');
const API = `${API_URL}/api/ode`;

// This safely fixes the Vite + Plotly "got object" crash!
const Plot = (PlotComponent as any).default || PlotComponent;

export default function ODESolver() {
  const [equation, setEquation] = useState("y'' + 5*y' + 6*y = 0");
  const [params, setParams] = useState({ y0: 1, v0: 0, time: 10 });
  const [solution, setSolution] = useState<any>(null);

  const insertText = (text: string) => setEquation((prev) => prev + text);

  const solve = async () => {
    try {
      const res = await axios.post(`${API}/solve`, { 
        equation, ...params 
      });
      setSolution(res.data);
    } catch (e) {
      alert("Error talking to backend");
    }
  };

  return (
    <div className="max-w-4xl bg-white p-8 rounded-xl shadow-sm border border-gray-200 text-gray-800">
      <h2 className="text-2xl font-bold mb-2">General ODE Solver</h2>
      
      <div className="bg-slate-50 p-6 rounded-lg mb-6 border">
        <label className="block text-sm font-bold text-gray-600 mb-2">Equation</label>
        <input 
          type="text" value={equation} onChange={(e) => setEquation(e.target.value)} 
          className="w-full border p-4 rounded outline-none focus:border-blue-500 font-mono text-xl bg-white mb-3"
        />
        
        <div className="flex gap-2 mb-6">
          <button onClick={() => insertText("y''")} className="px-3 py-1 bg-gray-200 rounded hover:bg-gray-300 font-mono">y''</button>
          <button onClick={() => insertText("y'")} className="px-3 py-1 bg-gray-200 rounded hover:bg-gray-300 font-mono">y'</button>
          <button onClick={() => insertText("y")} className="px-3 py-1 bg-gray-200 rounded hover:bg-gray-300 font-mono">y</button>
          <button onClick={() => insertText("sin(t)")} className="px-3 py-1 bg-gray-200 rounded hover:bg-gray-300 font-mono">sin(t)</button>
          <button onClick={() => insertText("exp(t)")} className="px-3 py-1 bg-gray-200 rounded hover:bg-gray-300 font-mono">e^t</button>
          <button onClick={() => insertText(" = 0")} className="px-3 py-1 bg-blue-100 text-blue-800 rounded hover:bg-blue-200 font-mono">= 0</button>
        </div>

        <div className="grid grid-cols-3 gap-4">
          <div><label className="block text-xs font-bold mb-1">y(0) [Initial Pos]</label>
            <input type="number" value={params.y0} onChange={(e) => setParams({...params, y0: parseFloat(e.target.value) || 0})} className="w-full border p-2 rounded" /></div>
          <div><label className="block text-xs font-bold mb-1">y'(0) [Initial Vel]</label>
            <input type="number" value={params.v0} onChange={(e) => setParams({...params, v0: parseFloat(e.target.value) || 0})} className="w-full border p-2 rounded" /></div>
          <div><label className="block text-xs font-bold mb-1">Graph Time Max</label>
            <input type="number" value={params.time} onChange={(e) => setParams({...params, time: parseFloat(e.target.value) || 0})} className="w-full border p-2 rounded" /></div>
        </div>
      </div>

      <button onClick={solve} className="w-full py-4 bg-slate-800 text-white font-bold rounded-lg hover:bg-slate-900 transition">
        Solve & Graph
      </button>

      {solution && (
        <div className="mt-8 border border-blue-200 rounded-lg overflow-hidden">
          {solution.error ? (
            <div className="p-6 text-red-600 font-bold bg-red-50">{solution.error}</div>
          ) : (
            <div>
              <div className="bg-blue-50 p-6 border-b border-blue-200">
                <p className="text-sm text-gray-500 font-bold uppercase mb-2">Exact Mathematical Solution</p>
                <div className="bg-white p-4 rounded border overflow-x-auto text-xl">
                  <BlockMath math={solution.latex_sol} />
                </div>
              </div>
              
              <div className="flex justify-center bg-white p-4">
                <Plot 
                  data={[{ x: solution.t, y: solution.y, type: 'scatter', mode: 'lines', name: 'y(t)', line: { color: '#2563eb', width: 3 } }]} 
                  layout={{ title: 'System Response', xaxis: {title: 'Time (t)'}, yaxis: {title: 'Amplitude y(t)'}, width: 750, height: 400, margin: { l: 50, r: 50, b: 50, t: 50 } }} 
                />
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}