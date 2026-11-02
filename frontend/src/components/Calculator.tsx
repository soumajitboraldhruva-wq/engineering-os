import { useState } from 'react';
import axios from 'axios';

export default function CalcModule() {
  const [expr, setExpr] = useState('');
  const [result, setResult] = useState<string | null>(null);

  const calculate = async () => {
    try {
      // Calls our Python backend
      const res = await axios.post('http://127.0.0.1:8000/api/calc/', { expression: expr });
      if (res.data.error) setResult('Error: ' + res.data.error);
      else setResult(res.data.result.toString());
    } catch (e) {
      setResult('Backend disconnected');
    }
  };

  return (
    <div className="max-w-2xl bg-white p-8 rounded-xl shadow-sm border border-gray-200">
      <h2 className="text-2xl font-bold text-gray-800 mb-6">Scientific Calculator</h2>
      
      <div className="bg-slate-100 p-4 rounded-lg mb-6 shadow-inner">
        <input 
          type="text" 
          value={expr}
          onChange={(e) => setExpr(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && calculate()}
          placeholder="e.g. sin(pi/4) + sqrt(16)"
          className="w-full bg-transparent text-2xl outline-none text-right font-mono text-slate-800"
        />
        <div className="text-right text-3xl font-bold text-blue-600 mt-2 min-h-[40px]">
          {result !== null ? `= ${result}` : ''}
        </div>
      </div>

      <button onClick={calculate} className="w-full py-4 bg-blue-600 text-white font-bold rounded-lg hover:bg-blue-700 transition">
        Calculate [Enter]
      </button>
    </div>
  );
}