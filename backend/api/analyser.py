
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
