import { useState, useEffect, useRef, useCallback } from "react";
import Plot from "react-plotly.js";

interface BenchmarkEntry {
  model: string;
  domain: string;
  method: string;
  accuracy: number;
  delta: number;
  stderr: number;
}

interface OptionScore {
  key: string;
  text: string;
  loglikelihood: number | null;
  selected: boolean;
}

interface SampleEntry {
  question: string;
  correct: string;
  correct_text: string;
  options: OptionScore[];
  results: Record<string, {
    answer: string;
    answer_text: string;
    correct: boolean;
    options: OptionScore[];
  }>;
}

const BENCHMARKS = [
  { key: "vectors", label: "Contrastive Vectors", desc: "Extract MMLU-Pro domain vectors (all layers)" },
  { key: "sae_train", label: "Train SAE", desc: "Train sparse autoencoder on residual stream" },
  { key: "sae_analysis", label: "SAE Analysis", desc: "Domain feature analysis + contrastive overlap" },
  { key: "feature_targeted", label: "Feature Benchmark", desc: "MMLU-Pro MC eval (baseline vs contrastive vs feature)" },
];

const DOMAIN_ORDER = ["math", "law", "history"];
const MODEL_ORDER = ["Qwen3-0.6B", "Qwen3-4B", "LFM2-700M"];

const PLOTLY_LIGHT = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "#fafafa",
  font: { color: "#525252", family: "Inter, sans-serif", size: 11 },
};

interface Props {
  modelLoaded: boolean;
}

export default function BenchmarkPanel({ modelLoaded }: Props) {
  const [data, setData] = useState<BenchmarkEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string>("all");
  const [selectedDomain, setSelectedDomain] = useState<string>("all");

  // Run state
  const [runModel, setRunModel] = useState("Qwen3-0.6B");
  const [runBenchmark, setRunBenchmark] = useState("feature_targeted");
  const [runDomain, setRunDomain] = useState("all");
  const [runLimit, setRunLimit] = useState(50);
  const [running, setRunning] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [runStatus, setRunStatus] = useState<string | null>(null);
  const logRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Sample viewer
  const [samples, setSamples] = useState<SampleEntry[] | null>(null);
  const [samplesLoading, setSamplesLoading] = useState(false);

  const loadResults = useCallback(() => {
    setLoading(true);
    fetch("/api/benchmarks/summary")
      .then((r) => r.json())
      .then((d) => { setData(d); setLoading(false); })
      .catch(() => setLoading(false));
  }, []);

  useEffect(() => { loadResults(); }, [loadResults]);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logs]);

  const handleRun = useCallback(async () => {
    setRunning(true);
    setLogs([]);
    setRunStatus(null);
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const res = await fetch("/api/benchmarks/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: runModel, benchmark: runBenchmark, limit: runLimit, domain: runDomain }),
        signal: controller.signal,
      });

      if (!res.ok) {
        const errText = await res.text();
        setLogs((prev) => [...prev, `ERROR: ${errText}`]);
        setRunStatus("error");
        setRunning(false);
        return;
      }

      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        let currentEvent = "";
        for (const line of lines) {
          if (line.startsWith("event: ")) {
            currentEvent = line.slice(7).trim();
          } else if (line.startsWith("data: ") && currentEvent) {
            try {
              const d = JSON.parse(line.slice(6));
              if (currentEvent === "log") setLogs((prev) => [...prev, d.line]);
              else if (currentEvent === "start") setLogs((prev) => [...prev, `$ ${d.cmd}`]);
              else if (currentEvent === "done") {
                setRunStatus(d.status);
                if (d.status === "success") loadResults();
              }
            } catch { /* skip */ }
            currentEvent = "";
          }
        }
      }
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        setLogs((prev) => [...prev, `ERROR: ${(err as Error).message}`]);
        setRunStatus("error");
      }
    } finally {
      setRunning(false);
    }
  }, [runModel, runBenchmark, runLimit, runDomain, loadResults]);

  const handleCancel = useCallback(async () => {
    abortRef.current?.abort();
    const job = `${runModel}:${runBenchmark}`;
    await fetch(`/api/benchmarks/cancel?job=${encodeURIComponent(job)}`, { method: "POST" }).catch(() => {});
    setRunning(false);
    setRunStatus("cancelled");
  }, [runModel, runBenchmark]);

  // Load samples
  const loadSamples = useCallback(async (model: string, domain: string) => {
    setSamplesLoading(true);
    try {
      const res = await fetch(`/api/benchmarks/samples?model=${encodeURIComponent(model)}&domain=${encodeURIComponent(domain)}`);
      if (res.ok) {
        const d = await res.json();
        setSamples(d.samples || []);
      } else {
        setSamples(null);
      }
    } catch {
      setSamples(null);
    } finally {
      setSamplesLoading(false);
    }
  }, []);

  // ── Filtered data for both chart and table ────────────────────────────
  const filtered = data.filter((d) =>
    (selectedModel === "all" || d.model === selectedModel) &&
    (selectedDomain === "all" || d.domain === selectedDomain)
  );

  const models = [...new Set(data.map((d) => d.model))];
  const domains = [...new Set(data.map((d) => d.domain))];

  // Build chart from FILTERED data
  const baselineByKey: Record<string, number> = {};
  const bestContrastiveByKey: Record<string, number> = {};
  const bestFeatureByKey: Record<string, number> = {};

  for (const entry of filtered) {
    const key = `${entry.model}|${entry.domain}`;
    if (entry.method === "baseline") baselineByKey[key] = entry.accuracy;
    if (entry.method.startsWith("contrastive")) {
      if ((bestContrastiveByKey[key] ?? -1) < entry.accuracy) bestContrastiveByKey[key] = entry.accuracy;
    }
    if (entry.method.startsWith("feat_")) {
      if ((bestFeatureByKey[key] ?? -1) < entry.accuracy) bestFeatureByKey[key] = entry.accuracy;
    }
  }

  const categories: string[] = [];
  const baselineVals: number[] = [];
  const contrastiveVals: number[] = [];
  const featureVals: number[] = [];

  for (const model of MODEL_ORDER) {
    for (const domain of DOMAIN_ORDER) {
      const key = `${model}|${domain}`;
      if (baselineByKey[key] == null) continue;
      categories.push(`${model}\n${domain}`);
      baselineVals.push(baselineByKey[key] * 100);
      contrastiveVals.push((bestContrastiveByKey[key] ?? 0) * 100);
      featureVals.push((bestFeatureByKey[key] ?? 0) * 100);
    }
  }

  const selectedBenchInfo = BENCHMARKS.find((b) => b.key === runBenchmark);

  return (
    <div>
      {/* ── Run Panel ─────────────────────────────────────────── */}
      <div style={{ border: "1px solid #e5e5e5", borderRadius: 4, padding: 24, marginBottom: 32 }}>
        <div style={sectionTitle}>RUN BENCHMARK</div>

        <div style={{ display: "flex", gap: 12, marginBottom: 16, flexWrap: "wrap" }}>
          <div>
            <div className="sidebar-label">MODEL</div>
            <select value={runModel} onChange={(e) => setRunModel(e.target.value)} style={{ width: 180 }}>
              {MODEL_ORDER.map((m) => <option key={m} value={m}>{m}</option>)}
            </select>
          </div>
          <div>
            <div className="sidebar-label">BENCHMARK</div>
            <select value={runBenchmark} onChange={(e) => setRunBenchmark(e.target.value)} style={{ width: 220 }}>
              {BENCHMARKS.map((b) => <option key={b.key} value={b.key}>{b.label}</option>)}
            </select>
          </div>
          {runBenchmark === "feature_targeted" && (
            <>
              <div>
                <div className="sidebar-label">DOMAIN</div>
                <select value={runDomain} onChange={(e) => setRunDomain(e.target.value)} style={{ width: 140 }}>
                  <option value="all">All domains</option>
                  {DOMAIN_ORDER.map((d) => <option key={d} value={d}>{d}</option>)}
                </select>
              </div>
              <div>
                <div className="sidebar-label">N SAMPLES</div>
                <select value={runLimit} onChange={(e) => setRunLimit(Number(e.target.value))} style={{ width: 100 }}>
                  {[20, 50, 100, 200].map((n) => <option key={n} value={n}>{n}</option>)}
                </select>
              </div>
            </>
          )}
        </div>

        {selectedBenchInfo && (
          <div style={{ fontSize: "0.8rem", color: "#737373", marginBottom: 12 }}>{selectedBenchInfo.desc}</div>
        )}

        <div style={{ fontSize: "0.75rem", color: "#a3a3a3", marginBottom: 12 }}>
          Benchmarks run on the server. For faster execution on GPU/MPS, use <code style={codeStyle}>just dev-backend</code>.
        </div>

        <div style={{ display: "flex", gap: 8 }}>
          <button className="btn-generate" style={{ width: "auto", marginBottom: 0 }} onClick={handleRun} disabled={running}>
            {running ? "RUNNING..." : "RUN"}
          </button>
          {running && (
            <button className="btn-generate" style={{ width: "auto", marginBottom: 0, background: "#dc2626" }} onClick={handleCancel}>
              CANCEL
            </button>
          )}
        </div>

        {logs.length > 0 && (
          <div ref={logRef} style={logConsoleStyle}>
            {logs.map((line, i) => (
              <div key={i} style={{ color: line.startsWith("ERROR") ? "#ef4444" : line.startsWith("$") ? "#22c55e" : "#a3a3a3" }}>
                {line}
              </div>
            ))}
            {runStatus && (
              <div style={{ marginTop: 8, fontWeight: 600, color: runStatus === "success" ? "#22c55e" : "#ef4444" }}>
                [{runStatus.toUpperCase()}]
              </div>
            )}
          </div>
        )}
      </div>

      {/* ── Results ───────────────────────────────────────────── */}
      {loading ? (
        <div style={{ color: "#888", padding: 40, textAlign: "center" }}>Loading...</div>
      ) : !data.length ? (
        <div style={{ color: "#888", padding: 40, textAlign: "center" }}>No benchmark results found.</div>
      ) : (
        <>
          {/* Filters */}
          <div style={{ display: "flex", gap: 12, marginBottom: 24 }}>
            <div>
              <div className="sidebar-label">FILTER MODEL</div>
              <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)} style={{ width: 180 }}>
                <option value="all">All models</option>
                {models.map((m) => <option key={m} value={m}>{m}</option>)}
              </select>
            </div>
            <div>
              <div className="sidebar-label">FILTER DOMAIN</div>
              <select value={selectedDomain} onChange={(e) => setSelectedDomain(e.target.value)} style={{ width: 180 }}>
                <option value="all">All domains</option>
                {domains.map((d) => <option key={d} value={d}>{d}</option>)}
              </select>
            </div>
            {/* Sample viewer trigger */}
            {selectedModel !== "all" && selectedDomain !== "all" && (
              <div style={{ display: "flex", alignItems: "flex-end" }}>
                <button
                  className="btn-generate"
                  style={{ width: "auto", marginBottom: 0, background: "#525252" }}
                  onClick={() => loadSamples(selectedModel, selectedDomain)}
                  disabled={samplesLoading}
                >
                  {samplesLoading ? "LOADING..." : "VIEW SAMPLES"}
                </button>
              </div>
            )}
          </div>

          {/* Chart — updates with filters */}
          {categories.length > 0 && (
            <div className="viz-card" style={{ marginBottom: 24 }}>
              <Plot
                data={[
                  { x: categories, y: baselineVals, name: "Baseline", type: "bar", marker: { color: "#d4d4d4" } } as any,
                  { x: categories, y: contrastiveVals, name: "Best Contrastive", type: "bar", marker: { color: "#0a0a0a" } } as any,
                  { x: categories, y: featureVals, name: "Best Feature", type: "bar", marker: { color: "#6d28d9" } } as any,
                ]}
                layout={{
                  title: { text: "MMLU-Pro MC — Baseline vs Best Steering" },
                  barmode: "group",
                  height: 400,
                  ...PLOTLY_LIGHT,
                  xaxis: { tickangle: -45, gridcolor: "#e5e5e5" },
                  yaxis: { title: { text: "Accuracy (%)" }, gridcolor: "#e5e5e5", range: [0, Math.max(55, ...baselineVals, ...contrastiveVals, ...featureVals) + 5] },
                  legend: { orientation: "h", y: 1.12 },
                }}
                config={{ responsive: true }}
                style={{ width: "100%" }}
              />
            </div>
          )}

          {/* Table */}
          <div style={{ border: "1px solid #e5e5e5", borderRadius: 4, overflow: "hidden", marginBottom: 24 }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.8rem", fontFamily: "var(--mono)" }}>
              <thead>
                <tr style={{ background: "#fafafa", borderBottom: "1px solid #e5e5e5" }}>
                  <th style={thStyle}>Model</th>
                  <th style={thStyle}>Domain</th>
                  <th style={thStyle}>Method</th>
                  <th style={{ ...thStyle, textAlign: "right" }}>Acc</th>
                  <th style={{ ...thStyle, textAlign: "right" }}>Delta</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((entry, i) => {
                  const isPositive = entry.delta > 0;
                  const isBaseline = entry.method === "baseline";
                  return (
                    <tr key={i} style={{ borderBottom: "1px solid #f0f0f0", background: isBaseline ? "#fafafa" : "transparent" }}>
                      <td style={tdStyle}>{entry.model}</td>
                      <td style={tdStyle}>{entry.domain}</td>
                      <td style={tdStyle}>{entry.method}</td>
                      <td style={{ ...tdStyle, textAlign: "right" }}>{(entry.accuracy * 100).toFixed(1)}%</td>
                      <td style={{
                        ...tdStyle, textAlign: "right", fontWeight: isPositive ? 600 : 400,
                        color: isBaseline ? "#a3a3a3" : isPositive ? "#16a34a" : entry.delta < -5 ? "#dc2626" : "#525252",
                      }}>
                        {isBaseline ? "—" : `${entry.delta > 0 ? "+" : ""}${entry.delta.toFixed(1)}pp`}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* ── Sample Viewer ─────────────────────────────────── */}
          {samples !== null && (
            <div style={{ border: "1px solid #e5e5e5", borderRadius: 4, padding: 24 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
                <div style={sectionTitle}>
                  SAMPLE RESPONSES — {selectedModel} / {selectedDomain}
                </div>
                <button
                  onClick={() => setSamples(null)}
                  style={{ background: "none", border: "1px solid #e5e5e5", borderRadius: 4, padding: "4px 12px", cursor: "pointer", fontSize: "0.75rem", color: "#737373" }}
                >
                  CLOSE
                </button>
              </div>

              {samples.length === 0 ? (
                <div style={{ color: "#a3a3a3", fontSize: "0.85rem" }}>
                  No saved samples. Run a benchmark with sample saving enabled to see per-question results.
                </div>
              ) : (
                samples.map((sample, i) => (
                  <div key={i} style={{ borderBottom: "1px solid #e5e5e5", paddingBottom: 20, marginBottom: 20 }}>
                    {/* Question */}
                    <div style={{ fontSize: "0.85rem", fontWeight: 600, marginBottom: 8, color: "#0a0a0a", lineHeight: 1.5 }}>
                      <span style={{ fontFamily: "var(--mono)", color: "#a3a3a3", fontSize: "0.75rem" }}>Q{i + 1}</span>{" "}
                      {sample.question}
                    </div>

                    {/* Options list */}
                    {sample.options?.length > 0 && (
                      <div style={{ marginBottom: 12, fontSize: "0.75rem", lineHeight: 1.7, color: "#525252" }}>
                        {sample.options.map((opt) => (
                          <div key={opt.key} style={{
                            padding: "2px 0",
                            fontWeight: opt.key === sample.correct ? 600 : 400,
                            color: opt.key === sample.correct ? "#16a34a" : "#525252",
                          }}>
                            <span style={{ fontFamily: "var(--mono)", color: "#a3a3a3", marginRight: 6 }}>{opt.key}.</span>
                            {opt.text}
                            {opt.key === sample.correct && " ✓"}
                          </div>
                        ))}
                      </div>
                    )}

                    {/* Correct answer text */}
                    <div style={{ fontSize: "0.75rem", color: "#737373", marginBottom: 10 }}>
                      Correct: <span style={{ color: "#16a34a", fontWeight: 600 }}>{sample.correct}</span>
                      {sample.correct_text && (
                        <span style={{ color: "#525252" }}> — {sample.correct_text}</span>
                      )}
                    </div>

                    {/* Method predictions grid */}
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: 6 }}>
                      {Object.entries(sample.results).map(([method, res]) => {
                        // Find max LL for bar normalization
                        const opts = res.options || [];
                        const maxLL = opts.length > 0 ? Math.max(...opts.map((o) => o.loglikelihood ?? -Infinity)) : 0;
                        const minLL = opts.length > 0 ? Math.min(...opts.map((o) => o.loglikelihood ?? 0)) : 0;
                        const range = maxLL - minLL || 1;

                        return (
                          <div key={method} style={{
                            padding: "10px 12px", borderRadius: 4, fontSize: "0.72rem",
                            background: res.correct ? "#f0fdf4" : "#fef2f2",
                            border: `1px solid ${res.correct ? "#dcfce7" : "#fee2e2"}`,
                          }}>
                            <div style={{ fontWeight: 600, marginBottom: 4, color: "#525252", fontFamily: "var(--mono)", fontSize: "0.65rem", letterSpacing: "0.04em" }}>
                              {method}
                            </div>
                            <div style={{ fontWeight: 600, fontSize: "0.85rem", color: res.correct ? "#16a34a" : "#dc2626", marginBottom: opts.length > 0 ? 6 : 0 }}>
                              {res.answer}{res.answer_text ? ` — ${res.answer_text}` : ""}
                            </div>

                            {/* LL bar chart for each option */}
                            {opts.length > 0 && (
                              <div style={{ marginTop: 4 }}>
                                {opts.slice(0, 6).map((opt) => {
                                  const barWidth = opt.loglikelihood != null
                                    ? Math.max(3, ((opt.loglikelihood - minLL) / range) * 100)
                                    : 0;
                                  const isCorrectOpt = opt.key === sample.correct;
                                  const isSelected = opt.selected;
                                  return (
                                    <div key={opt.key} style={{ display: "flex", alignItems: "center", gap: 4, marginBottom: 1 }}>
                                      <span style={{
                                        fontFamily: "var(--mono)", fontSize: "0.6rem", width: 12, color: isCorrectOpt ? "#16a34a" : "#a3a3a3",
                                        fontWeight: isSelected || isCorrectOpt ? 700 : 400,
                                      }}>
                                        {opt.key}
                                      </span>
                                      <div style={{ flex: 1, height: 4, background: "#e5e5e5", borderRadius: 2, overflow: "hidden" }}>
                                        <div style={{
                                          width: `${barWidth}%`, height: "100%", borderRadius: 2,
                                          background: isSelected ? (res.correct ? "#16a34a" : "#dc2626") : isCorrectOpt ? "#86efac" : "#d4d4d4",
                                        }} />
                                      </div>
                                      <span style={{ fontFamily: "var(--mono)", fontSize: "0.55rem", color: "#a3a3a3", width: 36, textAlign: "right" }}>
                                        {opt.loglikelihood != null ? opt.loglikelihood.toFixed(1) : ""}
                                      </span>
                                    </div>
                                  );
                                })}
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                ))
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}

const sectionTitle: React.CSSProperties = {
  fontSize: "0.65rem", fontWeight: 600, letterSpacing: "0.1em",
  textTransform: "uppercase", color: "#a3a3a3", marginBottom: 0,
};

const thStyle: React.CSSProperties = {
  padding: "10px 12px", textAlign: "left", fontWeight: 600,
  fontSize: "0.65rem", letterSpacing: "0.1em", textTransform: "uppercase", color: "#a3a3a3",
};

const tdStyle: React.CSSProperties = { padding: "8px 12px", color: "#525252" };

const codeStyle: React.CSSProperties = {
  fontFamily: "var(--mono)", fontSize: "0.7rem", background: "#f5f5f5", padding: "1px 4px", borderRadius: 2,
};

const logConsoleStyle: React.CSSProperties = {
  marginTop: 16, background: "#0a0a0a", color: "#a3a3a3", fontFamily: "var(--mono)",
  fontSize: "0.72rem", lineHeight: 1.6, padding: 16, borderRadius: 4,
  maxHeight: 300, overflowY: "auto", whiteSpace: "pre-wrap", wordBreak: "break-all",
};
