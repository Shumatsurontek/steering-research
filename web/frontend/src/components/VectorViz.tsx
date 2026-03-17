import { useState, useEffect } from "react";
import Plot from "react-plotly.js";
import { fetchVectorViz } from "../api";
import type { VizData } from "../types";

const PLOTLY_LIGHT = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "#fafafa",
  font: { color: "#525252", family: "Inter, sans-serif", size: 11 },
};

const AXIS_STYLE = { gridcolor: "#e5e5e5", zerolinecolor: "#d4d4d4", linecolor: "#e5e5e5" };

const DOMAIN_ICONS: Record<string, string> = { math: "∑", law: "§", history: "H" };

const SYMBOL_MAP: Record<string, string> = {
  contrastive: "diamond",
  "SAE-weighted": "circle",
  "SAE-uniform": "square",
  "SAE-single": "star",
};

interface Props {
  layer: number;
  modelLoaded: boolean;
}

export default function VectorViz({ layer, modelLoaded }: Props) {
  const [data, setData] = useState<VizData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    if (!modelLoaded) return;
    setLoading(true);
    setError(null);
    try {
      const viz = await fetchVectorViz(layer);
      setData(viz);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (modelLoaded) load();
  }, [modelLoaded, layer]);

  if (!modelLoaded) {
    return <div style={{ color: "#666", textAlign: "center", padding: 40 }}>Load a model to view vector visualizations.</div>;
  }

  if (loading) {
    return <div style={{ color: "#888", textAlign: "center", padding: 40 }}>Computing vector projections...</div>;
  }

  if (error) {
    return <div style={{ color: "#ff6b6b", textAlign: "center", padding: 40 }}>{error}</div>;
  }

  if (!data) return null;

  // Build PCA traces — use `any` to avoid Plotly's strict union types
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const pcaTraces: any[] = [];
  const points = data.pca.points;
  const types = [...new Set(points.map((p) => p.type))];
  const domains = [...new Set(points.map((p) => p.domain))];

  for (const vtype of types) {
    for (const domain of domains) {
      const filtered = points.filter((p) => p.type === vtype && p.domain === domain);
      if (!filtered.length) continue;
      pcaTraces.push({
        x: filtered.map((p) => p.x),
        y: filtered.map((p) => p.y),
        mode: "markers+text",
        type: "scatter",
        marker: {
          symbol: SYMBOL_MAP[vtype] || "circle",
          size: 13,
          color: filtered[0].color,
          line: { width: 1.5, color: "rgba(0,0,0,0.5)" },
        },
        text: filtered.map(() => vtype.replace("SAE-", "")),
        textposition: "top center",
        textfont: { size: 9, color: "#999" },
        name: `${domain} · ${vtype}`,
        legendgroup: domain,
      });
    }
  }

  const shortLabels = data.cosine.labels.map((l) =>
    l.replace("contrastive", "contr.").replace("weighted", "wt.").replace("uniform", "uni.")
  );

  return (
    <div>
      <button className="btn-generate" onClick={load} disabled={loading} style={{ marginBottom: 16 }}>
        {loading ? "COMPUTING..." : "REFRESH VISUALIZATIONS"}
      </button>

      <div className="viz-grid">
        <div className="viz-card">
          <Plot
            data={pcaTraces}
            layout={{
              title: { text: "Steering Vectors — PCA Projection" },
              xaxis: { title: { text: `PC1 (${(data.pca.variance[0] * 100).toFixed(1)}% var.)` }, ...AXIS_STYLE },
              yaxis: { title: { text: `PC2 (${(data.pca.variance[1] * 100).toFixed(1)}% var.)` }, ...AXIS_STYLE },
              height: 480,
              ...PLOTLY_LIGHT,
              legend: { font: { size: 10 }, bgcolor: "rgba(0,0,0,0)" },
            }}
            config={{ responsive: true }}
            style={{ width: "100%" }}
          />
        </div>
        <div className="viz-card">
          <Plot
            data={[
              {
                z: data.cosine.matrix,
                x: shortLabels,
                y: shortLabels,
                type: "heatmap",
                colorscale: [
                  [0, "#f5f3ff"],
                  [0.5, "#ddd6fe"],
                  [1, "#0a0a0a"],
                ],
                hovertemplate: "%{x}<br>%{y}<br>%{z:.2f}<extra></extra>",
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              } as any,
            ]}
            layout={{
              title: { text: "Cosine Similarity" },
              height: 480,
              ...PLOTLY_LIGHT,
              xaxis: { ...AXIS_STYLE, tickangle: -45 },
              yaxis: AXIS_STYLE,
              margin: { l: 100, b: 100, t: 40, r: 20 },
            }}
            config={{ responsive: true }}
            style={{ width: "100%" }}
          />
        </div>
      </div>

      <div className="viz-card" style={{ marginBottom: 16 }}>
        <Plot
          data={[
            {
              x: data.norms.bars.map((b) => b.label),
              y: data.norms.bars.map((b) => b.norm),
              type: "bar",
              marker: { color: data.norms.bars.map((b) => b.color), opacity: 0.85 },
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            } as any,
          ]}
          layout={{
            title: { text: "Vector L2 Norms" },
            height: 350,
            ...PLOTLY_LIGHT,
            xaxis: { ...AXIS_STYLE, tickangle: -45 },
            yaxis: { ...AXIS_STYLE, title: { text: "L2 Norm" } },
            showlegend: false,
          }}
          config={{ responsive: true }}
          style={{ width: "100%" }}
        />
      </div>

      {data.sae_available && data.feature_info && Object.keys(data.feature_info).length > 0 && (
        <div className="feature-grid">
          {["math", "law", "history"].map((d) => {
            const info = data.feature_info[d];
            if (!info) return null;
            const colors: Record<string, string> = { math: "#ff6b6b", law: "#4ecdc4", history: "#ffd93d" };
            return (
              <div key={d} className="feature-card" style={{ border: `1px solid ${colors[d]}25` }}>
                <div className="feature-title" style={{ color: colors[d] }}>
                  {DOMAIN_ICONS[d]} {d.toUpperCase()} — Top Features
                </div>
                {info.top_features.map((fid: number, i: number) => (
                  <div key={fid} className="feature-row">
                    <span className="feature-id">#{fid}</span>
                    <span className="feature-diff" style={{ color: colors[d] }}>
                      Δ={info.top_diffs[i]}
                    </span>
                  </div>
                ))}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
