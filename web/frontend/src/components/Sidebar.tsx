import type { ModelConfig, ModelStatus } from "../types";

const DOMAIN_ICONS: Record<string, string> = { math: "∑", law: "§", history: "H" };

interface Props {
  models: Record<string, ModelConfig>;
  selectedModel: string;
  onModelChange: (m: string) => void;
  domain: string;
  onDomainChange: (d: string) => void;
  layer: number;
  onLayerChange: (l: number) => void;
  alpha: number;
  onAlphaChange: (a: number) => void;
  strategy: string;
  onStrategyChange: (s: string) => void;
  topK: number;
  onTopKChange: (k: number) => void;
  maxTokens: number;
  onMaxTokensChange: (t: number) => void;
  status: ModelStatus | null;
  loading: boolean;
}

export default function Sidebar({
  models, selectedModel, onModelChange,
  domain, onDomainChange,
  layer, onLayerChange,
  alpha, onAlphaChange,
  strategy, onStrategyChange,
  topK, onTopKChange,
  maxTokens, onMaxTokensChange,
  status, loading,
}: Props) {
  const cfg = models[selectedModel];
  const saeLayer = cfg?.layer ?? 14;
  const layersTotal = cfg?.layers_total ?? 28;
  const isSaeLayer = layer === saeLayer;

  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <div className="title">STEERING ARENA</div>
        <div className="subtitle">Activation Steering Research</div>
      </div>

      <div className="sidebar-section">
        <div className="sidebar-label">MODEL</div>
        <select value={selectedModel} onChange={(e) => onModelChange(e.target.value)}>
          {Object.keys(models).map((k) => (
            <option key={k} value={k}>{k}</option>
          ))}
        </select>
      </div>

      <div className="sidebar-section">
        <div className="sidebar-label">DOMAIN</div>
        <select value={domain} onChange={(e) => onDomainChange(e.target.value)}>
          {["math", "law", "history"].map((d) => (
            <option key={d} value={d}>{DOMAIN_ICONS[d]} {d.toUpperCase()}</option>
          ))}
        </select>
      </div>

      <div className="sidebar-section">
        <div className="sidebar-label">LAYER</div>
        <div className="range-wrapper">
          <input
            type="range" min={0} max={layersTotal - 1} value={layer}
            onChange={(e) => onLayerChange(Number(e.target.value))}
          />
          <span className="range-value">{layer}</span>
        </div>
        <div className={`sae-badge ${isSaeLayer ? "active" : ""}`}>
          {isSaeLayer
            ? `SAE native layer (L${saeLayer})`
            : `SAE trained at L${saeLayer} — using cross-layer (retrain recommended)`}
        </div>
      </div>

      <div className="sidebar-section">
        <div className="sidebar-label">STEERING COEFFICIENT (α)</div>
        <div className="range-wrapper">
          <input
            type="range" min={0} max={60} step={1} value={alpha}
            onChange={(e) => onAlphaChange(Number(e.target.value))}
          />
          <span className="range-value">{alpha}</span>
        </div>
      </div>

      <div className="sidebar-section">
        <div className="sidebar-label">SAE STRATEGY</div>
        <select
          value={strategy} onChange={(e) => onStrategyChange(e.target.value)}
        >
          <option value="weighted">Weighted (top-k × diff)</option>
          <option value="uniform">Uniform (top-k equal)</option>
          <option value="single">Single (best feature)</option>
        </select>
      </div>

      <div className="sidebar-section">
        <div className="sidebar-label">TOP-K FEATURES</div>
        <div className="range-wrapper">
          <input
            type="range" min={1} max={50} value={topK}
            onChange={(e) => onTopKChange(Number(e.target.value))}
          />
          <span className="range-value">{topK}</span>
        </div>
      </div>

      <div className="sidebar-section">
        <div className="sidebar-label">MAX TOKENS</div>
        <div className="range-wrapper">
          <input
            type="range" min={32} max={512} step={32} value={maxTokens}
            onChange={(e) => onMaxTokensChange(Number(e.target.value))}
          />
          <span className="range-value">{maxTokens}</span>
        </div>
      </div>

      <div className="sidebar-divider" />

      <div className="sidebar-label" style={{ marginBottom: 8 }}>RESOURCES</div>
      <div className="status-badge">
        <div className="status-row">
          <div className="status-dot" style={{ background: status?.has_contrastive ? "#22c55e" : "#d4d4d4" }} />
          <span className="status-text">Contrastive vectors</span>
        </div>
        <div className="status-row">
          <div className="status-dot" style={{ background: status?.has_sae ? "#22c55e" : "#d4d4d4" }} />
          <span className="status-text">SAE features</span>
        </div>
        <div className="status-row">
          <div className="status-dot" style={{ background: status?.loaded_model ? "#22c55e" : loading ? "#ca8a04" : "#d4d4d4" }} />
          <span className="status-text">
            {status?.loaded_model ? status.loaded_model : loading ? "Loading..." : "No model loaded"}
          </span>
        </div>
      </div>

      <div style={{ marginTop: "auto", paddingTop: 32, fontSize: "0.65rem", color: "#a3a3a3", lineHeight: 1.6 }}>
        Developed by Arthur EDMOND
      </div>
    </aside>
  );
}
