const METHOD_STYLES: Record<string, { color: string; label: string }> = {
  baseline: { color: "#737373", label: "BASELINE" },
  contrastive: { color: "#0a0a0a", label: "CONTRASTIVE" },
  feature: { color: "#6d28d9", label: "SAE FEATURE" },
};

interface Props {
  method: string;
  text: string;
  alpha: number;
  streaming?: boolean;
  strategy?: string;
  stats?: { tokens: number; elapsed: number } | null;
}

export default function ChatCard({ method, text, alpha, streaming, strategy, stats }: Props) {
  const style = METHOD_STYLES[method] ?? METHOD_STYLES.baseline;
  const label = method === "feature" && strategy
    ? `SAE ${strategy.toUpperCase()}`
    : style.label;

  return (
    <div className="chat-card">
      <div className="chat-card-header">
        <div className="chat-card-method">
          <div className="method-dot" style={{ background: style.color }} />
          <span className="method-label" style={{ color: style.color }}>{label}</span>
        </div>
        <span className="alpha-badge">
          {alpha > 0 && method !== "baseline" ? `α=${alpha}` : "no steering"}
        </span>
      </div>

      <div className="chat-card-text">
        {text || <span className="chat-card-empty">Waiting for generation...</span>}
        {streaming && text && <span className="cursor" />}
      </div>

      {stats && (
        <div style={{ marginTop: 16, paddingTop: 12, borderTop: "1px solid #f0f0f0", fontSize: "0.7rem", fontFamily: "var(--mono)", color: "#a3a3a3" }}>
          {stats.tokens} tokens · {stats.elapsed}s · {(stats.tokens / stats.elapsed).toFixed(1)} tok/s
        </div>
      )}
    </div>
  );
}
