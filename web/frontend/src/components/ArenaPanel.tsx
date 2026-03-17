import { useState, useCallback, useRef, useEffect } from "react";
import ChatCard from "./ChatCard";
import { streamGenerate } from "../api";
import type { GenerateRequest, StreamState, StreamStats, SSEConfig } from "../types";

const DOMAIN_ICONS: Record<string, string> = { math: "∑", law: "§", history: "H" };
const DOMAIN_COLORS: Record<string, string> = { math: "#dc2626", law: "#0a0a0a", history: "#ca8a04" };

const DOMAIN_PROMPTS: Record<string, string[]> = {
  math: [
    "Solve the equation 3x + 7 = 22 for x.",
    "What is the derivative of sin(x) * cos(x)?",
    "Prove that the square root of 2 is irrational.",
    "Calculate the integral of e^x from 0 to 1.",
    "Find the eigenvalues of the matrix [[2, 1], [1, 2]].",
    "What is the probability of rolling two sixes with two dice?",
    "Simplify the expression (x^2 - 4)/(x - 2).",
    "How many ways can you arrange 5 books on a shelf?",
    "What is the Taylor series expansion of ln(1+x)?",
    "Solve the differential equation dy/dx = 2xy.",
  ],
  law: [
    "What is the difference between civil and criminal law?",
    "Explain the concept of habeas corpus.",
    "What are the elements of a valid contract?",
    "Define the legal principle of stare decisis.",
    "What is the Miranda warning and when must it be given?",
    "Explain the doctrine of sovereign immunity.",
    "What constitutes negligence in tort law?",
    "What is the difference between a felony and a misdemeanor?",
    "Explain the concept of due process under the 14th Amendment.",
    "What are the requirements for obtaining a patent?",
  ],
  history: [
    "What caused the fall of the Roman Empire?",
    "Describe the main events of the French Revolution.",
    "What was the significance of the Magna Carta?",
    "Explain the causes of World War I.",
    "What was the impact of the Industrial Revolution on society?",
    "Describe the civil rights movement in the United States.",
    "What were the consequences of the Treaty of Versailles?",
    "Explain the rise and fall of the Ottoman Empire.",
    "What was the significance of the Silk Road?",
    "Describe the colonization of the Americas by European powers.",
  ],
};

function randomPrompt(domain: string): string {
  const prompts = DOMAIN_PROMPTS[domain] ?? [];
  return prompts[Math.floor(Math.random() * prompts.length)] ?? "";
}

interface Props {
  domain: string;
  layer: number;
  alpha: number;
  strategy: string;
  maxTokens: number;
  topK: number;
  modelLoaded: boolean;
  saeLayer: number;
}

export default function ArenaPanel({ domain, layer, alpha, strategy, maxTokens, topK, modelLoaded, saeLayer }: Props) {
  const [prompt, setPrompt] = useState(() => randomPrompt(domain));
  const [streaming, setStreaming] = useState(false);
  const [texts, setTexts] = useState<StreamState>({ baseline: "", contrastive: "", feature: "" });
  const [stats, setStats] = useState<StreamStats>({});
  const [config, setConfig] = useState<SSEConfig | null>(null);
  const [submitted, setSubmitted] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  // Random prompt on domain change
  useEffect(() => {
    setPrompt(randomPrompt(domain));
  }, [domain]);

  const handleGenerate = useCallback(() => {
    if (!prompt.trim() || !modelLoaded) return;

    setTexts({ baseline: "", contrastive: "", feature: "" });
    setStats({});
    setConfig(null);
    setStreaming(true);
    setSubmitted(true);

    const body: GenerateRequest = {
      prompt: prompt.trim(),
      domain,
      layer,
      alpha,
      feature_strategy: strategy,
      max_tokens: maxTokens,
      top_k: topK,
    };

    abortRef.current = streamGenerate(
      body,
      (event, data) => {
        if (event === "config") {
          setConfig(data as unknown as SSEConfig);
          return;
        }

        const method = event.replace(/:done$/, "").replace(/:error$/, "") as keyof StreamState;

        if (event.endsWith(":done")) {
          setStats((prev) => ({
            ...prev,
            [method]: { tokens: data.tokens as number, elapsed: data.elapsed as number },
          }));
        } else if (!event.endsWith(":error") && (method === "baseline" || method === "contrastive" || method === "feature")) {
          setTexts((prev) => ({ ...prev, [method]: data.text as string }));
        }
      },
      () => setStreaming(false),
      (err) => {
        console.error("Stream error:", err);
        setStreaming(false);
      },
    );
  }, [prompt, domain, layer, alpha, strategy, maxTokens, topK, modelLoaded]);

  const methods = config?.methods ?? ["baseline", "contrastive", "feature"];
  const showFeature = methods.includes("feature");
  const saeNative = config?.sae_native_layer ?? (layer === saeLayer);
  const gridClass = showFeature ? "chat-grid" : "chat-grid two-col";

  const wordSet = (s: string) => new Set(s.toLowerCase().split(/\s+/).filter(Boolean));
  const jaccard = (a: Set<string>, b: Set<string>) => {
    const union = new Set([...a, ...b]);
    if (union.size === 0) return 0;
    const inter = new Set([...a].filter((x) => b.has(x)));
    return inter.size / union.size;
  };

  const bw = wordSet(texts.baseline);
  const cJac = jaccard(bw, wordSet(texts.contrastive));
  const fJac = jaccard(bw, wordSet(texts.feature));

  return (
    <div>
      <div className="prompt-area">
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder={`Ask a ${domain} question...`}
          onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleGenerate(); } }}
        />
      </div>

      <button
        className="btn-generate"
        onClick={handleGenerate}
        disabled={streaming || !modelLoaded || !prompt.trim()}
      >
        {streaming ? "GENERATING..." : !modelLoaded ? "LOAD MODEL FIRST" : "GENERATE"}
      </button>

      {/* SAE cross-layer warning */}
      {showFeature && !saeNative && submitted && (
        <div style={{
          padding: "12px 16px",
          marginBottom: 20,
          border: "1px solid #e5e5e5",
          borderRadius: 4,
          fontSize: "0.8rem",
          lineHeight: 1.5,
          color: "#525252",
        }}>
          <span style={{ fontWeight: 600, color: "#ca8a04" }}>Cross-layer SAE</span>{" — "}
          SAE was trained at layer {saeLayer} but applied at layer {layer}.
          Feature directions may be less meaningful. Retrain SAE at this layer with:{" "}
          <code style={{ fontFamily: "var(--mono)", fontSize: "0.75rem", background: "#f5f5f5", padding: "2px 4px", borderRadius: 2 }}>
            python -m src.steering.train_sae --layer {layer}
          </code>
        </div>
      )}

      {submitted && (
        <div className="user-prompt" style={{ borderColor: `${DOMAIN_COLORS[domain]}20` }}>
          <div className="user-prompt-icon">
            {DOMAIN_ICONS[domain] ?? "?"}
          </div>
          <div>
            <div className="user-prompt-label">{domain} prompt</div>
            <div className="user-prompt-text">{prompt}</div>
          </div>
        </div>
      )}

      {submitted && (
        <>
          <div className={gridClass}>
            <ChatCard method="baseline" text={texts.baseline} alpha={0} streaming={streaming} stats={stats.baseline} />
            {methods.includes("contrastive") && (
              <ChatCard method="contrastive" text={texts.contrastive} alpha={alpha} streaming={streaming} stats={stats.contrastive} />
            )}
            {showFeature && (
              <ChatCard method="feature" text={texts.feature} alpha={alpha} streaming={streaming} strategy={strategy} stats={stats.feature} />
            )}
          </div>

          {!streaming && texts.baseline && (
            <div className="stats-row">
              <div className="stat-card">
                <div className="stat-label">Baseline</div>
                <div className="stat-value" style={{ color: "#737373" }}>{texts.baseline.split(/\s+/).length} words</div>
                <div className="stat-sub">{texts.baseline.length} chars</div>
              </div>
              <div className="stat-card">
                <div className="stat-label">Contrastive</div>
                <div className="stat-value" style={{ color: "#0a0a0a" }}>{texts.contrastive.split(/\s+/).length} words</div>
                <div className="stat-sub">{texts.contrastive.length} chars</div>
              </div>
              {showFeature && (
                <div className="stat-card">
                  <div className="stat-label">Feature</div>
                  <div className="stat-value" style={{ color: "#6d28d9" }}>{texts.feature.split(/\s+/).length} words</div>
                  <div className="stat-sub">{texts.feature.length} chars</div>
                </div>
              )}
              <div className="stat-card">
                <div className="stat-label">Jaccard vs Base</div>
                <div className="stat-value" style={{ color: "#0a0a0a" }}>
                  {(cJac * 100).toFixed(0)}% / {(fJac * 100).toFixed(0)}%
                </div>
                <div className="stat-sub">contr. / feat.</div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
