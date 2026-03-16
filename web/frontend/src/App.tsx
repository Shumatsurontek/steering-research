import { useState, useEffect, useCallback } from "react";
import Sidebar from "./components/Sidebar";
import HeroBanner from "./components/HeroBanner";
import ArenaPanel from "./components/ArenaPanel";
import VectorViz from "./components/VectorViz";
import LoadingOverlay from "./components/LoadingOverlay";
import { fetchModels, fetchModelStatus, loadModel } from "./api";
import type { ModelConfig, ModelStatus, Tab } from "./types";

export default function App() {
  // Models
  const [models, setModels] = useState<Record<string, ModelConfig>>({});
  const [selectedModel, setSelectedModel] = useState("Qwen3-0.6B");
  const [status, setStatus] = useState<ModelStatus | null>(null);

  // Controls
  const [domain, setDomain] = useState("math");
  const [layer, setLayer] = useState(14);
  const [alpha, setAlpha] = useState(10);
  const [strategy, setStrategy] = useState("weighted");
  const [topK, setTopK] = useState(20);
  const [maxTokens, setMaxTokens] = useState(128);

  // UI state
  const [tab, setTab] = useState<Tab>("arena");
  const [loadingModel, setLoadingModel] = useState(false);
  const [loadingMsg, setLoadingMsg] = useState("");

  // Fetch models on mount, auto-load default model if nothing loaded
  useEffect(() => {
    fetchModels().then(setModels).catch(console.error);
    fetchModelStatus().then(async (s) => {
      setStatus(s);
      if (!s.loaded_model) {
        setLoadingModel(true);
        setLoadingMsg(`Loading ${selectedModel}...`);
        try {
          const updated = await loadModel(selectedModel);
          setStatus(updated);
        } catch (e) {
          console.error(e);
        } finally {
          setLoadingModel(false);
          setLoadingMsg("");
        }
      }
    }).catch(console.error);
  }, []);

  // Sync layer when model changes
  useEffect(() => {
    const cfg = models[selectedModel];
    if (cfg) setLayer(cfg.layer);
  }, [selectedModel, models]);

  const handleModelChange = useCallback(async (model: string) => {
    setSelectedModel(model);
    setLoadingModel(true);
    setLoadingMsg(`Loading ${model}...`);
    try {
      const s = await loadModel(model);
      setStatus(s);
    } catch (e) {
      console.error(e);
    } finally {
      setLoadingModel(false);
      setLoadingMsg("");
    }
  }, []);

  const cfg = models[selectedModel];

  return (
    <div className="app-layout">
      {loadingModel && (
        <LoadingOverlay message={loadingMsg} sub="Loading model, contrastive vectors, and SAE features..." />
      )}

      <Sidebar
        models={models}
        selectedModel={selectedModel}
        onModelChange={handleModelChange}
        domain={domain}
        onDomainChange={setDomain}
        layer={layer}
        onLayerChange={setLayer}
        alpha={alpha}
        onAlphaChange={setAlpha}
        strategy={strategy}
        onStrategyChange={setStrategy}
        topK={topK}
        onTopKChange={setTopK}
        maxTokens={maxTokens}
        onMaxTokensChange={setMaxTokens}
        status={status}
        loading={loadingModel}
      />

      <main className="main-content">
        {cfg && <HeroBanner modelName={selectedModel} config={cfg} layer={layer} />}

        <div className="tabs">
          <button className={`tab ${tab === "arena" ? "active" : ""}`} onClick={() => setTab("arena")}>
            ⚡ ARENA
          </button>
          <button className={`tab ${tab === "vectors" ? "active" : ""}`} onClick={() => setTab("vectors")}>
            📐 VECTOR SPACE
          </button>
        </div>

        {tab === "arena" && (
          <ArenaPanel
            domain={domain}
            layer={layer}
            alpha={alpha}
            strategy={strategy}
            maxTokens={maxTokens}
            topK={topK}
            modelLoaded={!!status?.loaded_model}
            saeLayer={cfg?.layer ?? 14}
          />
        )}

        {tab === "vectors" && (
          <VectorViz layer={layer} modelLoaded={!!status?.loaded_model} />
        )}
      </main>
    </div>
  );
}
