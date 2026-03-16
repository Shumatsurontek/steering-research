import type { ModelConfig } from "../types";

interface Props {
  modelName: string;
  config: ModelConfig;
  layer: number;
}

export default function HeroBanner({ modelName, config, layer }: Props) {
  return (
    <div className="hero-banner">
      <div>
        <div className="hero-label">Activation Steering Arena</div>
        <div className="hero-title">Compare steering methods in real time</div>
        <div className="hero-subtitle">
          Side-by-side generation with contrastive vectors, SAE features, and baseline
        </div>
      </div>
      <div className="hero-model-badge">
        <div className="hero-model-name">{modelName}</div>
        <div className="hero-model-info">
          {config.params} params · Layer {layer}/{config.layers_total} · SAE 8x
        </div>
      </div>
    </div>
  );
}
