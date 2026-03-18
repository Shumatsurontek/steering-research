export interface ModelConfig {
  model_id: string;
  layer: number;
  sae_dir: string;
  vectors: string;
  params: string;
  layers_total: number;
}

export interface ModelStatus {
  loaded_model: string | null;
  has_contrastive: boolean;
  has_sae: boolean;
}

export interface GenerateRequest {
  prompt: string;
  domain: string;
  layer: number;
  alpha: number;
  feature_strategy: string;
  max_tokens: number;
  top_k: number;
  steering_mode: string;
}

export interface SSEConfig {
  methods: string[];
  sae_available: boolean;
  sae_native_layer: boolean;
  sae_layer: number;
}

export interface StreamState {
  baseline: string;
  contrastive: string;
  feature: string;
}

export interface StreamStats {
  baseline?: { tokens: number; elapsed: number };
  contrastive?: { tokens: number; elapsed: number };
  feature?: { tokens: number; elapsed: number };
}

export interface PCAPoint {
  x: number;
  y: number;
  label: string;
  type: string;
  domain: string;
  color: string;
}

export interface VizData {
  pca: { points: PCAPoint[]; variance: number[] };
  cosine: { matrix: number[][]; labels: string[] };
  norms: { bars: { label: string; norm: number; color: string }[] };
  feature_info: Record<string, { top_features: number[]; top_diffs: string[] }>;
  layer: number;
  sae_available: boolean;
}

export type Tab = "arena" | "vectors" | "benchmarks";
