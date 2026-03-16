import type { ModelConfig, ModelStatus, GenerateRequest, VizData } from "./types";

const BASE = "/api";

export async function fetchModels(): Promise<Record<string, ModelConfig>> {
  const res = await fetch(`${BASE}/models`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function fetchModelStatus(): Promise<ModelStatus> {
  const res = await fetch(`${BASE}/models/status`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function loadModel(model: string): Promise<ModelStatus> {
  const res = await fetch(`${BASE}/models/load`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function recalculateSAE(model: string, top_k: number): Promise<ModelStatus> {
  const res = await fetch(`${BASE}/sae/recalculate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model, top_k }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function fetchVectorViz(layer?: number): Promise<VizData> {
  const params = layer != null ? `?layer=${layer}` : "";
  const res = await fetch(`${BASE}/vectors/visualizations${params}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

/**
 * POST-based SSE stream. Returns an AbortController so the caller can cancel.
 * `onEvent` receives parsed SSE events as they arrive.
 */
export function streamGenerate(
  body: GenerateRequest,
  onEvent: (event: string, data: Record<string, unknown>) => void,
  onDone: () => void,
  onError: (err: Error) => void,
): AbortController {
  const controller = new AbortController();

  (async () => {
    try {
      const res = await fetch(`${BASE}/generate/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });

      if (!res.ok) {
        onError(new Error(await res.text()));
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
              const data = JSON.parse(line.slice(6));
              onEvent(currentEvent, data);
            } catch {
              // skip malformed JSON
            }
            currentEvent = "";
          }
        }
      }

      onDone();
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        onError(err as Error);
      }
    }
  })();

  return controller;
}
