"""
Neuronpedia SAE Feature Exploration for Qwen3-4B

Fetches transcoder features from Neuronpedia API and maps them
to calendar-related concepts identified in Phase 2.
"""

import json
import time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
BASE_URL = "https://www.neuronpedia.org/api"
MODEL_ID = "qwen3-4b"
SOURCE_TEMPLATE = "{layer}-transcoder-hp"

# Calendar-related keywords to search for in feature explanations
CALENDAR_KEYWORDS = [
    "schedule", "calendar", "meeting", "appointment", "date", "time",
    "agenda", "event", "booking", "reminder", "invite", "attendee",
    "rendez-vous", "réunion", "heure", "planifier",
]

# Layers to explore (focus on top layers from Phase 2 + mid-range)
LAYERS_TO_EXPLORE = [15, 18, 20, 22, 25, 30, 33, 34, 35]

# Features to sample per layer (API rate limit friendly)
FEATURES_PER_LAYER = 50
FEATURE_STRIDE = 3000  # Sample every N-th feature out of 163,840


def api_get(endpoint: str, retries: int = 3) -> dict | None:
    """GET request to Neuronpedia API with retry."""
    url = f"{BASE_URL}/{endpoint}"
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "SteeringResearch/1.0"})
            with urlopen(req, timeout=15) as resp:
                return json.loads(resp.read())
        except HTTPError as e:
            if e.code == 429:
                wait = 2 ** attempt
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
            elif e.code == 500:
                return None
            else:
                raise
        except Exception:
            if attempt < retries - 1:
                time.sleep(1)
            else:
                return None
    return None


def api_post(endpoint: str, data: dict, retries: int = 3) -> dict | None:
    """POST request to Neuronpedia API."""
    url = f"{BASE_URL}/{endpoint}"
    body = json.dumps(data).encode()
    for attempt in range(retries):
        try:
            req = Request(url, data=body, method="POST",
                         headers={"Content-Type": "application/json",
                                  "User-Agent": "SteeringResearch/1.0"})
            with urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except HTTPError as e:
            if e.code in (429, 500):
                time.sleep(2 ** attempt)
            else:
                return None
        except Exception:
            if attempt < retries - 1:
                time.sleep(1)
            else:
                return None
    return None


def search_explanations(layer: int, query: str) -> list[dict]:
    """Search feature explanations for a keyword at a given layer."""
    source = SOURCE_TEMPLATE.format(layer=layer)
    data = {
        "modelId": MODEL_ID,
        "layers": [source],
        "query": query,
    }
    result = api_post("explanation/search", data)
    if result and isinstance(result, list):
        return result
    if result and "results" in result:
        return result["results"]
    return []


def fetch_feature(layer: int, index: int) -> dict | None:
    """Fetch a single feature's data."""
    source = SOURCE_TEMPLATE.format(layer=layer)
    return api_get(f"feature/{MODEL_ID}/{source}/{index}")


def extract_feature_info(feature_data: dict) -> dict:
    """Extract relevant info from a feature response."""
    explanations = feature_data.get("explanations", [])
    top_explanation = explanations[0].get("description", "") if explanations else ""

    # Top activating tokens
    pos_str = feature_data.get("pos_str", [])
    neg_str = feature_data.get("neg_str", [])

    return {
        "index": feature_data.get("index"),
        "layer": feature_data.get("layer"),
        "max_act": feature_data.get("maxActApprox", 0),
        "explanation": top_explanation,
        "pos_examples": pos_str[:5] if pos_str else [],
        "neg_examples": neg_str[:5] if neg_str else [],
        "neuron_alignment_indices": feature_data.get("neuron_alignment_indices", []),
        "neuron_alignment_values": feature_data.get("neuron_alignment_values", []),
    }


def is_calendar_related(info: dict) -> bool:
    """Check if a feature is related to calendar/scheduling."""
    text = (info.get("explanation", "") + " ".join(info.get("pos_examples", []))).lower()
    return any(kw in text for kw in CALENDAR_KEYWORDS)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("NEURONPEDIA SAE FEATURE EXPLORATION — Qwen3-4B")
    print("=" * 60)
    print(f"Model: {MODEL_ID}")
    print(f"Source: transcoder-hp (163,840 features/layer)")
    print(f"Layers to explore: {LAYERS_TO_EXPLORE}")

    all_features = {}
    calendar_features = []

    # --- Strategy 1: Search by explanation keywords ---
    print("\n--- STRATEGY 1: Search explanations for calendar keywords ---")
    search_results = {}
    for keyword in CALENDAR_KEYWORDS[:8]:  # Top 8 keywords to limit API calls
        print(f"\n  Searching '{keyword}'...")
        for layer in [20, 25, 30, 33, 35]:  # Focus layers
            results = search_explanations(layer, keyword)
            if results:
                for r in results[:3]:  # Top 3 per layer per keyword
                    feat_key = f"L{layer}_F{r.get('index', r.get('featureIndex', '?'))}"
                    if feat_key not in search_results:
                        search_results[feat_key] = {
                            "keyword": keyword,
                            "layer": layer,
                            "index": r.get("index", r.get("featureIndex")),
                            "description": r.get("description", ""),
                        }
                        print(f"    L{layer} #{r.get('index', r.get('featureIndex', '?'))}: "
                              f"{r.get('description', 'no desc')[:80]}")
            time.sleep(0.3)  # Rate limit

    print(f"\n  Found {len(search_results)} unique features via explanation search")

    # --- Strategy 2: Sample features and check for calendar relevance ---
    print("\n--- STRATEGY 2: Sample features across layers ---")
    for layer in LAYERS_TO_EXPLORE:
        print(f"\n  Layer {layer}:")
        layer_features = []
        sampled = 0

        for idx in range(0, 163840, FEATURE_STRIDE):
            if sampled >= FEATURES_PER_LAYER:
                break

            feature = fetch_feature(layer, idx)
            if feature is None:
                continue

            info = extract_feature_info(feature)
            layer_features.append(info)
            sampled += 1

            if is_calendar_related(info):
                calendar_features.append({**info, "source": "sampling"})
                print(f"    ★ #{idx}: {info['explanation'][:80] or 'no explanation'}")

            time.sleep(0.2)  # Rate limit

        all_features[layer] = layer_features
        n_cal = sum(1 for f in layer_features if is_calendar_related(f))
        print(f"    Sampled {sampled} features, {n_cal} calendar-related")

    # --- Strategy 3: Fetch specific high-activation features from our Phase 2 layers ---
    print("\n--- STRATEGY 3: High-index features at key layers ---")
    key_ranges = [
        (35, range(0, 500, 10)),
        (34, range(0, 500, 10)),
        (33, range(0, 500, 10)),
        (20, range(0, 500, 10)),
    ]
    for layer, indices in key_ranges:
        for idx in indices:
            feature = fetch_feature(layer, idx)
            if feature is None:
                continue
            info = extract_feature_info(feature)
            if is_calendar_related(info):
                calendar_features.append({**info, "source": "key_layer_scan"})
                print(f"    L{layer} #{idx}: {info['explanation'][:80] or info['pos_examples'][:3]}")
            time.sleep(0.15)

    # --- Summary ---
    print("\n" + "=" * 60)
    print(f"SUMMARY: {len(calendar_features)} calendar-related features found")
    print("=" * 60)

    # Deduplicate
    seen = set()
    unique_features = []
    for f in calendar_features:
        key = f"{f.get('layer')}_{f.get('index')}"
        if key not in seen:
            seen.add(key)
            unique_features.append(f)

    # Group by layer
    by_layer = {}
    for f in unique_features:
        layer = f.get("layer", "unknown")
        by_layer.setdefault(layer, []).append(f)

    for layer in sorted(by_layer.keys(), key=lambda x: str(x)):
        features = by_layer[layer]
        print(f"\n  Layer {layer}: {len(features)} features")
        for f in features[:5]:
            print(f"    #{f.get('index')}: {f.get('explanation', '')[:70] or 'no explanation'}")

    # --- Save ---
    output = {
        "model": MODEL_ID,
        "source_set": "transcoder-hp",
        "features_per_layer": 163840,
        "layers_explored": LAYERS_TO_EXPLORE,
        "search_results": search_results,
        "calendar_features": unique_features,
        "features_by_layer": {
            str(k): [
                {key: val for key, val in f.items()
                 if key not in ("pos_examples", "neg_examples")}
                for f in v
            ]
            for k, v in by_layer.items()
        },
        "total_sampled": sum(len(v) for v in all_features.values()),
        "total_calendar_related": len(unique_features),
    }

    path = RESULTS_DIR / "neuronpedia_features.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
