"""
MMLU-Pro Domain-Specific Steering Vectors.

Extracts contrastive steering vectors for all 14 MMLU-Pro categories,
computes pairwise cosine similarity, and selects the 3 most dissimilar
domains for targeted steering experiments.

Uses the same contrastive mean-difference methodology as domain_vectors.py.
"""

import gc
import json
import functools
import itertools
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

# ---------------------------------------------------------------------------
# Models to test
# ---------------------------------------------------------------------------
MODELS = {
    "qwen3_0.6b": "Qwen/Qwen3-0.6B",
    "qwen3_4b": "Qwen/Qwen3-4B",
    "llama3_3b": "meta-llama/Llama-3.2-3B-Instruct",
    "lfm2_1.2b": "LiquidAI/LFM2.5-1.2B-Instruct",
}

# ---------------------------------------------------------------------------
# Neutral prompts (domain-agnostic factual statements)
# ---------------------------------------------------------------------------
NEUTRAL_PROMPTS = [
    "The capital of France is Paris, which is known for the Eiffel Tower.",
    "Photosynthesis is the process by which plants convert sunlight into energy.",
    "Python is a high-level programming language known for its readability.",
    "The solar system has eight planets orbiting the Sun.",
    "DNA stands for deoxyribonucleic acid, which carries genetic information.",
    "Machine learning is a subset of artificial intelligence that learns from data.",
    "The French Revolution began in 1789 with the storming of the Bastille.",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    "Shakespeare wrote many famous plays including Hamlet and Romeo and Juliet.",
    "The speed of light in vacuum is approximately 299,792,458 meters per second.",
]

# ---------------------------------------------------------------------------
# MMLU-Pro domain prompts (10 per domain, reasoning-flavored)
# ---------------------------------------------------------------------------
DOMAIN_PROMPTS = {
    "math": [
        "To solve this integral, I'll apply integration by parts with u=ln(x) and dv=x dx.",
        "The eigenvalues of this 3x3 matrix can be found by computing det(A - λI) = 0.",
        "Using the chain rule, the derivative of f(g(x)) is f'(g(x)) · g'(x).",
        "This series converges by the ratio test since the limit of a_{n+1}/a_n < 1.",
        "The probability of event A given B is P(A|B) = P(A∩B)/P(B) by Bayes' theorem.",
        "To find the area between the curves, I integrate the difference from the left to right intersection.",
        "The gradient vector points in the direction of steepest ascent of the scalar field.",
        "By the fundamental theorem of algebra, this degree-5 polynomial has exactly 5 complex roots.",
        "The Taylor expansion of e^x about x=0 gives 1 + x + x²/2! + x³/3! + ...",
        "Using modular arithmetic, 7^100 mod 13 can be simplified via Fermat's little theorem.",
    ],
    "physics": [
        "Applying Newton's second law F=ma, the acceleration of this 5kg block is a = F_net/m.",
        "The electric field at distance r from a point charge is E = kQ/r² directed radially outward.",
        "Using conservation of energy, the kinetic energy at the bottom equals the potential energy at the top.",
        "The magnetic flux through the loop is changing, so by Faraday's law an EMF is induced.",
        "The wavelength of this photon can be found from E = hc/λ, giving λ = hc/E.",
        "In this collision, momentum is conserved: m1v1 + m2v2 = m1v1' + m2v2'.",
        "The period of a simple pendulum is T = 2π√(L/g), independent of mass.",
        "Using the ideal gas law PV = nRT, the pressure doubles when temperature doubles at constant volume.",
        "The relativistic energy-momentum relation is E² = (pc)² + (mc²)².",
        "By the uncertainty principle, ΔxΔp ≥ ℏ/2, so we cannot know both precisely.",
    ],
    "chemistry": [
        "Balancing this redox reaction requires identifying oxidation states of each element.",
        "The pH of a 0.01M HCl solution is -log(0.01) = 2, since HCl is a strong acid.",
        "Using Le Chatelier's principle, increasing pressure shifts equilibrium toward fewer moles of gas.",
        "The molecular orbital diagram shows that O₂ has two unpaired electrons, making it paramagnetic.",
        "The rate law for this second-order reaction is rate = k[A]², giving a half-life of 1/(k[A]₀).",
        "Electronegativity increases across a period and decreases down a group in the periodic table.",
        "The enthalpy of reaction can be calculated using Hess's law by summing formation enthalpies.",
        "This SN2 reaction proceeds with inversion of configuration at the chiral center.",
        "The ideal gas constant R = 8.314 J/(mol·K) connects pressure, volume, temperature, and moles.",
        "Hybridization of the carbon in methane is sp³, giving a tetrahedral geometry with 109.5° angles.",
    ],
    "law": [
        "Under the Fourth Amendment, this search requires probable cause and a valid warrant.",
        "The doctrine of stare decisis requires courts to follow precedent from higher courts.",
        "In contract law, consideration requires a bargained-for exchange between the parties.",
        "The mens rea element for murder requires proof of intent to kill or cause grievous harm.",
        "Under strict liability, the manufacturer is liable for defective products regardless of fault.",
        "The Commerce Clause grants Congress power to regulate interstate commercial activity.",
        "Negligence requires proving duty, breach, causation, and damages by a preponderance of evidence.",
        "The parol evidence rule bars extrinsic evidence that contradicts an integrated written agreement.",
        "Due process under the Fourteenth Amendment requires notice and an opportunity to be heard.",
        "Proximate cause limits liability to consequences that are reasonably foreseeable from the act.",
    ],
    "engineering": [
        "The bending stress in this beam is σ = My/I, where M is the moment and I the second moment of area.",
        "Using Kirchhoff's voltage law, the sum of voltage drops around the loop equals zero.",
        "The Nyquist sampling theorem requires sampling at twice the maximum signal frequency.",
        "This control system's transfer function has poles in the left half-plane, so it is stable.",
        "The Reynolds number Re = ρvD/μ determines whether flow is laminar or turbulent.",
        "Heat transfer through this wall follows Fourier's law: q = -kA(dT/dx).",
        "The efficiency of this Carnot engine is η = 1 - T_cold/T_hot.",
        "Using a Karnaugh map, I can simplify this 4-variable Boolean expression to minimize gates.",
        "The stress-strain curve shows the material yields at 250 MPa before entering plastic deformation.",
        "Signal-to-noise ratio in this communication channel limits the achievable data rate via Shannon's theorem.",
    ],
    "economics": [
        "At equilibrium, quantity supplied equals quantity demanded, determining the market price.",
        "The marginal cost curve intersects the average total cost curve at its minimum point.",
        "Using the IS-LM model, an increase in government spending shifts the IS curve rightward.",
        "The elasticity of demand measures how quantity demanded responds to a change in price.",
        "Comparative advantage shows that trade benefits both countries even when one has absolute advantage.",
        "The Phillips curve suggests an inverse relationship between unemployment and inflation in the short run.",
        "GDP is calculated as C + I + G + (X - M), where C is consumption and I is investment.",
        "The money multiplier is 1/reserve ratio, so a 10% reserve requirement gives a multiplier of 10.",
        "Deadweight loss from a price floor represents the inefficiency of the market distortion.",
        "The Solow model predicts that long-run growth depends on technological progress, not capital accumulation.",
    ],
    "health": [
        "The differential diagnosis includes myocardial infarction given the chest pain and elevated troponin.",
        "Type 2 diabetes results from insulin resistance, where cells fail to respond to insulin signaling.",
        "The inflammatory response involves neutrophil migration, vasodilation, and increased capillary permeability.",
        "Antibiotics target bacterial cell walls, protein synthesis, or DNA replication without harming host cells.",
        "The glomerular filtration rate estimates kidney function and is used to stage chronic kidney disease.",
        "Hypertension is diagnosed when systolic pressure exceeds 140mmHg or diastolic exceeds 90mmHg consistently.",
        "The autonomic nervous system divides into sympathetic (fight-or-flight) and parasympathetic (rest-and-digest).",
        "Vaccine-induced immunity relies on memory B and T cells recognizing the pathogen upon re-exposure.",
        "The oxygen dissociation curve shifts right with increased CO₂, temperature, or 2,3-BPG concentration.",
        "Pharmacokinetics describes absorption, distribution, metabolism, and excretion of drugs in the body.",
    ],
    "psychology": [
        "According to Piaget, children in the concrete operational stage can reason logically about concrete events.",
        "Classical conditioning pairs a neutral stimulus with an unconditioned stimulus to produce a conditioned response.",
        "The bystander effect shows that individuals are less likely to help when others are present.",
        "Working memory has limited capacity, typically holding 7±2 items according to Miller's research.",
        "Cognitive dissonance theory predicts that conflicting beliefs cause psychological discomfort motivating change.",
        "Maslow's hierarchy places physiological needs at the base and self-actualization at the top.",
        "The Stanford prison experiment demonstrated how situational factors can override individual dispositions.",
        "Confirmation bias leads people to seek information that confirms their existing beliefs.",
        "Attachment theory identifies secure, anxious, and avoidant patterns formed in early caregiver relationships.",
        "The limbic system, particularly the amygdala, plays a central role in emotional processing and fear.",
    ],
    "business": [
        "Porter's five forces analysis evaluates competitive intensity: rivalry, new entrants, substitutes, and bargaining power.",
        "The balanced scorecard measures performance across financial, customer, internal process, and learning perspectives.",
        "Net present value discounts future cash flows to determine if an investment exceeds its cost.",
        "The marketing mix of product, price, place, and promotion guides go-to-market strategy.",
        "SWOT analysis identifies internal strengths and weaknesses alongside external opportunities and threats.",
        "The break-even point occurs where total revenue equals total costs: Q = FC/(P-VC).",
        "Agile methodology uses iterative sprints to deliver incremental value and adapt to changing requirements.",
        "The weighted average cost of capital (WACC) blends the cost of equity and debt financing.",
        "Supply chain management optimizes the flow from raw materials to end customer delivery.",
        "The BCG matrix classifies business units as stars, cash cows, question marks, or dogs.",
    ],
    "biology": [
        "During mitosis, chromosomes align at the metaphase plate before sister chromatids separate in anaphase.",
        "The Krebs cycle produces NADH and FADH₂, which feed into the electron transport chain.",
        "Natural selection acts on phenotypic variation, favoring traits that increase reproductive fitness.",
        "The central dogma describes information flow from DNA to RNA to protein via transcription and translation.",
        "Speciation occurs when populations become reproductively isolated through geographic or behavioral barriers.",
        "The nephron is the functional unit of the kidney, filtering blood and producing urine.",
        "CRISPR-Cas9 uses a guide RNA to direct the Cas9 nuclease to specific DNA sequences for editing.",
        "Ecological succession begins with pioneer species colonizing bare substrate and progresses to climax communities.",
        "Mendel's law of segregation states that allele pairs separate during gamete formation.",
        "The endosymbiotic theory explains the origin of mitochondria and chloroplasts from engulfed prokaryotes.",
    ],
    "philosophy": [
        "Kant's categorical imperative requires acting only according to maxims you could universalize.",
        "The trolley problem illustrates the tension between utilitarian and deontological moral reasoning.",
        "Descartes' cogito ergo sum establishes the thinking self as the foundation of certain knowledge.",
        "Plato's allegory of the cave contrasts appearances with the reality of the Forms.",
        "Hume argues that we cannot derive ought from is — facts alone don't determine moral obligations.",
        "Existentialism holds that existence precedes essence: we define ourselves through our choices.",
        "The problem of induction questions whether past regularities justify predictions about the future.",
        "Rawls' veil of ignorance asks what principles of justice rational agents would choose without knowing their position.",
        "The hard problem of consciousness asks why physical processes give rise to subjective experience.",
        "Aristotle's virtue ethics emphasizes developing character traits that lie between excess and deficiency.",
    ],
    "computer_science": [
        "The time complexity of binary search is O(log n) because the search space halves each step.",
        "A hash table provides O(1) average-case lookup by mapping keys to array indices via a hash function.",
        "Dijkstra's algorithm finds shortest paths from a source node by greedily expanding the minimum-distance frontier.",
        "The halting problem proves that no general algorithm can decide if an arbitrary program terminates.",
        "TCP ensures reliable delivery through sequence numbers, acknowledgments, and retransmission of lost segments.",
        "Normalization to third normal form eliminates transitive dependencies between non-key attributes.",
        "A convolutional neural network applies learned filters to detect spatial features in the input.",
        "The CAP theorem states that a distributed system cannot simultaneously guarantee consistency, availability, and partition tolerance.",
        "Garbage collection reclaims memory occupied by objects no longer reachable from the root set.",
        "Public-key cryptography uses a pair of keys: the public key encrypts, the private key decrypts.",
    ],
    "history": [
        "The Treaty of Westphalia in 1648 established the principle of state sovereignty in international relations.",
        "The Industrial Revolution transformed manufacturing from cottage industries to factory-based mass production.",
        "The Marshall Plan provided American economic aid to rebuild Western European economies after World War II.",
        "The fall of Constantinople in 1453 marked the end of the Byzantine Empire and the rise of Ottoman power.",
        "The transatlantic slave trade forcibly transported millions of Africans to the Americas between the 16th and 19th centuries.",
        "Decolonization after World War II saw dozens of nations in Africa and Asia gain independence.",
        "The Cuban Missile Crisis of 1962 brought the US and Soviet Union to the brink of nuclear war.",
        "The Silk Road facilitated trade and cultural exchange between China, Central Asia, and the Mediterranean.",
        "The Reformation began with Luther's 95 Theses in 1517, challenging the authority of the Catholic Church.",
        "The Meiji Restoration of 1868 transformed Japan from a feudal society into an industrialized nation.",
    ],
    "other": [
        "The Doppler effect explains why a siren's pitch changes as an ambulance approaches and recedes.",
        "Fibonacci numbers appear in nature, from sunflower seed spirals to the branching of trees.",
        "The greenhouse effect traps solar radiation via atmospheric gases like CO₂ and methane.",
        "Tectonic plates move a few centimeters per year, causing earthquakes at convergent and transform boundaries.",
        "The Turing test evaluates whether a machine can exhibit intelligent behavior indistinguishable from a human.",
        "Cryptocurrency uses blockchain — a decentralized, immutable ledger — to record transactions.",
        "The Richter scale is logarithmic: each whole number increase represents a tenfold increase in amplitude.",
        "Cognitive load theory suggests that instructional design should minimize extraneous processing demands.",
        "The scientific method involves hypothesis formation, experimentation, observation, and falsification.",
        "Moore's law observed that the number of transistors on a chip doubles approximately every two years.",
    ],
}

MMLU_DOMAINS = list(DOMAIN_PROMPTS.keys())


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------
def _gather_hook(module, input, output, *, cache, layer_idx):
    hidden = output[0] if isinstance(output, tuple) else output
    cache[layer_idx] = hidden.detach().cpu()


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------
def get_device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def load_model(model_id, device, dtype):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype,
        device_map=device if device != "mps" else None,
        low_cpu_mem_usage=True,
    )
    if device == "mps":
        model = model.to(device)
    model.eval()
    return model, tokenizer


def get_layers(model):
    """Get the list of transformer layers, handling different architectures."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers  # Qwen, Llama
    if hasattr(model, "model") and hasattr(model.model, "blocks"):
        return model.model.blocks  # LFM2.5 might use blocks
    raise ValueError(f"Cannot find layers in {type(model)}")


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------
def extract_activations(model, tokenizer, prompts, device):
    layers = get_layers(model)
    n_layers = len(layers)
    all_acts = {i: [] for i in range(n_layers)}

    for prompt in prompts:
        cache = {}
        handles = []
        try:
            for i, layer in enumerate(layers):
                handles.append(layer.register_forward_hook(
                    functools.partial(_gather_hook, cache=cache, layer_idx=i)
                ))
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                model(**inputs)
            for i in range(n_layers):
                if i in cache:
                    all_acts[i].append(cache[i][0, -1, :])
        finally:
            for h in handles:
                h.remove()

    return {i: torch.stack(acts) for i, acts in all_acts.items() if acts}


def compute_mmlu_vectors(model, tokenizer, device):
    """Compute contrastive vectors for all 14 MMLU-Pro domains."""
    print("  Extracting neutral activations...")
    neutral_acts = extract_activations(model, tokenizer, NEUTRAL_PROMPTS, device)

    vectors = {}
    norms = {}

    for domain in MMLU_DOMAINS:
        print(f"  Extracting {domain}...", end="", flush=True)
        pos_acts = extract_activations(model, tokenizer, DOMAIN_PROMPTS[domain], device)

        vectors[domain] = {}
        norms[domain] = {}
        for i in neutral_acts:
            diff = pos_acts[i].mean(dim=0) - neutral_acts[i].mean(dim=0)
            vectors[domain][i] = diff
            norms[domain][i] = diff.norm().item()
        print(f" done (max L2={max(norms[domain].values()):.1f})")

    return vectors, norms


# ---------------------------------------------------------------------------
# Cosine similarity analysis
# ---------------------------------------------------------------------------
def compute_cosine_matrix(vectors, layer):
    """Compute pairwise cosine similarity between all domains at a given layer."""
    domains = list(vectors.keys())
    n = len(domains)
    matrix = torch.zeros(n, n)

    vecs = [vectors[d][layer] for d in domains]
    for i in range(n):
        for j in range(n):
            cos = torch.nn.functional.cosine_similarity(
                vecs[i].unsqueeze(0), vecs[j].unsqueeze(0)
            ).item()
            matrix[i, j] = cos

    return domains, matrix


def find_most_dissimilar_triplet(vectors, layer):
    """Find 3 domains with the lowest average pairwise cosine similarity."""
    domains, matrix = compute_cosine_matrix(vectors, layer)
    n = len(domains)

    best_triplet = None
    best_avg_cos = 1.0

    for i, j, k in itertools.combinations(range(n), 3):
        avg = (matrix[i, j] + matrix[i, k] + matrix[j, k]).item() / 3.0
        if avg < best_avg_cos:
            best_avg_cos = avg
            best_triplet = (domains[i], domains[j], domains[k])

    return best_triplet, best_avg_cos


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        help="Run only this model key (e.g. qwen3_4b)")
    args = parser.parse_args()

    device, dtype = get_device_and_dtype()
    print(f"Device: {device} | dtype: {dtype}\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    models_to_run = {args.model: MODELS[args.model]} if args.model else MODELS
    for model_key, model_id in models_to_run.items():
        print(f"\n{'='*60}")
        print(f"MODEL: {model_id}")
        print(f"{'='*60}")

        try:
            model, tokenizer = load_model(model_id, device, dtype)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue

        layers = get_layers(model)
        n_layers = len(layers)
        print(f"  Layers: {n_layers}")

        vectors, norms = compute_mmlu_vectors(model, tokenizer, device)

        # Find best layer for analysis (highest avg L2 across domains)
        avg_l2_per_layer = {}
        for layer_idx in range(n_layers):
            avg_l2 = sum(norms[d].get(layer_idx, 0) for d in MMLU_DOMAINS) / len(MMLU_DOMAINS)
            avg_l2_per_layer[layer_idx] = avg_l2
        best_layer = max(avg_l2_per_layer, key=avg_l2_per_layer.get)
        mid_layer = n_layers // 2  # ~50% depth

        print(f"\n  Best L2 layer: {best_layer} (avg L2={avg_l2_per_layer[best_layer]:.1f})")
        print(f"  Mid layer: {mid_layer} (avg L2={avg_l2_per_layer.get(mid_layer, 0):.1f})")

        # Cosine similarity at mid-layer
        analysis_layer = mid_layer
        domains, cos_matrix = compute_cosine_matrix(vectors, analysis_layer)

        print(f"\n  Cosine similarity matrix at layer {analysis_layer}:")
        print(f"  {'':>20s}", end="")
        for d in domains:
            print(f" {d[:6]:>6s}", end="")
        print()
        for i, d1 in enumerate(domains):
            print(f"  {d1:>20s}", end="")
            for j in range(len(domains)):
                val = cos_matrix[i, j].item()
                print(f" {val:6.3f}", end="")
            print()

        # Find most dissimilar triplet
        triplet, avg_cos = find_most_dissimilar_triplet(vectors, analysis_layer)
        print(f"\n  Most dissimilar triplet (layer {analysis_layer}):")
        print(f"    {triplet[0]}, {triplet[1]}, {triplet[2]}")
        print(f"    Average cosine: {avg_cos:.4f}")

        # Also check at best_layer
        triplet_best, avg_cos_best = find_most_dissimilar_triplet(vectors, best_layer)
        print(f"  Most dissimilar triplet (layer {best_layer}):")
        print(f"    {triplet_best[0]}, {triplet_best[1]}, {triplet_best[2]}")
        print(f"    Average cosine: {avg_cos_best:.4f}")

        # Save vectors
        vec_path = RESULTS_DIR / f"mmlu_pro_vectors_{model_key}.pt"
        torch.save(vectors, vec_path)
        print(f"\n  Saved vectors: {vec_path}")

        # Collect results
        all_results[model_key] = {
            "model_id": model_id,
            "n_layers": n_layers,
            "best_l2_layer": best_layer,
            "mid_layer": mid_layer,
            "triplet_mid": {
                "domains": list(triplet),
                "avg_cosine": round(avg_cos, 4),
                "layer": analysis_layer,
            },
            "triplet_best": {
                "domains": list(triplet_best),
                "avg_cosine": round(avg_cos_best, 4),
                "layer": best_layer,
            },
            "norms": {d: {str(k): round(v, 2) for k, v in norms[d].items()}
                      for d in MMLU_DOMAINS},
        }

        # Cleanup
        del model, tokenizer, vectors, norms
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Save results
    results_path = RESULTS_DIR / "mmlu_pro_vectors_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results: {results_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY — Most Dissimilar Triplets")
    print(f"{'='*60}")
    for model_key, res in all_results.items():
        t = res["triplet_mid"]
        print(f"  {model_key}: {t['domains']} (cos={t['avg_cosine']:.4f} @ L{t['layer']})")


if __name__ == "__main__":
    main()
