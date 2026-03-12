"""
Phase 4: Prompt Engineering Baselines for Calendar Event Extraction

Defines 5 prompting strategies and a 30-case evaluation dataset
for comparing against steering vector approaches.
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# ---------------------------------------------------------------------------
# Tool definition (OpenAI-compatible, works with Ollama)
# ---------------------------------------------------------------------------

CALENDAR_TOOL = {
    "type": "function",
    "function": {
        "name": "create_calendar_event",
        "description": "Create a new calendar event with the given details.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Event title"},
                "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
                "start_time": {"type": "string", "description": "Start time in HH:MM format"},
                "end_time": {"type": "string", "description": "End time in HH:MM format (optional)"},
                "location": {"type": "string", "description": "Event location (optional)"},
                "attendees": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of attendee names",
                },
            },
            "required": ["title", "date", "start_time"],
        },
    },
}

# ---------------------------------------------------------------------------
# Evaluation dataset — 30 calendar extraction test cases
# Reference date: 2026-03-12 (Thursday)
# ---------------------------------------------------------------------------

EVAL_DATASET = [
    # --- Simple FR ---
    {
        "id": "simple_fr_01",
        "input": "Crée un rendez-vous demain à 14h avec Marie.",
        "lang": "fr",
        "complexity": "simple",
        "expected": {
            "title": "Rendez-vous avec Marie",
            "date": "2026-03-13",
            "start_time": "14:00",
            "end_time": None,
            "location": None,
            "attendees": ["Marie"],
        },
    },
    {
        "id": "simple_fr_02",
        "input": "Planifie une réunion lundi prochain à 10h.",
        "lang": "fr",
        "complexity": "simple",
        "expected": {
            "title": "Réunion",
            "date": "2026-03-16",
            "start_time": "10:00",
            "end_time": None,
            "location": None,
            "attendees": [],
        },
    },
    {
        "id": "simple_fr_03",
        "input": "Ajoute un déjeuner avec Paul vendredi à midi.",
        "lang": "fr",
        "complexity": "simple",
        "expected": {
            "title": "Déjeuner avec Paul",
            "date": "2026-03-13",
            "start_time": "12:00",
            "end_time": None,
            "location": None,
            "attendees": ["Paul"],
        },
    },
    # --- Simple EN ---
    {
        "id": "simple_en_01",
        "input": "Schedule a meeting tomorrow at 2pm with John.",
        "lang": "en",
        "complexity": "simple",
        "expected": {
            "title": "Meeting with John",
            "date": "2026-03-13",
            "start_time": "14:00",
            "end_time": None,
            "location": None,
            "attendees": ["John"],
        },
    },
    {
        "id": "simple_en_02",
        "input": "Book a call with Sarah on Monday at 3pm.",
        "lang": "en",
        "complexity": "simple",
        "expected": {
            "title": "Call with Sarah",
            "date": "2026-03-16",
            "start_time": "15:00",
            "end_time": None,
            "location": None,
            "attendees": ["Sarah"],
        },
    },
    {
        "id": "simple_en_03",
        "input": "Set up a 1:1 with Alex next Wednesday at 11am.",
        "lang": "en",
        "complexity": "simple",
        "expected": {
            "title": "1:1 with Alex",
            "date": "2026-03-18",
            "start_time": "11:00",
            "end_time": None,
            "location": None,
            "attendees": ["Alex"],
        },
    },
    # --- Complex FR (location, multiple attendees, time range) ---
    {
        "id": "complex_fr_01",
        "input": "Bloque le créneau de 10h à 11h30 lundi dans la salle Confluence pour la rétrospective sprint avec Jean-Pierre et Fatima.",
        "lang": "fr",
        "complexity": "complex",
        "expected": {
            "title": "Rétrospective sprint",
            "date": "2026-03-16",
            "start_time": "10:00",
            "end_time": "11:30",
            "location": "Salle Confluence",
            "attendees": ["Jean-Pierre", "Fatima"],
        },
    },
    {
        "id": "complex_fr_02",
        "input": "Organise une session de brainstorming le 24 mars de 14h à 16h en salle Innovation avec toute l'équipe produit.",
        "lang": "fr",
        "complexity": "complex",
        "expected": {
            "title": "Session de brainstorming",
            "date": "2026-03-24",
            "start_time": "14:00",
            "end_time": "16:00",
            "location": "Salle Innovation",
            "attendees": ["équipe produit"],
        },
    },
    {
        "id": "complex_fr_03",
        "input": "Prévois un point d'avancement mercredi 18 mars à 9h30 avec Sophie, Marc et le directeur technique dans le bureau 4B.",
        "lang": "fr",
        "complexity": "complex",
        "expected": {
            "title": "Point d'avancement",
            "date": "2026-03-18",
            "start_time": "09:30",
            "end_time": None,
            "location": "Bureau 4B",
            "attendees": ["Sophie", "Marc", "directeur technique"],
        },
    },
    {
        "id": "complex_fr_04",
        "input": "Planifie la revue trimestrielle le 31 mars de 10h à 12h en visioconférence avec les équipes Paris et Lyon.",
        "lang": "fr",
        "complexity": "complex",
        "expected": {
            "title": "Revue trimestrielle",
            "date": "2026-03-31",
            "start_time": "10:00",
            "end_time": "12:00",
            "location": "Visioconférence",
            "attendees": ["équipe Paris", "équipe Lyon"],
        },
    },
    # --- Complex EN ---
    {
        "id": "complex_en_01",
        "input": "Book the main conference room for a product demo on March 28th from 10am to 11:30am with the sales team and two clients.",
        "lang": "en",
        "complexity": "complex",
        "expected": {
            "title": "Product demo",
            "date": "2026-03-28",
            "start_time": "10:00",
            "end_time": "11:30",
            "location": "Main conference room",
            "attendees": ["sales team", "clients"],
        },
    },
    {
        "id": "complex_en_02",
        "input": "Set up a 2-hour architecture review on April 2nd at 2pm in Room 301 with David, Emma, and the backend team.",
        "lang": "en",
        "complexity": "complex",
        "expected": {
            "title": "Architecture review",
            "date": "2026-04-02",
            "start_time": "14:00",
            "end_time": "16:00",
            "location": "Room 301",
            "attendees": ["David", "Emma", "backend team"],
        },
    },
    # --- Relative dates ---
    {
        "id": "relative_01",
        "input": "Cale-moi un truc avec Antoine la semaine prochaine, plutôt le matin.",
        "lang": "fr",
        "complexity": "ambiguous",
        "expected": {
            "title": "Rendez-vous avec Antoine",
            "date": "2026-03-16",
            "start_time": "09:00",
            "end_time": None,
            "location": None,
            "attendees": ["Antoine"],
        },
    },
    {
        "id": "relative_02",
        "input": "Dans 3 jours, j'ai besoin d'un créneau d'une heure pour préparer la présentation.",
        "lang": "fr",
        "complexity": "ambiguous",
        "expected": {
            "title": "Préparation présentation",
            "date": "2026-03-15",
            "start_time": "09:00",
            "end_time": "10:00",
            "location": None,
            "attendees": [],
        },
    },
    {
        "id": "relative_03",
        "input": "After-demain après-midi, bloque 2h pour le code review.",
        "lang": "fr",
        "complexity": "ambiguous",
        "expected": {
            "title": "Code review",
            "date": "2026-03-14",
            "start_time": "14:00",
            "end_time": "16:00",
            "location": None,
            "attendees": [],
        },
    },
    {
        "id": "relative_04",
        "input": "Can we do lunch next Thursday?",
        "lang": "en",
        "complexity": "ambiguous",
        "expected": {
            "title": "Lunch",
            "date": "2026-03-19",
            "start_time": "12:00",
            "end_time": None,
            "location": None,
            "attendees": [],
        },
    },
    # --- Absolute dates with various formats ---
    {
        "id": "absolute_01",
        "input": "Réunion le 15/04/2026 à 16h45 avec le comité de direction.",
        "lang": "fr",
        "complexity": "complex",
        "expected": {
            "title": "Réunion comité de direction",
            "date": "2026-04-15",
            "start_time": "16:45",
            "end_time": None,
            "location": None,
            "attendees": ["comité de direction"],
        },
    },
    {
        "id": "absolute_02",
        "input": "Board meeting on April 15, 2026 at 4:45 PM in the executive suite.",
        "lang": "en",
        "complexity": "complex",
        "expected": {
            "title": "Board meeting",
            "date": "2026-04-15",
            "start_time": "16:45",
            "end_time": None,
            "location": "Executive suite",
            "attendees": [],
        },
    },
    # --- Edge cases ---
    {
        "id": "edge_recurring",
        "input": "Mets en place un standup quotidien à 9h30 du lundi au vendredi.",
        "lang": "fr",
        "complexity": "edge",
        "expected": {
            "title": "Standup quotidien",
            "date": "2026-03-16",
            "start_time": "09:30",
            "end_time": None,
            "location": None,
            "attendees": [],
        },
    },
    {
        "id": "edge_multiday",
        "input": "Block March 25-27 for the team offsite in Bordeaux.",
        "lang": "en",
        "complexity": "edge",
        "expected": {
            "title": "Team offsite",
            "date": "2026-03-25",
            "start_time": "09:00",
            "end_time": None,
            "location": "Bordeaux",
            "attendees": [],
        },
    },
    {
        "id": "edge_timezone",
        "input": "Schedule a call at 9am EST / 3pm CET with the New York office on March 20th.",
        "lang": "en",
        "complexity": "edge",
        "expected": {
            "title": "Call with New York office",
            "date": "2026-03-20",
            "start_time": "15:00",
            "end_time": None,
            "location": None,
            "attendees": ["New York office"],
        },
    },
    {
        "id": "edge_informal_fr",
        "input": "Faut qu'on se voit avec Léa, genre mardi ou mercredi aprèm si t'es dispo.",
        "lang": "fr",
        "complexity": "edge",
        "expected": {
            "title": "Rendez-vous avec Léa",
            "date": "2026-03-17",
            "start_time": "14:00",
            "end_time": None,
            "location": None,
            "attendees": ["Léa"],
        },
    },
    {
        "id": "edge_minimal",
        "input": "Dentiste jeudi 8h.",
        "lang": "fr",
        "complexity": "simple",
        "expected": {
            "title": "Dentiste",
            "date": "2026-03-19",
            "start_time": "08:00",
            "end_time": None,
            "location": None,
            "attendees": [],
        },
    },
    {
        "id": "edge_cancel",
        "input": "Annule le rendez-vous de demain avec Marie.",
        "lang": "fr",
        "complexity": "edge",
        "expected": {
            "title": "Rendez-vous avec Marie",
            "date": "2026-03-13",
            "start_time": None,
            "end_time": None,
            "location": None,
            "attendees": ["Marie"],
        },
    },
    # --- More variety ---
    {
        "id": "variety_01",
        "input": "Je voudrais réserver le créneau de 17h à 18h ce vendredi pour un appel avec le fournisseur.",
        "lang": "fr",
        "complexity": "complex",
        "expected": {
            "title": "Appel fournisseur",
            "date": "2026-03-13",
            "start_time": "17:00",
            "end_time": "18:00",
            "location": None,
            "attendees": ["fournisseur"],
        },
    },
    {
        "id": "variety_02",
        "input": "Put a 45-minute planning session on my calendar for next Monday afternoon.",
        "lang": "en",
        "complexity": "ambiguous",
        "expected": {
            "title": "Planning session",
            "date": "2026-03-16",
            "start_time": "14:00",
            "end_time": "14:45",
            "location": None,
            "attendees": [],
        },
    },
    {
        "id": "variety_03",
        "input": "Déjeuner d'équipe au restaurant Le Petit Zinc, vendredi 20 mars à 12h30. On sera 6.",
        "lang": "fr",
        "complexity": "complex",
        "expected": {
            "title": "Déjeuner d'équipe",
            "date": "2026-03-20",
            "start_time": "12:30",
            "end_time": None,
            "location": "Le Petit Zinc",
            "attendees": [],
        },
    },
    {
        "id": "variety_04",
        "input": "Reminder: submit the quarterly report by end of day March 31st.",
        "lang": "en",
        "complexity": "edge",
        "expected": {
            "title": "Submit quarterly report",
            "date": "2026-03-31",
            "start_time": "17:00",
            "end_time": None,
            "location": None,
            "attendees": [],
        },
    },
    {
        "id": "variety_05",
        "input": "Entretien annuel avec RH le 2 avril à 11h, bureau de Mme Dupont.",
        "lang": "fr",
        "complexity": "complex",
        "expected": {
            "title": "Entretien annuel",
            "date": "2026-04-02",
            "start_time": "11:00",
            "end_time": None,
            "location": "Bureau de Mme Dupont",
            "attendees": ["RH", "Mme Dupont"],
        },
    },
]


# ---------------------------------------------------------------------------
# Prompt strategies
# ---------------------------------------------------------------------------

SYSTEM_BASE = (
    "You are a calendar assistant. Extract event details from the user's message "
    "and return a JSON object with: title, date (YYYY-MM-DD), start_time (HH:MM), "
    "end_time (HH:MM or null), location (string or null), attendees (list of strings). "
    "Today's date is 2026-03-12 (Thursday)."
)

FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": "Meeting with Bob tomorrow at 3pm in room A.",
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "title": "Meeting with Bob",
            "date": "2026-03-13",
            "start_time": "15:00",
            "end_time": None,
            "location": "Room A",
            "attendees": ["Bob"],
        }),
    },
    {
        "role": "user",
        "content": "Organise un point de 10h à 11h lundi avec Claire et Thomas.",
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "title": "Point",
            "date": "2026-03-16",
            "start_time": "10:00",
            "end_time": "11:00",
            "location": None,
            "attendees": ["Claire", "Thomas"],
        }),
    },
    {
        "role": "user",
        "content": "Lunch next Friday at noon.",
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "title": "Lunch",
            "date": "2026-03-20",
            "start_time": "12:00",
            "end_time": None,
            "location": None,
            "attendees": [],
        }),
    },
    {
        "role": "user",
        "content": "Réserve la salle B2 pour un workshop le 25 mars de 9h à 12h avec l'équipe data.",
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "title": "Workshop",
            "date": "2026-03-25",
            "start_time": "09:00",
            "end_time": "12:00",
            "location": "Salle B2",
            "attendees": ["équipe data"],
        }),
    },
    {
        "role": "user",
        "content": "Quick sync with Lisa on Wednesday at 4:30pm.",
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "title": "Quick sync with Lisa",
            "date": "2026-03-18",
            "start_time": "16:30",
            "end_time": None,
            "location": None,
            "attendees": ["Lisa"],
        }),
    },
]


def strategy_zero_shot(user_input: str) -> list[dict]:
    """Zero-shot: system prompt + user message only."""
    return [
        {"role": "system", "content": SYSTEM_BASE},
        {"role": "user", "content": user_input},
    ]


def strategy_few_shot_3(user_input: str) -> list[dict]:
    """Few-shot with 3 examples."""
    return [
        {"role": "system", "content": SYSTEM_BASE},
        *FEW_SHOT_EXAMPLES[:6],  # 3 pairs
        {"role": "user", "content": user_input},
    ]


def strategy_few_shot_5(user_input: str) -> list[dict]:
    """Few-shot with 5 examples."""
    return [
        {"role": "system", "content": SYSTEM_BASE},
        *FEW_SHOT_EXAMPLES[:10],  # 5 pairs
        {"role": "user", "content": user_input},
    ]


def strategy_cot(user_input: str) -> list[dict]:
    """Chain-of-thought prompting."""
    cot_system = (
        f"{SYSTEM_BASE}\n\n"
        "Think step by step before outputting JSON:\n"
        "1. Identify the event type and title\n"
        "2. Resolve the date (convert relative dates using today = 2026-03-12, Thursday)\n"
        "3. Extract start and end times (convert to 24h HH:MM)\n"
        "4. Identify location if mentioned\n"
        "5. List all attendees\n"
        "6. Output the final JSON"
    )
    return [
        {"role": "system", "content": cot_system},
        {"role": "user", "content": user_input},
    ]


def strategy_tool_use(user_input: str) -> dict:
    """Function calling / tool use format (Ollama-compatible)."""
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a calendar assistant. Use the create_calendar_event tool "
                    "to create events from user requests. Today is 2026-03-12 (Thursday)."
                ),
            },
            {"role": "user", "content": user_input},
        ],
        "tools": [CALENDAR_TOOL],
    }


STRATEGIES = {
    "zero_shot": strategy_zero_shot,
    "few_shot_3": strategy_few_shot_3,
    "few_shot_5": strategy_few_shot_5,
    "cot": strategy_cot,
    "tool_use": strategy_tool_use,
}


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def exact_match(predicted: dict, expected: dict) -> bool:
    """All fields match exactly."""
    for key in ["title", "date", "start_time", "end_time", "location"]:
        if predicted.get(key) != expected.get(key):
            return False
    pred_att = set(predicted.get("attendees", []))
    exp_att = set(expected.get("attendees", []))
    return pred_att == exp_att


def field_accuracy(predicted: dict, expected: dict) -> dict[str, bool]:
    """Per-field accuracy."""
    results = {}
    for key in ["title", "date", "start_time", "end_time", "location"]:
        results[key] = predicted.get(key) == expected.get(key)
    pred_att = set(predicted.get("attendees", []))
    exp_att = set(expected.get("attendees", []))
    results["attendees"] = pred_att == exp_att
    return results


def partial_score(predicted: dict, expected: dict) -> float:
    """
    Weighted partial score.
    date=0.30, start_time=0.25, title=0.20, attendees=0.15, location=0.10
    """
    weights = {
        "date": 0.30,
        "start_time": 0.25,
        "title": 0.20,
        "attendees": 0.15,
        "location": 0.10,
    }
    fa = field_accuracy(predicted, expected)
    return sum(weights[k] * (1.0 if fa[k] else 0.0) for k in weights)


def aggregate_results(
    results: list[tuple[dict, dict]],  # (predicted, expected)
) -> dict:
    """Aggregate evaluation metrics."""
    n = len(results)
    if n == 0:
        return {}

    em_count = sum(1 for p, e in results if exact_match(p, e))

    field_counts = {k: 0 for k in ["title", "date", "start_time", "end_time", "location", "attendees"]}
    partial_scores = []

    for pred, exp in results:
        fa = field_accuracy(pred, exp)
        for k, v in fa.items():
            if v:
                field_counts[k] += 1
        partial_scores.append(partial_score(pred, exp))

    return {
        "n_samples": n,
        "exact_match_rate": round(em_count / n, 3),
        "mean_partial_score": round(sum(partial_scores) / n, 3),
        "field_accuracy": {k: round(v / n, 3) for k, v in field_counts.items()},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save dataset
    dataset_path = DATA_DIR / "calendar_eval_dataset.json"
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(EVAL_DATASET, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print("PHASE 4: PROMPT ENGINEERING BASELINES")
    print("=" * 60)

    print(f"\nDataset: {len(EVAL_DATASET)} test cases saved to {dataset_path}")

    # Stats
    langs = {}
    complexities = {}
    for case in EVAL_DATASET:
        langs[case["lang"]] = langs.get(case["lang"], 0) + 1
        complexities[case["complexity"]] = complexities.get(case["complexity"], 0) + 1

    print(f"  Languages: {langs}")
    print(f"  Complexity: {complexities}")

    print(f"\nStrategies defined: {list(STRATEGIES.keys())}")

    # Show example for each strategy
    example_input = "Crée un rendez-vous demain à 14h avec Marie."
    print(f"\nExample prompts for: \"{example_input}\"")
    for name, fn in STRATEGIES.items():
        result = fn(example_input)
        if isinstance(result, dict):
            n_msg = len(result["messages"])
            print(f"  {name:15s} → {n_msg} messages + 1 tool definition")
        else:
            print(f"  {name:15s} → {len(result)} messages")

    print(f"\nEvaluation metrics: exact_match, field_accuracy, partial_score")
    print("Ready for inference with Ollama/llama.cpp.")


if __name__ == "__main__":
    main()
