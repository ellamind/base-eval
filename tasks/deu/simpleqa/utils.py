"""Utility functions for German SimpleQA Verified tasks (Gen, MC, RC, BPB).

Dataset: ellamind/simpleqa-verified-multilingual (deu subset)
- 1,000 short-form factual QA prompts testing parametric knowledge
- Fields: question, answer, answer_aliases, hard_distractors, easy_distractors
- 5-shot from first 5 eval examples (filtered from evaluation)
"""

import math
import random
import re
import unicodedata

LABELS = "ABCDE"
OLMES_SHUFFLE_SEED = 111

FEWSHOT_IDS = {
    "simpleqa_verified_1000",
    "simpleqa_verified_1001",
    "simpleqa_verified_1007",
    "simpleqa_verified_1010",
    "simpleqa_verified_1013",
}

_FEWSHOT_RAW = [
    {
        "id": "simpleqa_verified_1000",
        "question": (
            "Nennen Sie den Tag, den Monat und das Jahr, an dem Canon zwei "
            "neue tragbare UHDgc 2/3-Zoll-Zoomobjektive vorgestellt hat, die "
            "für 4K-UHD-Broadcast-Kameras entwickelt wurden."
        ),
        "answer": "2. April 2019",
        "answer_aliases": ["02.04.2019", "02. April 2019", "2.04.2019"],
        "hard_distractors": [
            "4. April 2018",
            "2. Mai 2019",
            "12. September 2019",
            "18. April 2019",
        ],
    },
    {
        "id": "simpleqa_verified_1001",
        "question": (
            "Wie lautet der Vor- und Nachname der Preisträgerin des Annie "
            "Jump Cannon Award in Astronomy im Jahr 1952?"
        ),
        "answer": "Ida Barney",
        "answer_aliases": ["Dr. Ida Barney"],
        "hard_distractors": [
            "Helen Sawyer Hogg",
            "Cecilia Payne-Gaposchkin",
            "Charlotte Moore Sitterly",
            "Margaret Burbidge",
        ],
    },
    {
        "id": "simpleqa_verified_1007",
        "question": (
            "Wie lautet der Nachname der Person, die 1970 den Faraday "
            "Lectureship Prize, früher bekannt als Faraday Lectureship, "
            "gewonnen hat?"
        ),
        "answer": "Herzberg",
        "answer_aliases": ["Gerhard Herzberg"],
        "hard_distractors": ["Coulson", "Eigen", "Porter", "Barton"],
    },
    {
        "id": "simpleqa_verified_1010",
        "question": "Wie viele totale Mondfinsternisse gab es im Jahr 1982?",
        "answer": "3",
        "answer_aliases": ["drei", "Drei"],
        "hard_distractors": ["0", "1", "2", "4"],
    },
    {
        "id": "simpleqa_verified_1013",
        "question": (
            "Wie viele Kinder hatte der Schweizer Maler Johann Caspar "
            "Füssli mit seiner Frau Elisabeth?"
        ),
        "answer": "18",
        "answer_aliases": ["achtzehn", "18 Kinder", "achtzehn Kinder"],
        "hard_distractors": ["12", "14", "16", "20"],
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
    """Add shuffled choices (answer + 4 hard distractors) to doc."""
    all_choices = [doc["answer"]] + doc["hard_distractors"]
    rng = random.Random(OLMES_SHUFFLE_SEED + idx)
    rng.shuffle(all_choices)

    doc["choices"] = all_choices
    doc["answer_idx"] = all_choices.index(doc["answer"])
    doc["answer_label"] = LABELS[all_choices.index(doc["answer"])]
    return doc


def process_docs(dataset):
    """Filter flagged rows and fewshot IDs (for gen/bpb variants)."""
    return _filter_base(dataset)


def process_docs_mc(dataset):
    """Filter + shuffle choices for MC/RC variants."""
    return _filter_base(dataset).map(_shuffle_choices, with_indices=True)


# ---------------------------------------------------------------------------
# Fewshot
# ---------------------------------------------------------------------------

def _build_fewshot_mc():
    """Pre-shuffle fewshot examples for MC/RC format."""
    results = []
    for i, raw in enumerate(_FEWSHOT_RAW):
        all_choices = [raw["answer"]] + raw["hard_distractors"]
        rng = random.Random(OLMES_SHUFFLE_SEED + i)
        rng.shuffle(all_choices)
        results.append({
            "question": raw["question"],
            "answer": raw["answer"],
            "choices": all_choices,
            "answer_idx": all_choices.index(raw["answer"]),
            "answer_label": LABELS[all_choices.index(raw["answer"])],
        })
    return results


def list_fewshot_gen():
    """5 fewshot examples for generative format."""
    return [
        {"question": raw["question"], "answer": raw["answer"]}
        for raw in _FEWSHOT_RAW
    ]


def list_fewshot_mc():
    """5 fewshot examples for MC/RC format (shuffled choices)."""
    return _build_fewshot_mc()


# ---------------------------------------------------------------------------
# Gen helpers
# ---------------------------------------------------------------------------

def doc_to_text(doc):
    return f"Frage: {doc['question']}\nAntwort:"


def doc_to_target_gen(doc):
    return f" {doc['answer']}"


def _normalize(text):
    """Lowercase, strip, remove punctuation for flexible matching."""
    text = text.lower().strip()
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def process_results_gen(doc, results):
    """Check generated answer against answer + all aliases.

    Returns exact_match=1 if the normalized generation matches any
    acceptable answer form.
    """
    pred = results[0] if isinstance(results, list) else results
    pred_norm = _normalize(str(pred))

    candidates = [doc["answer"]] + (doc.get("answer_aliases") or [])
    match = any(_normalize(c) == pred_norm for c in candidates)

    return {"exact_match": int(match)}


# ---------------------------------------------------------------------------
# MC helpers (letter-labeled generation)
# ---------------------------------------------------------------------------

def doc_to_text_mc(doc):
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
# RC helpers (likelihood-based rank choice)
# ---------------------------------------------------------------------------

def doc_to_choice_rc(doc):
    return doc["choices"]


# ---------------------------------------------------------------------------
# BPB helpers
# ---------------------------------------------------------------------------

def doc_to_target_bpb(doc):
    return f" {doc['answer']}"


def process_results_bpb(doc, results):
    """BPB = -log2 P(answer | context) / bytes(answer)."""
    ll, _is_greedy = results[0]

    gold_bytes = len(f" {doc['answer']}".encode("utf-8"))
    bpb = -ll / (math.log(2) * max(gold_bytes, 1))

    return {"bits_per_byte": bpb}
