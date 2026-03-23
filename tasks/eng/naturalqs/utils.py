"""Natural Questions utilities for English evaluation."""

import math
import re
import string
from typing import List, Dict


# =============================================================================
# Token-level F1 metrics (SQuAD-style)
# =============================================================================

def _normalize_answer(text: str) -> str:
    """Normalize answer for comparison."""
    text = text.lower()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = ''.join(ch for ch in text if ch not in string.punctuation)
    text = ' '.join(text.split())
    return text.strip()


def _get_tokens(text: str) -> List[str]:
    """Tokenize normalized text."""
    return _normalize_answer(text).split()


def _compute_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1."""
    pred_tokens = _get_tokens(prediction)
    ref_tokens = _get_tokens(reference)

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = set(pred_tokens) & set(ref_tokens)
    num_common = sum(min(pred_tokens.count(t), ref_tokens.count(t)) for t in common)

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return (2 * precision * recall) / (precision + recall)


def _compute_exact_match(prediction: str, reference: str) -> float:
    """Compute exact match."""
    return 1.0 if _normalize_answer(prediction) == _normalize_answer(reference) else 0.0


def _max_over_references(metric_fn, prediction: str, references: List[str]) -> float:
    """Return max metric score over all reference answers."""
    if not references:
        return 0.0
    return max(metric_fn(prediction, ref) for ref in references)


def process_results_naturalqs(doc: dict, results: List[str]) -> dict:
    """Process results for Natural Questions generative task."""
    prediction = results[0] if results else ""

    # nq_open dataset has 'answer' as a list of valid answers
    references = doc.get("answer", [])
    if isinstance(references, str):
        references = [references]

    if not references:
        return {"em": 0.0, "f1": 0.0}

    em = _max_over_references(_compute_exact_match, prediction, references)
    f1 = _max_over_references(_compute_f1, prediction, references)

    return {"em": em, "f1": f1}


def get_naturalqs_fewshot() -> List[Dict]:
    """5-shot examples for Natural Questions (MC)."""
    return [
        {"question": "When was the last time the USA men's national soccer team missed the World Cup?", "answer": ["1986"], "uuid": "9b8ed2f1-35eb-4af4-b162-b5c4e8b31fb9", "choices": ["1986", "1990", "1994", "2002"], "label": 0},
        {"question": "What does barium do in a ct scan?", "answer": ["to minimize the thickness of the gastrointestinal wall"], "uuid": "28ae5a13-bbf0-4c41-907a-31e2dec09fca", "choices": ["minimizes gastrointestinal wall thickness", "enhances blood vessels", "highlights bone structure", "reduces radiation exposure"], "label": 0},
        {"question": "Where was the fort located at which the first shot of the civil war was fired?", "answer": ["Charleston Harbor, South Carolina"], "uuid": "e9ecb201-cfe9-49ad-bcad-4d8c0b6f64f1", "choices": ["Charleston Harbor, South Carolina", "Gettysburg, Pennsylvania", "Richmond, Virginia", "Atlanta, Georgia"], "label": 0},
        {"question": "When did nando's come to the UK?", "answer": ["1992"], "uuid": "f9b8d794-e7c7-451a-913b-b9a0f013e30e", "choices": ["1992", "1987", "1995", "2000"], "label": 0},
        {"question": "Who plays uni-kitty in the LEGO movie?", "answer": ["Alison Brie"], "uuid": "9c8c3f2c-2c0c-492e-ae99-42a1ec8c3072", "choices": ["Alison Brie", "Elizabeth Banks", "Tiffany Haddish", "Maya Rudolph"], "label": 0},
    ]


def get_naturalqs_rc_fewshot() -> List[Dict]:
    """10 fixed OLMES fewshot examples for Natural Questions RC (gen2mc format).

    Source: OLMES:naturalqs_mc (fewshot_sources.py).
    """
    return [
        {
            "question": "Which side of the White House is the front?",
            "choices": {"text": ["East", "North", "South", "West"], "label": ["A", "B", "C", "D"]},
            "answerKey": "B",
        },
        {
            "question": "Where was the Super Bowl hosted in 2019?",
            "choices": {"text": ["Atlanta, Georgia", "Houston, Texas", "Los Angeles, California", "Miami, Florida"], "label": ["A", "B", "C", "D"]},
            "answerKey": "A",
        },
        {
            "question": "What is the origin of the name Cynthia?",
            "choices": {"text": ["Latin", "Hebrew", "Norse", "Greek"], "label": ["A", "B", "C", "D"]},
            "answerKey": "D",
        },
        {
            "question": "Who composed and performed the theme song for Miami Vice?",
            "choices": {"text": ["Danny Elfman", "Hans Zimmer", "Jan Hammer", "John Williams"], "label": ["A", "B", "C", "D"]},
            "answerKey": "C",
        },
        {
            "question": "What is the size of the angles of an equilateral triangle?",
            "choices": {"text": ["90\u00b0", "60\u00b0", "120\u00b0", "45\u00b0"], "label": ["A", "B", "C", "D"]},
            "answerKey": "B",
        },
        {
            "question": "Who plays Mavis in the movie Hotel Transylvania?",
            "choices": {"text": ["Selena Gomez", "Emma Stone", "Kristen Bell", "Anne Hathaway"], "label": ["A", "B", "C", "D"]},
            "answerKey": "A",
        },
        {
            "question": "Which West Ham players participated in the 1966 World Cup?",
            "choices": {"text": ["Wayne Rooney", "David Beckham", "Alan Shearer", "Bobby Moore"], "label": ["A", "B", "C", "D"]},
            "answerKey": "D",
        },
        {
            "question": "Ice sheets and tundra are typical of which K\u00f6ppen climate category?",
            "choices": {"text": ["Arid", "Temperate", "Polar", "Tropical"], "label": ["A", "B", "C", "D"]},
            "answerKey": "C",
        },
        {
            "question": "What is the legal marriage age in New York?",
            "choices": {"text": ["18", "21", "16", "25"], "label": ["A", "B", "C", "D"]},
            "answerKey": "A",
        },
    ]


def get_naturalqs_gen_fewshot() -> List[Dict]:
    return [
    {"question": "which side of the white house is the front", "answer": ["North"]},
    {"question": "who's hosting the super bowl in 2019", "answer": ["Atlanta, Georgia"]},
    {"question": "what is the origin of the name cynthia", "answer": ["Greek"]},
    {
        "question": "who is the guy who voiced disney channel",
        "answer": ['"Buzz" Brainard', "Cameron"],
    },
    {"question": "what is the size of the angles of an equilateral triangle", "answer": ["60°"]},
    {
        "question": "who plays mavis in the movie hotel transylvania",
        "answer": ["Sadie Sandler", "Selena Gomez"],
    },
    {
        "question": "west ham players in the 1966 world cup",
        "answer": ["Martin Peters", "Geoff Hurst", "Bobby Moore"],
    },
    {"question": "who sings the theme song for miami vice", "answer": ["Jan Hammer"]},
    {
        "question": "ice sheets and tundra are typical of which koppen climate category",
        "answer": ["polar"],
    },
    {"question": "what's the legal marriage age in new york", "answer": ["18"]},
]

def get_naturalqs_mc_fewshot() -> List[Dict]:
    # from olmes FEWSHOT_SOURCES["OLMES:naturalqs_mc"]
    return [
    {
        "id": "nq_open_mc_format_fewshot_0",
        "question": "Which side of the White House is the front?",
        "choices": {"text": ["East", "North", "South", "West"], "label": ["A", "B", "C", "D"]},
        "answerKey": "B",
        "question_original": "which side of the white house is the front",
        "answer_original": ["North"],
    },
    {
        "id": "nq_open_mc_format_fewshot_1",
        "question": "Where was the Super Bowl hosted in 2019?",
        "choices": {
            "text": [
                "Atlanta, Georgia",
                "Houston, Texas",
                "Los Angeles, California",
                "Miami, Florida",
            ],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "A",
        "question_original": "who's hosting the super bowl in 2019",
        "answer_original": ["Atlanta, Georgia"],
    },
    {
        "id": "nq_open_mc_format_fewshot_2",
        "question": "What is the origin of the name Cynthia?",
        "choices": {"text": ["Latin", "Hebrew", "Norse", "Greek"], "label": ["A", "B", "C", "D"]},
        "answerKey": "D",
        "question_original": "what is the origin of the name cynthia",
        "answer_original": ["Greek"],
    },
    {
        "id": "nq_open_mc_format_fewshot_7",
        "question": "Who composed and performed the theme song for Miami Vice?",
        "choices": {
            "text": ["Danny Elfman", "Hans Zimmer", "Jan Hammer", "John Williams"],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "C",
        "question_original": "who sings the theme song for miami vice",
        "answer_original": ["Jan Hammer"],
    },
    {
        "id": "nq_open_mc_format_fewshot_4",
        "question": "What is the size of the angles of an equilateral triangle?",
        "choices": {
            "text": ["90\u00b0", "60\u00b0", "120\u00b0", "45\u00b0"],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "B",
        "question_original": "what is the size of the angles of an equilateral triangle",
        "answer_original": ["60\u00b0"],
    },
    {
        "id": "nq_open_mc_format_fewshot_5",
        "question": "Who plays Mavis in the movie Hotel Transylvania?",
        "choices": {
            "text": ["Selena Gomez", "Emma Stone", "Kristen Bell", "Anne Hathaway"],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "A",
        "question_original": "who plays mavis in the movie hotel transylvania",
        "answer_original": ["Sadie Sandler", "Selena Gomez"],
    },
    {
        "id": "nq_open_mc_format_fewshot_6",
        "question": "Which West Ham players participated in the 1966 World Cup?",
        "choices": {
            "text": ["Wayne Rooney", "David Beckham", "Alan Shearer", "Bobby Moore"],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "D",
        "question_original": "west ham players in the 1966 world cup",
        "answer_original": ["Martin Peters", "Geoff Hurst", "Bobby Moore"],
    },
    {
        "id": "nq_open_mc_format_fewshot_8",
        "question": "Ice sheets and tundra are typical of which K\u00f6ppen climate category?",
        "choices": {
            "text": ["Arid", "Temperate", "Polar", "Tropical"],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "C",
        "question_original": "ice sheets and tundra are typical of which koppen climate category",
        "answer_original": ["polar"],
    },
    {
        "id": "nq_open_mc_format_fewshot_9",
        "question": "What is the legal marriage age in New York?",
        "choices": {"text": ["18", "21", "16", "25"], "label": ["A", "B", "C", "D"]},
        "answerKey": "A",
        "question_original": "what's the legal marriage age in new york",
        "answer_original": ["18"],
    },
]


def process_results_bpb(doc, results):
    """Compute answer-only BPB (OLMES-style) for Natural Questions."""
    ll, _ = results[0]
    gold_text = doc["answer"][0]
    gold_bytes = len((" " + gold_text).encode("utf-8"))
    return {"bits_per_byte": -ll / (math.log(2) * max(gold_bytes, 1))}
