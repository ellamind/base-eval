"""Jeopardy utilities for English evaluation."""

import datasets
import math
import re
import string
from typing import List, Dict


def cap_dataset(dataset: datasets.Dataset) -> datasets.Dataset:
    """Cap the Jeopardy dataset to 10,000 examples (matching OLMES)."""
    if len(dataset) > 10000:
        return dataset.select(range(10000))
    return dataset


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


def process_results_jeopardy(doc: dict, results: List[str]) -> dict:
    """Process results for Jeopardy generative task."""
    prediction = results[0] if results else ""

    # soldni/jeopardy with mosaicml_gauntlet has 'continuation' as the answer
    reference = doc.get("continuation", "")

    if not reference:
        return {"em": 0.0, "f1": 0.0}

    em = _compute_exact_match(prediction, reference)
    f1 = _compute_f1(prediction, reference)

    return {"em": em, "f1": f1}


def get_jeopardy_gen_fewshot_v2() -> List[Dict]:
    """5-shot examples for Jeopardy (generative). Matches soldni/jeopardy mosaicml_gauntlet format."""
    return [
        {"context": "HISTORY: For the last 8 years of his life, Galileo was under house arrest for espousing this man's theory", "continuation": "Copernicus"},
        {"context": "ESPN's TOP 10 ALL-TIME ATHLETES: No. 2: 1912 Olympian; football star at Carlisle Indian School; 6 MLB seasons with the Reds, Giants & Braves", "continuation": "Jim Thorpe"},
        {"context": "EVERYBODY TALKS ABOUT IT...: The city of Yuma in this state has a record average of 4,055 hours of sunshine each year", "continuation": "Arizona"},
        {"context": "THE COMPANY LINE: In 1963, live on \"The Art Linkletter Show\", this company served its billionth burger", "continuation": "McDonald's"},
        {"context": "EPITAPHS & TRIBUTES: Signer of the Dec. of Indep., framer of the Constitution of Mass., second President of the United States", "continuation": "John Adams"},
    ]


def get_jeopardy_fewshot() -> List[Dict]:
    """5-shot examples for Jeopardy (MC)."""
    return [
        {"category": "HISTORY", "air_date": "2004-12-31", "question": "'For the last 8 years of his life, Galileo was under house arrest for espousing this man\\'s theory'", "value": "$200", "answer": "Copernicus", "round": "Jeopardy!", "show_number": "4680", "uuid": "0a9b32a6-9d9c-4dc9-9ce0-8eb8a52b4dc8", "choices": ["Copernicus", "Newton", "Galileo", "Kepler"], "label": 0},
        {"category": "ESPN's TOP 10 ALL-TIME ATHLETES", "air_date": "2004-12-31", "question": "'No. 2: 1912 Olympian; football star at Carlisle Indian School; 6 MLB seasons with the Reds, Giants & Braves'", "value": "$200", "answer": "Jim Thorpe", "round": "Jeopardy!", "show_number": "4680", "uuid": "3ff4eae5-50b1-47c3-843a-e51b75b37e81", "choices": ["Jim Thorpe", "Babe Ruth", "Jackie Robinson", "Lou Gehrig"], "label": 0},
        {"category": "EVERYBODY TALKS ABOUT IT...", "air_date": "2004-12-31", "question": "'The city of Yuma in this state has a record average of 4,055 hours of sunshine each year'", "value": "$200", "answer": "Arizona", "round": "Jeopardy!", "show_number": "4680", "uuid": "852c3a1f-77dc-4562-96b6-a78451307f0e", "choices": ["Arizona", "California", "Nevada", "New Mexico"], "label": 0},
        {"category": "THE COMPANY LINE", "air_date": "2004-12-31", "question": "'In 1963, live on \"The Art Linkletter Show\", this company served its billionth burger'", "value": "$200", "answer": "McDonald\\'s", "round": "Jeopardy!", "show_number": "4680", "uuid": "3bf2b6b1-a8f9-48a3-8a41-8d1bb78df6ca", "choices": ["McDonald's", "Burger King", "Wendy's", "White Castle"], "label": 0},
        {"category": "EPITAPHS & TRIBUTES", "air_date": "2004-12-31", "question": "'Signer of the Dec. of Indep., framer of the Constitution of Mass., second President of the United States'", "value": "$200", "answer": "John Adams", "round": "Jeopardy!", "show_number": "4680", "uuid": "56e53e6a-fd68-4d4e-9c8b-f8c6b8c1ea9b", "choices": ["John Adams", "Thomas Jefferson", "Benjamin Franklin", "George Washington"], "label": 0},
    ]


def get_jeopardy_gen_fewshot() -> List[Dict]:
    """5-shot examples for Jeopardy (generative)."""
    return [
        {"category": "HISTORY", "question": "For the last 8 years of his life, Galileo was under house arrest for espousing this man's theory", "answer": "Copernicus"},
        {"category": "ESPN's TOP 10 ALL-TIME ATHLETES", "question": "No. 2: 1912 Olympian; football star at Carlisle Indian School; 6 MLB seasons with the Reds, Giants & Braves", "answer": "Jim Thorpe"},
        {"category": "EVERYBODY TALKS ABOUT IT...", "question": "The city of Yuma in this state has a record average of 4,055 hours of sunshine each year", "answer": "Arizona"},
        {"category": "THE COMPANY LINE", "question": "In 1963, live on \"The Art Linkletter Show\", this company served its billionth burger", "answer": "McDonald's"},
        {"category": "EPITAPHS & TRIBUTES", "question": "Signer of the Dec. of Indep., framer of the Constitution of Mass., second President of the United States", "answer": "John Adams"},
    ]


def get_jeopardy_bpb_fewshot() -> List[Dict]:
    """5-shot examples for Jeopardy BPB (soldni/jeopardy mosaicml_gauntlet format)."""
    return [
        {"context": "HISTORY: For the last 8 years of his life, Galileo was under house arrest for espousing this man's theory", "continuation": "Copernicus"},
        {"context": "ESPN's TOP 10 ALL-TIME ATHLETES: No. 2: 1912 Olympian; football star at Carlisle Indian School; 6 MLB seasons with the Reds, Giants & Braves", "continuation": "Jim Thorpe"},
        {"context": "EVERYBODY TALKS ABOUT IT...: The city of Yuma in this state has a record average of 4,055 hours of sunshine each year", "continuation": "Arizona"},
        {"context": "THE COMPANY LINE: In 1963, live on \"The Art Linkletter Show\", this company served its billionth burger", "continuation": "McDonald's"},
        {"context": "EPITAPHS & TRIBUTES: Signer of the Dec. of Indep., framer of the Constitution of Mass., second President of the United States", "continuation": "John Adams"},
    ]


def get_jeopardy_rc_fewshot() -> List[Dict]:
    """10 fixed OLMES fewshot examples for Jeopardy RC (gen2mc format).

    Source: OLMES:jeopardy_mc (fewshot_sources.py).
    """
    return [
        {
            "context_original": "HISTORY: Under the 1814 Treaty of Kiel, this country gave Norway to Sweden but kept Greenland & other islands",
            "choices": {"text": ["Iceland", "Denmark", "Finland", "Netherlands"], "label": ["A", "B", "C", "D"]},
            "answerKey": "B",
        },
        {
            "context_original": "U.S. HISTORY: In the 1968 election, he won 13 1/2 percent of the popular vote & carried 5 southern states",
            "choices": {"text": ["George Wallace", "Barry Goldwater", "Strom Thurmond", "Hubert Humphrey"], "label": ["A", "B", "C", "D"]},
            "answerKey": "A",
        },
        {
            "context_original": 'SHAKESPEARE: In "The Merchant of Venice" he tells his friend Tubal, "Meet me at our synagogue"',
            "choices": {"text": ["Antonio", "Bassanio", "Gratiano", "Shylock"], "label": ["A", "B", "C", "D"]},
            "answerKey": "D",
        },
        {
            "context_original": 'SCIENCE & NATURE: Sir Humphry Davy named this yellowish-green gas from a Greek word meaning "greenish-yellow"',
            "choices": {"text": ["Fluorine", "Sulfur", "Chlorine", "Radon"], "label": ["A", "B", "C", "D"]},
            "answerKey": "C",
        },
        {
            "context_original": "HISTORY: In the midst of the Korean War, this South Korean president was elected to his second of 4 terms",
            "choices": {"text": ["Syngman Rhee", "Kim Il-sung", "Moon Jae-in", "Park Chung-hee"], "label": ["A", "B", "C", "D"]},
            "answerKey": "A",
        },
        {
            "context_original": "U.S. HISTORY: In 1878 an amendment for this was introduced in Congress; its adoption didn't occur until 1920",
            "choices": {"text": ["Income tax reform", "Civil rights for African Americans", "Prohibition", "Women's suffrage"], "label": ["A", "B", "C", "D"]},
            "answerKey": "D",
        },
        {
            "context_original": 'IN THE DICTIONARY: This car name may come from an abbreviation of "general purpose vehicle"',
            "choices": {"text": ["Coupe", "Jeep", "SUV", "Sedan"], "label": ["A", "B", "C", "D"]},
            "answerKey": "B",
        },
        {
            "context_original": "SHAKESPEARE: Othello kills himself on this island",
            "choices": {"text": ["Malta", "Crete", "Cyprus", "Sicily"], "label": ["A", "B", "C", "D"]},
            "answerKey": "C",
        },
        {
            "context_original": "SCIENCE & NATURE: Take the fibrinogen out of blood plasma & you're left with a fluid called this",
            "choices": {"text": ["Serum", "Hemoglobin", "Platelets", "Plasma"], "label": ["A", "B", "C", "D"]},
            "answerKey": "A",
        },
        {
            "context_original": 'ADDICTED TO ADJECTIVES: This adjective derives from the name of the author of "Martin Chuzzlewit"',
            "choices": {"text": ["Shakespearean", "Austenian", "Dickensian", "Hemingwayesque"], "label": ["A", "B", "C", "D"]},
            "answerKey": "C",
        },
    ]


def get_jeopardy_mc_fewshot() -> List[Dict]:
    """10 fixed OLMES fewshot examples for Jeopardy RC (gen2mc format).

    Source: OLMES:jeopardy_mc (fewshot_sources.py).
    """
    return [
    {
        "id": "jeopardy_mc_format_fewshot_0",
        "choices": {
            "text": ["Iceland", "Denmark", "Finland", "Netherlands"],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "B",
        "context_original": "HISTORY: Under the 1814 Treaty of Kiel, this country gave Norway to Sweden but kept Greenland & other islands",
        "continuation_original": "Denmark",
        "category_original": "history",
    },
    {
        "id": "jeopardy_mc_format_fewshot_1",
        "choices": {
            "text": ["George Wallace", "Barry Goldwater", "Strom Thurmond", "Hubert Humphrey"],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "A",
        "context_original": "U.S. HISTORY: In the 1968 election, he won 13 1/2 percent of the popular vote & carried 5 southern states",
        "continuation_original": "George Wallace",
        "category_original": "us_history",
    },
    {
        "id": "jeopardy_mc_format_fewshot_2",
        "choices": {
            "text": ["Antonio", "Bassanio", "Gratiano", "Shylock"],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "D",
        "context_original": 'SHAKESPEARE: In "The Merchant of Venice" he tells his friend Tubal, "Meet me at our synagogue"',
        "continuation_original": "Shylock",
        "category_original": "shakespeare",
    },
    {
        "id": "jeopardy_mc_format_fewshot_3",
        "choices": {
            "text": ["Fluorine", "Sulfur", "Chlorine", "Radon"],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "C",
        "context_original": 'SCIENCE & NATURE: Sir Humphry Davy named this yellowish-green gas from a Greek word meaning "greenish-yellow"',
        "continuation_original": "Chlorine",
        "category_original": "science_nature",
    },
    {
        "id": "jeopardy_mc_format_fewshot_5",
        "choices": {
            "text": ["Syngman Rhee", "Kim Il-sung", "Moon Jae-in", "Park Chung-hee"],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "A",
        "context_original": "HISTORY: In the midst of the Korean War, this South Korean president was elected to his second of 4 terms",
        "continuation_original": "Syngman Rhee",
        "category_original": "history",
    },
    {
        "id": "jeopardy_mc_format_fewshot_6",
        "choices": {
            "text": [
                "Income tax reform",
                "Civil rights for African Americans",
                "Prohibition",
                "Women's suffrage",
            ],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "D",
        "context_original": "U.S. HISTORY: In 1878 an amendment for this was introduced in Congress; its adoption didn't occur until 1920",
        "continuation_original": "Women's suffrage",
        "category_original": "us_history",
    },
    {
        "id": "jeopardy_mc_format_fewshot_4",
        "choices": {"text": ["Coupe", "Jeep", "SUV", "Sedan"], "label": ["A", "B", "C", "D"]},
        "answerKey": "B",
        "context_original": 'IN THE DICTIONARY: This car name may come from an abbreviation of "general purpose vehicle"',
        "continuation_original": "jeep",
        "category_original": "in_the_dictionary",
    },
    {
        "id": "jeopardy_mc_format_fewshot_7",
        "choices": {"text": ["Malta", "Crete", "Cyprus", "Sicily"], "label": ["A", "B", "C", "D"]},
        "answerKey": "C",
        "context_original": "SHAKESPEARE: Othello kills himself on this island",
        "continuation_original": "Cyprus",
        "category_original": "shakespeare",
    },
    {
        "id": "jeopardy_mc_format_fewshot_8",
        "choices": {
            "text": ["Serum", "Hemoglobin", "Platelets", "Plasma"],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "A",
        "context_original": "SCIENCE & NATURE: Take the fibrinogen out of blood plasma & you're left with a fluid called this",
        "continuation_original": "Serum",
        "category_original": "science_nature",
    },
    {
        "id": "jeopardy_mc_format_fewshot_9",
        "choices": {
            "text": ["Shakespearean", "Austenian", "Dickensian", "Hemingwayesque"],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "C",
        "context_original": 'ADDICTED TO ADJECTIVES: This adjective derives from the name of the author of "Martin Chuzzlewit"',
        "continuation_original": "Dickensian",
        "category_original": "addicted_to_adjectives",
    },
]

def process_results_jeopardy_gen(doc: dict, results: List[str]) -> dict:
    """Process results for Jeopardy generative task (OLMES-aligned, all_questions config)."""
    prediction = results[0] if results else ""

    reference = doc.get("continuation", "")

    if not reference:
        return {"em": 0.0, "f1": 0.0}

    em = _compute_exact_match(prediction, reference)
    f1 = _compute_f1(prediction, reference)

    return {"em": em, "f1": f1}


def get_jeopardy_gen_fewshot_gen() -> List[Dict]:
    """5-shot examples for Jeopardy generative (OLMES-aligned).

    Uses soldni/jeopardy all_questions schema (context, continuation).
    Source: oe_eval/tasks/fewshot_sources.py OLMES:jeopardy (first 5).
    """
    return [
        {"context": "HISTORY: Under the 1814 Treaty of Kiel, this country gave Norway to Sweden but kept Greenland & other islands", "continuation": "Denmark"},
        {"context": "U.S. HISTORY: In the 1968 election, he won 13 1/2 percent of the popular vote & carried 5 southern states", "continuation": "George Wallace"},
        {"context": 'SHAKESPEARE: In "The Merchant of Venice" he tells his friend Tubal, "Meet me at our synagogue"', "continuation": "Shylock"},
        {"context": 'SCIENCE & NATURE: Sir Humphry Davy named this yellowish-green gas from a Greek word meaning "greenish-yellow"', "continuation": "Chlorine"},
        {"context": 'IN THE DICTIONARY: This car name may come from an abbreviation of "general purpose vehicle"', "continuation": "jeep"},
    ]


def process_results_bpb(doc, results):
    """Compute answer-only BPB (OLMES-style) for Jeopardy."""
    ll, _ = results[0]
    gold_text = doc["continuation"]
    gold_bytes = len((" " + gold_text).encode("utf-8"))
    return {"bits_per_byte": -ll / (math.log(2) * max(gold_bytes, 1))}
