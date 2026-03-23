"""CoQA utilities for German evaluation.

Mirrors the English CoQA implementation (tasks/eng/coqa/utils.py) which follows
the OLMES CoQA task structure.

Dataset: ellamind/coqa-multilingual (deu subset)
- 500 stories with 12-16 conversational QA turns each
- Fields: story, turns[].question, turns[].answer
- Evaluation: each turn is a separate instance with gold history
"""

import math
import re
import string
from collections import Counter
from typing import List

from datasets import Dataset


# =============================================================================
# Dataset processing — flatten turns into individual instances (matches OLMES)
# =============================================================================

def _process_doc_to_multi(doc):
    """Explode each CoQA document into per-turn instances.

    Mirrors OLMES CoQA._process_doc_to_multi() prompt format:
      Passage: {story}
      Preceding questions:
      Question: {q1}
      Answer: {a1}
      Final question:
      Question: {current_q}
      Answer:

    Translated to German:
      Textabschnitt: {story}
      Vorherige Fragen:
      Frage: {q1}
      Antwort: {a1}
      Letzte Frage:
      Frage: {current_q}
      Antwort:
    """
    new_docs = []
    doc_id = doc.get("id", "")
    story = doc["story"]
    turns = doc["turns"]
    previous_qa: list = []
    for turn_idx, turn in enumerate(turns):
        question = turn["question"]
        answer = turn["answer"]

        query = f"Textabschnitt: {story}"
        if previous_qa:
            query += "\n\nVorherige Fragen:"
            for prev in previous_qa:
                query += f"\n\nFrage: {prev['question']}\nAntwort: {prev['answer']}"
        query += "\n\nLetzte Frage:"
        query += f"\n\nFrage: {question}\nAntwort:"

        new_doc = {
            "id": f"{doc_id}_turn{turn_idx}",
            "story": story,
            "query": query,
            "question": question,
            "answers": [answer],
        }
        previous_qa.append({"question": question, "answer": answer})
        new_docs.append(new_doc)
    return new_docs


def process_docs(dataset):
    """Flatten multi-turn stories into per-turn instances."""
    new_docs = []
    for doc in dataset:
        new_docs.extend(_process_doc_to_multi(doc))
    return Dataset.from_list(new_docs)


# =============================================================================
# Token-level F1 metrics (SQuAD-style, adapted for German)
# =============================================================================

_GERMAN_ARTICLES = re.compile(
    r"\b(der|die|das|den|dem|des|ein|eine|einen|einem|einer|eines)\b"
)


def _normalize_answer(text: str) -> str:
    """Normalize answer for comparison (SQuAD-style, German articles)."""
    text = text.lower()
    text = _GERMAN_ARTICLES.sub(" ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = " ".join(text.split())
    return text.strip()


def _get_tokens(text: str) -> List[str]:
    return _normalize_answer(text).split()


def _compute_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1 (SQuAD-style)."""
    pred_tokens = _get_tokens(prediction)
    ref_tokens = _get_tokens(reference)

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return (2 * precision * recall) / (precision + recall)


def _compute_exact_match(prediction: str, reference: str) -> float:
    return 1.0 if _normalize_answer(prediction) == _normalize_answer(reference) else 0.0


def process_results_gen(doc: dict, results: list) -> dict:
    """Process results for CoQA generative task.

    Computes max F1/EM across all reference answers,
    matching OLMES SQuADF1EMRecallMetric behavior.
    """
    prediction = results[0] if results else ""
    references = doc.get("answers", [])
    if not references:
        return {"em": 0.0, "f1": 0.0}

    best_f1 = max(_compute_f1(prediction, ref) for ref in references)
    best_em = max(_compute_exact_match(prediction, ref) for ref in references)

    return {"em": best_em, "f1": best_f1}


# =============================================================================
# BPB helpers (first turn only, matches English coqa_bpb)
# =============================================================================

def process_results_bpb(doc, results):
    """Compute answer-only BPB (OLMES-style) for CoQA."""
    ll, _ = results[0]
    gold_text = doc["turns"][0]["answer"]
    gold_bytes = len((" " + gold_text).encode("utf-8"))
    return {"bits_per_byte": -ll / (math.log(2) * max(gold_bytes, 1))}
