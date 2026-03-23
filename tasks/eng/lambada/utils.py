"""LAMBADA utilities for English evaluation."""

from typing import List


def _normalize_answer(text: str) -> str:
    """Normalize answer for comparison - strip whitespace and lowercase."""
    return text.strip().lower()

def doc_to_text(doc: dict):
    return " ".join(doc["text"].split()[:-1])

def doc_to_target(doc: dict):
    return " " + doc["text"].split()[-1]

def process_results_lambada(doc: dict, results: List[str]) -> dict:
    """Process results for LAMBADA generative task.

    LAMBADA task: predict the last word of a passage.
    Returns acc (accuracy) as 1.0 if exact match, 0.0 otherwise.

    Note: Generation might include a leading space (from doc_to_target format),
    so we strip both prediction and target before comparison.
    """
    prediction = results[0] if results else ""

    # Get the target (last word)
    text = doc.get("text", "")
    target = text.rsplit(" ", 1)[1] if " " in text else text

    # Normalize both for comparison (strip handles the leading space)
    pred_normalized = _normalize_answer(prediction)
    target_normalized = _normalize_answer(target)

    # Exact match
    acc = 1.0 if pred_normalized == target_normalized else 0.0

    return {"acc": acc}
