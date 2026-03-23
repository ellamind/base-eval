"""MMLU utilities for English evaluation."""

import math


# ---------------------------------------------------------------------------
# RC (cloze) format helpers — aligned with OLMES mmlu:rc
# ---------------------------------------------------------------------------

def _format_subject(dataset_name: str) -> str:
    """Convert dataset_name slug to readable subject name."""
    return dataset_name.replace("_", " ")


def doc_to_text_rc(doc):
    """Cloze prompt: 'Question: {question}\nAnswer:'
    Note: subject header is set via per-task 'description' field so it
    appears only once at the top of the few-shot context."""
    return f"Question: {doc['question'].strip()}\nAnswer:"


def doc_to_choice_rc(doc):
    """Return full answer texts as choices (not letter labels)."""
    return doc["choices"]


# ---------------------------------------------------------------------------
# BPB helpers
# ---------------------------------------------------------------------------

def process_results_bpb(doc, results):
    """Compute answer-only BPB (OLMES-style) for MMLU."""
    ll, _ = results[0]
    gold_text = doc["choices"][doc["answer"]]
    gold_bytes = len((" " + gold_text).encode("utf-8"))
    return {"bits_per_byte": -ll / (math.log(2) * max(gold_bytes, 1))}
