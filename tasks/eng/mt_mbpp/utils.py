"""Multilingual MBPP utilities for English evaluation.

Prompt format aligned with OLMES mt_mbpp_v2fix:
  {text}
  ```{language}
  {code}
  ```
"""

import math


def _clean(text: str) -> str:
    """Normalize \\r\\n to \\n (v2fix)."""
    return text.replace("\r\n", "\n")


def process_docs(dataset):
    """Clean line endings in all documents (v2fix)."""
    def _fix(doc):
        doc["text"] = _clean(doc["text"]).strip()
        doc["code"] = _clean(doc["code"]).strip()
        return doc
    return dataset.map(_fix)


def doc_to_text(doc):
    """Format prompt: text + open code fence with language tag."""
    return doc["text"] + f"\n```{doc['language']}\n"


def doc_to_target(doc):
    """Format target: code + closing fence."""
    return doc["code"] + "\n```"


def process_results_bpb(doc, results):
    """Compute answer-only BPB (OLMES-style) for Multilingual MBPP."""
    ll, _ = results[0]
    target = doc["code"] + "\n```"
    gold_bytes = len(target.encode("utf-8"))
    return {"bits_per_byte": -ll / (math.log(2) * max(gold_bytes, 1))}
