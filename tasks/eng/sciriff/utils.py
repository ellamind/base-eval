"""SciRIFF utilities for English evaluation."""

import math


def process_docs_sciriff_yesno(dataset):
    """
    Process SciRIFF Yes/No dataset to convert answer to gold index.

    SciRIFF Yes/No (allenai/sciriff-yesno):
      - id: string, unique identifier
      - context: string, background text/passage
      - question: string, yes/no question
      - answer: "Yes" or "No"

    Transforms to:
      - gold: int (0 for "Yes", 1 for "No")
    """
    def _process_doc(doc):
        gold = 0 if doc["answer"] == "Yes" else 1
        return {
            "id": doc["id"],
            "context": doc["context"],
            "question": doc["question"],
            "answer": doc["answer"],
            "gold": gold,
        }

    return dataset.map(_process_doc)


def process_results_bpb(doc, results):
    """Compute answer-only BPB (OLMES-style) for SciRIFF."""
    ll, _ = results[0]
    gold_text = ["Yes", "No"][doc["gold"]]
    gold_bytes = len((" " + gold_text).encode("utf-8"))
    return {"bits_per_byte": -ll / (math.log(2) * max(gold_bytes, 1))}
