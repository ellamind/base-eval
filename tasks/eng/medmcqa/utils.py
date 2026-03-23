"""MedMCQA utilities for English evaluation."""

import math


def process_results_bpb(doc, results):
    """Compute answer-only BPB (OLMES-style) for MedMCQA."""
    ll, _ = results[0]
    gold_text = [doc["opa"], doc["opb"], doc["opc"], doc["opd"]][doc["cop"]]
    gold_bytes = len((" " + gold_text).encode("utf-8"))
    return {"bits_per_byte": -ll / (math.log(2) * max(gold_bytes, 1))}
