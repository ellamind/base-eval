"""LAB-Bench utilities for English evaluation."""

import math


def process_results_bpb(doc, results):
    """Compute answer-only BPB (OLMES-style) for LAB-Bench."""
    ll, _ = results[0]
    gold_text = doc["ideal"]
    gold_bytes = len((" " + gold_text).encode("utf-8"))
    return {"bits_per_byte": -ll / (math.log(2) * max(gold_bytes, 1))}
