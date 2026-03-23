"""SciQ utilities for English evaluation."""

import math
import random
from typing import List, Dict



def sciq_doc_to_choice(doc: dict) -> list:
    """Extract choices from SciQ document."""
    return [doc["correct_answer"], doc["distractor1"], doc["distractor2"], doc["distractor3"]]


def sciq_doc_to_target(doc: dict) -> int:
    """Get target index from SciQ document (correct answer is always first after shuffle)."""
    return 0


def process_sciq_docs(dataset):
    """Process SciQ dataset to add shuffled choices and gold index.

    Uses index-based seeding like OLMES to ensure:
    1. Deterministic shuffling for reproducibility
    2. Correct answer isn't always in the same position (prevents position bias)
    """

    def _process_doc(doc, index):
        # Create choices list: [distractor1, distractor2, distractor3, correct_answer]
        # This matches OLMES convention where correct answer is last before shuffle
        choices = [
            doc["distractor1"],
            doc["distractor2"],
            doc["distractor3"],
            doc["correct_answer"],
        ]

        # Use index as seed for consistent randomization (matches OLMES)
        rng = random.Random(index)
        num_choices = len(choices)
        positions = list(range(num_choices))
        rng.shuffle(positions)

        shuffled_choices = [choices[i] for i in positions]
        # Correct answer was at index 3 (last position), find where it moved to
        gold_idx = positions.index(num_choices - 1)

        return {
            **doc,
            "choices": shuffled_choices,
            "gold": gold_idx,
        }

    return dataset.map(_process_doc, with_indices=True)


def process_results_bpb(doc, results):
    """Compute answer-only BPB (OLMES-style) for SciQ."""
    ll, _ = results[0]
    gold_text = doc["correct_answer"]
    gold_bytes = len((" " + gold_text).encode("utf-8"))
    return {"bits_per_byte": -ll / (math.log(2) * max(gold_bytes, 1))}
