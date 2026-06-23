"""Utility functions for the German GRIPS base-model tasks (MC, RC, BPB).

Dataset: ellamind/grips-base
- German reasoning, idioms, puzzles & wordplay quiz.
- Each item carries baked-in, deterministically shuffled `choices` with the
  gold option at `answer_idx` (and letter `answer_key`), a clean `answer`
  string, and an `explanation`.
- A dedicated `fewshot` split (5 items) supplies the few-shot context for all
  formulations; the eval `test` split excludes those items.

MC/RC use inline Jinja templates (no helpers needed). Only the two BPB variants
need utils, to keep the scored continuation and its byte count exactly aligned.
"""

import math


def doc_to_target_bpb(doc):
    """Gold continuation: the answer only."""
    return f" {doc['answer']}"


def doc_to_target_bpb_expl(doc):
    """Gold continuation: the answer followed by its explanation."""
    return f" {doc['answer']}\n{doc['explanation']}"


def _bits_per_byte(ll, continuation):
    gold_bytes = len(continuation.encode("utf-8"))
    return -ll / (math.log(2) * max(gold_bytes, 1))


def process_results_bpb(doc, results):
    """BPB = -log2 P(answer | context) / bytes(answer)."""
    ll, _is_greedy = results[0]
    return {"answer_bits_per_byte": _bits_per_byte(ll, doc_to_target_bpb(doc))}


def process_results_bpb_expl(doc, results):
    """BPB over the answer + explanation continuation.

    Emits a distinct metric name so this diagnostic variant is not averaged
    together with the plain `answer_bits_per_byte` tasks in suite aggregates.
    """
    ll, _is_greedy = results[0]
    return {"answer_expl_bits_per_byte": _bits_per_byte(ll, doc_to_target_bpb_expl(doc))}
