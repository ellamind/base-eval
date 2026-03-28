"""HumanEval utilities for English evaluation."""

import math
from typing import List, Dict

_compute = None


def doc_to_target(doc):
    """Return the gold code completion.

    For fewshot examples (marked with is_fewshot), returns code + newline.
    For evaluation examples, returns the test cases for execution.
    """
    return doc['test']+"\ncheck(" + doc['entry_point'] + ")"


def doc_to_target_olmo3(doc):
    """Return the OLMo3-style fenced code completion target."""
    return doc["canonical_solution"] + "```"


def doc_to_target_fewshot(doc):
    return doc_to_target_olmo3(doc)


def _get_code_eval():
    """Lazy-load the code_eval metric (requires HF_ALLOW_CODE_EVAL=1)."""
    global _compute
    if _compute is None:
        import evaluate as hf_evaluate
        _compute = hf_evaluate.load("code_eval")
    return _compute


def pass_at_k(references: list[str], predictions: list[list[str]], k: list[int] = None):
    """Compute pass@k metric using code execution.

    Done at a per doc basis

    Args:
        references: List of test case strings (one per problem), contains one element, i.e. the test case string.
        predictions: List of lists of code completions (multiple samples per problem), predictions has one element which is the list holding the n-repeat number of predictions
        k: List of k values to compute pass@k for (e.g., [1, 10, 100])

    Returns:
        Dict with pass@k scores
    """
    assert k is not None
    if isinstance(k, int):
        k = [k]

    res = _get_code_eval().compute(
        references=references,
        predictions=predictions,
        k=k,
        num_workers=4,
        timeout=5.0,
    )
    return res[0]

def build_predictions_olmo(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    """Combine prompt with model responses for code execution (olmo3 variant).
    This basically strips the few shots
    """
    return [[doc["prompt"] + r for r in resp] for resp, doc in zip(resps, docs)]

def extract_code_completion(text: str, entry_point: str = None) -> str:
    """Extract code completion from model response."""
    stop_sequences = [
        "\nclass ",
        "\ndef ",
        "\n#",
        "\nif __name__",
        "\nprint(",
        "\nassert ",
        "```",
    ]
    for stop in stop_sequences:
        if stop in text:
            text = text[:text.index(stop)]
    return text.strip()


def process_results_bpb(doc, results):
    """Compute answer-only BPB (OLMES-style) for HumanEval."""
    ll, _ = results[0]
    gold_text = doc_to_target_olmo3(doc)
    gold_bytes = len(gold_text.encode("utf-8"))
    return {"bits_per_byte": -ll / (math.log(2) * max(gold_bytes, 1))}
