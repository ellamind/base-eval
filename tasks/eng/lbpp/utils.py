"""LBPP (Less Basic Python Programming) utilities for English evaluation.

Dataset: ellamind/lbpp (162 Python problems with unit tests)
Mirrors OLMES default base-model config: instruction + signature prompt.
"""

import random

_compute = None
_rng = random.Random(1234)

# Assistant prefix prepended to model generation (set in YAML via target_delimiter)
ASSISTANT_PREFIX = "Here is the completed function:\n\n```python\n"


def _get_code_eval():
    """Lazy-load the code_eval metric (requires HF_ALLOW_CODE_EVAL=1)."""
    global _compute
    if _compute is None:
        import evaluate as hf_evaluate
        _compute = hf_evaluate.load("code_eval")
    return _compute


def doc_to_text(doc):
    """Build base-model prompt: instruction + signature + assistant prefix.

    Mirrors OLMES default variant (lbpp.py line 209 + assistant_prefix line 71):
        instruction + "\\n" + signature + assistant_prefix
    The assistant prefix primes the model to output code directly.
    """
    return (
        doc["instruction"] + "\n" + doc["signature"]
        + ASSISTANT_PREFIX
    )


def doc_to_target(doc):
    """Return the test cases for execution."""
    # Build test string: test_setup (minus the 'from code import ...' line) + test_list
    lines = []
    if doc["test_setup"]:
        setup_lines = doc["test_setup"].split("\n")
        # Skip the 'from code import ...' line (LBPP convention)
        lines.extend(l for l in setup_lines if not l.startswith("from code import"))
    lines.extend(doc["test_list"])
    return "\n".join(lines)


def build_predictions(resps, docs):
    """Combine model responses into complete programs for code execution.

    The model generates after the assistant prefix (``python\\n),
    typically starting with the function signature. Use the raw output directly.
    """
    results = []
    for resp, doc in zip(resps, docs):
        preds = []
        for r in resp:
            preds.append(r)
        results.append(preds)
    return results


def pass_at_k(references, predictions, k=None):
    """Compute pass@k metric using code execution.

    Args:
        references: List of test case strings (one per problem).
        predictions: List of lists of code completions.
        k: List of k values to compute pass@k for.

    Returns:
        Dict with pass@k scores.
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
