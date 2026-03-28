"""DeepSeek LeetCode utilities for English evaluation."""

_compute = None


def doc_to_text(doc):
    """Return the dataset prompt directly.

    The prompt already ends with the class/method signature,
    so the model naturally continues with the method body.
    """
    return doc["prompt"]


def doc_to_target(doc):
    """Return the test cases for execution."""
    return doc["test"]


def _get_code_eval():
    """Lazy-load the code_eval metric (requires HF_ALLOW_CODE_EVAL=1)."""
    global _compute
    if _compute is None:
        import evaluate as hf_evaluate
        _compute = hf_evaluate.load("code_eval")
    return _compute


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    """Combine prompt with model continuation for code execution.

    The model generates the method body after the prompt's class/method
    signature. Prepend the prompt to form a complete program.
    """
    results = []
    for resp, doc in zip(resps, docs):
        prompt = doc["prompt"]
        results.append([prompt + r for r in resp])
    return results


def pass_at_k(references: list[str], predictions: list[list[str]], k: list[int] = None):
    """Compute pass@k metric using code execution.

    Args:
        references: List of test case strings (one per problem).
        predictions: List of lists of code completions (multiple samples per problem).
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
