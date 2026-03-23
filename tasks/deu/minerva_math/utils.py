"""
Minerva Math utilities for German - uses lm-eval-harness functions with German prompts.

Aligned with ENG minerva_math_gen: sampling n=4, pass@k / maj@k evaluation.
Overrides answer extraction to handle both German and English patterns.
"""

import math
import re
from collections import Counter

import datasets
import numpy as np

# Import lm-eval-harness minerva_math utilities
from lm_eval.tasks.minerva_math.utils import (
    is_equiv,
    last_boxed_only_string,
    normalize_final_answer,
    process_docs as _minerva_process_docs,
    remove_boxed,
)

try:
    from math_verify import parse, verify
except ImportError:
    parse = None
    verify = None


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Process documents using minerva_math utilities."""
    return _minerva_process_docs(dataset)


def _filter_subject(dataset: datasets.Dataset, subject: str) -> datasets.Dataset:
    """Filter dataset to a specific problem_type and process docs."""
    filtered = dataset.filter(lambda x: x["problem_type"] == subject)
    return _minerva_process_docs(filtered)


def process_algebra(dataset):
    return _filter_subject(dataset, "Algebra")

def process_counting_and_probability(dataset):
    return _filter_subject(dataset, "Counting & Probability")

def process_geometry(dataset):
    return _filter_subject(dataset, "Geometry")

def process_intermediate_algebra(dataset):
    return _filter_subject(dataset, "Intermediate Algebra")

def process_number_theory(dataset):
    return _filter_subject(dataset, "Number Theory")

def process_prealgebra(dataset):
    return _filter_subject(dataset, "Prealgebra")

def process_precalculus(dataset):
    return _filter_subject(dataset, "Precalculus")


def doc_to_text(doc: dict) -> str:
    """Format the problem with German prompt."""
    return "Aufgabe:\n" + doc["problem"] + "\n\nLösung:"


def get_unnormalized_answer(text: str) -> str:
    """Extract answer from model output, handling both German and English patterns.

    Looks for patterns like:
    - German: "Endgültige Antwort: Die endgültige Antwort ist $X$."
    - English: "Final Answer: The final answer is $X$. I hope it is correct."
    - Fallback: Last \\boxed{X} in the text
    """
    INVALID_ANSWER = "[invalidanswer]"

    # Try German pattern first
    german_match = re.search(
        r"(?:Endgültige Antwort|Die endgültige Antwort)[:\s]+(?:Die endgültige Antwort ist\s*)?(.*?)(?:\.|$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if german_match:
        answer = german_match.group(1).strip()
        # Clean up trailing punctuation
        answer = re.sub(r"[.\s]+$", "", answer)
        if answer:
            return answer

    # Try English pattern
    text_with_end = text + " I hope it is correct."
    english_match = re.search(
        r"Final Answer: The final answer is(.*?)\. I hope it is correct\.",
        text_with_end,
    )
    if english_match:
        return english_match.group(1).strip()

    # Fallback: try to extract from \boxed{}
    boxed_match = re.search(r"\\boxed\{([^}]+)\}", text)
    if boxed_match:
        return boxed_match.group(1).strip()

    return INVALID_ANSWER


def _estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Estimate pass@k using the unbiased estimator from the Codex paper.
    1 - comb(n-c, k) / comb(n, k)
    """
    if n < k:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def _extract_answer(text: str) -> str:
    """Extract and normalize answer from a model response.

    Tries multiple extraction strategies:
    1. German/English 'Final Answer' pattern (via get_unnormalized_answer)
    2. Last \\boxed{} expression
    3. Last dollar-sign delimited expression
    """
    # Strategy 1: Minerva "Final Answer" pattern (handles German & English)
    minerva_answer = get_unnormalized_answer(text)
    if minerva_answer != "[invalidanswer]":
        return normalize_final_answer(minerva_answer)

    # Strategy 2: Last boxed expression
    boxed = last_boxed_only_string(text)
    if boxed:
        return normalize_final_answer(remove_boxed(boxed))

    # Strategy 3: Dollar sign delimited
    dollars = [m.start() for m in re.finditer("\\$", text)]
    if len(dollars) > 1:
        answer = text[dollars[-2] + 1 : dollars[-1]]
        return normalize_final_answer(answer)

    return normalize_final_answer(text)


def _check_answer(prediction: str, gold_answer: str) -> bool:
    """Check if a prediction matches the gold answer."""
    pred_answer = _extract_answer(prediction)
    if not pred_answer or not gold_answer:
        return False

    # Try direct string match first
    if pred_answer == gold_answer:
        return True

    # Try sympy equivalence
    return is_equiv(pred_answer, gold_answer)


def process_results(doc: dict, results) -> dict:
    """
    Process Minerva Math results with pass@k and maj@k evaluation.

    Aligned with ENG minerva_math_gen: with repeats=4 and keep_all_responses
    filter, results is [[resp1, resp2, ..., resp4]].
    Returns pass@k for k=1,2,4 and maj@k for k=1,2,4.
    """
    # Unpack: with keep_all_responses filter, results is [list_of_all_responses]
    all_responses = results[0] if results and isinstance(results[0], list) else results

    # Get the gold answer from the doc
    solution = doc.get("solution", "")
    boxed = last_boxed_only_string(solution)
    gold_answer = normalize_final_answer(remove_boxed(boxed)) if boxed else ""

    n = len(all_responses)

    # Extract answers and check correctness for each response
    extracted_answers = []
    correct_flags = []
    for prediction in all_responses:
        is_correct = _check_answer(prediction, gold_answer)
        correct_flags.append(1 if is_correct else 0)
        extracted_answers.append(_extract_answer(prediction))

    total_correct = sum(correct_flags)

    # Compute pass@k
    metrics = {}
    for k in [1, 2, 4]:
        metrics[f"pass_at_{k}"] = _estimate_pass_at_k(n, total_correct, k)

    # Compute maj@k (majority vote at k)
    for k in [1, 2, 4]:
        sample_answers = extracted_answers[:k] if k <= n else extracted_answers
        if sample_answers:
            majority_answer = Counter(sample_answers).most_common(1)[0][0]
            if majority_answer and gold_answer:
                if majority_answer == gold_answer or is_equiv(majority_answer, gold_answer):
                    metrics[f"maj_at_{k}"] = 1.0
                else:
                    metrics[f"maj_at_{k}"] = 0.0
            else:
                metrics[f"maj_at_{k}"] = 0.0
        else:
            metrics[f"maj_at_{k}"] = 0.0

    # Also include exact_match for backward compatibility
    metrics["exact_match"] = total_correct / max(n, 1)

    return metrics


def keep_all_responses(resps, docs):
    """Custom filter that preserves all responses (for pass@k / maj@k with repeats)."""
    return resps


def list_fewshot_samples() -> list[dict]:
    """German few-shot examples for Minerva Math evaluation (localized).
    
    Note: Uses 'solution' field for doc_to_target in few-shot context,
    showing the model the expected output format.
    """
    return [
        {
            "problem": "Bestimme den Definitionsbereich des Ausdrucks $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.",
            "solution": "Die Ausdrücke unter jeder Quadratwurzel müssen nicht-negativ sein. Daher gilt $x-2 \\ge 0$, also $x \\ge 2$, und $5 - x \\ge 0$, also $x \\le 5$. Außerdem darf der Nenner nicht gleich Null sein, also $5-x>0$, was $x<5$ ergibt. Daher ist der Definitionsbereich des Ausdrucks $\\boxed{[2,5)}$.\nEndgültige Antwort: Die endgültige Antwort ist $[2,5)$.",
        },
        {
            "problem": "Wenn $\\det \\mathbf{A} = 2$ und $\\det \\mathbf{B} = 12$ ist, bestimme $\\det (\\mathbf{A} \\mathbf{B})$.",
            "solution": "Es gilt $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}$.\nEndgültige Antwort: Die endgültige Antwort ist $24$.",
        },
        {
            "problem": "Tobias hebt beim Training normalerweise zwei 10-Kilogramm-Hanteln jeweils 12 Mal. Wenn er stattdessen zwei 7,5-Kilogramm-Hanteln verwendet, wie oft muss Tobias sie heben, um das gleiche Gesamtgewicht zu bewegen?",
            "solution": "Wenn Tobias zwei 10-Kilogramm-Hanteln 12 Mal hebt, bewegt er insgesamt $2\\cdot 12\\cdot 10=240$ Kilogramm. Wenn er stattdessen zwei 7,5-Kilogramm-Hanteln $n$ Mal hebt, bewegt er insgesamt $2\\cdot 7{,}5\\cdot n=15n$ Kilogramm. Setzen wir dies gleich 240 Kilogramm, können wir nach $n$ auflösen:\n\\begin{align*}\n15n&=240\\\\\n\\Rightarrow\\qquad n&=240/15=\\boxed{16}\n\\end{align*}\nEndgültige Antwort: Die endgültige Antwort ist $16$.",
        },
        {
            "problem": "Wenn das Gleichungssystem\n\n\\begin{align*}\n6x-4y&=a,\\\\\n6y-9x &=b\n\\end{align*}eine Lösung $(x, y)$ hat, bei der $x$ und $y$ beide ungleich Null sind, bestimme $\\frac{a}{b}$, unter der Annahme, dass $b$ ungleich Null ist.",
            "solution": "Wenn wir die erste Gleichung mit $-\\frac{3}{2}$ multiplizieren, erhalten wir\n\n$$6y-9x=-\\frac{3}{2}a.$$Da wir auch wissen, dass $6y-9x=b$ ist, haben wir\n\n$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$\nEndgültige Antwort: Die endgültige Antwort ist $-\\frac{2}{3}$.",
        },
    ]


def process_results_bpb(doc, results):
    """Compute answer-only BPB (OLMES-style) for Minerva Math."""
    ll, _ = results[0]
    gold_text = doc["solution"]
    # Leading space is part of the scored target (matches doc_to_target " {{solution}}")
    gold_bytes = len((" " + gold_text).encode("utf-8"))
    return {"bits_per_byte": -ll / (math.log(2) * max(gold_bytes, 1))}
