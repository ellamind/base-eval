"""Minerva Math utilities for English evaluation."""

import math
import re
from collections import Counter

import numpy as np
from typing import List, Dict

# Import is_equiv from lm-eval (has 5-second signal.alarm timeout on sympy.simplify)
# to match OLMES scoring behavior and prevent hangs on pathological LaTeX.
from lm_eval.tasks.minerva_math.utils import is_equiv


def _last_boxed_only_string(text: str) -> str:
    """Extract the last \\boxed{...} expression from text, handling nested braces."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return ""

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(text):
        if text[i] == "{":
            num_left_braces_open += 1
        if text[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return ""
    return text[idx : right_brace_idx + 1]


def _remove_boxed(s: str) -> str:
    """Remove \\boxed{...} wrapper."""
    if "\\boxed " in s:
        left = "\\boxed "
        return s[len(left):]
    left = "\\boxed{"
    if s.startswith(left) and s.endswith("}"):
        return s[len(left):-1]
    return s


def get_unnormalized_answer(text: str) -> str:
    """Extract answer using the Minerva 'Final Answer' pattern."""
    INVALID_ANSWER = "[invalidanswer]"
    end_seq = "I hope it is correct."
    text += end_seq
    match = re.search(
        r"Final Answer: The final answer is(.*?). I hope it is correct.",
        text,
    )
    if match:
        return match.group(1).strip()
    return INVALID_ANSWER


SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]
REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """
    Normalize a final answer to a quantitative reasoning question.
    Copied from lm_eval/tasks/minerva_math/utils.py (appendix D of Lewkowycz et al. 2022).
    """
    final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract answer that is in LaTeX math, is bold, is surrounded by a box, etc.
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize 100,000 -> 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer




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
    """Extract and normalize answer from a model response, matching OLMES behavior.

    Tries multiple extraction strategies:
    1. Minerva 'Final Answer' pattern
    2. Last \\boxed{} expression
    3. Last dollar-sign delimited expression
    """
    # Strategy 1: Minerva "Final Answer" pattern
    minerva_answer = get_unnormalized_answer(text)
    if minerva_answer != "[invalidanswer]":
        return normalize_final_answer(minerva_answer)

    # Strategy 2: Last boxed expression
    boxed = _last_boxed_only_string(text)
    if boxed:
        return normalize_final_answer(_remove_boxed(boxed))

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
    Process Minerva Math results with pass@k and maj@k evaluation (matching OLMES behavior).

    With repeats=N and keep_all_responses filter, results is [[resp1, resp2, ..., respN]].
    Returns pass@k for k=1,2,4 and maj@k for k=1,2,4.
    """
    # Unpack: with keep_all_responses filter, results is [list_of_all_responses]
    # With default take_first filter, results is [single_string]
    all_responses = results[0] if results and isinstance(results[0], list) else results

    # Get the gold answer from the doc
    solution = doc.get("solution", "")
    boxed = _last_boxed_only_string(solution)
    gold_answer = normalize_final_answer(_remove_boxed(boxed)) if boxed else ""

    n = len(all_responses)

    # Extract answers and check correctness for each response
    extracted_answers = []
    correct_flags = []
    for prediction in all_responses:
        is_correct = _check_answer(prediction, gold_answer)
        correct_flags.append(1 if is_correct else 0)
        extracted_answers.append(_extract_answer(prediction))

    total_correct = sum(correct_flags)

    metrics = _compute_pass_and_maj(n, total_correct, extracted_answers, gold_answer, ks=[1, 4])
    metrics["exact_match"] = total_correct / max(n, 1)
    return metrics


def process_results_n32(doc: dict, results) -> dict:
    """Process results for n=32 sampling (MATH-500: pass@1,16 and maj@1,16)."""
    all_responses = results[0] if results and isinstance(results[0], list) else results

    solution = doc.get("solution", "")
    boxed = _last_boxed_only_string(solution)
    gold_answer = normalize_final_answer(_remove_boxed(boxed)) if boxed else ""

    n = len(all_responses)
    extracted_answers = []
    correct_flags = []
    for prediction in all_responses:
        is_correct = _check_answer(prediction, gold_answer)
        correct_flags.append(1 if is_correct else 0)
        extracted_answers.append(_extract_answer(prediction))

    total_correct = sum(correct_flags)

    metrics = _compute_pass_and_maj(n, total_correct, extracted_answers, gold_answer, ks=[1, 16])
    metrics["exact_match"] = total_correct / max(n, 1)
    return metrics


def _compute_pass_and_maj(n, total_correct, extracted_answers, gold_answer, ks):
    """Compute pass@k and maj@k for given k values."""
    metrics = {}
    for k in ks:
        metrics[f"pass_at_{k}"] = _estimate_pass_at_k(n, total_correct, k)

    for k in ks:
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

    return metrics


def get_minerva_math_fewshot() -> List[Dict]:
    """4-shot examples for Minerva Math (matching OLMES Minerva:MATH:fixed)."""
    return [
        {
            "problem": "Find the domain of the expression  $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.}",
            "solution": "The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{[2,5)}$.\nFinal Answer: The final answer is $[2,5)$. I hope it is correct.",
        },
        {
            "problem": "If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$",
            "solution": "We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$\nFinal Answer: The final answer is $24$. I hope it is correct.",
        },
        {
            "problem": "Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?",
            "solution": "If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$:\n\\begin{align*}\n30n&=480\\\n\\Rightarrow\\qquad n&=480/30=\\boxed{16}\n\\end{align*}\nFinal Answer: The final answer is $16$. I hope it is correct.",
        },
        {
            "problem": "If the system of equations\n\\begin{align*}\n6x-4y&=a,\\\n6y-9x &=b.\n\\end{align*}\nhas a solution $(x, y)$ where $x$ and $y$ are both nonzero, find $\\frac{a}{b},$ assuming $b$ is nonzero.",
            "solution": "If we multiply the first equation by $-\\frac{3}{2}$, we obtain $$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have $$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$\nFinal Answer: The final answer is $-\\frac{2}{3}$. I hope it is correct.",
        }
    ]


def get_minerva_math_fewshot_bpb() -> List[Dict]:
    """4-shot examples for Minerva Math BPB evaluation (original format)."""
    return [
        {
            "problem": "Find the domain of the expression $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.}",
            "solution": "The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{[2,5)}$. Final Answer: The final answer is $[2,5)$. I hope it is correct.",
            "answer": "[2,5)",
            "subject": "Algebra",
            "level": "Level 5",
            "unique_id": "train/algebra/25722.json"
        },
        {
            "problem": "If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$",
            "solution": "We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$ Final Answer: The final answer is $24$. I hope it is correct.",
            "answer": "24",
            "subject": "Precalculus",
            "level": "Level 1",
            "unique_id": "train/precalculus/263.json"
        },
        {
            "problem": "Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?",
            "solution": "If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight. If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight. Equating this to 480 pounds, we can solve for $n$: \\begin{align*} 30n&=480\\\\ \\Rightarrow\\qquad n&=480/30=\\boxed{16} \\end{align*} Final Answer: The final answer is $16$. I hope it is correct.",
            "answer": "16",
            "subject": "Algebra",
            "level": "Level 1",
            "unique_id": "train/algebra/1152.json"
        },
        {
            "problem": "If the system of equations \\begin{align*} 6x-4y&=a,\\\\ 6y-9x &=b. \\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero, find $\\frac{a}{b},$ assuming $b$ is nonzero.",
            "solution": "If we multiply the first equation by $-\\frac{3}{2}$, we obtain $$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have $$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$ Final Answer: The final answer is $-\\frac{2}{3}$. I hope it is correct.",
            "answer": "-\\frac{2}{3}",
            "subject": "Algebra",
            "level": "Level 5",
            "unique_id": "train/algebra/1594.json"
        }
    ]


def keep_all_responses(resps, docs):
    """Custom filter that preserves all responses (for pass@k / maj@k with repeats)."""
    return resps


def process_results_bpb(doc, results):
    """Compute answer-only BPB (OLMES-style) for Minerva Math."""
    ll, _ = results[0]
    gold_text = doc["solution"]
    # Leading space is part of the scored target (matches doc_to_target " {{solution}}")
    gold_bytes = len((" " + gold_text).encode("utf-8"))
    return {"bits_per_byte": -ll / (math.log(2) * max(gold_bytes, 1))}
