"""DeepMind Math utilities for English evaluation.

Dataset: ellamind/deepmind-math-sample (56 categories, 100 per category)
Single parquet with 'category' column. Pre-sampled with OLMES seed=1234.
Answer checking uses SymPy for mathematical equivalence (mirrors OLMES).
"""

import re
from functools import partial

import numpy as np
import sympy
from sympy.parsing.sympy_parser import TokenError


def _clean_bytes(s):
    """Clean byte-string artifacts from deepmind/math_dataset.

    The HF dataset stores strings as b'...\\n' — strip the wrapping.
    """
    if isinstance(s, bytes):
        return s.decode("utf-8").strip()
    if isinstance(s, str) and s.startswith("b'") and s.endswith("\\n'"):
        return s[2:-3].strip()
    if isinstance(s, str):
        return s.strip()
    return str(s).strip()


def doc_to_text(doc):
    """Build prompt for a math question."""
    return "Problem:\n" + _clean_bytes(doc["question"]) + "\n\nAnswer:"


def doc_to_target(doc):
    """Return the gold answer."""
    return " " + _clean_bytes(doc["answer"])


def _clean_prediction(prediction):
    """Clean model prediction for comparison."""
    res = re.sub(r"\s*\.?\s*I hope it is correct.*", "", prediction)
    for token in ["<end_of_turn>", "<|eot_id|>", "<s>", "</s>"]:
        res = res.replace(token, "").strip()
    res = re.sub(r"\.\s*$", "", res).strip()
    for left, right in [("$", "$"), ("\\(", "\\)"), ("**", "**"), ("***", "***"), ("\\[", "\\]")]:
        left_re = re.escape(left)
        right_re = re.escape(right)
        res = re.sub(f"^{left_re}(.*){right_re}$", r"\1", res).strip()
    return res


def _extract_answer(text):
    """Extract answer from CoT response.

    Looks for 'the final answer is ...' or 'answer is ...' patterns.
    Falls back to the full text.
    """
    for prefix in ["the final answer is", "answer is"]:
        pattern = re.compile(rf"(?i){prefix}\s*(.*)", re.DOTALL)
        m = pattern.search(text)
        if m:
            answer = m.group(1).strip()
            if answer.endswith("."):
                answer = answer[:-1].strip()
            return answer
    return text.strip()


def _check_sympy_equal(expr1, expr2):
    """Check if two SymPy expressions are equal using numerical evaluation."""
    if not isinstance(expr1, sympy.Expr) or not isinstance(expr2, sympy.Expr):
        return expr1 == expr2
    if expr1 is None or expr2 is None:
        return False
    if expr1.free_symbols != expr2.free_symbols:
        return False
    free_vars = expr1.free_symbols
    values = np.random.random(len(free_vars))
    e1 = expr1
    e2 = expr2
    for symbol, number in zip(free_vars, values):
        e1 = e1.subs(symbol, sympy.Float(number))
        e2 = e2.subs(symbol, sympy.Float(number))
    e1 = float(e1)
    e2 = float(e2)
    if not np.allclose(e1, e2):
        return False
    return bool(expr1.equals(expr2))


def process_results(doc, results):
    """Process results with SymPy-based answer checking (mirrors OLMES)."""
    raw = results[0]
    prediction = _extract_answer(raw)
    prediction = _clean_prediction(prediction)
    answer = _clean_bytes(doc["answer"])

    if answer.lower() == prediction.lower():
        acc = 1
    elif answer.lower() in ["true", "false"]:
        pred_lower = prediction.lower()
        ans_lower = answer.lower()
        acc_str = ans_lower == pred_lower
        acc_yn = (pred_lower == "yes" and ans_lower == "true") or (
            pred_lower == "no" and ans_lower == "false"
        )
        acc = int(acc_str or acc_yn)
    else:
        try:
            answer_expr = sympy.parse_expr(answer)
            prediction_expr = sympy.parse_expr(prediction)
            acc = 1 if _check_sympy_equal(answer_expr, prediction_expr) else 0
        except (TypeError, SyntaxError, TokenError):
            acc = 0
        except Exception:
            acc = 0

    return {"exact_match": acc}


def _filter_category(dataset, category):
    """Filter dataset to a single category."""
    indices = [i for i, doc in enumerate(dataset) if doc["category"] == category]
    return dataset.select(indices)


# 56 category filter functions — one per deepmind math category
CATEGORIES = [
    "algebra__linear_1d", "algebra__linear_1d_composed", "algebra__linear_2d",
    "algebra__linear_2d_composed", "algebra__polynomial_roots",
    "algebra__polynomial_roots_composed", "algebra__sequence_next_term",
    "algebra__sequence_nth_term", "arithmetic__add_or_sub",
    "arithmetic__add_or_sub_in_base", "arithmetic__add_sub_multiple",
    "arithmetic__div", "arithmetic__mixed", "arithmetic__mul",
    "arithmetic__mul_div_multiple", "arithmetic__nearest_integer_root",
    "arithmetic__simplify_surd", "calculus__differentiate",
    "calculus__differentiate_composed", "comparison__closest",
    "comparison__closest_composed", "comparison__kth_biggest",
    "comparison__kth_biggest_composed", "comparison__pair",
    "comparison__pair_composed", "comparison__sort", "comparison__sort_composed",
    "measurement__conversion", "measurement__time", "numbers__base_conversion",
    "numbers__div_remainder", "numbers__div_remainder_composed", "numbers__gcd",
    "numbers__gcd_composed", "numbers__is_factor", "numbers__is_factor_composed",
    "numbers__is_prime", "numbers__is_prime_composed", "numbers__lcm",
    "numbers__lcm_composed", "numbers__list_prime_factors",
    "numbers__list_prime_factors_composed", "numbers__place_value",
    "numbers__place_value_composed", "numbers__round_number",
    "numbers__round_number_composed", "polynomials__add",
    "polynomials__coefficient_named", "polynomials__collect",
    "polynomials__compose", "polynomials__evaluate",
    "polynomials__evaluate_composed", "polynomials__expand",
    "polynomials__simplify_power", "probability__swr_p_level_set",
    "probability__swr_p_sequence",
]


def _make_filter(cat):
    return partial(_filter_category, category=cat)


# Generate filter_<category> for each category
for _cat in CATEGORIES:
    globals()[f"filter_{_cat}"] = _make_filter(_cat)
