"""GSM8K utilities for English evaluation."""

import re
from typing import List, Dict
import numpy as np
import string
from collections import Counter


def extract_gsm8k_answer(text: str) -> str:
    """
    Extract final answer from GSM8K-style response.

    GSM8K answers end with "#### <number>" pattern.
    """
    # Look for #### pattern
    match = re.search(r'####\s*(-?[\d,]+(?:\.\d+)?)', text)
    if match:
        answer = match.group(1).replace(',', '')
        return answer

    # Fallback: try to find last number in text
    numbers = re.findall(r'-?[\d,]+(?:\.\d+)?', text)
    if numbers:
        return numbers[-1].replace(',', '')

    return ""


def _normalize_number(s: str) -> str:
    """Normalize a number string for comparison."""
    s = s.replace(',', '').strip()
    # Handle decimals - try to convert to float and back to clean format
    try:
        val = float(s)
        # Return integer format if it's a whole number
        if val == int(val):
            return str(int(val))
        return str(val)
    except ValueError:
        return s
    

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


def process_results(doc: dict, results: List[str]) -> dict:
    """
    Process GSM8K results with lenient answer extraction (matching OLMES behavior).

    OLMES extracts answers in this priority:
    1. Look for #### <number> pattern
    2. Fallback to last number in the response
    """
    prediction = results[0] if results else ""

    # Get the gold answer from the doc
    answer_text = doc.get("answer", "")
    # GSM8K target is after ####
    gold_match = re.search(r'####\s*(-?[\d,]+(?:\.\d+)?)', answer_text)
    gold = gold_match.group(1).replace(',', '') if gold_match else ""

    # Extract predicted answer with fallback
    pred = extract_gsm8k_answer(prediction)

    # Normalize both for comparison
    gold_norm = _normalize_number(gold)
    pred_norm = _normalize_number(pred)

    # Exact match after normalization
    em = 1.0 if gold_norm == pred_norm and gold_norm != "" else 0.0

    return {"exact_match": em}



# ---------------------------------------------------------------------------
# Shared helpers — ported from OLMES
# ---------------------------------------------------------------------------

def _clean_short_answer(continuation: str) -> str:
    """
    Mirrors GSM8K._clean_short_answer (gsm8k.py:268-277).
    Strips commas between digits and returns the last number.
    """
    output = re.sub(r"(\d),(\d)", r"\1\2", continuation)
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)
    if numbers:
        return numbers[-1]
    else:
        return output


def _extract_answer(continuation: str) -> str:
    """
    Mirrors GSM8K._extract_answer (gsm8k.py:232-266) for the code path
    where answer_regexes is NOT set (which is the case for gsm8k::olmo3:n8:v2).
    Strips commas between digits and returns the last number.
    """
    # Replace commas between digits  (gsm8k.py:238)
    output = re.sub(r"(\d),(\d)", r"\1\2", continuation)
    # No answer_regexes → take the else branch  (gsm8k.py:262-266)
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)
    if numbers:
        return numbers[-1]
    else:
        return output


def _exact_match_hf(prediction: str, reference: str,
                    regexes_to_ignore=None, ignore_case=False,
                    ignore_punctuation=False) -> float:
    """
    Mirrors exact_match_hf_evaluate (exact_match.py:25-57) for a single
    prediction/reference pair.
    """
    predictions = [prediction]
    references = [reference]
    if regexes_to_ignore is not None:
        for s in regexes_to_ignore:
            predictions = [re.sub(s, "", x) for x in predictions]
            references = [re.sub(s, "", x) for x in references]
    if ignore_case:
        predictions = [x.lower() for x in predictions]
        references = [x.lower() for x in references]
    if ignore_punctuation:
        table = str.maketrans("", "", string.punctuation)
        predictions = [x.translate(table) for x in predictions]
        references = [x.translate(table) for x in references]
    return 1.0 if predictions[0] == references[0] else 0.0


def _estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Mirrors estimate_pass_at_k_1doc (metric_utils.py:37-49).
    Calculates 1 - comb(n-c, k) / comb(n, k).
    """
    if n < k:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


# OLMES regexes_to_ignore for gsm8k::olmo3:n8:v2  (from the task config)
_REGEXES_TO_IGNORE = [",", "\\$", "(?s).*#### ", "\\.$"]


def _gold_answer(doc: dict) -> str:
    """
    Mirrors _process_doc gold extraction (gsm8k.py:165-172):
      short_answer = doc["answer"].split("####")[-1].strip()
      cleaned_short_answer = _clean_short_answer(short_answer)
    """
    short_answer = doc["answer"].split("####")[-1].strip()
    return _clean_short_answer(short_answer)



# ---------------------------------------------------------------------------
# Final process results for gen
# ---------------------------------------------------------------------------

def keep_all_responses(resps, docs):
    """Custom filter that preserves all responses (for pass@k / maj@k with repeats)."""
    return resps

def process_results_gen(doc: dict, results: List[str]) -> dict:
    """
    Score GSM8K generations following the exact OLMES pipeline for
    gsm8k::olmo3:n8:v2.

    Per-repeat flow (mirrors OLMES):
      1. _extract_answer(continuation)        — gsm8k.py:232-266
         strips commas between digits, returns last number.
      2. ExactMatch.process_one_doc            — metric.py:536-550
         calls exact_match_hf_evaluate(pred, label,
               regexes_to_ignore=[",","\\$","(?s).*#### ","\\.$"],
               ignore_case=True)
      3. PassAtK.process_one_doc               — metric.py:680-714
         counts correct, calls estimate_pass_at_k_1doc(n, c, k).
      4. MajAtK.process_one_doc                — metric.py:605-645
         majority vote over first k extracted answers, then
         exact_match_hf_evaluate against label.

    label = _clean_short_answer(doc["answer"].split("####")[-1].strip())
    """
    # Gold label — same as gsm8k.py:168-172
    label = _gold_answer(doc)

    # Unpack: with keep_all_responses filter + repeats, results is [[resp1, ..., respN]]
    all_responses = results[0] if results and isinstance(results[0], list) else results

    n = len(all_responses)
    # Step 1: extract answers from all repeats  (gsm8k.py:238,262-264)
    model_answers = [_extract_answer(cont) for cont in all_responses]

    # Step 2: per-repeat exact match  (metric.py:544-550)
    correct = 0
    for model_answer in model_answers:
        em = _exact_match_hf(
            prediction=model_answer,
            reference=label,
            regexes_to_ignore=_REGEXES_TO_IGNORE,
            ignore_case=True,
            ignore_punctuation=False,
        )
        correct += int(em)

    metrics = {}

    # Step 3: pass@k (OLMES: pass@1, pass@4)
    for k in [1, 4]:
        metrics[f"pass_at_{k}"] = _estimate_pass_at_k(n, correct, k)

    # Step 5: exact_match — OLMES ExactMatch.process_one_doc (metric.py:536-550)
    # uses ONLY the first repeat (group_lst[0]), not an average across all repeats.
    metrics["exact_match"] = _exact_match_hf(
        prediction=model_answers[0] if model_answers else "",
        reference=label,
        regexes_to_ignore=_REGEXES_TO_IGNORE,
        ignore_case=True,
        ignore_punctuation=False,
    )

    return metrics




def get_gsm8k_fewshot() -> List[Dict]:
    """8-shot CoT examples for GSM8K."""
    return [
        {
            "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
            "answer": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. #### 6"
        },
        {
            "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
            "answer": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. #### 5"
        },
        {
            "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
            "answer": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. #### 39"
        },
        {
            "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
            "answer": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. #### 8"
        },
        {
            "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
            "answer": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. #### 9"
        },
        {
            "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
            "answer": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 = 29. #### 29"
        },
        {
            "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
            "answer": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. #### 33"
        },
        {
            "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
            "answer": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 = 8 dollars left. #### 8"
        },
    ]



def get_gsm8k_gen_fewshot() -> List[Dict]:
    """
    8-shot CoT examples for GSM8K generative task (OLMES STD:GSM8k format).

    These match the OLMES fewshot_source "STD:GSM8k" which uses full chain-of-thought
    reasoning ending with "So the answer is X." instead of "#### X".
    """
    return [
        {
            "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
            "answer": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. So the answer is 6.",
        },
        {
            "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
            "answer": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. So the answer is 5.",
        },
        {
            "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
            "answer": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. So the answer is 39.",
        },
        {
            "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
            "answer": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. So the answer is 8.",
        },
        {
            "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
            "answer": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. So the answer is 9.",
        },
        {
            "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
            "answer": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. So the answer is 29.",
        },
        {
            "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
            "answer": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. So the answer is 33.",
        },
        {
            "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
            "answer": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. So the answer is 8.",
        },
    ]