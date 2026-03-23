"""MBPP utilities for German evaluation.

Aligned with OLMES mbpp:3shot::none (inloop_bpb) configuration for base models.
Clean prompt: plain German text description + ```python code block.
"""

import math
import re
from typing import List, Dict

_code_eval = None


def _get_code_eval():
    """Lazy-load the code_eval metric (requires HF_ALLOW_CODE_EVAL=1)."""
    global _code_eval
    if _code_eval is None:
        import evaluate as hf_evaluate
        _code_eval = hf_evaluate.load("code_eval")
    return _code_eval


def pass_at_k(
    references: list[str], predictions: list[list[str]], k: list[int] = None
):
    """Compute pass@k metrics using code execution.

    Uses the unbiased estimator from the Codex paper:
        pass@k = 1 - comb(n-c, k) / comb(n, k)

    This matches the OLMES CodePassAtK metric computation.

    Args:
        references: List of test case strings (one per problem)
        predictions: List of lists of code completions (multiple samples per problem)
        k: List of k values to compute pass@k for (e.g., [1, 2, 4, 8, 16])

    Returns:
        Dict with pass@k scores (e.g., {"pass@1": 0.5, "pass@2": 0.6, ...})
    """

    if isinstance(k, int):
        k = [k]
    res = _get_code_eval().compute(
        references=references,
        predictions=predictions,
        k=k,
        num_workers=4,
        timeout=10.0,
    )
    return res[0]


def _split_code(doc):
    """Split gold code into (preamble, body) at the main function's def line.

    Extracts the tested function name from the first assert, then finds
    that function's def line in the code. Returns:
      - preamble: everything up to and including 'def func_name(...):'
      - body: everything after that colon

    Handles imports, constants, and helper functions before the main def.
    """
    code = doc["code"].replace("\r", "")
    test_case = doc["test_list"][0]

    # Extract function name from assert: "assert func_name(...)" or "assert set(func_name(...))"
    m = re.search(r"assert\s+(?:set\()?\s*(\w+)\s*\(", test_case)
    if m:
        func_name = m.group(1)
        # Find "def func_name(" in code
        pattern = re.compile(rf"^(\s*def\s+{re.escape(func_name)}\s*\(.*?\)\s*:)", re.MULTILINE)
        match = pattern.search(code)
        if match:
            end = match.end()
            return code[:end], code[end:]

    # Fallback: split at first "def ...(...):"
    m = re.search(r"^(\s*def\s+\w+\s*\(.*?\)\s*:)", code, re.MULTILINE)
    if m:
        end = m.end()
        return code[:end], code[end:]

    # Last resort: split at first colon
    idx = code.index(":")
    return code[:idx + 1], code[idx + 1:]


def doc_to_text(doc):
    """Description + first test case + ```python + code preamble.

    Prefills everything up to the main function signature so the model
    only generates the function body. Handles imports, constants, and
    helper functions that appear before the main def.
    """
    text = doc.get("text", doc.get("prompt", "")).strip()
    test_case = doc["test_list"][0]
    preamble, _ = _split_code(doc)
    return text + "\n" + test_case + "\n```python\n" + preamble


def doc_to_target(doc):
    """Return the test cases for execution (eval examples)."""
    return "\n".join(doc["test_list"])


def doc_to_target_fewshot(doc):
    """Return the function body (after main def signature) + closing ```."""
    _, body = _split_code(doc)
    return body.rstrip() + "\n```"


def get_mbpp_fewshot() -> List[Dict]:
    """All 7 examples from the MBPP 'prompt' split (Original:MBPP source).

    These match the OLMES fewshot_source 'Original:MBPP'. The framework's
    default random sampler will pick 3 from these 7 using the fewshot seed.
    German text, proper multi-line code (matching the actual dataset format).
    """
    examples = [
        {
            "text": "Schreibe eine Funktion, um die gemeinsamen Elemente aus den gegebenen zwei Listen zu finden.",
            "prompt": "Schreibe eine Funktion, um die gemeinsamen Elemente aus den gegebenen zwei Listen zu finden.",
            "code": "def similar_elements(test_tup1, test_tup2):\n  res = tuple(set(test_tup1) & set(test_tup2))\n  return (res) ",
            "test_list": [
                "assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))",
                "assert set(similar_elements((1, 2, 3, 4),(5, 4, 3, 7))) == set((3, 4))",
                "assert set(similar_elements((11, 12, 14, 13),(17, 15, 14, 13))) == set((13, 14))",
            ],
        },
        {
            "text": "Schreibe eine Python-Funktion, um Nicht-Primzahlen zu identifizieren.",
            "prompt": "Schreibe eine Python-Funktion, um Nicht-Primzahlen zu identifizieren.",
            "code": "import math\ndef is_not_prime(n):\n    result = False\n    for i in range(2,int(math.sqrt(n)) + 1):\n        if n % i == 0:\n            result = True\n    return result",
            "test_list": [
                "assert is_not_prime(2) == False",
                "assert is_not_prime(10) == True",
                "assert is_not_prime(35) == True",
            ],
        },
        {
            "text": "Schreibe eine Funktion, um die n größten Ganzzahlen aus einer gegebenen Zahlenliste in absteigender Reihenfolge zurückzugeben.",
            "prompt": "Schreibe eine Funktion, um die n größten Ganzzahlen aus einer gegebenen Zahlenliste in absteigender Reihenfolge zurückzugeben.",
            "code": "import heapq as hq\ndef heap_queue_largest(nums,n):\n  largest_nums = hq.nlargest(n, nums)\n  return largest_nums",
            "test_list": [
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65]",
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75]",
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]",
            ],
        },
        {
            "text": "Schreibe eine Funktion, um die Anzahl der Möglichkeiten zu finden, ein 3 x n Brett mit 2 x 1 Dominosteinen zu füllen.",
            "prompt": "Schreibe eine Funktion, um die Anzahl der Möglichkeiten zu finden, ein 3 x n Brett mit 2 x 1 Dominosteinen zu füllen.",
            "code": "def count_ways(n):\n    A = [0] * (n + 1)\n    B = [0] * (n + 1)\n    A[0] = 1\n    A[1] = 0\n    B[0] = 0\n    B[1] = 1\n    for i in range(2, n+1):\n        A[i] = A[i - 2] + 2 * B[i - 1]\n        B[i] = A[i - 1] + B[i - 2]\n    return A[n]",
            "test_list": [
                "assert count_ways(2) == 3",
                "assert count_ways(8) == 153",
                "assert count_ways(12) == 2131",
            ],
        },
        {
            "text": "Schreibe eine Python-Funktion, um zu prüfen, ob sich die zwei Zahlen nur an einer Bitposition unterscheiden oder nicht.",
            "prompt": "Schreibe eine Python-Funktion, um zu prüfen, ob sich die zwei Zahlen nur an einer Bitposition unterscheiden oder nicht.",
            "code": "def is_Power_Of_Two (x):\n    return x and (not(x & (x - 1)))\ndef differ_At_One_Bit_Pos(a,b):\n    return is_Power_Of_Two(a ^ b)",
            "test_list": [
                "assert differ_At_One_Bit_Pos(13,9) == True",
                "assert differ_At_One_Bit_Pos(15,8) == False",
                "assert differ_At_One_Bit_Pos(2,4) == False",
            ],
        },
        {
            "text": "Schreibe eine Funktion, um mit Regex alle Wörter zu finden, die in einem String mindestens 4 Zeichen lang sind.",
            "prompt": "Schreibe eine Funktion, um mit Regex alle Wörter zu finden, die in einem String mindestens 4 Zeichen lang sind.",
            "code": "import re\ndef find_char_long(text):\n    return (re.findall(r'\\b\\w{4,}\\b', text))",
            "test_list": [
                "assert find_char_long('Please move back to stream') == ['Please', 'move', 'back', 'stream']",
                "assert find_char_long('Jing Eco and Tech') == ['Jing', 'Tech']",
                "assert find_char_long('Jhingai wulu road Zone 3') == ['Jhingai', 'wulu', 'road', 'Zone']",
            ],
        },
        {
            "text": "Schreibe eine Funktion, um mithilfe einer Lambda-Funktion die Quadrate einzelner Elemente in einer Liste zu finden.",
            "prompt": "Schreibe eine Funktion, um mithilfe einer Lambda-Funktion die Quadrate einzelner Elemente in einer Liste zu finden.",
            "code": "def square_nums(nums):\n    square_nums = list(map(lambda x: x ** 2, nums))\n    return square_nums",
            "test_list": [
                "assert square_nums([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])==[1, 4, 9, 16, 25, 36, 49, 64, 81, 100]",
                "assert square_nums([10,20,30])==([100,400,900])",
                "assert square_nums([12,15])==([144,225])",
            ],
        },
    ]
    return examples

def prepend_preamble(resps, docs):
    """Prepend the code preamble to each model response.

    The model only generates the function body. We reconstruct the full
    code (imports + helpers + main function signature + body) for execution.
    """
    results = []
    for doc_resps, doc in zip(resps, docs):
        preamble, _ = _split_code(doc)
        results.append([preamble + "\n" + r for r in doc_resps])
    return results


def strip_cr(dataset):
    """Strip \\r from code field for consistent line endings."""
    return dataset.map(lambda doc: {**doc, "code": doc["code"].replace("\r", "")})


def doc_to_text_bpb(doc):
    """OLMES-style context: description + function signature.

    Example:
      Schreibe eine Funktion, um die ähnlichen Elemente...
      def similar_elements(test_tup1, test_tup2):
    """
    text = doc.get("text", doc.get("prompt", "")).strip()
    code = doc["code"]
    func_sig = code.split(":")[0] + ":"
    return text + "\n" + func_sig


def doc_to_target_bpb(doc):
    """Code body after function signature (no overlap with context)."""
    code = doc["code"]
    _, _, body = code.partition(":")
    return body


def process_results_bpb(doc, results):
    """Compute answer-only BPB (OLMES-style) for MBPP."""
    ll, _ = results[0]
    # Target is the code body after the function signature
    _, _, body = doc["code"].partition(":")
    gold_bytes = len(body.encode("utf-8"))
    return {"bits_per_byte": -ll / (math.log(2) * max(gold_bytes, 1))}
