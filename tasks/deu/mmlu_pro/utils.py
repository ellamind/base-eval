"""Utility functions for German MMLU-ProX tasks (MC, RC, CoT).

Dataset: li-lab/MMLU-ProX (de config)
- 11,759 test questions across 14 categories
- 70 validation examples (5 per category) with cot_content for fewshot
- Options stored as individual option_0..option_9 fields (None-padded)
- 5-shot chain-of-thought, generate_until output, exact_match metric
"""

import re
from functools import partial

LABELS = "ABCDEFGHIJ"

COT_PREFIX_RAW = "Denken wir Schritt für Schritt nach."
COT_PREFIX_CLEAN = "Lass uns Schritt für Schritt vorgehen. "


def _get_options(doc):
    """Extract valid options from individual option_N fields."""
    opts = []
    for i in range(10):
        val = doc[f"option_{i}"]
        if val is not None and val != "N/A":
            opts.append(val)
    return opts


def _format_choices(options):
    """Format options as ' A. ... \\n B. ...' matching _make_mcq_prompt convention."""
    return "\n".join(f" {LABELS[i]}. {opt}" for i, opt in enumerate(options))


def _process_cot(cot):
    """Clean up raw cot_content from the dataset.

    Fixes applied:
    1. Remove leading "A: " prefix
    2. Replace awkward "Denken wir Schritt für Schritt nach." with
       "Lass uns Schritt für Schritt vorgehen. " (trailing space)
    3. Normalize whitespace before "Die Antwort ist" (ensure single space)
    """
    cot = cot.removeprefix("A: ")
    cot = cot.replace(COT_PREFIX_RAW, COT_PREFIX_CLEAN)
    cot = re.sub(r"\s*Die Antwort ist", " Die Antwort ist", cot)
    return cot.strip()


def doc_to_text(doc):
    """Build COT prompt for test questions."""
    choices_text = _format_choices(_get_options(doc))
    return (
        f"Frage: {doc['question']}\n"
        f"{choices_text}\n"
        f"Antwort: Lass uns Schritt für Schritt vorgehen."
    )


def fewshot_to_text(doc):
    """Build fewshot example with processed CoT reasoning."""
    choices_text = _format_choices(_get_options(doc))
    cot = _process_cot(doc["cot_content"])
    return f"Frage: {doc['question']}\n{choices_text}\nAntwort: {cot}"


# --- MC format helpers ---


def mc_doc_to_text(doc):
    """MC prompt: question + lettered options."""
    choices_text = _format_choices(_get_options(doc))
    return f"Frage: {doc['question']}\n{choices_text}\nAntwort:"


def mc_doc_to_choice(doc):
    """Return letter labels matching the number of options."""
    n = len(_get_options(doc))
    return list(LABELS[:n])


# --- RC format helpers ---


def rc_doc_to_text(doc):
    """RC prompt: question only, no lettered options."""
    return f"Frage: {doc['question']}\nAntwort:"


def rc_doc_to_choice(doc):
    """Return option texts for rank classification."""
    return _get_options(doc)


def _filter_category(dataset, subject):
    return dataset.filter(lambda x: x["category"] == subject)


process_biology = partial(_filter_category, subject="biology")
process_business = partial(_filter_category, subject="business")
process_chemistry = partial(_filter_category, subject="chemistry")
process_computer_science = partial(_filter_category, subject="computer science")
process_economics = partial(_filter_category, subject="economics")
process_engineering = partial(_filter_category, subject="engineering")
process_health = partial(_filter_category, subject="health")
process_history = partial(_filter_category, subject="history")
process_law = partial(_filter_category, subject="law")
process_math = partial(_filter_category, subject="math")
process_other = partial(_filter_category, subject="other")
process_philosophy = partial(_filter_category, subject="philosophy")
process_physics = partial(_filter_category, subject="physics")
process_psychology = partial(_filter_category, subject="psychology")
