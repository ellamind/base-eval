"""Utility functions for English MMLU Pro tasks (MC, RC).

Dataset: TIGER-Lab/MMLU-Pro
- 12,032 test questions across 14 categories
- 70 validation examples (5 per category) for fewshot
- Options stored as list in 'options' field
"""

from functools import partial

LABELS = "ABCDEFGHIJ"


def _get_options(doc):
    """Extract valid options from the options list."""
    return [opt for opt in doc["options"] if opt is not None]


def _format_choices(options):
    """Format options as ' A. ... \\n B. ...' matching OLMES convention."""
    return "\n".join(f" {LABELS[i]}. {opt}" for i, opt in enumerate(options))


# --- MC format helpers ---


def mc_doc_to_text(doc):
    """MC prompt: question + lettered options."""
    choices_text = _format_choices(_get_options(doc))
    return f"Question: {doc['question']}\n{choices_text}\nAnswer:"


def mc_doc_to_choice(doc):
    """Return letter labels matching the number of options."""
    n = len(_get_options(doc))
    return list(LABELS[:n])


# --- RC format helpers ---


def rc_doc_to_text(doc):
    """RC prompt: question only, no lettered options."""
    return f"Question: {doc['question']}\nAnswer:"


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
