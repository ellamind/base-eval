import json
import math
from functools import partial
from pathlib import Path

SUBJECTS_PATH = Path(__file__).resolve().parent / "subjects.json"
with SUBJECTS_PATH.open(encoding="utf-8") as f:
    SUBJECTS = json.load(f)

FEWSHOT_PATH = Path(__file__).resolve().parent / "mmlu_dev_fewshot.json"
with FEWSHOT_PATH.open(encoding="utf-8") as f:
    _FEWSHOT_DATA = json.load(f)


def _normalize_subject_name(name: str) -> str:
    """
    Some MMMLU configs embed CSV filenames in the Subject column (e.g.
    `college_mathematics_test.csv_sw-KE.csv`). Strip the `_test.csv` suffix and
    anything that follows so we always compare against the canonical subject id.
    """
    if not isinstance(name, str):
        return name
    for marker in ("_test.csv", "_test-"):
        idx = name.find(marker)
        if idx != -1:
            return name[:idx]
    return name


def _filter_subject(dataset, subject):
    normalized = subject

    def _predicate(row, target=normalized):
        row_subject = _normalize_subject_name(row["Subject"])
        return row_subject == target

    return dataset.filter(_predicate)


def _register_subject_filters():
    for subject in SUBJECTS:
        globals()[f"process_{subject}"] = partial(_filter_subject, subject=subject)


_register_subject_filters()


# ---------------------------------------------------------------------------
# Fewshot helpers — curated translated examples from MMLU dev set
# ---------------------------------------------------------------------------

def _fewshot_for_subject(subject):
    """Return 5 curated German fewshot examples for the given subject."""
    return _FEWSHOT_DATA[subject]


def _register_fewshot_functions():
    for subject in SUBJECTS:
        globals()[f"fewshot_{subject}"] = partial(_fewshot_for_subject, subject=subject)


_register_fewshot_functions()


# ---------------------------------------------------------------------------
# RC (cloze) format helpers — aligned with OLMES mmlu:rc
# ---------------------------------------------------------------------------

SUBJECT_DE = {
    "abstract_algebra": "abstrakte Algebra",
    "anatomy": "Anatomie",
    "astronomy": "Astronomie",
    "business_ethics": "Wirtschaftsethik",
    "clinical_knowledge": "klinisches Wissen",
    "college_biology": "Biologie (Hochschule)",
    "college_chemistry": "Chemie (Hochschule)",
    "college_computer_science": "Informatik (Hochschule)",
    "college_mathematics": "Mathematik (Hochschule)",
    "college_medicine": "Medizin (Hochschule)",
    "college_physics": "Physik (Hochschule)",
    "computer_security": "Computersicherheit",
    "conceptual_physics": "konzeptionelle Physik",
    "econometrics": "Ökonometrie",
    "electrical_engineering": "Elektrotechnik",
    "elementary_mathematics": "elementare Mathematik",
    "formal_logic": "formale Logik",
    "global_facts": "globale Fakten",
    "high_school_biology": "Biologie (Oberstufe)",
    "high_school_chemistry": "Chemie (Oberstufe)",
    "high_school_computer_science": "Informatik (Oberstufe)",
    "high_school_european_history": "europäische Geschichte (Oberstufe)",
    "high_school_geography": "Geographie (Oberstufe)",
    "high_school_government_and_politics": "Politik und Gesellschaft (Oberstufe)",
    "high_school_macroeconomics": "Makroökonomie (Oberstufe)",
    "high_school_mathematics": "Mathematik (Oberstufe)",
    "high_school_microeconomics": "Mikroökonomie (Oberstufe)",
    "high_school_physics": "Physik (Oberstufe)",
    "high_school_psychology": "Psychologie (Oberstufe)",
    "high_school_statistics": "Statistik (Oberstufe)",
    "high_school_us_history": "US-Geschichte (Oberstufe)",
    "high_school_world_history": "Weltgeschichte (Oberstufe)",
    "human_aging": "menschliches Altern",
    "human_sexuality": "menschliche Sexualität",
    "international_law": "Völkerrecht",
    "jurisprudence": "Rechtswissenschaft",
    "logical_fallacies": "logische Fehlschlüsse",
    "machine_learning": "maschinelles Lernen",
    "management": "Management",
    "marketing": "Marketing",
    "medical_genetics": "medizinische Genetik",
    "miscellaneous": "Verschiedenes",
    "moral_disputes": "moralische Streitfragen",
    "moral_scenarios": "moralische Szenarien",
    "nutrition": "Ernährung",
    "philosophy": "Philosophie",
    "prehistory": "Vorgeschichte",
    "professional_accounting": "Rechnungswesen",
    "professional_law": "Rechtswesen",
    "professional_medicine": "Medizin (Facharzt)",
    "professional_psychology": "Psychologie (Fachgebiet)",
    "public_relations": "Öffentlichkeitsarbeit",
    "security_studies": "Sicherheitsstudien",
    "sociology": "Soziologie",
    "us_foreign_policy": "US-Außenpolitik",
    "virology": "Virologie",
    "world_religions": "Weltreligionen",
}


def _format_subject(subject_slug: str) -> str:
    """Convert subject slug to readable German display name."""
    return SUBJECT_DE.get(subject_slug, subject_slug.replace("_", " "))


def _get_subject(doc) -> str:
    """Extract the canonical subject from a document."""
    return _normalize_subject_name(doc["Subject"])


def doc_to_text_rc(doc):
    """Cloze prompt: 'Frage: {question}\nAntwort:'
    Note: subject header is set via per-task 'description' field so it
    appears only once at the top of the few-shot context."""
    return f"Frage: {doc['Question'].strip()}\nAntwort:"


def doc_to_choice_rc(doc):
    """Return full answer texts as choices (not letter labels)."""
    return [doc["A"], doc["B"], doc["C"], doc["D"]]


def doc_to_target_rc(doc):
    """Return index of correct answer (0-3)."""
    return ["A", "B", "C", "D"].index(doc["Answer"])


# ---------------------------------------------------------------------------
# BPB helpers
# ---------------------------------------------------------------------------

def doc_to_text_bpb(doc):
    """BPB prompt: 'Frage: {question}\nAntwort:'
    Note: subject header is set via per-task 'description' field so it
    appears only once at the top of the few-shot context."""
    return f"Frage: {doc['Question'].strip()}\nAntwort:"

def process_results_bpb(doc, results):
    """Compute answer-only BPB (OLMES-style) for MMMLU."""
    ll, _ = results[0]
    gold_text = doc[doc["Answer"]]
    # Include leading space to match doc_to_target " {answer_text}"
    gold_bytes = len((" " + gold_text).encode("utf-8"))
    return {"bits_per_byte": -ll / (math.log(2) * max(gold_bytes, 1))}
