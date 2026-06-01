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
    """Return 5 curated Italian fewshot examples for the given subject."""
    return _FEWSHOT_DATA[subject]


def _register_fewshot_functions():
    for subject in SUBJECTS:
        globals()[f"fewshot_{subject}"] = partial(_fewshot_for_subject, subject=subject)


_register_fewshot_functions()


# ---------------------------------------------------------------------------
# RC (cloze) format helpers — aligned with OLMES mmlu:rc
# ---------------------------------------------------------------------------

SUBJECT_ITA = {
    "abstract_algebra": "algebra astratta",
    "anatomy": "anatomia",
    "astronomy": "astronomia",
    "business_ethics": "etica aziendale",
    "clinical_knowledge": "conoscenze cliniche",
    "college_biology": "biologia (università)",
    "college_chemistry": "chimica (università)",
    "college_computer_science": "informatica (università)",
    "college_mathematics": "matematica (università)",
    "college_medicine": "medicina (università)",
    "college_physics": "fisica (università)",
    "computer_security": "sicurezza informatica",
    "conceptual_physics": "fisica concettuale",
    "econometrics": "econometria",
    "electrical_engineering": "ingegneria elettrica",
    "elementary_mathematics": "matematica elementare",
    "formal_logic": "logica formale",
    "global_facts": "fatti globali",
    "high_school_biology": "biologia (liceo)",
    "high_school_chemistry": "chimica (liceo)",
    "high_school_computer_science": "informatica (liceo)",
    "high_school_european_history": "storia europea (liceo)",
    "high_school_geography": "geografia (liceo)",
    "high_school_government_and_politics": "politica e governo (liceo)",
    "high_school_macroeconomics": "macroeconomia (liceo)",
    "high_school_mathematics": "matematica (liceo)",
    "high_school_microeconomics": "microeconomia (liceo)",
    "high_school_physics": "fisica (liceo)",
    "high_school_psychology": "psicologia (liceo)",
    "high_school_statistics": "statistica (liceo)",
    "high_school_us_history": "storia degli USA (liceo)",
    "high_school_world_history": "storia mondiale (liceo)",
    "human_aging": "invecchiamento umano",
    "human_sexuality": "sessualità umana",
    "international_law": "diritto internazionale",
    "jurisprudence": "giurisprudenza",
    "logical_fallacies": "fallacie logiche",
    "machine_learning": "apprendimento automatico",
    "management": "gestione aziendale",
    "marketing": "marketing",
    "medical_genetics": "genetica medica",
    "miscellaneous": "varie",
    "moral_disputes": "controversie morali",
    "moral_scenarios": "scenari morali",
    "nutrition": "nutrizione",
    "philosophy": "filosofia",
    "prehistory": "preistoria",
    "professional_accounting": "contabilità professionale",
    "professional_law": "diritto professionale",
    "professional_medicine": "medicina professionale",
    "professional_psychology": "psicologia professionale",
    "public_relations": "relazioni pubbliche",
    "security_studies": "studi sulla sicurezza",
    "sociology": "sociologia",
    "us_foreign_policy": "politica estera degli USA",
    "virology": "virologia",
    "world_religions": "religioni del mondo",
}


def _format_subject(subject_slug: str) -> str:
    """Convert subject slug to readable Italian display name."""
    return SUBJECT_ITA.get(subject_slug, subject_slug.replace("_", " "))


def _get_subject(doc) -> str:
    """Extract the canonical subject from a document."""
    return _normalize_subject_name(doc["Subject"])


def doc_to_text_rc(doc):
    """Cloze prompt: 'Domanda: {question}\nRisposta:'
    Note: subject header is set via per-task 'description' field so it
    appears only once at the top of the few-shot context."""
    return f"Domanda: {doc['Question'].strip()}\nRisposta:"


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
    """BPB prompt: 'Domanda: {question}\nRisposta:'
    Note: subject header is set via per-task 'description' field so it
    appears only once at the top of the few-shot context."""
    return f"Domanda: {doc['Question'].strip()}\nRisposta:"

def process_results_bpb(doc, results):
    """Compute answer-only BPB (OLMES-style) for MMMLU."""
    ll, _ = results[0]
    gold_text = doc[doc["Answer"]]
    # Include leading space to match doc_to_target " {answer_text}"
    gold_bytes = len((" " + gold_text).encode("utf-8"))
    return {"bits_per_byte": -ll / (math.log(2) * max(gold_bytes, 1))}
