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
    """Return 5 curated French fewshot examples for the given subject."""
    return _FEWSHOT_DATA[subject]


def _register_fewshot_functions():
    for subject in SUBJECTS:
        globals()[f"fewshot_{subject}"] = partial(_fewshot_for_subject, subject=subject)


_register_fewshot_functions()


# ---------------------------------------------------------------------------
# RC (cloze) format helpers — aligned with OLMES mmlu:rc
# ---------------------------------------------------------------------------

SUBJECT_FRA = {
    "abstract_algebra": "l'algèbre abstraite",
    "anatomy": "l'anatomie",
    "astronomy": "l'astronomie",
    "business_ethics": "l'éthique des affaires",
    "clinical_knowledge": "les connaissances cliniques",
    "college_biology": "la biologie (université)",
    "college_chemistry": "la chimie (université)",
    "college_computer_science": "l'informatique (université)",
    "college_mathematics": "les mathématiques (université)",
    "college_medicine": "la médecine (université)",
    "college_physics": "la physique (université)",
    "computer_security": "la sécurité informatique",
    "conceptual_physics": "la physique conceptuelle",
    "econometrics": "l'économétrie",
    "electrical_engineering": "l'ingénierie électrique",
    "elementary_mathematics": "les mathématiques élémentaires",
    "formal_logic": "la logique formelle",
    "global_facts": "les faits mondiaux",
    "high_school_biology": "la biologie (lycée)",
    "high_school_chemistry": "la chimie (lycée)",
    "high_school_computer_science": "l'informatique (lycée)",
    "high_school_european_history": "l'histoire européenne (lycée)",
    "high_school_geography": "la géographie (lycée)",
    "high_school_government_and_politics": "la politique et le gouvernement (lycée)",
    "high_school_macroeconomics": "la macroéconomie (lycée)",
    "high_school_mathematics": "les mathématiques (lycée)",
    "high_school_microeconomics": "la microéconomie (lycée)",
    "high_school_physics": "la physique (lycée)",
    "high_school_psychology": "la psychologie (lycée)",
    "high_school_statistics": "les statistiques (lycée)",
    "high_school_us_history": "l'histoire des États-Unis (lycée)",
    "high_school_world_history": "l'histoire mondiale (lycée)",
    "human_aging": "le vieillissement humain",
    "human_sexuality": "la sexualité humaine",
    "international_law": "le droit international",
    "jurisprudence": "la jurisprudence",
    "logical_fallacies": "les sophismes logiques",
    "machine_learning": "l'apprentissage automatique",
    "management": "la gestion",
    "marketing": "le marketing",
    "medical_genetics": "la génétique médicale",
    "miscellaneous": "des sujets divers",
    "moral_disputes": "les controverses morales",
    "moral_scenarios": "les scénarios moraux",
    "nutrition": "la nutrition",
    "philosophy": "la philosophie",
    "prehistory": "la préhistoire",
    "professional_accounting": "la comptabilité professionnelle",
    "professional_law": "le droit professionnel",
    "professional_medicine": "la médecine professionnelle",
    "professional_psychology": "la psychologie professionnelle",
    "public_relations": "les relations publiques",
    "security_studies": "les études de sécurité",
    "sociology": "la sociologie",
    "us_foreign_policy": "la politique étrangère des États-Unis",
    "virology": "la virologie",
    "world_religions": "les religions du monde",
}


def _format_subject(subject_slug: str) -> str:
    """Convert subject slug to readable French display name."""
    return SUBJECT_FRA.get(subject_slug, subject_slug.replace("_", " "))


def _get_subject(doc) -> str:
    """Extract the canonical subject from a document."""
    return _normalize_subject_name(doc["Subject"])


def doc_to_text_rc(doc):
    """Cloze prompt: 'Question: {question}\nRéponse:'
    Note: subject header is set via per-task 'description' field so it
    appears only once at the top of the few-shot context."""
    return f"Question: {doc['Question'].strip()}\nRéponse:"


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
    """BPB prompt: 'Question: {question}\nRéponse:'
    Note: subject header is set via per-task 'description' field so it
    appears only once at the top of the few-shot context."""
    return f"Question: {doc['Question'].strip()}\nRéponse:"

def process_results_bpb(doc, results):
    """Compute answer-only BPB (OLMES-style) for MMMLU."""
    ll, _ = results[0]
    gold_text = doc[doc["Answer"]]
    # Include leading space to match doc_to_target " {answer_text}"
    gold_bytes = len((" " + gold_text).encode("utf-8"))
    return {"bits_per_byte": -ll / (math.log(2) * max(gold_bytes, 1))}
