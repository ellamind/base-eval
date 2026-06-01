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
    """Return 5 curated Spanish fewshot examples for the given subject."""
    return _FEWSHOT_DATA[subject]


def _register_fewshot_functions():
    for subject in SUBJECTS:
        globals()[f"fewshot_{subject}"] = partial(_fewshot_for_subject, subject=subject)


_register_fewshot_functions()


# ---------------------------------------------------------------------------
# RC (cloze) format helpers — aligned with OLMES mmlu:rc
# ---------------------------------------------------------------------------

SUBJECT_SPA = {
    "abstract_algebra": "álgebra abstracta",
    "anatomy": "anatomía",
    "astronomy": "astronomía",
    "business_ethics": "ética empresarial",
    "clinical_knowledge": "conocimiento clínico",
    "college_biology": "biología (universidad)",
    "college_chemistry": "química (universidad)",
    "college_computer_science": "informática (universidad)",
    "college_mathematics": "matemáticas (universidad)",
    "college_medicine": "medicina (universidad)",
    "college_physics": "física (universidad)",
    "computer_security": "seguridad informática",
    "conceptual_physics": "física conceptual",
    "econometrics": "econometría",
    "electrical_engineering": "ingeniería eléctrica",
    "elementary_mathematics": "matemáticas elementales",
    "formal_logic": "lógica formal",
    "global_facts": "hechos globales",
    "high_school_biology": "biología (secundaria)",
    "high_school_chemistry": "química (secundaria)",
    "high_school_computer_science": "informática (secundaria)",
    "high_school_european_history": "historia europea (secundaria)",
    "high_school_geography": "geografía (secundaria)",
    "high_school_government_and_politics": "política y gobierno (secundaria)",
    "high_school_macroeconomics": "macroeconomía (secundaria)",
    "high_school_mathematics": "matemáticas (secundaria)",
    "high_school_microeconomics": "microeconomía (secundaria)",
    "high_school_physics": "física (secundaria)",
    "high_school_psychology": "psicología (secundaria)",
    "high_school_statistics": "estadística (secundaria)",
    "high_school_us_history": "historia de EE. UU. (secundaria)",
    "high_school_world_history": "historia mundial (secundaria)",
    "human_aging": "envejecimiento humano",
    "human_sexuality": "sexualidad humana",
    "international_law": "derecho internacional",
    "jurisprudence": "jurisprudencia",
    "logical_fallacies": "falacias lógicas",
    "machine_learning": "aprendizaje automático",
    "management": "gestión empresarial",
    "marketing": "marketing",
    "medical_genetics": "genética médica",
    "miscellaneous": "miscelánea",
    "moral_disputes": "disputas morales",
    "moral_scenarios": "escenarios morales",
    "nutrition": "nutrición",
    "philosophy": "filosofía",
    "prehistory": "prehistoria",
    "professional_accounting": "contabilidad profesional",
    "professional_law": "derecho profesional",
    "professional_medicine": "medicina profesional",
    "professional_psychology": "psicología profesional",
    "public_relations": "relaciones públicas",
    "security_studies": "estudios de seguridad",
    "sociology": "sociología",
    "us_foreign_policy": "política exterior de EE. UU.",
    "virology": "virología",
    "world_religions": "religiones del mundo",
}


def _format_subject(subject_slug: str) -> str:
    """Convert subject slug to readable Spanish display name."""
    return SUBJECT_SPA.get(subject_slug, subject_slug.replace("_", " "))


def _get_subject(doc) -> str:
    """Extract the canonical subject from a document."""
    return _normalize_subject_name(doc["Subject"])


def doc_to_text_rc(doc):
    """Cloze prompt: 'Pregunta: {question}\nRespuesta:'
    Note: subject header is set via per-task 'description' field so it
    appears only once at the top of the few-shot context."""
    return f"Pregunta: {doc['Question'].strip()}\nRespuesta:"


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
    """BPB prompt: 'Pregunta: {question}\nRespuesta:'
    Note: subject header is set via per-task 'description' field so it
    appears only once at the top of the few-shot context."""
    return f"Pregunta: {doc['Question'].strip()}\nRespuesta:"

def process_results_bpb(doc, results):
    """Compute answer-only BPB (OLMES-style) for MMMLU."""
    ll, _ = results[0]
    gold_text = doc[doc["Answer"]]
    # Include leading space to match doc_to_target " {answer_text}"
    gold_bytes = len((" " + gold_text).encode("utf-8"))
    return {"bits_per_byte": -ll / (math.log(2) * max(gold_bytes, 1))}
