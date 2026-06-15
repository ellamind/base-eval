"""Utility functions for INCLUDE Spanish tasks (MC, RC, BPB).

Fixes the upstream lm-eval-harness INCLUDE task by:
- Using Spanish prompts (Pregunta:/Respuesta:) instead of English
- Shuffling answer options deterministically to avoid position bias
- Adding RC (rank choice) and BPB (bits-per-byte) variants
- 5-shot from validation split (same domain)
"""

import hashlib
import math
import random
from functools import partial

LABELS = "ABCD"

DOMAINS = ["Arts & Humanities", "Health oriented education", "STEM", "Social Science"]


# ---------------------------------------------------------------------------
# Dataset preprocessing
# ---------------------------------------------------------------------------

def _shuffle_options(doc):
    """Shuffle the 4 options deterministically and update the answer index."""
    options = [doc["option_a"], doc["option_b"], doc["option_c"], doc["option_d"]]
    correct_idx = doc["answer"]
    correct_text = options[correct_idx]

    seed = int(hashlib.md5(doc["question"].encode()).hexdigest(), 16)
    rng = random.Random(seed)
    rng.shuffle(options)

    doc["shuffled_options"] = options
    doc["shuffled_answer"] = options.index(correct_text)
    return doc


def _filter_domain(dataset, domain):
    """Filter dataset to a single domain, then shuffle options."""
    return dataset.filter(lambda x: x["domain"] == domain).map(_shuffle_options)


def _raw_to_fewshot(raw):
    """Convert a raw validation example into a shuffled few-shot dict."""
    return _shuffle_options({
        "question": raw["question"],
        "option_a": raw["option_a"],
        "option_b": raw["option_b"],
        "option_c": raw["option_c"],
        "option_d": raw["option_d"],
        "answer": raw["answer"],
    })


process_arts_humanities = partial(_filter_domain, domain="Arts & Humanities")
process_health_education = partial(_filter_domain, domain="Health oriented education")
process_stem = partial(_filter_domain, domain="STEM")
process_social_science = partial(_filter_domain, domain="Social Science")


# ---------------------------------------------------------------------------
# 5-shot examples from validation split (curated per domain)
# ---------------------------------------------------------------------------

_FEWSHOT_ARTS_HUMANITIES_RAW = [
    {"question": "¿Cuántos adjetivos calificativos presenta el texto? El explotado trabajador, muy molesto e indignado, criticó las propuestas horrorosas de esa empresa textil con alma capitalista.", "option_a": "cuatro", "option_b": "seis", "option_c": "cinco", "option_d": "siete", "answer": 1},
    {"question": "De acuerdo con la guía CCN-STIC-802, de auditoría del ENS, señale cuál es la respuesta correcta:", "option_a": "Los sistemas de categoría básica y media requerirán de una autoevaluación para su declaración de la conformidad que", "option_b": "deberá realizarse al menos cada dos años, o cuando se produzcan modificaciones sustanciales en el sistema.", "option_c": "Los sistemas de categoría básica requerirán de una autoevaluación para su declaración de la conformidad que deberá", "option_d": "realizarse al menos cada año, o cuando se produzcan modificaciones sustanciales en el sistema.", "answer": 3},
    {"question": "¿Cuántos verbos encontramos en el siguiente texto? Ella era hermosa, hermosa con esa hermosura que inspira el vértigo; hermosa con esa hermosura que no se parece a nada a la que soñamos en los ángeles, que; sin embargo, es sobrenatural; hermosura diabólica, que tal vez presta el demonio a algunos seres...", "option_a": "seis", "option_b": "tres", "option_c": "cuatro", "option_d": "cinco", "answer": 0},
    {"question": "Todo proceso comunicativo humano se caracteriza por ser primariamente", "option_a": "escrito.", "option_b": "gestual.", "option_c": "oral.", "option_d": "sígnico.", "answer": 2},
    {"question": "En la obra La metamorfosis, la transformación de Gregorio Samsa se asocia al", "option_a": "animismo.", "option_b": "carácter aristocrático.", "option_c": "uso de la alegoría.", "option_d": "esteticismo.", "answer": 2},
]

_FEWSHOT_HEALTH_EDUCATION_RAW = [
    {"question": "Tipo de músculo que mueve un hueso hacia la línea media", "option_a": "Depresor", "option_b": "Pronador", "option_c": "Aductor", "option_d": "Tensor", "answer": 2},
    {"question": "Glándula que interviene en la producción de glucocorticoides:", "option_a": "Páncreas", "option_b": "Suprarrenal", "option_c": "Ovario", "option_d": "Testículo", "answer": 1},
    {"question": "Las fosas nasales presentan todos los componentes, excepto:", "option_a": "Fosa nasal", "option_b": "Tabique nasal", "option_c": "Vestíbulo nasal", "option_d": "Cornete", "answer": 0},
    {"question": "Hormona que actúa sobre el metabolismo de agua, sodio, potasio y cloruro de sodio:", "option_a": "Aldosterona", "option_b": "Cortisol", "option_c": "Corticosterona", "option_d": "Cortisona", "answer": 0},
    {"question": "Nervio que inerva a los músculos esternocleidomastoideo y trapecio:", "option_a": "Hipogloso", "option_b": "Espinal", "option_c": "Vago", "option_d": "Acústico", "answer": 1},
]

_FEWSHOT_STEM_RAW = [
    {"question": "La estructura presente en tejidos vegetales, formada por pectatos de calcio y magnesio se denomina", "option_a": "lámina nuclear.", "option_b": "desmosoma.", "option_c": "lámina media.", "option_d": "plasmodemo.", "answer": 2},
    {"question": "El yuyo es un tipo de alga comestible que pertenece al grupo de las algas", "option_a": "rojas.", "option_b": "doradas.", "option_c": "verdes.", "option_d": "azul verdosas.", "answer": 0},
    {"question": "La transcripción y la traducción en las células bacterianas son procesos", "option_a": "acoplados.", "option_b": "semiconservativos.", "option_c": "discontinuos.", "option_d": "bidireccionales.", "answer": 0},
    {"question": "Si cruzamos un organismo con genotipo AABB con un aabb, ¿qué proporción de la F2 son doblemente homocigotes recesivos?", "option_a": "1/2", "option_b": "1/16", "option_c": "1/4", "option_d": "1/8", "answer": 1},
    {"question": "Respecto a los alelos, marque la alternativa incorrecta.", "option_a": "son las variantes de un gen", "option_b": "pueden ser homocigotos", "option_c": "ocupan diferentes locus", "option_d": "pueden ser híbridos", "answer": 2},
]

_FEWSHOT_SOCIAL_SCIENCE_RAW = [
    {"question": "Cuando se nos presentan muchas necesidades en un determinado momento, asumimos que las necesidades", "option_a": "son concurrentes", "option_b": "varían en intensidad", "option_c": "son complementarias", "option_d": "tienden a fijarse", "answer": 0},
    {"question": "Según la Constitución Política del Perú, el presupuesto público no se puede aprobar si es que antes no se destina un porcentaje al pago", "option_a": "de las reservas de contingencia.", "option_b": "de los gastos de capital.", "option_c": "del servicio de la deuda.", "option_d": "solo de amortizaciones de la deuda.", "answer": 2},
    {"question": "Cuando un recurso potencial para la producción es extraído de la naturaleza, por la acción del hombre, se le considera a este como", "option_a": "materia prima", "option_b": "insumos", "option_c": "bien final", "option_d": "materia bruta", "answer": 0},
    {"question": "Si el precio del bien sustituto disminuye, la curva de la demanda se", "option_a": "expandirá", "option_b": "incrementará", "option_c": "desplazará hacia la izquierda", "option_d": "mantendrá constante", "answer": 2},
    {"question": "Empresas como Sedapal y Edelnor son ejemplos de modelos de mercado denominados", "option_a": "monopolio bilateral", "option_b": "monopolio natural", "option_c": "oligopolio", "option_d": "competencia monopolística", "answer": 1},
]


def list_fewshot_arts_humanities():
    return [_raw_to_fewshot(r) for r in _FEWSHOT_ARTS_HUMANITIES_RAW]


def list_fewshot_health_education():
    return [_raw_to_fewshot(r) for r in _FEWSHOT_HEALTH_EDUCATION_RAW]


def list_fewshot_stem():
    return [_raw_to_fewshot(r) for r in _FEWSHOT_STEM_RAW]


def list_fewshot_social_science():
    return [_raw_to_fewshot(r) for r in _FEWSHOT_SOCIAL_SCIENCE_RAW]


# ---------------------------------------------------------------------------
# MC variant — letter-labeled choices (A/B/C/D) in prompt
# ---------------------------------------------------------------------------

def doc_to_text_mc(doc):
    options = doc["shuffled_options"]
    choices_text = "\n".join(
        f" {LABELS[i]}. {options[i]}" for i in range(len(options))
    )
    return f"Pregunta: {doc['question'].strip()}\n{choices_text}\nRespuesta:"


def doc_to_choice_mc(doc):
    return list(LABELS[:len(doc["shuffled_options"])])


def doc_to_target_mc(doc):
    return doc["shuffled_answer"]


# ---------------------------------------------------------------------------
# RC variant — question only, score full answer texts
# ---------------------------------------------------------------------------

def doc_to_text_rc(doc):
    return f"Pregunta: {doc['question'].strip()}\nRespuesta:"


def doc_to_choice_rc(doc):
    return doc["shuffled_options"]


# ---------------------------------------------------------------------------
# BPB variant — score only the gold answer, normalize by byte length
# ---------------------------------------------------------------------------

def doc_to_target_bpb(doc):
    return f" {doc['shuffled_options'][doc['shuffled_answer']]}"


def process_results_bpb(doc, results):
    """BPB = -log2 P(answer | context) / bytes(answer)."""
    ll, _is_greedy = results[0]

    gold_text = doc["shuffled_options"][doc["shuffled_answer"]]
    gold_bytes = len((" " + gold_text).encode("utf-8"))
    bpb = -ll / (math.log(2) * max(gold_bytes, 1))

    return {"answer_bits_per_byte": bpb}
