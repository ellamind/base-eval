"""Utility functions for INCLUDE French tasks (MC, RC, BPB).

Fixes the upstream lm-eval-harness INCLUDE task by:
- Using French prompts (Question :/Reponse :) instead of English
- Shuffling answer options deterministically to avoid position bias
- Adding RC (rank choice) and BPB (bits-per-byte) variants
- 5-shot from validation split (same domain; "Other" for Driving License)
"""

import hashlib
import math
import random
from functools import partial

LABELS = "ABCD"

DOMAINS = ["Arts & Humanities", "Driving License", "Health oriented education", "STEM", "Social Science"]


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
process_driving_license = partial(_filter_domain, domain="Driving License")
process_health_education = partial(_filter_domain, domain="Health oriented education")
process_stem = partial(_filter_domain, domain="STEM")
process_social_science = partial(_filter_domain, domain="Social Science")


# ---------------------------------------------------------------------------
# 5-shot examples from validation split (curated per domain)
# ---------------------------------------------------------------------------

_FEWSHOT_ARTS_HUMANITIES_RAW = [
    {"question": "Qui a r\u00e9alis\u00e9 le film \u00ab L\u00e9on \u00bb ?", "option_a": "Costa-Gavras", "option_b": "Luc Besson", "option_c": "Martin Scorsese", "option_d": "Steven Spielberg", "answer": 1},
    {"question": "Qui est le dernier Pr\u00e9sident de la IV\u00e8me R\u00e9publique ?", "option_a": "Ren\u00e9 Coty", "option_b": "F\u00e9lix Gaillard", "option_c": "Charles de Gaulle", "option_d": "Alain Poher", "answer": 0},
    {"question": "Comment est appel\u00e9e la grenadille ?", "option_a": "La goyave", "option_b": "La mangue", "option_c": "Le fruit de la passion", "option_d": "La papaye", "answer": 2},
    {"question": "Le drapeau europ\u00e9en est constitu\u00e9 de", "option_a": "6 \u00e9toiles", "option_b": "12 \u00e9toiles", "option_c": "18 \u00e9toiles", "option_d": "27 \u00e9toiles", "answer": 1},
    {"question": "Qui est \u00e0 l\u2019origine de cette formule \"carpe diem\" ?", "option_a": "Virgile", "option_b": "Pline le Jeune", "option_c": "Horace", "option_d": "Sophocle", "answer": 2},
]

_FEWSHOT_DRIVING_LICENSE_RAW = [
    {"question": "Pour un v\u00e9hicule essence, la consommation de carburant augmente si :", "option_a": "Les pneus sont sous gonfl\u00e9s.", "option_b": "La batterie est d\u00e9charg\u00e9e.", "option_c": "Les bougies sont us\u00e9es.", "option_d": "Le filtre \u00e0 air est encrass\u00e9", "answer": 0},
    {"question": "Pour consulter mon solde de points, je me rends sur le site internet :", "option_a": "Allopoints.", "option_b": "Info-point.", "option_c": "Telepoint.", "option_d": "Point-permis.", "answer": 2},
    {"question": "En circulant \u00e0 110 km/h je parcours en une seconde la distance de", "option_a": "33 m\u00e8tres environ", "option_b": "66 m\u00e8tres environ", "option_c": "88 m\u00e8tres environ", "option_d": "110 m\u00e8tres environ", "answer": 0},
    {"question": "J\u2019ai mon permis depuis 8 mois. Je peux circuler \u00e0", "option_a": "130km/h", "option_b": "110km/h", "option_c": "100km/h", "option_d": "90km/h", "answer": 1},
    {"question": "En conduite je peux utiliser des lunettes de soleil de cat\u00e9gorie", "option_a": "2", "option_b": "3", "option_c": "4", "option_d": "5", "answer": 0},
]

_FEWSHOT_HEALTH_EDUCATION_RAW = [
    {"question": "Toutes les causes suivantes peuvent favoriser la survenue d\u2019une thrombose veineuse des membres inf\u00e9rieurs sauf une. Laquelle ?", "option_a": "D\u00e9ficit en anti", "option_b": "Anti", "option_c": "D\u00e9cubitus prolong\u00e9", "option_d": "Prise d\u2019oestro", "answer": 1},
    {"question": "Lequel de ces cancers est li\u00e9 au tabagisme ?", "option_a": "Sein", "option_b": "C\u00f4lon", "option_c": "Vessie", "option_d": "Rein", "answer": 2},
    {"question": "L\u2019inflammation :", "option_a": "Est l\u2019envahissement de l\u2019organisme par un agent pathog\u00e8ne vivant", "option_b": "Peut se produire dans tous les tissus", "option_c": "Est synonyme d\u2019infection", "option_d": "A une correspondance morphologique pr\u00e9cise", "answer": 1},
    {"question": "Le sympt\u00f4me r\u00e9v\u00e9lateur le plus fr\u00e9quent d\u2019un cancer du rein de l\u2019adulte est :", "option_a": "La d\u00e9couverte d\u2019une masse abdominale", "option_b": "Une h\u00e9maturie totale", "option_c": "Fi\u00e8vre inexpliqu\u00e9e", "option_d": "An\u00e9mie", "answer": 1},
    {"question": "Dans le m\u00e9lanome malin primitif, quel est le crit\u00e8re histologique essentiel du pronostic, quelle que soit la vari\u00e9t\u00e9 anatomoclinique ?", "option_a": "L\u2019inflammation p\u00e9ritumorale", "option_b": "L\u2019indice mitotique", "option_c": "L\u2019\u00e9paisseur maximum de la tumeur", "option_d": "Le niveau d\u2019invasion du derme", "answer": 2},
]

_FEWSHOT_STEM_RAW = [
    {"question": "Trois milli\u00e8mes est \u00e9gal \u00e0 :", "option_a": "0,003", "option_b": "300", "option_c": "0,3", "option_d": "0,03", "answer": 0},
    {"question": "La taille moyenne d\u2019un groupe d\u2019enfants augmenterait de 6 cm si 12 des enfants du groupe mesuraient 8 cm de plus. Combien d\u2019enfants y a-t-il dans le groupe ?", "option_a": "14", "option_b": "21", "option_c": "16", "option_d": "26", "answer": 2},
    {"question": "Soit a et b des entiers strictement positifs qui v\u00e9rifient 45a + b = 2021. Quelle est la valeur minimale possible de a + b ?", "option_a": "44", "option_b": "85", "option_c": "82", "option_d": "86", "answer": 1},
    {"question": "Quelle est la valeur de 0,9 + 0,09 ?", "option_a": "0,99", "option_b": "1,08", "option_c": "0,909", "option_d": "1,8", "answer": 0},
    {"question": "Dans la figure ci-contre, PQRS est un carr\u00e9 et M est le milieu de PS. Quel est le rapport de l\u2019aire du triangle QMS \u00e0 l\u2019aire du carr\u00e9 PQRS?", "option_a": "1 : 6", "option_b": "1 : 3", "option_c": "1 : 4", "option_d": "1 : 8", "answer": 2},
]

_FEWSHOT_SOCIAL_SCIENCE_RAW = [
    {"question": "L\u2019accroissement naturel de la population\u2026", "option_a": "peut \u00eatre calcul\u00e9 de la somme du nombre de naissances vivantes et de d\u00e9c\u00e8s.", "option_b": "est un indice d\u00e9mographique dont la valeur ne peut \u00eatre qu\u2019un nombe entier positif.", "option_c": "est la diff\u00e9rence entre le nombre de naissances vivantes et le nombre de d\u00e9c\u00e8s.", "option_d": "peut \u00eatre calcul\u00e9 de la croissance d\u00e9mographique sur une unit\u00e9 de territoire.", "answer": 2},
    {"question": "Les bermudes appartiennent :", "option_a": "au Royaume-Uni", "option_b": "aux Etats-Unis", "option_c": "\u00e0 la Chine", "option_d": "Au Mexique", "answer": 0},
    {"question": "Augmentation de l\u2019allocation familiale. ELEMENTS BUDGETAIRES:", "option_a": "Imp\u00f4ts et taxes", "option_b": "Achats de marchandises, investissements du gouvernement", "option_c": "Aides, allocations", "option_d": "Aucun des \u00e9l\u00e9ments pr\u00e9cit\u00e9s Justifier la r\u00e9ponse donn\u00e9e \u00e0 la mesure 2.", "answer": 2},
    {"question": "Quel est le plus grand fleuve au monde ?", "option_a": "l\u2019Amazone", "option_b": "le Nil", "option_c": "le Mississippi", "option_d": "le Gange", "answer": 0},
    {"question": "La publicit\u00e9 comparative est caract\u00e9ris\u00e9e par le fait", "option_a": "qu\u2019elle ne peut pas influencer les acheteurs.", "option_b": "qu\u2019uniquement des produits d\u2019origine diff\u00e9rente peuvent \u00eatre compar\u00e9s.", "option_c": "qu\u2019elle est sanctionn\u00e9e par la loi.", "option_d": "qu\u2019elle ne peut pas l\u00e9ser \u00e0 la r\u00e9putation d\u2019une autre entreprise.", "answer": 3},
]


def list_fewshot_arts_humanities():
    return [_raw_to_fewshot(r) for r in _FEWSHOT_ARTS_HUMANITIES_RAW]


def list_fewshot_driving_license():
    return [_raw_to_fewshot(r) for r in _FEWSHOT_DRIVING_LICENSE_RAW]


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
    return f"Question : {doc['question'].strip()}\n{choices_text}\nR\u00e9ponse :"


def doc_to_choice_mc(doc):
    return list(LABELS[:len(doc["shuffled_options"])])


def doc_to_target_mc(doc):
    return doc["shuffled_answer"]


# ---------------------------------------------------------------------------
# RC variant — question only, score full answer texts
# ---------------------------------------------------------------------------

def doc_to_text_rc(doc):
    return f"Question : {doc['question'].strip()}\nR\u00e9ponse :"


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

    return {"bits_per_byte": bpb}
