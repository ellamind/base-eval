import hashlib
import math
import random


def list_fewshot_samples():
    """5-shot curated examples for CommonsenseQA (German).

    Hand-written examples designed to sound natural in German rather than
    translated from English.
    """
    return [
        {
            "question": "Die Schule hatte strenge neue Regeln eingeführt, aber die Schüler hielten sich kaum daran. Was fehlte den Regeln offenbar?",
            "correct_answer": "Akzeptanz",
            "choices": ["Papier", "Akzeptanz", "Lautstärke", "Farbe", "Gewicht"],
            "answer_idx": 1,
        },
        {
            "question": "Leon wollte unter Leute kommen und neue Bekanntschaften machen. Wohin sollte er gehen?",
            "correct_answer": "auf eine Veranstaltung",
            "choices": ["in den Keller", "auf eine Veranstaltung", "in die Wüste", "auf einen Berggipfel", "in eine Lagerhalle"],
            "answer_idx": 1,
        },
        {
            "question": "Um einen Ring zu kaufen, der weder im Kaufhaus noch online zu finden war, wohin würde man gehen?",
            "correct_answer": "zum Goldschmied",
            "choices": ["zum Goldschmied", "zum Friseur", "in die Apotheke", "zum Kaufhaus", "ins Reisebüro"],
            "answer_idx": 0,
        },
        {
            "question": "Navis und Karten-Apps auf dem Handy haben bei den meisten Autofahrern etwas überflüssig gemacht. Was?",
            "correct_answer": "Den Straßenatlas",
            "choices": ["Das Lenkrad", "Den Führerschein", "Den Straßenatlas", "Den Sicherheitsgurt", "Die Hupe"],
            "answer_idx": 2,
        },
        {
            "question": "Ein Reh verlässt das Stadtgebiet und läuft in den Wald zurück. Was sucht es dort?",
            "correct_answer": "Schutz und Ruhe",
            "choices": ["einen Parkplatz", "ein Geschäft", "Schutz und Ruhe", "eine Straße", "ein Gebäude"],
            "answer_idx": 2,
        },
    ]


def _prepare_choices(doc, distractor_key="hard_distractors"):
    """Build choices list and find correct answer index.

    Args:
        doc: Dataset document.
        distractor_key: Which distractors to include
            ("hard_distractors" or "easy_distractors").
    """
    choices = [doc["correct_answer"]] + doc[distractor_key]

    # Deterministic shuffle based on seed_id (using md5 for consistent hash across sessions)
    seed_str = doc.get("seed_id", doc["question"])
    seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16)
    rng = random.Random(seed)
    rng.shuffle(choices)

    doc["choices"] = choices
    doc["answer_idx"] = choices.index(doc["correct_answer"])
    return doc


def prepare_all(dataset):
    return dataset.map(lambda x: _prepare_choices(x, distractor_key="hard_distractors"))


def prepare_easy(dataset):
    return dataset.map(lambda x: _prepare_choices(x, distractor_key="easy_distractors"))


def process_results_bpb(doc, results):
    """Compute answer-only BPB (OLMES-style) for CSQA."""
    ll, _ = results[0]
    gold_text = doc["correct_answer"]
    gold_bytes = len((" " + gold_text).encode("utf-8"))
    return {"bits_per_byte": -ll / (math.log(2) * max(gold_bytes, 1))}
