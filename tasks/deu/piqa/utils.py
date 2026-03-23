import hashlib
import math
import random


def list_fewshot_samples():
    """5-shot curated examples for PiQA (German).

    Hand-written examples covering practical/physical tasks, designed to
    sound natural in German rather than translated from English.
    """
    return [
        {
            "goal": "Wenn man Butter ausgelassen hat, wie bewahrt man das Butterschmalz am besten auf?",
            "correct_solution": "Man füllt es in ein Schraubglas und lässt es abkühlen.",
            "choices": [
                "Man gießt es auf einen flachen Teller und stellt ihn in den Kühlschrank.",
                "Man füllt es in ein Schraubglas und lässt es abkühlen.",
            ],
            "answer_idx": 1,
        },
        {
            "goal": "Um das Eigelb vom Eiweiß zu trennen, kann man eine leere Plastikflasche verwenden. Wie geht das?",
            "correct_solution": "Man drückt die Flasche zusammen, hält sie an das Eigelb und lässt los. Der Sog zieht das Eigelb in die Flasche.",
            "choices": [
                "Man drückt die Flasche zusammen, hält sie an das Eigelb und lässt los. Der Sog zieht das Eigelb in die Flasche.",
                "Man drückt die Flasche auf das Eigelb und drückt weiter. Der Druck schiebt das Eigelb hinein.",
            ],
            "answer_idx": 0,
        },
        {
            "goal": "Wie schüttelt man eine Flasche Salatdressing richtig durch?",
            "correct_solution": "Man bewegt die Flasche schnell auf und ab und hin und her, bis sich alles vermischt hat.",
            "choices": [
                "Man rührt langsam mit einem Löffel um.",
                "Man bewegt die Flasche schnell auf und ab und hin und her, bis sich alles vermischt hat.",
            ],
            "answer_idx": 1,
        },
        {
            "goal": "Wie reinigt man eine Spülmaschine von innen?",
            "correct_solution": "Man stellt eine Tasse Essig ins obere Fach und lässt den heißesten Spülgang durchlaufen.",
            "choices": [
                "Man stellt eine Tasse Essig ins obere Fach und lässt den heißesten Spülgang durchlaufen.",
                "Man stellt eine Tasse Essig ins obere Fach und lässt den kältesten Spülgang durchlaufen.",
            ],
            "answer_idx": 0,
        },
        {
            "goal": "Wie träufelt man Olivenöl über einen Salat?",
            "correct_solution": "Man kippt die Flasche leicht und lässt das Öl in einem dünnen Strahl fließen.",
            "choices": [
                "Man kippt die Flasche leicht und lässt das Öl in einem dünnen Strahl fließen.",
                "Man gießt das Öl schnell und großzügig über den Salat.",
            ],
            "answer_idx": 0,
        },
    ]


def _prepare_choices(doc, use_easy=False):
    """Build choices list and find correct answer index.

    Args:
        doc: Dataset document.
        use_easy: If True, pair correct with easy_distractor;
            otherwise pair with hard_distractor.

    PiQA has single distractors (strings), not lists.
    """
    distractor = doc["easy_distractor"] if use_easy else doc["hard_distractor"]
    choices = [doc["correct_solution"], distractor]

    # Deterministic shuffle based on seed_id (using md5 for consistent hash across sessions)
    seed_str = doc.get("seed_id", doc["goal"])
    seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16)
    rng = random.Random(seed)
    rng.shuffle(choices)

    doc["choices"] = choices
    doc["answer_idx"] = choices.index(doc["correct_solution"])
    return doc


def prepare_all(dataset):
    return dataset.map(lambda x: _prepare_choices(x, use_easy=False))


def prepare_easy(dataset):
    return dataset.map(lambda x: _prepare_choices(x, use_easy=True))


def process_results_bpb(doc, results):
    """Compute answer-only BPB (OLMES-style) for PiQA."""
    ll, _ = results[0]
    gold_text = doc["correct_solution"]
    gold_bytes = len((" " + gold_text).encode("utf-8"))
    return {"bits_per_byte": -ll / (math.log(2) * max(gold_bytes, 1))}
