import hashlib
import math
import random


def list_fewshot_samples():
    """5-shot curated examples for HellaSwag (German).

    Hand-written examples covering everyday scenarios, designed to sound
    natural in German rather than translated from English.
    """
    return [
        {
            "context": "Ein Koch steht in der Küche und schneidet Zwiebeln auf einem Holzbrett. Danach",
            "correct_ending": "gibt er die Zwiebelstücke in eine heiße Pfanne mit Olivenöl.",
            "choices": [
                "legt er die Zwiebeln zurück in den Kühlschrank und geht einkaufen.",
                "wischt er die Arbeitsfläche ab und räumt das Geschirr in den Schrank.",
                "gibt er die Zwiebelstücke in eine heiße Pfanne mit Olivenöl.",
                "stellt er den Backofen auf 200 Grad und schiebt ein leeres Blech hinein.",
            ],
            "answer_idx": 2,
        },
        {
            "context": "Eine Studentin wacht morgens auf und stellt fest, dass ihr Wecker nicht geklingelt hat. Sie",
            "correct_ending": "schaut erschrocken auf die Uhr und springt schnell aus dem Bett.",
            "choices": [
                "schaut erschrocken auf die Uhr und springt schnell aus dem Bett.",
                "dreht sich gemütlich um und schläft noch eine Stunde weiter.",
                "ruft sofort ihre Mutter an und beschwert sich über den Wecker.",
                "steht auf und fängt an, das Badezimmer zu putzen.",
            ],
            "answer_idx": 0,
        },
        {
            "context": "Ein älteres Ehepaar spaziert an einem Sonntagmorgen durch den Park. Als sie an den See kommen,",
            "correct_ending": "setzen sie sich auf eine Bank und beobachten die Enten auf dem Wasser.",
            "choices": [
                "ziehen sie ihre Schuhe aus und waten durch das kalte Wasser.",
                "setzen sie sich auf eine Bank und beobachten die Enten auf dem Wasser.",
                "holen sie ihre Angelruten heraus und fangen an zu fischen.",
                "laufen sie schnell zurück zum Auto, weil es anfängt zu regnen.",
            ],
            "answer_idx": 1,
        },
        {
            "context": "Ein Mann steht an der Kasse im Supermarkt und legt seine Einkäufe auf das Band. Als er bezahlen will,",
            "correct_ending": "zückt er seine Bankkarte und hält sie an das Lesegerät.",
            "choices": [
                "merkt er, dass er sein Portemonnaie zu Hause vergessen hat, und lässt alles stehen.",
                "fängt er an, mit der Kassiererin über das Wetter zu reden.",
                "packt er alles wieder ein und stellt es zurück ins Regal.",
                "zückt er seine Bankkarte und hält sie an das Lesegerät.",
            ],
            "answer_idx": 3,
        },
        {
            "context": "Zwei Kinder spielen im Garten Fußball. Eines von ihnen schießt den Ball über den Zaun. Das andere",
            "correct_ending": "klettert über den Zaun, um den Ball aus dem Nachbargarten zu holen.",
            "choices": [
                "setzt sich ins Gras und wartet, bis jemand den Ball zurückbringt.",
                "fängt an, mit Steinen statt mit dem Ball weiterzuspielen.",
                "klettert über den Zaun, um den Ball aus dem Nachbargarten zu holen.",
                "tritt gegen den Zaun und fängt an zu weinen.",
            ],
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
    choices = [doc["correct_ending"]] + doc[distractor_key]

    # Deterministic shuffle based on seed_id (using md5 for consistent hash across sessions)
    seed_str = doc.get("seed_id", doc["context"])
    seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16)
    rng = random.Random(seed)
    rng.shuffle(choices)

    doc["choices"] = choices
    doc["answer_idx"] = choices.index(doc["correct_ending"])
    return doc


def prepare_all(dataset):
    return dataset.map(lambda x: _prepare_choices(x, distractor_key="hard_distractors"))


def prepare_easy(dataset):
    return dataset.map(lambda x: _prepare_choices(x, distractor_key="easy_distractors"))


def process_results_bpb(doc, results):
    """Compute answer-only BPB (OLMES-style) for HellaSwag."""
    ll, _ = results[0]
    gold_text = doc["correct_ending"]
    gold_bytes = len((" " + gold_text).encode("utf-8"))
    return {"bits_per_byte": -ll / (math.log(2) * max(gold_bytes, 1))}
