import hashlib
import math
import random


def list_fewshot_samples():
    """5-shot curated examples for SocialIQA (German).

    Hand-written examples covering social situations, designed to sound
    natural in German rather than translated from English.
    """
    return [
        {
            "context": "Lena wollte abends nicht allein nach Hause laufen, weil die Straßen schlecht beleuchtet waren.",
            "question": "Was muss Lena vorher tun?",
            "correct_answer": "Sich eine sichere Möglichkeit suchen, nach Hause zu kommen",
            "choices": [
                "Einfach trotzdem allein losgehen",
                "Sich eine sichere Möglichkeit suchen, nach Hause zu kommen",
                "Bis zum nächsten Morgen auf der Arbeit bleiben",
            ],
            "answer_idx": 1,
        },
        {
            "context": "Katharina organisierte ein großes Sommerfest und lud alle ihre Freunde und Nachbarn ein.",
            "question": "Wie würdest du Katharina beschreiben?",
            "correct_answer": "Gesellig",
            "choices": [
                "Gesellig",
                "Schüchtern",
                "Gleichgültig",
            ],
            "answer_idx": 0,
        },
        {
            "context": "Anton sagte vor Gericht gegen seinen ehemaligen Geschäftspartner Julian aus.",
            "question": "Was wird wahrscheinlich zwischen Anton und Julian passieren?",
            "correct_answer": "Julian wird Anton das nicht verzeihen",
            "choices": [
                "Julian wird Anton das nicht verzeihen",
                "Julian wird Anton zum Essen einladen",
                "Sie werden beste Freunde",
            ],
            "answer_idx": 0,
        },
        {
            "context": "Tobias zahlte jeden Monat pünktlich seine Miete und hielt die Wohnung stets in gutem Zustand.",
            "question": "Wie würdest du Tobias beschreiben?",
            "correct_answer": "Verantwortungsbewusst",
            "choices": [
                "Nachlässig",
                "Verantwortungsbewusst",
                "Verschwenderisch",
            ],
            "answer_idx": 1,
        },
        {
            "context": "Luisa nahm ihre neue Kollegin mit zur Geburtstagsfeier, weil diese in der Stadt noch niemanden kannte.",
            "question": "Warum hat Luisa das getan?",
            "correct_answer": "Um ihr zu helfen, Anschluss zu finden",
            "choices": [
                "Um sie vor den anderen bloßzustellen",
                "Um selbst nicht allein hingehen zu müssen",
                "Um ihr zu helfen, Anschluss zu finden",
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
    """Compute answer-only BPB (OLMES-style) for SocialIQA."""
    ll, _ = results[0]
    gold_text = doc["correct_answer"]
    gold_bytes = len((" " + gold_text).encode("utf-8"))
    return {"bits_per_byte": -ll / (math.log(2) * max(gold_bytes, 1))}
