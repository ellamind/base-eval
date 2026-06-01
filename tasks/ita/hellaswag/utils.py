import hashlib
import math
import random


def list_fewshot_samples():
    """5-shot curated examples for HellaSwag (Italian).

    Hand-written examples covering everyday scenarios, designed to sound
    natural in Italian rather than translated from English.
    """
    return [
        {
            "context": "Un cuoco è in cucina e taglia le cipolle su un tagliere di legno. Poi,",
            "correct_ending": "mette i pezzi di cipolla in una padella calda con olio d'oliva.",
            "choices": [
                "rimette le cipolle nel frigorifero e va a fare la spesa.",
                "pulisce il piano di lavoro e ripone i piatti nell'armadietto.",
                "mette i pezzi di cipolla in una padella calda con olio d'oliva.",
                "imposta il forno a 200 gradi e vi infila una teglia vuota.",
            ],
            "answer_idx": 2,
        },
        {
            "context": "Una studentessa si sveglia la mattina e si accorge che la sveglia non ha suonato. Lei",
            "correct_ending": "guarda l'ora spaventata e salta giù dal letto.",
            "choices": [
                "guarda l'ora spaventata e salta giù dal letto.",
                "si gira tranquillamente e dorme ancora un'ora.",
                "chiama subito sua madre per lamentarsi della sveglia.",
                "si alza e comincia a pulire il bagno.",
            ],
            "answer_idx": 0,
        },
        {
            "context": "Una coppia di anziani passeggia nel parco una domenica mattina. Arrivati al lago,",
            "correct_ending": "si siedono su una panchina e osservano le anatre sull'acqua.",
            "choices": [
                "si tolgono le scarpe e camminano nell'acqua fredda.",
                "si siedono su una panchina e osservano le anatre sull'acqua.",
                "tirano fuori le canne da pesca e si mettono a pescare.",
                "tornano in fretta alla macchina perché inizia a piovere.",
            ],
            "answer_idx": 1,
        },
        {
            "context": "Un uomo è alla cassa del supermercato e mette la spesa sul nastro. Al momento di pagare,",
            "correct_ending": "tira fuori la carta bancaria e la avvicina al lettore.",
            "choices": [
                "si accorge di aver dimenticato il portafoglio a casa e lascia tutto lì.",
                "si mette a parlare del tempo con la cassiera.",
                "rimette tutto a posto sugli scaffali.",
                "tira fuori la carta bancaria e la avvicina al lettore.",
            ],
            "answer_idx": 3,
        },
        {
            "context": "Due bambini giocano a calcio in giardino. Uno di loro tira il pallone oltre la recinzione. L'altro",
            "correct_ending": "scavalca la recinzione per recuperare il pallone dal giardino del vicino.",
            "choices": [
                "si siede sull'erba e aspetta che qualcuno riporti il pallone.",
                "comincia a giocare con i sassi invece che con il pallone.",
                "scavalca la recinzione per recuperare il pallone dal giardino del vicino.",
                "dà un calcio alla recinzione e si mette a piangere.",
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
