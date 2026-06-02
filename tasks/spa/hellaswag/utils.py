import hashlib
import math
import random


def list_fewshot_samples():
    """5-shot curated examples for HellaSwag (Spanish).

    Hand-written examples covering everyday scenarios, designed to sound
    natural in Spanish rather than translated from English.
    """
    return [
        {
            "context": "Un cocinero está en la cocina cortando cebollas sobre una tabla de madera. Después,",
            "correct_ending": "echa los trozos de cebolla en una sartén caliente con aceite de oliva.",
            "choices": [
                "vuelve a meter las cebollas en el refrigerador y se va a hacer la compra.",
                "limpia la encimera y guarda los platos en el armario.",
                "echa los trozos de cebolla en una sartén caliente con aceite de oliva.",
                "pone el horno a 200 grados y mete dentro una bandeja vacía.",
            ],
            "answer_idx": 2,
        },
        {
            "context": "Una estudiante se despierta por la mañana y se da cuenta de que su despertador no ha sonado. Ella",
            "correct_ending": "mira el reloj asustada y salta rápidamente de la cama.",
            "choices": [
                "mira el reloj asustada y salta rápidamente de la cama.",
                "se da la vuelta tranquilamente y sigue durmiendo una hora más.",
                "llama enseguida a su madre para quejarse del despertador.",
                "se levanta y empieza a limpiar el baño.",
            ],
            "answer_idx": 0,
        },
        {
            "context": "Una pareja de ancianos pasea por el parque un domingo por la mañana. Cuando llegan al lago,",
            "correct_ending": "se sientan en un banco y observan a los patos en el agua.",
            "choices": [
                "se quitan los zapatos y se meten en el agua fría.",
                "se sientan en un banco y observan a los patos en el agua.",
                "sacan sus cañas de pescar y se ponen a pescar.",
                "vuelven corriendo al coche porque empieza a llover.",
            ],
            "answer_idx": 1,
        },
        {
            "context": "Un hombre está en la caja del supermercado y pone sus compras en la cinta. Cuando va a pagar,",
            "correct_ending": "saca su tarjeta bancaria y la acerca al lector.",
            "choices": [
                "se da cuenta de que ha olvidado la cartera en casa y lo deja todo allí.",
                "se pone a hablar del tiempo con la cajera.",
                "vuelve a meter todo y lo devuelve a las estanterías.",
                "saca su tarjeta bancaria y la acerca al lector.",
            ],
            "answer_idx": 3,
        },
        {
            "context": "Dos niños juegan al fútbol en el jardín. Uno de ellos lanza el balón por encima de la valla. El otro",
            "correct_ending": "trepa por la valla para recuperar el balón del jardín del vecino.",
            "choices": [
                "se sienta en la hierba y espera a que alguien le devuelva el balón.",
                "empieza a jugar con piedras en lugar del balón.",
                "trepa por la valla para recuperar el balón del jardín del vecino.",
                "da una patada a la valla y se pone a llorar.",
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
    return {"answer_bits_per_byte": -ll / (math.log(2) * max(gold_bytes, 1))}
