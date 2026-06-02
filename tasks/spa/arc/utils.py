import math


def filter_easy(dataset):
    return dataset.filter(lambda x: x["arc_config"] == "ARC-Easy")


def filter_challenge(dataset):
    return dataset.filter(lambda x: x["arc_config"] == "ARC-Challenge")


def list_fewshot_easy():
    """5-shot curated examples for ARC Easy (Spanish).

    Translated from the OLMES curated set (OLMES:ARC-Easy).
    """
    return [
        {
            "question": "Los líquenes son organismos simbióticos formados por algas verdes y hongos. ¿Qué proporcionan las algas verdes a los hongos en esta relación simbiótica?",
            "choices": ["Dióxido de carbono", "Alimento", "Protección", "Agua"],
            "answer_key": "B",
        },
        {
            "question": "Cuando se usa un interruptor en un circuito eléctrico, el interruptor puede",
            "choices": [
                "acumular la carga.",
                "aumentar y disminuir el voltaje.",
                "cambiar la dirección de la corriente.",
                "detener e iniciar el flujo de corriente.",
            ],
            "answer_key": "D",
        },
        {
            "question": "¿Cuál es un ejemplo de dispositivo médico?",
            "choices": ["una lente de contacto", "una motocicleta", "un impermeable", "una cafetera"],
            "answer_key": "A",
        },
        {
            "question": "Las rocas se clasifican como ígneas, metamórficas o sedimentarias según",
            "choices": [
                "su color",
                "su forma",
                "cómo se formaron",
                "los minerales que contienen",
            ],
            "answer_key": "C",
        },
        {
            "question": "Una tableta masticable de carbonato de calcio es un remedio común para las molestias estomacales. El carbonato de calcio se usa muy probablemente como medicamento porque el carbonato de calcio",
            "choices": [
                "tiene un sabor agradable.",
                "es barato de fabricar.",
                "neutraliza el ácido del estómago.",
                "se encuentra naturalmente en el cuerpo.",
            ],
            "answer_key": "C",
        },
    ]


def list_fewshot_challenge():
    """5-shot curated examples for ARC Challenge (Spanish).

    Translated from the OLMES curated set (OLMES:ARC-Challenge).
    """
    return [
        {
            "question": "Jorge quiere calentarse las manos rápidamente frotándolas. ¿Qué superficie de la mano producirá más calor?",
            "choices": [
                "palmas secas",
                "palmas mojadas",
                "palmas cubiertas de aceite",
                "palmas cubiertas de loción",
            ],
            "answer_key": "A",
        },
        {
            "question": "Un barco es impulsado por una corriente de río que fluye hacia el norte y por el viento en sus velas. El barco se desplaza hacia el noreste. ¿En qué dirección está aplicando el viento con mayor probabilidad una fuerza sobre las velas del barco?",
            "choices": ["oeste", "este", "norte", "sur"],
            "answer_key": "B",
        },
        {
            "question": "¿Qué accidente geográfico se crea por la fuerza constructiva de un glaciar?",
            "choices": [
                "valles excavados por un glaciar en movimiento",
                "montones de rocas depositados por un glaciar que se derrite",
                "surcos creados en la roca por un glaciar",
                "colinas de lecho rocoso desgastadas por un glaciar",
            ],
            "answer_key": "B",
        },
        {
            "question": "¿Cuál de estos reflejos ocurre exclusivamente en los seres humanos?",
            "choices": ["estornudar", "estremecerse", "llorar", "parpadear"],
            "answer_key": "C",
        },
        {
            "question": "El uso de recursos no renovables para obtener energía produce desechos que pueden tener efectos negativos a largo plazo en los subsistemas de la Tierra. ¿Qué fuente de energía produce desechos cuyos efectos pueden durar más tiempo?",
            "choices": ["gas natural", "uranio", "petróleo crudo", "carbón"],
            "answer_key": "B",
        },
    ]


def process_results_bpb(doc, results):
    """Compute answer-only bits-per-byte (OLMES-style).

    BPB is computed over the *gold answer text only*, conditioned on the
    question context.  This matches the OLMES definition:
        BPB = -log_2 P(answer | context) / bytes(answer)

    Uses ``output_type: loglikelihood`` which sends a single
    (context, gold_answer) request -- only the correct answer is scored.
    ``results`` is [(loglikelihood, is_greedy)].
    """
    ll, _is_greedy = results[0]

    gold = "ABCDE".index(doc["answer_key"])
    gold_text = doc["choices"][gold]
    # Include leading space to match the scored continuation (" answer")
    gold_bytes = len((" " + gold_text).encode("utf-8"))

    bpb = -ll / (math.log(2) * max(gold_bytes, 1))

    return {
        "answer_bits_per_byte": bpb,
    }
