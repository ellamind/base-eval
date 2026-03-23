import math


def filter_easy(dataset):
    return dataset.filter(lambda x: x["arc_config"] == "ARC-Easy")


def filter_challenge(dataset):
    return dataset.filter(lambda x: x["arc_config"] == "ARC-Challenge")


def list_fewshot_easy():
    """5-shot curated examples for ARC Easy (German).

    Translated from the OLMES curated set (OLMES:ARC-Easy).
    """
    return [
        {
            "question": "Flechten sind symbiotische Organismen aus Grünalgen und Pilzen. Was liefern die Grünalgen den Pilzen in dieser symbiotischen Beziehung?",
            "choices": ["Kohlendioxid", "Nahrung", "Schutz", "Wasser"],
            "answer_key": "B",
        },
        {
            "question": "Wenn ein Schalter in einem Stromkreis verwendet wird, kann der Schalter",
            "choices": [
                "die Ladung aufbauen.",
                "die Spannung erhöhen und verringern.",
                "die Stromrichtung ändern.",
                "den Stromfluss stoppen und starten.",
            ],
            "answer_key": "D",
        },
        {
            "question": "Was ist ein Beispiel für ein medizinisches Hilfsmittel?",
            "choices": ["Kontaktlinse", "Motorrad", "Regenmantel", "Kaffeekanne"],
            "answer_key": "A",
        },
        {
            "question": "Gesteine werden als magmatisch, metamorph oder sedimentär klassifiziert nach",
            "choices": [
                "ihrer Farbe",
                "ihrer Form",
                "wie sie entstanden sind",
                "den Mineralien, die sie enthalten",
            ],
            "answer_key": "C",
        },
        {
            "question": "Eine Kalziumkarbonat-Kautablette ist ein gängiges Mittel gegen Magenbeschwerden. Kalziumkarbonat wird höchstwahrscheinlich als Medikament verwendet, weil Kalziumkarbonat",
            "choices": [
                "einen angenehmen Geschmack hat.",
                "günstig herzustellen ist.",
                "Magensäure neutralisiert.",
                "natürlich im Körper vorkommt.",
            ],
            "answer_key": "C",
        },
    ]


def list_fewshot_challenge():
    """5-shot curated examples for ARC Challenge (German).

    Translated from the OLMES curated set (OLMES:ARC-Challenge).
    """
    return [
        {
            "question": "Georg möchte seine Hände schnell durch Reiben aufwärmen. Welche Handoberfläche erzeugt die meiste Wärme?",
            "choices": [
                "trockene Handflächen",
                "nasse Handflächen",
                "mit Öl bedeckte Handflächen",
                "mit Lotion bedeckte Handflächen",
            ],
            "answer_key": "A",
        },
        {
            "question": "Ein Boot wird von einer nach Norden fließenden Strömung und vom Wind in seinen Segeln angetrieben. Das Boot fährt nach Nordosten. In welche Richtung übt der Wind höchstwahrscheinlich Kraft auf die Segel des Bootes aus?",
            "choices": ["Westen", "Osten", "Norden", "Süden"],
            "answer_key": "B",
        },
        {
            "question": "Welche Landschaftsform entsteht durch die konstruktive Kraft eines Gletschers?",
            "choices": [
                "Täler, die von einem sich bewegenden Gletscher ausgehöhlt wurden",
                "Gesteinsansammlungen, die von einem schmelzenden Gletscher abgelagert wurden",
                "Rillen, die von einem Gletscher in Fels erzeugt wurden",
                "Grundgebirgshügel, die von einem Gletscher aufgeraut wurden",
            ],
            "answer_key": "B",
        },
        {
            "question": "Welcher der folgenden Reflexe tritt ausschließlich beim Menschen auf?",
            "choices": ["Niesen", "Zusammenzucken", "Weinen", "Blinzeln"],
            "answer_key": "C",
        },
        {
            "question": "Die Nutzung nicht erneuerbarer Ressourcen zur Energiegewinnung erzeugt Abfallprodukte, die langfristige negative Auswirkungen auf die Teilsysteme der Erde haben können. Welche Energiequelle erzeugt Abfallprodukte, die diese Auswirkungen am längsten haben können?",
            "choices": ["Erdgas", "Uran", "Rohöl", "Kohle"],
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
        "bits_per_byte": bpb,
    }
