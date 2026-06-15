import math


def filter_easy(dataset):
    return dataset.filter(lambda x: x["arc_config"] == "ARC-Easy")


def filter_challenge(dataset):
    return dataset.filter(lambda x: x["arc_config"] == "ARC-Challenge")


def list_fewshot_easy():
    """5-shot curated examples for ARC Easy (French).

    Translated from the OLMES curated set (OLMES:ARC-Easy).
    """
    return [
        {
            "question": "Les lichens sont des organismes symbiotiques composés d'algues vertes et de champignons. Que fournissent les algues vertes aux champignons dans cette relation symbiotique ?",
            "choices": ["Du dioxyde de carbone", "De la nourriture", "Une protection", "De l'eau"],
            "answer_key": "B",
        },
        {
            "question": "Lorsqu'un interrupteur est utilisé dans un circuit électrique, l'interrupteur peut",
            "choices": [
                "accumuler la charge.",
                "augmenter et diminuer la tension.",
                "changer le sens du courant.",
                "arrêter et démarrer le flux de courant.",
            ],
            "answer_key": "D",
        },
        {
            "question": "Lequel est un exemple de dispositif médical ?",
            "choices": ["une lentille de contact", "une moto", "un imperméable", "une cafetière"],
            "answer_key": "A",
        },
        {
            "question": "Les roches sont classées comme ignées, métamorphiques ou sédimentaires selon",
            "choices": [
                "leur couleur",
                "leur forme",
                "la façon dont elles se sont formées",
                "les minéraux qu'elles contiennent",
            ],
            "answer_key": "C",
        },
        {
            "question": "Un comprimé à croquer de carbonate de calcium est un remède courant contre les maux d'estomac. Le carbonate de calcium est très probablement utilisé comme médicament parce que le carbonate de calcium",
            "choices": [
                "a un goût agréable.",
                "est peu coûteux à produire.",
                "neutralise l'acide gastrique.",
                "est naturellement présent dans le corps.",
            ],
            "answer_key": "C",
        },
    ]


def list_fewshot_challenge():
    """5-shot curated examples for ARC Challenge (French).

    Translated from the OLMES curated set (OLMES:ARC-Challenge).
    """
    return [
        {
            "question": "Georges veut réchauffer rapidement ses mains en les frottant. Quelle surface de la main produira le plus de chaleur ?",
            "choices": [
                "des paumes sèches",
                "des paumes mouillées",
                "des paumes recouvertes d'huile",
                "des paumes recouvertes de lotion",
            ],
            "answer_key": "A",
        },
        {
            "question": "Un bateau est poussé par un courant de rivière qui s'écoule vers le nord et par le vent dans ses voiles. Le bateau se déplace vers le nord-est. Dans quelle direction le vent applique-t-il le plus probablement une force sur les voiles du bateau ?",
            "choices": ["l'ouest", "l'est", "le nord", "le sud"],
            "answer_key": "B",
        },
        {
            "question": "Quelle forme de relief est créée par la force constructive d'un glacier ?",
            "choices": [
                "des vallées creusées par un glacier en mouvement",
                "des amas de roches déposés par un glacier en fonte",
                "des sillons creusés dans la roche par un glacier",
                "des collines de socle rocheux rendues rugueuses par un glacier",
            ],
            "answer_key": "B",
        },
        {
            "question": "Lequel de ces réflexes se produit exclusivement chez l'être humain ?",
            "choices": ["éternuer", "tressaillir", "pleurer", "cligner des yeux"],
            "answer_key": "C",
        },
        {
            "question": "L'utilisation de ressources non renouvelables pour produire de l'énergie génère des déchets qui peuvent avoir des effets négatifs à long terme sur les sous-systèmes de la Terre. Quelle source d'énergie produit des déchets dont les effets peuvent durer le plus longtemps ?",
            "choices": ["le gaz naturel", "l'uranium", "le pétrole brut", "le charbon"],
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
