import hashlib
import math
import random


def list_fewshot_samples():
    """5-shot curated examples for HellaSwag (French).

    Hand-written examples covering everyday scenarios, designed to sound
    natural in French rather than translated from English.
    """
    return [
        {
            "context": "Un cuisinier est dans la cuisine et coupe des oignons sur une planche en bois. Ensuite,",
            "correct_ending": "il met les morceaux d'oignon dans une poêle chaude avec de l'huile d'olive.",
            "choices": [
                "il remet les oignons dans le réfrigérateur et part faire les courses.",
                "il essuie le plan de travail et range la vaisselle dans le placard.",
                "il met les morceaux d'oignon dans une poêle chaude avec de l'huile d'olive.",
                "il règle le four à 200 degrés et y enfourne une plaque vide.",
            ],
            "answer_idx": 2,
        },
        {
            "context": "Une étudiante se réveille le matin et se rend compte que son réveil n'a pas sonné. Elle",
            "correct_ending": "regarde l'heure avec effroi et bondit hors du lit.",
            "choices": [
                "regarde l'heure avec effroi et bondit hors du lit.",
                "se retourne tranquillement et dort encore une heure.",
                "appelle aussitôt sa mère pour se plaindre du réveil.",
                "se lève et commence à nettoyer la salle de bain.",
            ],
            "answer_idx": 0,
        },
        {
            "context": "Un couple âgé se promène dans le parc un dimanche matin. En arrivant au bord du lac,",
            "correct_ending": "ils s'assoient sur un banc et observent les canards sur l'eau.",
            "choices": [
                "ils enlèvent leurs chaussures et marchent dans l'eau froide.",
                "ils s'assoient sur un banc et observent les canards sur l'eau.",
                "ils sortent leurs cannes à pêche et se mettent à pêcher.",
                "ils retournent vite à la voiture parce qu'il commence à pleuvoir.",
            ],
            "answer_idx": 1,
        },
        {
            "context": "Un homme est à la caisse du supermarché et pose ses achats sur le tapis. Au moment de payer,",
            "correct_ending": "il sort sa carte bancaire et l'approche du lecteur.",
            "choices": [
                "il se rend compte qu'il a oublié son portefeuille à la maison et laisse tout sur place.",
                "il se met à parler de la pluie et du beau temps avec la caissière.",
                "il remballe tout et le remet dans les rayons.",
                "il sort sa carte bancaire et l'approche du lecteur.",
            ],
            "answer_idx": 3,
        },
        {
            "context": "Deux enfants jouent au football dans le jardin. L'un d'eux envoie le ballon par-dessus la clôture. L'autre",
            "correct_ending": "escalade la clôture pour aller récupérer le ballon dans le jardin du voisin.",
            "choices": [
                "s'assoit dans l'herbe et attend que quelqu'un rapporte le ballon.",
                "se met à jouer avec des cailloux au lieu du ballon.",
                "escalade la clôture pour aller récupérer le ballon dans le jardin du voisin.",
                "donne un coup de pied dans la clôture et se met à pleurer.",
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
