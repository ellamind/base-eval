import hashlib
import math
import random


# ---------------------------------------------------------------------------
# Subset filters — one per AGIEval subset
# ---------------------------------------------------------------------------

SUBSETS = [
    "aqua_rat",
    "gaokao_english",
    "logiqa_en",
    "lsat_ar",
    "lsat_lr",
    "lsat_rc",
    "sat_en",
    "sat_en_without_passage",
    "sat_math",
]


def _filter_subset(dataset, subset):
    return dataset.filter(lambda x: x["subset"] == subset)


def filter_aqua_rat(dataset):
    return _prepare(_filter_subset(dataset, "aqua_rat"))


def filter_gaokao_english(dataset):
    return _prepare(_filter_subset(dataset, "gaokao_english"))


def filter_logiqa_en(dataset):
    return _prepare(_filter_subset(dataset, "logiqa_en"))


def filter_lsat_ar(dataset):
    return _prepare(_filter_subset(dataset, "lsat_ar"))


def filter_lsat_lr(dataset):
    return _prepare(_filter_subset(dataset, "lsat_lr"))


def filter_lsat_rc(dataset):
    return _prepare(_filter_subset(dataset, "lsat_rc"))


def filter_sat_en(dataset):
    return _prepare(_filter_subset(dataset, "sat_en"))


def filter_sat_en_without_passage(dataset):
    return _prepare(_filter_subset(dataset, "sat_en_without_passage"))


def filter_sat_math(dataset):
    return _prepare(_filter_subset(dataset, "sat_math"))


# ---------------------------------------------------------------------------
# Choice assembly & deterministic shuffle
# ---------------------------------------------------------------------------


def _prepare(dataset):
    """Combine correct_answer + incorrect_answers into a shuffled choices list."""
    return dataset.map(_prepare_doc)


def _prepare_doc(doc):
    choices = [doc["correct_answer"]] + doc["incorrect_answers"]

    # Deterministic shuffle keyed on question id
    seed_str = doc.get("id", doc["question"])
    seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16)
    rng = random.Random(seed)
    rng.shuffle(choices)

    doc["choices"] = choices
    doc["answer_idx"] = choices.index(doc["correct_answer"])
    return doc


# ---------------------------------------------------------------------------
# BPB processing (answer-only bits-per-byte, OLMES-style)
# ---------------------------------------------------------------------------


def process_results_bpb(doc, results):
    """BPB = -log2 P(answer | context) / bytes(answer)."""
    ll, _is_greedy = results[0]
    gold_text = doc["correct_answer"]
    gold_bytes = len((" " + gold_text).encode("utf-8"))
    return {"answer_bits_per_byte": -ll / (math.log(2) * max(gold_bytes, 1))}


# ---------------------------------------------------------------------------
# Few-shot examples — curated per subset, following OLMES prompt style
#
# Format matches the processed doc schema:
#   question, choices (shuffled), answer_idx, correct_answer
#
# 3-shot for most subsets (matching OLMES default), 5-shot for sat_math
# and aqua_rat (matching OLMES).
# ---------------------------------------------------------------------------


def list_fewshot_aqua_rat():
    """5-shot curated examples for AQUA-RAT (French)."""
    return [
        {
            "question": "Un train roule a 60 km/h et parcourt une distance de 240 km. Combien de temps dure le trajet ?",
            "correct_answer": "4 heures",
            "choices": ["2 heures", "4 heures", "6 heures", "3 heures", "5 heures"],
            "answer_idx": 1,
        },
        {
            "question": "Si 6 ouvriers mettent 8 jours pour construire un mur, combien de jours faut-il a 12 ouvriers pour construire le meme mur ?",
            "correct_answer": "4 jours",
            "choices": ["4 jours", "6 jours", "8 jours", "2 jours", "10 jours"],
            "answer_idx": 0,
        },
        {
            "question": "Un commercant achete un article pour 80 euros et le revend avec un benefice de 25 %. Quel est le prix de vente ?",
            "correct_answer": "100 euros",
            "choices": ["90 euros", "95 euros", "100 euros", "105 euros", "110 euros"],
            "answer_idx": 2,
        },
        {
            "question": "Le rapport entre les garcons et les filles dans une classe est de 3:5. S'il y a 24 garcons, combien y a-t-il de filles ?",
            "correct_answer": "40",
            "choices": ["30", "35", "40", "45", "50"],
            "answer_idx": 2,
        },
        {
            "question": "Quel est l'interet simple sur 5000 euros a un taux de 4 % pendant 3 ans ?",
            "correct_answer": "600 euros",
            "choices": ["400 euros", "500 euros", "600 euros", "700 euros", "800 euros"],
            "answer_idx": 2,
        },
    ]


def list_fewshot_gaokao_english():
    """3-shot curated examples for Gaokao English (French)."""
    return [
        {
            "question": "TELECOMMANDE-MONTRE\nVoici une montre que James Bond porterait avec fierte ! Votre TELECOMMANDE-MONTRE electronique PENGO sert de telecommande pour televiseurs et magnetoscopes.\nA l'aide d'un Mr. H, vous pouvez ___.",
            "correct_answer": "faire vos devoirs a temps.",
            "choices": [
                "arreter d'utiliser des piles.",
                "faire vos devoirs a temps.",
                "retenir les instructions de votre professeur.",
                "faire ranger votre chambre en rentrant chez vous.",
            ],
            "answer_idx": 1,
        },
        {
            "question": "Un panneau sur la porte d'un magasin indique : 'Nous sommes ouverts pour vous 7 jours sur 7 et 365 jours par an.' Quel est l'objectif principal de ce panneau ?",
            "correct_answer": "Communiquer les horaires d'ouverture du magasin.",
            "choices": [
                "Communiquer les horaires d'ouverture du magasin.",
                "Recruter de nouveaux employes.",
                "Faire la publicite d'un nouveau produit.",
                "Signaler un changement de prix.",
            ],
            "answer_idx": 0,
        },
        {
            "question": "Selon le texte, la principale raison pour laquelle les gens font du benevolat est ___.",
            "correct_answer": "qu'ils souhaitent aider les autres et redonner a la communaute",
            "choices": [
                "qu'ils veulent gagner de l'argent",
                "qu'ils souhaitent aider les autres et redonner a la communaute",
                "qu'ils veulent acquerir de nouvelles competences",
                "qu'ils veulent nouer des contacts professionnels",
            ],
            "answer_idx": 1,
        },
    ]


def list_fewshot_logiqa_en():
    """3-shot curated examples for LogiQA (French)."""
    return [
        {
            "question": "Dans un bureau, quatre personnes sont assises : A, B, C et D. A est assis en face de B. C est assis a droite de A. Qui est assis a gauche de B ?",
            "correct_answer": "C",
            "choices": ["A", "C", "D", "Personne"],
            "answer_idx": 1,
        },
        {
            "question": "Tous les philosophes sont des penseurs. Certains penseurs sont des ecrivains. Quelle conclusion est necessairement correcte ?",
            "correct_answer": "Certains philosophes pourraient etre des ecrivains.",
            "choices": [
                "Tous les ecrivains sont des philosophes.",
                "Aucun philosophe n'est ecrivain.",
                "Certains philosophes pourraient etre des ecrivains.",
                "Tous les penseurs sont des philosophes.",
            ],
            "answer_idx": 2,
        },
        {
            "question": "S'il pleut, la route est mouillee. La route est mouillee. Quelle conclusion est correcte ?",
            "correct_answer": "On ne peut pas savoir avec certitude s'il a plu.",
            "choices": [
                "Il a plu.",
                "Il n'a pas plu.",
                "On ne peut pas savoir avec certitude s'il a plu.",
                "La route a ete nettoyee.",
            ],
            "answer_idx": 2,
        },
    ]


def list_fewshot_lsat_ar():
    """3-shot curated examples for LSAT-AR (French)."""
    return [
        {
            "question": "Cinq conferences -- F, G, H, J et K -- sont donnees successivement au cours d'une journee. G est donnee avant H. J est donnee immediatement apres F. Quel ordre est possible ?",
            "correct_answer": "G, H, F, J, K",
            "choices": [
                "F, J, G, K, H",
                "G, H, F, J, K",
                "H, G, F, J, K",
                "J, F, G, H, K",
            ],
            "answer_idx": 1,
        },
        {
            "question": "Un fleuriste dispose sept bouquets -- S, T, U, V, W, X et Y -- en une rangee. V est en troisieme position. T est immediatement a gauche de U. Quel bouquet pourrait etre en premiere position ?",
            "correct_answer": "T",
            "choices": ["U", "V", "T", "Y"],
            "answer_idx": 2,
        },
        {
            "question": "Trois equipes -- Rouge, Bleu et Vert -- jouent chacune deux matchs. Rouge joue avant Bleu. Vert ne joue pas en premier. Quel ordre des premiers matchs est possible ?",
            "correct_answer": "Rouge, Vert, Bleu",
            "choices": [
                "Bleu, Rouge, Vert",
                "Vert, Rouge, Bleu",
                "Rouge, Vert, Bleu",
                "Rouge, Bleu, Vert",
            ],
            "answer_idx": 2,
        },
    ]


def list_fewshot_lsat_lr():
    """3-shot curated examples for LSAT-LR (French)."""
    return [
        {
            "question": "Editorial : Comme la population vieillit, les couts de sante augmentent. Par consequent, le gouvernement doit investir davantage dans la prevention. Quelle hypothese sous-tend cet argument ?",
            "correct_answer": "Les mesures de prevention peuvent reduire les couts de sante d'une population vieillissante.",
            "choices": [
                "La population ne vieillira plus a l'avenir.",
                "Les mesures de prevention peuvent reduire les couts de sante d'une population vieillissante.",
                "Le gouvernement ne depense actuellement rien pour la prevention.",
                "L'augmentation des couts de sante est inevitable.",
            ],
            "answer_idx": 1,
        },
        {
            "question": "Critique : Ce musee n'expose que des oeuvres d'artistes celebres. Par consequent, il neglige la promotion des talents emergents. Quelle affirmation affaiblit le plus cet argument ?",
            "correct_answer": "Le musee dispose d'un espace d'exposition dedie aux nouveaux artistes.",
            "choices": [
                "Les artistes celebres attirent davantage de visiteurs.",
                "Le musee dispose d'un espace d'exposition dedie aux nouveaux artistes.",
                "D'autres musees n'exposent egalement que des artistes celebres.",
                "Les artistes emergents preferent les petites galeries.",
            ],
            "answer_idx": 1,
        },
        {
            "question": "Si tous les employes d'une entreprise sont ponctuels et que Stefan est un employe de cette entreprise, alors Stefan doit etre ponctuel. Stefan arrive souvent en retard. Qu'en decoule-t-il ?",
            "correct_answer": "Tous les employes de cette entreprise ne sont pas ponctuels.",
            "choices": [
                "Stefan n'est pas un employe de l'entreprise.",
                "Tous les employes de cette entreprise ne sont pas ponctuels.",
                "Stefan est toujours ponctuel.",
                "L'entreprise n'a pas de regles de ponctualite.",
            ],
            "answer_idx": 1,
        },
    ]


def list_fewshot_lsat_rc():
    """3-shot curated examples for LSAT-RC (French)."""
    return [
        {
            "question": "Les avocats ont le devoir de defendre au mieux leurs clients. En meme temps, ils ont une responsabilite envers la societe. Quelle affirmation decrit le mieux l'idee principale du texte ?",
            "correct_answer": "Les avocats doivent tenir compte a la fois des interets de leurs clients et de ceux de la societe.",
            "choices": [
                "Les avocats ne devraient representer que les interets de leurs clients.",
                "Les avocats doivent tenir compte a la fois des interets de leurs clients et de ceux de la societe.",
                "La societe devrait controler davantage le travail des avocats.",
                "Les clients devraient pouvoir choisir librement leurs avocats.",
            ],
            "answer_idx": 1,
        },
        {
            "question": "L'auteur discute de differentes approches de la reforme du droit d'auteur. Quelle position l'auteur defend-il principalement ?",
            "correct_answer": "Une approche equilibree, protegeant a la fois les auteurs et le public, est necessaire.",
            "choices": [
                "Le droit d'auteur devrait etre entierement aboli.",
                "Une approche equilibree, protegeant a la fois les auteurs et le public, est necessaire.",
                "Seules les entreprises devraient pouvoir detenir des droits d'auteur.",
                "Le systeme actuel fonctionne parfaitement.",
            ],
            "answer_idx": 1,
        },
        {
            "question": "Le texte decrit l'evolution de la legislation environnementale. Selon le texte, quelle a ete la principale raison du durcissement des lois ?",
            "correct_answer": "L'accumulation de connaissances scientifiques sur les dommages environnementaux.",
            "choices": [
                "L'accumulation de connaissances scientifiques sur les dommages environnementaux.",
                "Les interets economiques de l'industrie.",
                "Les accords politiques internationaux.",
                "Les protestations de citoyens individuels.",
            ],
            "answer_idx": 0,
        },
    ]


def list_fewshot_sat_en():
    """3-shot curated examples for SAT-EN (French)."""
    return [
        {
            "question": "Akira arriva directement et rompit avec toute tradition. Il frappa a la porte par un soir d'hiver. 'Je voudrais epouser votre fille Naomi', dit-il. Quelle affirmation decrit le mieux ce qui se passe dans le texte ?",
            "correct_answer": "Un personnage recoit une demande surprenante d'un autre personnage.",
            "choices": [
                "Un personnage se dispute avec un autre personnage.",
                "Un personnage recoit une demande surprenante d'un autre personnage.",
                "Un personnage repense a des decisions passees.",
                "Un personnage critique un autre pour son comportement inattendu.",
            ],
            "answer_idx": 1,
        },
        {
            "question": "Le narrateur decrit un voyage a travers une ville etrangere. Les rues etaient etroites et les batiments anciens. Quelle est l'atmosphere principale du texte ?",
            "correct_answer": "Curiosite melee d'incertitude.",
            "choices": [
                "Joie et enthousiasme.",
                "Curiosite melee d'incertitude.",
                "Profonde tristesse.",
                "Colere et frustration.",
            ],
            "answer_idx": 1,
        },
        {
            "question": "Dans le texte, l'auteur compare deux theories scientifiques. Quel est l'objectif principal de cette comparaison ?",
            "correct_answer": "Mettre en evidence les forces et les faiblesses des deux approches.",
            "choices": [
                "Prouver qu'une theorie est fausse.",
                "Mettre en evidence les forces et les faiblesses des deux approches.",
                "Proposer une theorie entierement nouvelle.",
                "Resumer l'histoire de la science.",
            ],
            "answer_idx": 1,
        },
    ]


def list_fewshot_sat_en_without_passage():
    """3-shot curated examples for SAT-EN without passage (French)."""
    return [
        {
            "question": "Quelle affirmation decrit le mieux ce qui se passe dans le texte ?",
            "correct_answer": "Un personnage recoit une demande surprenante d'un autre personnage.",
            "choices": [
                "Un personnage se dispute avec un autre personnage.",
                "Un personnage recoit une demande surprenante d'un autre personnage.",
                "Un personnage repense a des decisions passees.",
                "Un personnage critique un autre pour son comportement inattendu.",
            ],
            "answer_idx": 1,
        },
        {
            "question": "Quel role joue le troisieme paragraphe dans l'ensemble du texte ?",
            "correct_answer": "Il fournit un exemple concret de la these avancee precedemment.",
            "choices": [
                "Il refute l'argument principal.",
                "Il fournit un exemple concret de la these avancee precedemment.",
                "Il introduit un sujet entierement nouveau.",
                "Il resume l'ensemble du texte.",
            ],
            "answer_idx": 1,
        },
        {
            "question": "Quel mot decrit le mieux le ton de l'auteur ?",
            "correct_answer": "objectif",
            "choices": ["enthousiaste", "objectif", "sarcastique", "indifferent"],
            "answer_idx": 1,
        },
    ]


def list_fewshot_sat_math():
    """5-shot curated examples for SAT-Math (French)."""
    return [
        {
            "question": "Si $\\frac{x-1}{3}=k$ et $k=3$, quelle est la valeur de $x$ ?",
            "correct_answer": "10",
            "choices": ["2", "4", "9", "10"],
            "answer_idx": 3,
        },
        {
            "question": "Si $3x + 2 = 14$, quelle est la valeur de $x$ ?",
            "correct_answer": "4",
            "choices": ["2", "3", "4", "6"],
            "answer_idx": 2,
        },
        {
            "question": "Une fonction est definie par $f(x) = 2x^2 - 3x + 1$. Quelle est la valeur de $f(2)$ ?",
            "correct_answer": "3",
            "choices": ["1", "3", "5", "7"],
            "answer_idx": 1,
        },
        {
            "question": "La circonference d'un cercle est de $10\\pi$. Quel est le rayon ?",
            "correct_answer": "5",
            "choices": ["3", "5", "10", "20"],
            "answer_idx": 1,
        },
        {
            "question": "Si $y = 3x - 7$ et $y = 5$, quelle est la valeur de $x$ ?",
            "correct_answer": "4",
            "choices": ["2", "3", "4", "5"],
            "answer_idx": 2,
        },
    ]
