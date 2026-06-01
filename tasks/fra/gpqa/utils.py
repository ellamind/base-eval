"""Utility functions for French GPQA tasks (MC and COT variants).

Dataset: ellamind/gpqa-multilingual (fra subset)
- 448 graduate-level STEM questions (Biology, Chemistry, Physics)
- Diamond subset: 198 hardest questions (is_diamond == True)
- Fields: question, correct_answer, incorrect_answers (list of 3)

MC tasks: 0-shot, multiple_choice output, acc/acc_norm metrics
COT tasks: 5-shot chain-of-thought, generate_until output, exact_match metric

Following the deu config: the COT instruction and "Let's think step by step" /
"The correct answer is:" scaffolding stay in English (format guidance, shared
across languages); only the few-shot example content is translated.
"""

import random

LABELS = "ABCD"
OLMES_SHUFFLE_SEED = 111


# ---------------------------------------------------------------------------
# Dataset processing
# ---------------------------------------------------------------------------

def _filter_flagged(dataset):
    """Remove rows flagged for review (translation quality issues)."""
    return dataset.filter(lambda x: not x.get("flag_for_review", False))


def process_docs(dataset):
    """Remove flagged rows, shuffle choices deterministically (OLMES seed)."""
    return _filter_flagged(dataset).map(_process_doc, with_indices=True)


def process_docs_diamond(dataset):
    """Filter to diamond subset, remove flagged rows, then shuffle choices."""
    return process_docs(dataset.filter(lambda x: x["is_diamond"]))


def _process_doc(doc, idx):
    correct = doc["correct_answer"]
    distractors = doc["incorrect_answers"]
    all_choices = [correct] + distractors

    rng = random.Random(OLMES_SHUFFLE_SEED + idx)
    rng.shuffle(all_choices)

    doc["choices"] = all_choices
    doc["answer_idx"] = all_choices.index(correct)
    doc["answer_label"] = LABELS[all_choices.index(correct)]
    return doc


# ---------------------------------------------------------------------------
# MC helpers
# ---------------------------------------------------------------------------

def doc_to_text_mc(doc):
    """Build MC prompt: Question : ... A. ... B. ... Réponse :"""
    choices = doc["choices"]
    choices_text = "\n".join(
        f" {LABELS[i]}. {text}" for i, text in enumerate(choices)
    )
    return f"Question : {doc['question']}\n{choices_text}\nRéponse :"


def doc_to_target_mc(doc):
    return doc["answer_idx"]


def doc_to_choice_mc(doc):
    return list(LABELS[: len(doc["choices"])])


# ---------------------------------------------------------------------------
# COT helpers
# ---------------------------------------------------------------------------

def doc_to_text_cot(doc):
    """Build COT prompt with instruction and choices."""
    choices = doc["choices"]
    choices_text = "\n".join(
        f" ({LABELS[i]}) {text}" for i, text in enumerate(choices)
    )
    return (
        "Answer the following multiple choice question. The last line of your "
        "response should be of the following format: 'The correct answer is: "
        "($LETTER)' where LETTER is one of ABCD. "
        "Think step by step before answering.\n\n"
        f"{doc['question']}\n{choices_text}"
    )


def doc_to_target_cot(doc):
    """Target for scoring: just the answer letter."""
    return doc["answer_label"]


def doc_to_target_cot_fewshot(doc):
    """Target for fewshot context: explanation + answer letter."""
    explanation = doc.get("explanation", "")
    label = doc["answer_label"]
    prefix = "\nLet's think step by step:\n"
    if explanation:
        return f"{prefix}{explanation}\nThe correct answer is: ({label})"
    return f"{prefix}The correct answer is: ({label})"


# ---------------------------------------------------------------------------
# COT fewshot examples (from GPQA original, translated to French)
# ---------------------------------------------------------------------------

def list_fewshot_cot():
    """5 curated GPQA fewshot examples with explanations (French).

    Translation of the 5 OLMES fewshot examples (Original:GPQA) from
    olmes/oe_eval/tasks/fewshot_sources.py into French. LaTeX formulas and
    chemical designations are kept in their international standard form.
    Choice order follows the OLMES shuffle convention (seed 111 + id), matching
    the deu config. Note: examples 3 and 4 overlap with the evaluation set —
    this is consistent with the OLMES procedure.
    """
    return [
        {
            "question": (
                "Dans une population donnée, 1 personne sur 400 est atteinte "
                "d'un cancer causé par un allèle entièrement récessif, b. En "
                "supposant que la population est en équilibre de "
                "Hardy-Weinberg : laquelle des valeurs suivantes correspond à "
                "la proportion attendue d'individus qui portent l'allèle b mais "
                "ne devraient pas développer le cancer ?"
            ),
            "choices": ["1/400", "19/400", "38/400", "20/400"],
            "answer_idx": 2,
            "answer_label": "C",
            "explanation": (
                "La proportion attendue d'individus qui portent l'allèle b mais "
                "ne développent pas le cancer correspond à la fréquence des "
                "hétérozygotes dans la population. Selon l'équation de "
                "Hardy-Weinberg p^2 + 2pq + q^2 = 1, où p est la fréquence de "
                "l'allèle dominant et q celle de l'allèle récessif. Comme "
                "q^2 = 1/400, on a q = 0,05 et p = 1 - q = 0,95. La fréquence "
                "des hétérozygotes est 2pq = 2 * 0,05 * 0,95 = 38/400."
            ),
        },
        {
            "question": (
                "Une pastille de Fe de 0,056 g est d'abord dissoute dans 10 mL "
                "d'acide bromhydrique HBr (0,1 M). La solution obtenue est "
                "ensuite titrée par KMnO4 (0,02 M). Combien y a-t-il de points "
                "d'équivalence ?"
            ),
            "choices": [
                "Un point, à 25 mL",
                "Deux points, à 25 mL et 35 mL",
                "Un point, à 10 mL",
                "Deux points, à 25 mL et 30 mL",
            ],
            "answer_idx": 1,
            "answer_label": "B",
            "explanation": (
                "HBr réagit avec Fe pour produire Fe2+. Dans le bécher, les "
                "espèces présentes sont Fe2+ et Br-. Lors d'une titration "
                "impliquant deux analytes, il faut déterminer quelle réaction "
                "se produit en premier. D'après les potentiels de réduction "
                "E°(Br2/Br-) = 1,09 V, E°(MnO4-/Mn2+) = 1,49 V et "
                "E°(Fe3+/Fe2+) = 0,77 V, MnO4- réagit d'abord avec Fe2+ "
                "(potentiel plus faible) dans un rapport 1:5 ; le premier point "
                "d'équivalence se situe à 10 mL. Une fois Fe2+ épuisé, MnO4- "
                "réagit avec Br- dans un rapport 2:10, ce qui nécessite 25 mL "
                "supplémentaires — le second point d'équivalence se situe à "
                "35 mL."
            ),
        },
        {
            "question": (
                "Considérez un système de mécanique quantique contenant une "
                "particule de masse $m$ se déplaçant dans un potentiel "
                "tridimensionnel isotrope de la forme "
                "$V(r) = 1/2 \\, m \\omega^2 r^2$, correspondant à une force "
                "obéissant à la loi de Hooke. Ici, $\\omega$ est la fréquence "
                "angulaire d'oscillation et $r$ la distance radiale de la "
                "particule par rapport à l'origine en coordonnées sphériques. "
                "Quelle est la valeur de l'énergie du troisième état excité, et "
                "combien de fonctions propres linéairement indépendantes "
                "sont possibles pour la même valeur propre d'énergie ?"
            ),
            "choices": [
                "11 \\pi^2 \\hbar^2 / (2m r^2), 3",
                "(9/2) \\hbar \\omega , 10",
                "11 \\pi^2 \\hbar^2 / (2m r^2), 10",
                "(9/2) \\hbar \\omega, 3",
            ],
            "answer_idx": 1,
            "answer_label": "B",
            "explanation": (
                "Il s'agit du problème de l'oscillateur harmonique "
                "tridimensionnel. Le spectre d'énergie est "
                "E_n = (n + 3/2) hbar omega avec n = 0, 1, 2, 3, … "
                "Pour le troisième état excité, n = 3 : "
                "E = (3 + 3/2) hbar omega = (9/2) hbar omega. "
                "La dégénérescence de l'état est g_n = (n+1)(n+2)/2. "
                "Pour n = 3 : g = (3+1)*(3+2)/2 = 4*5/2 = 10."
            ),
        },
        {
            "question": (
                "Vous surprenez une conversation entre deux chimistes qui "
                "sortent d'un laboratoire de chimie organique de synthèse. "
                "L'un demande : « Alors, comment ça s'est passé ? » L'autre "
                "répond : « Pas bien — mes composés se superposent. » À quoi le "
                "second chimiste fait-il très probablement référence ?"
            ),
            "choices": [
                "Les composés avec lesquels ils travaillent ont des points d'ébullition similaires.",
                "Les composés avec lesquels ils travaillent ont des polarités similaires.",
                "Les composés avec lesquels ils travaillent se lient entre eux par des interactions non covalentes / de van der Waals.",
                "Les composés avec lesquels ils travaillent ont des pouvoirs rotatoires optiques similaires.",
            ],
            "answer_idx": 1,
            "answer_label": "B",
            "explanation": (
                "« Se superposer » fait généralement référence à deux composés "
                "présentant des valeurs de Rf similaires en chromatographie "
                "(une opération courante en chimie de synthèse). Des valeurs de "
                "Rf similaires apparaissent pour des composés de polarités "
                "similaires."
            ),
        },
        {
            "question": (
                "Les mitochondries sont des organites cellulaires "
                "semi-autonomes chargés de la production d'énergie. Elles "
                "codent une partie de leur propre machinerie de traduction et "
                "de leurs complexes respiratoires. La fonction mitochondriale "
                "est régie par plus d'un millier de protéines importées depuis "
                "la cellule, qui contribuent à des processus tels que le "
                "transport des protéines, la biogenèse des ribosomes et la "
                "régulation de la traduction, l'oxydation respiratoire, le "
                "métabolisme et la cascade de signalisation apoptotique. Des "
                "mutations dans le code des réseaux de protéines "
                "mitochondriales peuvent provoquer de nombreuses maladies "
                "héréditaires chez l'être humain. Les mutations de laquelle des "
                "protéines mitochondriales listées ci-dessous sont les moins "
                "susceptibles d'être transmises génétiquement d'un père à ses "
                "enfants ?"
            ),
            "choices": [
                "Translocase de la membrane mitochondriale interne 17B",
                "NADH déshydrogénase 2",
                "Cassette de liaison à l'ATP, sous-famille B, membre 8",
                "Facteur d'élongation de la traduction Tu, mitochondrial",
            ],
            "answer_idx": 1,
            "answer_label": "B",
            "explanation": (
                "Les mitochondries du père ne sont pratiquement jamais "
                "transmises à la descendance. Par conséquent, les protéines "
                "codées par le génome mitochondrial paternel ne sont très "
                "probablement pas héritées. La NADH déshydrogénase 2 est la "
                "seule des protéines listées à être codée par le génome "
                "mitochondrial (gène MT-ND2). Les mutations du gène ND2 sont "
                "associées au syndrome de Leigh, à l'acidose lactique et à des "
                "maladies métaboliques. ABCB8 est codée sur le chromosome 7, "
                "TUFM sur le chromosome 16 et TIMM17B sur le chromosome X. Il "
                "n'existe aucune preuve d'empreinte maternelle de ce gène, de "
                "sorte que les filles peuvent hériter de la copie du gène "
                "paternel dans un rapport de 50:50."
            ),
        },
    ]
