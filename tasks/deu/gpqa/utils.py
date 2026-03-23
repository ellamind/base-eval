"""Utility functions for German GPQA tasks (MC and COT variants).

Dataset: ellamind/gpqa-multilingual (deu subset)
- 448 graduate-level STEM questions (Biology, Chemistry, Physics)
- Diamond subset: 198 hardest questions (is_diamond == True)
- Fields: question, correct_answer, incorrect_answers (list of 3)

MC tasks: 0-shot, multiple_choice output, acc/acc_norm metrics
COT tasks: 5-shot chain-of-thought, generate_until output, exact_match metric
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
    """Build MC prompt: Frage: ... A. ... B. ... Antwort:"""
    choices = doc["choices"]
    choices_text = "\n".join(
        f" {LABELS[i]}. {text}" for i, text in enumerate(choices)
    )
    return f"Frage: {doc['question']}\n{choices_text}\nAntwort:"


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
# COT fewshot examples (from GPQA original, translated to German)
# ---------------------------------------------------------------------------

def list_fewshot_cot():
    """5 kuratierte GPQA-Fewshot-Beispiele mit Erklärungen.

    Übersetzung der 5 OLMES-Fewshot-Beispiele (Original:GPQA) aus
    olmes/oe_eval/tasks/fewshot_sources.py ins Deutsche. LaTeX-Formeln
    und chemische Bezeichnungen bleiben im internationalen Standard.
    Hinweis: Beispiele 3 und 4 überschneiden sich mit dem
    Evaluierungsdatensatz — dies ist konsistent mit dem OLMES-Vorgehen.
    """
    return [
        {
            "question": (
                "In einer bestimmten Population hat 1 von 400 Personen eine "
                "Krebserkrankung, die durch ein vollständig rezessives Allel b "
                "verursacht wird. Unter der Annahme, dass sich die Population "
                "im Hardy-Weinberg-Gleichgewicht befindet: Welcher der "
                "folgenden Werte ist der erwartete Anteil der Individuen, die "
                "das b-Allel tragen, aber voraussichtlich nicht daran erkranken?"
            ),
            "choices": [
                "1/400",
                "19/400",
                "38/400",
                "20/400",
            ],
            "answer_idx": 2,
            "answer_label": "C",
            "explanation": (
                "Der erwartete Anteil der Träger des b-Allels, die nicht "
                "erkranken, entspricht der Häufigkeit des heterozygoten "
                "Genotyps in der Population. Nach der Hardy-Weinberg-Gleichung "
                "gilt p^2 + 2pq + q^2 = 1, wobei p die Häufigkeit des "
                "dominanten und q die des rezessiven Allels ist. "
                "Gegeben ist q^2 = 1/400, also q = 0,05 und p = 1 - q = 0,95. "
                "Die Häufigkeit der Heterozygoten beträgt "
                "2pq = 2 * 0,05 * 0,95 = 38/400."
            ),
        },
        {
            "question": (
                "Ein Fe-Pellet von 0,056 g wird zunächst in 10 mL "
                "Bromwasserstoffsäure HBr (0,1 M) gelöst. Die resultierende "
                "Lösung wird anschließend mit KMnO4 (0,02 M) titriert. "
                "Wie viele Äquivalenzpunkte gibt es?"
            ),
            "choices": [
                "Einen Punkt bei 25 mL",
                "Zwei Punkte bei 25 mL und 35 mL",
                "Einen Punkt bei 10 mL",
                "Zwei Punkte bei 25 mL und 30 mL",
            ],
            "answer_idx": 1,
            "answer_label": "B",
            "explanation": (
                "HBr reagiert mit Fe zu Fe2+. Im Becherglas liegen Fe2+ und "
                "Br- vor. Bei einer Titration mit zwei Analyten muss bestimmt "
                "werden, welche Reaktion zuerst abläuft. Anhand der "
                "Reduktionspotenziale E°(Br2/Br-) = 1,09 V, "
                "E°(MnO4-/Mn2+) = 1,49 V und E°(Fe3+/Fe2+) = 0,77 V "
                "reagiert MnO4- zuerst mit Fe2+ (niedrigeres Potenzial) im "
                "Verhältnis 1:5; der erste Äquivalenzpunkt liegt bei 10 mL. "
                "Sobald Fe2+ verbraucht ist, reagiert MnO4- mit Br- im "
                "Verhältnis 2:10, was weitere 25 mL erfordert — der zweite "
                "Äquivalenzpunkt liegt bei 35 mL."
            ),
        },
        {
            "question": (
                "Betrachten Sie ein quantenmechanisches System mit einem "
                "Teilchen der Masse $m$, das sich in einem isotropen "
                "dreidimensionalen Potenzial der Form "
                "$V(r) = 1/2 \\, m \\omega^2 r^2$ bewegt, entsprechend einer "
                "Kraft nach dem Hookeschen Gesetz. Dabei ist $\\omega$ die "
                "Kreisfrequenz der Schwingung und $r$ der radiale Abstand des "
                "Teilchens vom Ursprung in Kugelkoordinaten. Welchen Wert hat "
                "die Energie des dritten angeregten Zustands, und wie viele "
                "linear unabhängige Eigenfunktionen gibt es für denselben "
                "Energieeigenwert?"
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
                "Dies ist das Problem des dreidimensionalen harmonischen "
                "Oszillators. Das Energiespektrum lautet "
                "E_n = (n + 3/2) hbar omega mit n = 0, 1, 2, 3, … "
                "Für den dritten angeregten Zustand gilt n = 3: "
                "E = (3 + 3/2) hbar omega = (9/2) hbar omega. "
                "Die Entartung des Zustands beträgt g_n = (n+1)(n+2)/2. "
                "Für n = 3: g = (3+1)*(3+2)/2 = 4*5/2 = 10."
            ),
        },
        {
            "question": (
                "Sie hören zufällig, wie sich zwei Chemiker beim Verlassen "
                "eines Labors für synthetische organische Chemie unterhalten. "
                "Der eine fragt: \"Und, wie lief es?\" Der andere antwortet: "
                "\"Nicht gut — meine Verbindungen liegen übereinander.\" "
                "Worauf bezieht sich der zweite Chemiker höchstwahrscheinlich?"
            ),
            "choices": [
                "Die Verbindungen, mit denen sie arbeiten, haben ähnliche Siedepunkte.",
                "Die Verbindungen, mit denen sie arbeiten, haben ähnliche Polaritäten.",
                "Die Verbindungen, mit denen sie arbeiten, binden über nicht-kovalente/Van-der-Waals-Wechselwirkungen aneinander.",
                "Die Verbindungen, mit denen sie arbeiten, haben ähnliche optische Drehwerte.",
            ],
            "answer_idx": 1,
            "answer_label": "B",
            "explanation": (
                "\"Übereinander liegen\" bezieht sich üblicherweise darauf, "
                "dass zwei Verbindungen ähnliche Rf-Werte in der "
                "Chromatographie aufweisen (ein gängiges Verfahren in der "
                "synthetischen Chemie). Ähnliche Rf-Werte ergeben sich bei "
                "Verbindungen mit ähnlichen Polaritäten."
            ),
        },
        {
            "question": (
                "Mitochondrien sind semi-autonome Zellorganellen, die für die "
                "Energieproduktion zuständig sind. Sie kodieren einen Teil "
                "ihrer eigenen Translationsmaschinerie und "
                "Atmungskettenkomplexe. Die mitochondriale Funktion wird von "
                "über tausend aus der Zelle importierten Proteinen gesteuert, "
                "die zu Prozessen wie Proteintransport, Ribosomenbiogenese "
                "und Translationsregulation, respiratorischer Oxidation, "
                "Metabolismus und der apoptotischen Signalkaskade beitragen. "
                "Mutationen im Code mitochondrialer Proteinnetzwerke können "
                "zahlreiche vererbbare Krankheiten beim Menschen verursachen. "
                "Mutationen welcher der unten aufgeführten mitochondrialen "
                "Proteine werden am wenigsten wahrscheinlich genetisch von "
                "einem Vater an seine Kinder weitergegeben?"
            ),
            "choices": [
                "Translokase der inneren Mitochondrienmembran 17B",
                "NADH-Dehydrogenase 2",
                "ATP-bindende Kassette, Unterfamilie B, Mitglied 8",
                "Tu-Translationselongationsfaktor, mitochondrial",
            ],
            "answer_idx": 1,
            "answer_label": "B",
            "explanation": (
                "Mitochondrien des Vaters werden so gut wie nie an die "
                "Nachkommen weitergegeben. Daher werden Proteine, die vom "
                "väterlichen mitochondrialen Genom kodiert werden, mit hoher "
                "Wahrscheinlichkeit nicht vererbt. NADH-Dehydrogenase 2 ist "
                "das einzige der aufgelisteten Proteine, das vom "
                "mitochondrialen Genom kodiert wird (MT-ND2-Gen). "
                "Mutationen im ND2-Gen sind mit dem Leigh-Syndrom, "
                "Laktatazidose und Stoffwechselerkrankungen assoziiert. "
                "ABCB8 wird auf Chromosom 7 kodiert, TUFM auf Chromosom 16, "
                "und TIMM17B auf dem X-Chromosom. "
                "Es gibt keine Hinweise auf maternales Imprinting dieses Gens, "
                "sodass Töchter die Genkopie des Vaters im Verhältnis 50:50 "
                "erben können."
            ),
        },
    ]
