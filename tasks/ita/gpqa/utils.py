"""Utility functions for Italian GPQA tasks (MC and COT variants).

Dataset: ellamind/gpqa-multilingual (ita subset)
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
    """Build MC prompt: Domanda: ... A. ... B. ... Risposta:"""
    choices = doc["choices"]
    choices_text = "\n".join(
        f" {LABELS[i]}. {text}" for i, text in enumerate(choices)
    )
    return f"Domanda: {doc['question']}\n{choices_text}\nRisposta:"


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
# COT fewshot examples (from GPQA original, translated to Italian)
# ---------------------------------------------------------------------------

def list_fewshot_cot():
    """5 curated GPQA fewshot examples with explanations (Italian).

    Translation of the 5 OLMES fewshot examples (Original:GPQA) from
    olmes/oe_eval/tasks/fewshot_sources.py into Italian. LaTeX formulas and
    chemical designations are kept in their international standard form.
    Choice order follows the OLMES shuffle convention (seed 111 + id), matching
    the deu config. Note: examples 3 and 4 overlap with the evaluation set —
    this is consistent with the OLMES procedure.
    """
    return [
        {
            "question": (
                "In una data popolazione, 1 persona su 400 ha un cancro "
                "causato da un allele completamente recessivo, b. Supponendo "
                "che la popolazione sia in equilibrio di Hardy-Weinberg: quale "
                "dei seguenti valori è la proporzione attesa di individui che "
                "portano l'allele b ma che non dovrebbero sviluppare il cancro?"
            ),
            "choices": ["1/400", "19/400", "38/400", "20/400"],
            "answer_idx": 2,
            "answer_label": "C",
            "explanation": (
                "La proporzione attesa di individui che portano l'allele b ma "
                "non sviluppano il cancro corrisponde alla frequenza degli "
                "eterozigoti nella popolazione. Secondo l'equazione di "
                "Hardy-Weinberg p^2 + 2pq + q^2 = 1, dove p è la frequenza "
                "dell'allele dominante e q quella dell'allele recessivo. Dato "
                "che q^2 = 1/400, si ha q = 0,05 e p = 1 - q = 0,95. La "
                "frequenza degli eterozigoti è "
                "2pq = 2 * 0,05 * 0,95 = 38/400."
            ),
        },
        {
            "question": (
                "Una pallina di Fe da 0,056 g viene prima sciolta in 10 mL di "
                "acido bromidrico HBr (0,1 M). La soluzione risultante viene "
                "poi titolata con KMnO4 (0,02 M). Quanti punti di equivalenza "
                "ci sono?"
            ),
            "choices": [
                "Un punto, a 25 mL",
                "Due punti, a 25 mL e 35 mL",
                "Un punto, a 10 mL",
                "Due punti, a 25 mL e 30 mL",
            ],
            "answer_idx": 1,
            "answer_label": "B",
            "explanation": (
                "HBr reagisce con Fe per produrre Fe2+. Nel becher le specie "
                "presenti sono Fe2+ e Br-. In una titolazione con due analiti "
                "occorre determinare quale reazione avviene per prima. In base "
                "ai potenziali di riduzione E°(Br2/Br-) = 1,09 V, "
                "E°(MnO4-/Mn2+) = 1,49 V e E°(Fe3+/Fe2+) = 0,77 V, MnO4- "
                "reagisce prima con Fe2+ (potenziale più basso) in un rapporto "
                "1:5; il primo punto di equivalenza è a 10 mL. Una volta "
                "esaurito il Fe2+, MnO4- reagisce con Br- in un rapporto 2:10, "
                "il che richiede altri 25 mL — il secondo punto di equivalenza "
                "è a 35 mL."
            ),
        },
        {
            "question": (
                "Si consideri un sistema di meccanica quantistica contenente "
                "una particella di massa $m$ che si muove in un potenziale "
                "tridimensionale isotropo della forma "
                "$V(r) = 1/2 \\, m \\omega^2 r^2$, corrispondente a una forza "
                "che obbedisce alla legge di Hooke. Qui $\\omega$ è la "
                "frequenza angolare di oscillazione e $r$ è la distanza radiale "
                "della particella dall'origine in coordinate sferiche. Qual è "
                "il valore dell'energia del terzo stato eccitato e quante "
                "autofunzioni linearmente indipendenti sono possibili per lo "
                "stesso autovalore di energia?"
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
                "Si tratta del problema dell'oscillatore armonico "
                "tridimensionale. Lo spettro di energia è "
                "E_n = (n + 3/2) hbar omega con n = 0, 1, 2, 3, … "
                "Per il terzo stato eccitato, n = 3: "
                "E = (3 + 3/2) hbar omega = (9/2) hbar omega. "
                "La degenerazione dello stato è g_n = (n+1)(n+2)/2. "
                "Per n = 3: g = (3+1)*(3+2)/2 = 4*5/2 = 10."
            ),
        },
        {
            "question": (
                "Senti per caso due chimici che parlano mentre escono da un "
                "laboratorio di chimica organica sintetica. Uno chiede: "
                "«Allora, com'è andata?» L'altro risponde: «Non bene — i miei "
                "composti sono uno sopra l'altro.» A che cosa si riferisce "
                "molto probabilmente il secondo chimico?"
            ),
            "choices": [
                "I composti con cui lavorano hanno punti di ebollizione simili.",
                "I composti con cui lavorano hanno polarità simili.",
                "I composti con cui lavorano si legano tra loro tramite interazioni non covalenti / di van der Waals.",
                "I composti con cui lavorano hanno poteri rotatori ottici simili.",
            ],
            "answer_idx": 1,
            "answer_label": "B",
            "explanation": (
                "«Essere uno sopra l'altro» di solito si riferisce a due "
                "composti che presentano valori di Rf simili in cromatografia "
                "(un'operazione comune in chimica sintetica). Valori di Rf "
                "simili si presentano per composti con polarità simili."
            ),
        },
        {
            "question": (
                "I mitocondri sono organuli cellulari semi-autonomi "
                "responsabili della produzione di energia. Codificano una "
                "parte del proprio apparato di traduzione e dei complessi "
                "respiratori. La funzione mitocondriale è governata da oltre "
                "mille proteine importate dalla cellula, che contribuiscono a "
                "processi quali il trasporto delle proteine, la biogenesi dei "
                "ribosomi e la regolazione della traduzione, l'ossidazione "
                "respiratoria, il metabolismo e la cascata di segnalazione "
                "apoptotica. Le mutazioni nel codice delle reti di proteine "
                "mitocondriali possono causare numerose malattie ereditarie "
                "nell'essere umano. Le mutazioni di quale delle proteine "
                "mitocondriali elencate di seguito hanno la minore probabilità "
                "di essere trasmesse geneticamente da un padre ai suoi figli?"
            ),
            "choices": [
                "Traslocasi della membrana mitocondriale interna 17B",
                "NADH deidrogenasi 2",
                "Cassetta legante l'ATP, sottofamiglia B, membro 8",
                "Fattore di allungamento della traduzione Tu, mitocondriale",
            ],
            "answer_idx": 1,
            "answer_label": "B",
            "explanation": (
                "I mitocondri del padre non vengono quasi mai trasmessi alla "
                "prole. Di conseguenza, le proteine codificate dal genoma "
                "mitocondriale paterno molto probabilmente non vengono "
                "ereditate. La NADH deidrogenasi 2 è l'unica tra le proteine "
                "elencate a essere codificata dal genoma mitocondriale (gene "
                "MT-ND2). Le mutazioni del gene ND2 sono associate alla "
                "sindrome di Leigh, all'acidosi lattica e a malattie "
                "metaboliche. ABCB8 è codificata sul cromosoma 7, TUFM sul "
                "cromosoma 16 e TIMM17B sul cromosoma X. Non vi è alcuna prova "
                "di imprinting materno di questo gene, per cui le figlie "
                "possono ereditare la copia del gene paterno in un rapporto di "
                "50:50."
            ),
        },
    ]
