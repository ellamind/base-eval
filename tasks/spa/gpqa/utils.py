"""Utility functions for Spanish GPQA tasks (MC and COT variants).

Dataset: ellamind/gpqa-multilingual (spa subset)
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
    """Build MC prompt: Pregunta: ... A. ... B. ... Respuesta:"""
    choices = doc["choices"]
    choices_text = "\n".join(
        f" {LABELS[i]}. {text}" for i, text in enumerate(choices)
    )
    return f"Pregunta: {doc['question']}\n{choices_text}\nRespuesta:"


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
# COT fewshot examples (from GPQA original, translated to Spanish)
# ---------------------------------------------------------------------------

def list_fewshot_cot():
    """5 curated GPQA fewshot examples with explanations (Spanish).

    Translation of the 5 OLMES fewshot examples (Original:GPQA) from
    olmes/oe_eval/tasks/fewshot_sources.py into Spanish. LaTeX formulas and
    chemical designations are kept in their international standard form.
    Choice order follows the OLMES shuffle convention (seed 111 + id), matching
    the deu config. Note: examples 3 and 4 overlap with the evaluation set —
    this is consistent with the OLMES procedure.
    """
    return [
        {
            "question": (
                "En una población dada, 1 de cada 400 personas tiene un cáncer "
                "causado por un alelo completamente recesivo, b. Suponiendo que "
                "la población está en equilibrio de Hardy-Weinberg: ¿cuál de "
                "los siguientes valores es la proporción esperada de individuos "
                "que portan el alelo b pero que no se espera que desarrollen el "
                "cáncer?"
            ),
            "choices": ["1/400", "19/400", "38/400", "20/400"],
            "answer_idx": 2,
            "answer_label": "C",
            "explanation": (
                "La proporción esperada de individuos que portan el alelo b "
                "pero no desarrollan el cáncer corresponde a la frecuencia de "
                "los heterocigotos en la población. Según la ecuación de "
                "Hardy-Weinberg p^2 + 2pq + q^2 = 1, donde p es la frecuencia "
                "del alelo dominante y q la del alelo recesivo. Dado que "
                "q^2 = 1/400, entonces q = 0,05 y p = 1 - q = 0,95. La "
                "frecuencia de los heterocigotos es "
                "2pq = 2 * 0,05 * 0,95 = 38/400."
            ),
        },
        {
            "question": (
                "Una pastilla de Fe de 0,056 g se disuelve primero en 10 mL de "
                "ácido bromhídrico HBr (0,1 M). La solución resultante se "
                "titula luego con KMnO4 (0,02 M). ¿Cuántos puntos de "
                "equivalencia hay?"
            ),
            "choices": [
                "Un punto, a 25 mL",
                "Dos puntos, a 25 mL y 35 mL",
                "Un punto, a 10 mL",
                "Dos puntos, a 25 mL y 30 mL",
            ],
            "answer_idx": 1,
            "answer_label": "B",
            "explanation": (
                "HBr reacciona con Fe para producir Fe2+. En el vaso de "
                "precipitados, las especies presentes son Fe2+ y Br-. En una "
                "titulación con dos analitos hay que determinar qué reacción "
                "ocurre primero. Según los potenciales de reducción "
                "E°(Br2/Br-) = 1,09 V, E°(MnO4-/Mn2+) = 1,49 V y "
                "E°(Fe3+/Fe2+) = 0,77 V, MnO4- reacciona primero con Fe2+ "
                "(potencial más bajo) en una proporción 1:5; el primer punto de "
                "equivalencia está a 10 mL. Una vez agotado el Fe2+, MnO4- "
                "reacciona con Br- en una proporción 2:10, lo que requiere "
                "25 mL adicionales — el segundo punto de equivalencia está a "
                "35 mL."
            ),
        },
        {
            "question": (
                "Considere un sistema de mecánica cuántica que contiene una "
                "partícula de masa $m$ que se mueve en un potencial "
                "tridimensional isótropo de la forma "
                "$V(r) = 1/2 \\, m \\omega^2 r^2$, correspondiente a una fuerza "
                "que obedece la ley de Hooke. Aquí, $\\omega$ es la frecuencia "
                "angular de oscilación y $r$ es la distancia radial de la "
                "partícula al origen en coordenadas esféricas. ¿Cuál es el "
                "valor de la energía del tercer estado excitado y cuántas "
                "funciones propias linealmente independientes son posibles para "
                "el mismo autovalor de energía?"
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
                "Se trata del problema del oscilador armónico tridimensional. "
                "El espectro de energía es E_n = (n + 3/2) hbar omega con "
                "n = 0, 1, 2, 3, … Para el tercer estado excitado, n = 3: "
                "E = (3 + 3/2) hbar omega = (9/2) hbar omega. "
                "La degeneración del estado es g_n = (n+1)(n+2)/2. "
                "Para n = 3: g = (3+1)*(3+2)/2 = 4*5/2 = 10."
            ),
        },
        {
            "question": (
                "Escuchas por casualidad a dos químicos que conversan al salir "
                "de un laboratorio de química orgánica sintética. Uno pregunta: "
                "«¿Y bien, cómo fue?» El otro responde: «No muy bien: mis "
                "compuestos están uno encima del otro.» ¿A qué se refiere muy "
                "probablemente el segundo químico?"
            ),
            "choices": [
                "Los compuestos con los que trabajan tienen puntos de ebullición similares.",
                "Los compuestos con los que trabajan tienen polaridades similares.",
                "Los compuestos con los que trabajan se unen entre sí mediante interacciones no covalentes / de van der Waals.",
                "Los compuestos con los que trabajan tienen rotaciones ópticas similares.",
            ],
            "answer_idx": 1,
            "answer_label": "B",
            "explanation": (
                "«Estar uno encima del otro» suele referirse a dos compuestos "
                "que presentan valores de Rf similares en cromatografía (una "
                "operación habitual en química sintética). Valores de Rf "
                "similares aparecen en compuestos con polaridades similares."
            ),
        },
        {
            "question": (
                "Las mitocondrias son orgánulos celulares semiautónomos "
                "encargados de la producción de energía. Codifican una parte de "
                "su propia maquinaria de traducción y de sus complejos "
                "respiratorios. La función mitocondrial está regida por más de "
                "mil proteínas importadas desde la célula, que contribuyen a "
                "procesos como el transporte de proteínas, la biogénesis de "
                "ribosomas y la regulación de la traducción, la oxidación "
                "respiratoria, el metabolismo y la cascada de señalización "
                "apoptótica. Las mutaciones en el código de las redes de "
                "proteínas mitocondriales pueden causar numerosas enfermedades "
                "hereditarias en los seres humanos. ¿Las mutaciones de cuál de "
                "las proteínas mitocondriales enumeradas a continuación tienen "
                "la menor probabilidad de transmitirse genéticamente de un "
                "padre a sus hijos?"
            ),
            "choices": [
                "Translocasa de la membrana mitocondrial interna 17B",
                "NADH deshidrogenasa 2",
                "Casete de unión a ATP, subfamilia B, miembro 8",
                "Factor de elongación de la traducción Tu, mitocondrial",
            ],
            "answer_idx": 1,
            "answer_label": "B",
            "explanation": (
                "Las mitocondrias del padre casi nunca se transmiten a la "
                "descendencia. Por lo tanto, las proteínas codificadas por el "
                "genoma mitocondrial paterno muy probablemente no se heredan. "
                "La NADH deshidrogenasa 2 es la única de las proteínas listadas "
                "codificada por el genoma mitocondrial (gen MT-ND2). Las "
                "mutaciones del gen ND2 se asocian con el síndrome de Leigh, la "
                "acidosis láctica y enfermedades metabólicas. ABCB8 se codifica "
                "en el cromosoma 7, TUFM en el cromosoma 16 y TIMM17B en el "
                "cromosoma X. No hay evidencia de impronta materna de este gen, "
                "por lo que las hijas pueden heredar la copia del gen paterno "
                "en una proporción de 50:50."
            ),
        },
    ]
