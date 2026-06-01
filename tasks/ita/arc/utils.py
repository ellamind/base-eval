import math


def filter_easy(dataset):
    return dataset.filter(lambda x: x["arc_config"] == "ARC-Easy")


def filter_challenge(dataset):
    return dataset.filter(lambda x: x["arc_config"] == "ARC-Challenge")


def list_fewshot_easy():
    """5-shot curated examples for ARC Easy (Italian).

    Translated from the OLMES curated set (OLMES:ARC-Easy).
    """
    return [
        {
            "question": "I licheni sono organismi simbiotici composti da alghe verdi e funghi. Che cosa forniscono le alghe verdi ai funghi in questa relazione simbiotica?",
            "choices": ["Anidride carbonica", "Cibo", "Protezione", "Acqua"],
            "answer_key": "B",
        },
        {
            "question": "Quando si usa un interruttore in un circuito elettrico, l'interruttore può",
            "choices": [
                "accumulare la carica.",
                "aumentare e diminuire la tensione.",
                "cambiare la direzione della corrente.",
                "interrompere e avviare il flusso di corrente.",
            ],
            "answer_key": "D",
        },
        {
            "question": "Quale è un esempio di dispositivo medico?",
            "choices": ["una lente a contatto", "una motocicletta", "un impermeabile", "una caffettiera"],
            "answer_key": "A",
        },
        {
            "question": "Le rocce sono classificate come ignee, metamorfiche o sedimentarie in base",
            "choices": [
                "al loro colore",
                "alla loro forma",
                "a come si sono formate",
                "ai minerali che contengono",
            ],
            "answer_key": "C",
        },
        {
            "question": "Una compressa masticabile di carbonato di calcio è un rimedio comune per i disturbi di stomaco. Il carbonato di calcio è molto probabilmente usato come medicinale perché il carbonato di calcio",
            "choices": [
                "ha un sapore gradevole.",
                "è economico da produrre.",
                "neutralizza l'acido dello stomaco.",
                "è naturalmente presente nel corpo.",
            ],
            "answer_key": "C",
        },
    ]


def list_fewshot_challenge():
    """5-shot curated examples for ARC Challenge (Italian).

    Translated from the OLMES curated set (OLMES:ARC-Challenge).
    """
    return [
        {
            "question": "Giorgio vuole scaldarsi rapidamente le mani strofinandole. Quale superficie della mano produrrà più calore?",
            "choices": [
                "palmi asciutti",
                "palmi bagnati",
                "palmi coperti di olio",
                "palmi coperti di lozione",
            ],
            "answer_key": "A",
        },
        {
            "question": "Una barca è spinta da una corrente fluviale che scorre verso nord e dal vento nelle sue vele. La barca si muove verso nord-est. In quale direzione il vento sta più probabilmente applicando una forza sulle vele della barca?",
            "choices": ["ovest", "est", "nord", "sud"],
            "answer_key": "B",
        },
        {
            "question": "Quale forma del territorio è creata dalla forza costruttiva di un ghiacciaio?",
            "choices": [
                "valli scavate da un ghiacciaio in movimento",
                "cumuli di rocce depositati da un ghiacciaio che si scioglie",
                "solchi creati nella roccia da un ghiacciaio",
                "colline di roccia madre rese ruvide da un ghiacciaio",
            ],
            "answer_key": "B",
        },
        {
            "question": "Quale di questi riflessi si verifica esclusivamente negli esseri umani?",
            "choices": ["starnutire", "sussultare", "piangere", "sbattere le palpebre"],
            "answer_key": "C",
        },
        {
            "question": "L'uso di risorse non rinnovabili per produrre energia genera prodotti di scarto che possono avere effetti negativi a lungo termine sui sottosistemi della Terra. Quale fonte di energia produce prodotti di scarto i cui effetti possono durare più a lungo?",
            "choices": ["gas naturale", "uranio", "petrolio greggio", "carbone"],
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
