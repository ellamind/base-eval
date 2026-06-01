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
    return {"bits_per_byte": -ll / (math.log(2) * max(gold_bytes, 1))}


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
    """5-shot curated examples for AQUA-RAT (Italian)."""
    return [
        {
            "question": "Un treno viaggia a 60 km/h e percorre una distanza di 240 km. Quanto dura il viaggio?",
            "correct_answer": "4 ore",
            "choices": ["2 ore", "4 ore", "6 ore", "3 ore", "5 ore"],
            "answer_idx": 1,
        },
        {
            "question": "Se 6 operai impiegano 8 giorni per costruire un muro, quanti giorni servono a 12 operai per lo stesso muro?",
            "correct_answer": "4 giorni",
            "choices": ["4 giorni", "6 giorni", "8 giorni", "2 giorni", "10 giorni"],
            "answer_idx": 0,
        },
        {
            "question": "Un commerciante acquista un articolo a 80 \u20ac e lo vende con un profitto del 25%. Qual \u00e8 il prezzo di vendita?",
            "correct_answer": "100 \u20ac",
            "choices": ["90 \u20ac", "95 \u20ac", "100 \u20ac", "105 \u20ac", "110 \u20ac"],
            "answer_idx": 2,
        },
        {
            "question": "Il rapporto tra ragazzi e ragazze in una classe \u00e8 3:5. Se ci sono 24 ragazzi, quante ragazze ci sono?",
            "correct_answer": "40",
            "choices": ["30", "35", "40", "45", "50"],
            "answer_idx": 2,
        },
        {
            "question": "Qual \u00e8 l'interesse semplice su 5000 \u20ac a un tasso del 4% per 3 anni?",
            "correct_answer": "600 \u20ac",
            "choices": ["400 \u20ac", "500 \u20ac", "600 \u20ac", "700 \u20ac", "800 \u20ac"],
            "answer_idx": 2,
        },
    ]


def list_fewshot_gaokao_english():
    """3-shot curated examples for Gaokao English (Italian)."""
    return [
        {
            "question": "OROLOGIO TELECOMANDO\nQuesto \u00e8 un orologio che James Bond indosserebbe con orgoglio! Il tuo OROLOGIO TELECOMANDO elettronico PENGO funziona come telecomando per televisori e videoregistratori.\nCon l'aiuto di un Mr. H puoi ___.",
            "correct_answer": "finire i compiti in tempo.",
            "choices": [
                "smettere di usare le batterie.",
                "finire i compiti in tempo.",
                "ricordare le istruzioni del tuo insegnante.",
                "far pulire la tua stanza mentre torni a casa.",
            ],
            "answer_idx": 1,
        },
        {
            "question": "Un cartello sulla porta di un negozio recita: 'Siamo aperti per voi 7 giorni su 7, 365 giorni all'anno.' Qual \u00e8 lo scopo principale del cartello?",
            "correct_answer": "Comunicare gli orari di apertura del negozio.",
            "choices": [
                "Comunicare gli orari di apertura del negozio.",
                "Reclutare nuovi dipendenti.",
                "Pubblicizzare un nuovo prodotto.",
                "Segnalare una variazione di prezzo.",
            ],
            "answer_idx": 0,
        },
        {
            "question": "Secondo il testo, il motivo principale per cui le persone fanno volontariato \u00e8 ___.",
            "correct_answer": "che vogliono aiutare gli altri e restituire qualcosa alla comunit\u00e0",
            "choices": [
                "che vogliono guadagnare denaro",
                "che vogliono aiutare gli altri e restituire qualcosa alla comunit\u00e0",
                "che vogliono acquisire nuove competenze",
                "che vogliono creare contatti professionali",
            ],
            "answer_idx": 1,
        },
    ]


def list_fewshot_logiqa_en():
    """3-shot curated examples for LogiQA (Italian)."""
    return [
        {
            "question": "In un ufficio siedono quattro persone: A, B, C e D. A siede di fronte a B. C siede alla destra di A. Chi siede alla sinistra di B?",
            "correct_answer": "C",
            "choices": ["A", "C", "D", "Nessuno"],
            "answer_idx": 1,
        },
        {
            "question": "Tutti i filosofi sono pensatori. Alcuni pensatori sono scrittori. Quale conclusione \u00e8 necessariamente corretta?",
            "correct_answer": "Alcuni filosofi potrebbero essere scrittori.",
            "choices": [
                "Tutti gli scrittori sono filosofi.",
                "Nessun filosofo \u00e8 scrittore.",
                "Alcuni filosofi potrebbero essere scrittori.",
                "Tutti i pensatori sono filosofi.",
            ],
            "answer_idx": 2,
        },
        {
            "question": "Se piove, la strada si bagna. La strada \u00e8 bagnata. Quale conclusione \u00e8 corretta?",
            "correct_answer": "Non si pu\u00f2 dire con certezza se ha piovuto.",
            "choices": [
                "Ha piovuto.",
                "Non ha piovuto.",
                "Non si pu\u00f2 dire con certezza se ha piovuto.",
                "La strada \u00e8 stata lavata.",
            ],
            "answer_idx": 2,
        },
    ]


def list_fewshot_lsat_ar():
    """3-shot curated examples for LSAT-AR (Italian)."""
    return [
        {
            "question": "Cinque conferenze \u2013 F, G, H, J e K \u2013 si tengono in successione nello stesso giorno. G si tiene prima di H. J si tiene subito dopo F. Quale ordine \u00e8 possibile?",
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
            "question": "Un negozio di fiori dispone sette mazzi \u2013 S, T, U, V, W, X e Y \u2013 in fila. V \u00e8 al terzo posto. T \u00e8 immediatamente a sinistra di U. Quale mazzo potrebbe trovarsi al primo posto?",
            "correct_answer": "T",
            "choices": ["U", "V", "T", "Y"],
            "answer_idx": 2,
        },
        {
            "question": "Tre squadre \u2013 Rossa, Blu e Verde \u2013 giocano ciascuna due partite. La Rossa gioca prima della Blu. La Verde non gioca per prima. Quale ordine delle prime partite \u00e8 possibile?",
            "correct_answer": "Rossa, Verde, Blu",
            "choices": [
                "Blu, Rossa, Verde",
                "Verde, Rossa, Blu",
                "Rossa, Verde, Blu",
                "Rossa, Blu, Verde",
            ],
            "answer_idx": 2,
        },
    ]


def list_fewshot_lsat_lr():
    """3-shot curated examples for LSAT-LR (Italian)."""
    return [
        {
            "question": "Editoriale: Poich\u00e9 la popolazione invecchia, i costi sanitari aumentano. Pertanto il governo deve investire di pi\u00f9 nella prevenzione. Quale presupposto \u00e8 alla base di questo argomento?",
            "correct_answer": "Le misure di prevenzione possono ridurre i costi sanitari di una popolazione che invecchia.",
            "choices": [
                "La popolazione non invecchier\u00e0 pi\u00f9 in futuro.",
                "Le misure di prevenzione possono ridurre i costi sanitari di una popolazione che invecchia.",
                "Il governo attualmente non spende nulla per la prevenzione.",
                "L'aumento dei costi sanitari \u00e8 inevitabile.",
            ],
            "answer_idx": 1,
        },
        {
            "question": "Critico: Questo museo espone solo opere di artisti famosi. Pertanto non riesce a promuovere i talenti emergenti. Quale affermazione indebolisce maggiormente questo argomento?",
            "correct_answer": "Il museo dispone di un'area espositiva dedicata ai nuovi artisti.",
            "choices": [
                "Gli artisti famosi attirano pi\u00f9 visitatori.",
                "Il museo dispone di un'area espositiva dedicata ai nuovi artisti.",
                "Anche altri musei espongono solo artisti famosi.",
                "Gli artisti emergenti preferiscono gallerie pi\u00f9 piccole.",
            ],
            "answer_idx": 1,
        },
        {
            "question": "Se tutti i dipendenti di un'azienda sono puntuali e Stefano \u00e8 un dipendente di questa azienda, allora Stefano deve essere puntuale. Stefano arriva spesso in ritardo. Cosa ne consegue?",
            "correct_answer": "Non tutti i dipendenti di questa azienda sono puntuali.",
            "choices": [
                "Stefano non \u00e8 un dipendente dell'azienda.",
                "Non tutti i dipendenti di questa azienda sono puntuali.",
                "Stefano \u00e8 sempre puntuale.",
                "L'azienda non ha regole sulla puntualit\u00e0.",
            ],
            "answer_idx": 1,
        },
    ]


def list_fewshot_lsat_rc():
    """3-shot curated examples for LSAT-RC (Italian)."""
    return [
        {
            "question": "Gli avvocati hanno il dovere di difendere al meglio i propri clienti. Allo stesso tempo hanno una responsabilit\u00e0 verso la societ\u00e0. Quale affermazione descrive meglio il messaggio principale del testo?",
            "correct_answer": "Gli avvocati devono tenere conto sia degli interessi dei loro clienti sia di quelli della societ\u00e0.",
            "choices": [
                "Gli avvocati dovrebbero rappresentare solo gli interessi dei loro clienti.",
                "Gli avvocati devono tenere conto sia degli interessi dei loro clienti sia di quelli della societ\u00e0.",
                "La societ\u00e0 dovrebbe controllare maggiormente il lavoro degli avvocati.",
                "I clienti dovrebbero poter scegliere liberamente il proprio avvocato.",
            ],
            "answer_idx": 1,
        },
        {
            "question": "L'autore discute diversi approcci alla riforma del diritto d'autore. Quale posizione sostiene principalmente l'autore?",
            "correct_answer": "\u00c8 necessario un approccio equilibrato che tuteli sia gli autori sia il pubblico.",
            "choices": [
                "Il diritto d'autore dovrebbe essere completamente abolito.",
                "\u00c8 necessario un approccio equilibrato che tuteli sia gli autori sia il pubblico.",
                "Solo le aziende dovrebbero poter detenere diritti d'autore.",
                "Il sistema attuale funziona perfettamente.",
            ],
            "answer_idx": 1,
        },
        {
            "question": "Nel testo viene descritta l'evoluzione della legislazione ambientale. Secondo il testo, qual \u00e8 stata la ragione principale dell'inasprimento delle leggi?",
            "correct_answer": "Le crescenti conoscenze scientifiche sui danni ambientali.",
            "choices": [
                "Le crescenti conoscenze scientifiche sui danni ambientali.",
                "Gli interessi economici dell'industria.",
                "Gli accordi politici internazionali.",
                "Le proteste di singoli cittadini.",
            ],
            "answer_idx": 0,
        },
    ]


def list_fewshot_sat_en():
    """3-shot curated examples for SAT-EN (Italian)."""
    return [
        {
            "question": "Akira arriv\u00f2 diretto e ruppe con ogni tradizione. Buss\u00f2 alla porta in una sera d'inverno. 'Voglio sposare vostra figlia Naomi', disse. Quale affermazione descrive meglio ci\u00f2 che accade nel testo?",
            "correct_answer": "Un personaggio riceve una richiesta sorprendente da un altro personaggio.",
            "choices": [
                "Un personaggio litiga con un altro personaggio.",
                "Un personaggio riceve una richiesta sorprendente da un altro personaggio.",
                "Un personaggio ripensa a decisioni passate.",
                "Un personaggio critica un altro per il suo comportamento inaspettato.",
            ],
            "answer_idx": 1,
        },
        {
            "question": "Il narratore descrive un viaggio attraverso una citt\u00e0 sconosciuta. Le strade erano strette e gli edifici antichi. Qual \u00e8 l'atmosfera principale del testo?",
            "correct_answer": "Curiosit\u00e0 mista a incertezza.",
            "choices": [
                "Gioia ed entusiasmo.",
                "Curiosit\u00e0 mista a incertezza.",
                "Profonda tristezza.",
                "Rabbia e frustrazione.",
            ],
            "answer_idx": 1,
        },
        {
            "question": "Nel testo l'autore confronta due teorie scientifiche. Qual \u00e8 lo scopo principale di questo confronto?",
            "correct_answer": "Evidenziare i punti di forza e di debolezza di entrambi gli approcci.",
            "choices": [
                "Dimostrare che una teoria \u00e8 sbagliata.",
                "Evidenziare i punti di forza e di debolezza di entrambi gli approcci.",
                "Proporre una teoria completamente nuova.",
                "Riassumere la storia della scienza.",
            ],
            "answer_idx": 1,
        },
    ]


def list_fewshot_sat_en_without_passage():
    """3-shot curated examples for SAT-EN without passage (Italian)."""
    return [
        {
            "question": "Quale affermazione descrive meglio ci\u00f2 che accade nel testo?",
            "correct_answer": "Un personaggio riceve una richiesta sorprendente da un altro personaggio.",
            "choices": [
                "Un personaggio litiga con un altro personaggio.",
                "Un personaggio riceve una richiesta sorprendente da un altro personaggio.",
                "Un personaggio ripensa a decisioni passate.",
                "Un personaggio critica un altro per il suo comportamento inaspettato.",
            ],
            "answer_idx": 1,
        },
        {
            "question": "Quale funzione svolge il terzo paragrafo nel contesto complessivo del testo?",
            "correct_answer": "Fornisce un esempio concreto della tesi precedentemente esposta.",
            "choices": [
                "Confuta l'argomento principale.",
                "Fornisce un esempio concreto della tesi precedentemente esposta.",
                "Introduce un argomento completamente nuovo.",
                "Riassume l'intero testo.",
            ],
            "answer_idx": 1,
        },
        {
            "question": "Quale parola descrive meglio il tono dell'autore?",
            "correct_answer": "oggettivo",
            "choices": ["entusiasta", "oggettivo", "sarcastico", "indifferente"],
            "answer_idx": 1,
        },
    ]


def list_fewshot_sat_math():
    """5-shot curated examples for SAT-Math (Italian)."""
    return [
        {
            "question": "Se $\\frac{x-1}{3}=k$ e $k=3$, qual \u00e8 il valore di $x$?",
            "correct_answer": "10",
            "choices": ["2", "4", "9", "10"],
            "answer_idx": 3,
        },
        {
            "question": "Se $3x + 2 = 14$, qual \u00e8 il valore di $x$?",
            "correct_answer": "4",
            "choices": ["2", "3", "4", "6"],
            "answer_idx": 2,
        },
        {
            "question": "Una funzione \u00e8 definita come $f(x) = 2x^2 - 3x + 1$. Quanto vale $f(2)$?",
            "correct_answer": "3",
            "choices": ["1", "3", "5", "7"],
            "answer_idx": 1,
        },
        {
            "question": "La circonferenza di un cerchio \u00e8 $10\\pi$. Quanto misura il raggio?",
            "correct_answer": "5",
            "choices": ["3", "5", "10", "20"],
            "answer_idx": 1,
        },
        {
            "question": "Se $y = 3x - 7$ e $y = 5$, qual \u00e8 il valore di $x$?",
            "correct_answer": "4",
            "choices": ["2", "3", "4", "5"],
            "answer_idx": 2,
        },
    ]
