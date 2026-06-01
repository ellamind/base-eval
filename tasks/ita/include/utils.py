"""Utility functions for INCLUDE Italian tasks (MC, RC, BPB).

Fixes the upstream lm-eval-harness INCLUDE task by:
- Using Italian prompts (Domanda:/Risposta:) instead of English
- Shuffling answer options deterministically to avoid position bias
- Adding RC (rank choice) and BPB (bits-per-byte) variants
- 5-shot from validation split (same domain; "Other" for Professional certification)
"""

import hashlib
import math
import random
from functools import partial

LABELS = "ABCD"

DOMAINS = [
    "Applied Science",
    "Arts & Humanities",
    "Health oriented education",
    "Professional certification",
    "STEM",
    "Social Science",
]


# ---------------------------------------------------------------------------
# Dataset preprocessing
# ---------------------------------------------------------------------------

def _shuffle_options(doc):
    """Shuffle the 4 options deterministically and update the answer index."""
    options = [doc["option_a"], doc["option_b"], doc["option_c"], doc["option_d"]]
    correct_idx = doc["answer"]
    correct_text = options[correct_idx]

    seed = int(hashlib.md5(doc["question"].encode()).hexdigest(), 16)
    rng = random.Random(seed)
    rng.shuffle(options)

    doc["shuffled_options"] = options
    doc["shuffled_answer"] = options.index(correct_text)
    return doc


def _filter_domain(dataset, domain):
    """Filter dataset to a single domain, then shuffle options."""
    return dataset.filter(lambda x: x["domain"] == domain).map(_shuffle_options)


def _raw_to_fewshot(raw):
    """Convert a raw validation example into a shuffled few-shot dict."""
    return _shuffle_options({
        "question": raw["question"],
        "option_a": raw["option_a"],
        "option_b": raw["option_b"],
        "option_c": raw["option_c"],
        "option_d": raw["option_d"],
        "answer": raw["answer"],
    })


process_applied_science = partial(_filter_domain, domain="Applied Science")
process_arts_humanities = partial(_filter_domain, domain="Arts & Humanities")
process_health_education = partial(_filter_domain, domain="Health oriented education")
process_professional_certification = partial(_filter_domain, domain="Professional certification")
process_stem = partial(_filter_domain, domain="STEM")
process_social_science = partial(_filter_domain, domain="Social Science")


# ---------------------------------------------------------------------------
# 5-shot examples from validation split (curated per domain)
# ---------------------------------------------------------------------------

_FEWSHOT_APPLIED_SCIENCE_RAW = [
    {"question": "Nella fermentazione del mosto d'uva si sviluppa:", "option_a": "anidride carbonica", "option_b": "acido solfidrico", "option_c": "acido nitrico", "option_d": "acido cloridrico", "answer": 0},
    {"question": "Con il termine condensazione s'intende:", "option_a": "il passaggio di una sostanza dallo stato gassoso a quello liquido", "option_b": "il passaggio di una sostanza dallo stato di vapore a quello liquido", "option_c": "il passaggio di una sostanza dallo stato solido a quello gassoso", "option_d": "il passaggio di una sostanza dallo stato solido a quello liquido", "answer": 1},
    {"question": "L'elettronegatività è:", "option_a": "la capacità che ha un atomo di attrarre gli elettroni di legame", "option_b": "la capacità che ha un atomo di cedere elettroni", "option_c": "una proprietà intrinseca degli elettroni", "option_d": "una proprietà intrinseca dei neutroni", "answer": 0},
    {"question": "Quale, fra i seguenti composti, ha legami con maggiore carattere ionico?", "option_a": "NaCl", "option_b": "HCl", "option_c": "CCl_4", "option_d": "AlCl_3", "answer": 0},
    {"question": "Il processo di traduzione che porta alla sintesi di una catena polipeptidica richiede tutti i fattori elencati tranne uno. Quale?", "option_a": "Ribosomi", "option_b": "DNA stampo", "option_c": "RNA messaggero", "option_d": "Amminoacidi", "answer": 1},
]

_FEWSHOT_ARTS_HUMANITIES_RAW = [
    {"question": "Come si chiama la tomba sopra la quale viene innalzato un poggio in terra o in pietra?", "option_a": "Gradina", "option_b": "Labirinto", "option_c": "Mastaba", "option_d": "Tumulo", "answer": 3},
    {"question": "Kojoj religiji pripada učenje o četirima plemenitim istinama i osmerostrukome putu oslobođenja od patnje?", "option_a": "kršćanstvu", "option_b": "budizmu", "option_c": "islamu", "option_d": "židovstvu", "answer": 1},
    {"question": "Un pacchetto azionario, del valore iniziale di 50.000 euro, ha fruttato il primo anno il 50%, il secondo il 10% e il terzo il 20%. Qual è il valore finale del pacchetto?", "option_a": "99.000 euro", "option_b": "90.750 euro", "option_c": "115.500 euro", "option_d": "49.000 euro", "answer": 0},
    {"question": "Koja je od navedenih maksima u skladu s Kantovom etičkom teorijom?", "option_a": "Smijem dati lažno obećanje ako time izbjegavam neko zlo.", "option_b": "Ne smijem nikada dati lažno obećanje.", "option_c": "Smijem dati lažno obećanje samo ako će laž proizvesti veće dobro od govorenja istine.", "option_d": "Smijem dati lažno obećanje samo ako time postižem neko dobro.", "answer": 1},
    {"question": "Quale bano bosniaco incentivava lo sviluppo dell'attività mineraria in Bosnia invitando i Sassoni all'apertura delle miniere di oro, d'argento e di piombo?", "option_a": "Tvrtko I Kotromanić", "option_b": "Kulin", "option_c": "Borić", "option_d": "Stjepan II Kotromanić", "answer": 3},
]

_FEWSHOT_HEALTH_EDUCATION_RAW = [
    {"question": "Quale dei seguenti processi fisiologici distingue i vegetali dagli animali?", "option_a": "Fotosintesi", "option_b": "Assorbimento di sostanze nutritive esogene", "option_c": "Metabolismo anaerobico", "option_d": "Fermentazione", "answer": 0},
    {"question": "Nelle prime fasi della ricerca scientifica l'impegno principale risiede nel descrivere i fenomeni osservati e nel classificarli a seconda delle loro caratteristiche. Successivamente le misure (quantificazioni) sostituiscono le descrizioni qualitative. In una fase ancora successiva, i dati quantitativi possono essere descritti da alcune concise affermazioni (o equazioni matematiche) chiamate leggi. È possibile, talvolta, costruire una teoria che spiega più leggi tra loro differenti mediante pochi principi generali. Esempi di teorie o principi generali unificanti sono, in biologia, la teoria dell'evoluzione, ed in chimica la teoria atomica e molecolare della materia. Teorie e leggi sono spesso soggette a modifiche più o meno rilevanti man mano che vengono eseguiti nuovi esperimenti e fatte nuove osservazioni. Ad esempio, la teoria della gravitazione di Newton fu modificata dalle teorie di Einstein, che, a loro volta, possono essere oggetto di perfezionamenti e modifiche. Quale delle seguenti affermazioni PUÒ essere dedotta dalla lettura del brano di cui sopra?", "option_a": "Una legge consiste in uno o più principi generali unificanti", "option_b": "L'insieme di più leggi costituisce sempre una teoria", "option_c": "È possibile enunciare una legge mediante un'equazione matematica", "option_d": "Solo a seguito delle osservazioni di Einstein è stato possibile formulare la teoria atomica e molecolare della materia", "answer": 2},
    {"question": "Due rette di equazioni y = mx e y = nx sono tra loro sempre perpendicolari se:", "option_a": "mn = 1", "option_b": "m = n", "option_c": "mn = - 1", "option_d": "mn = 0,5", "answer": 2},
    {"question": "Quale serie di segni posta nell'ordine davanti ai seguenti numeri 12; 17; 9; 17; 18 dà come risultato 15?", "option_a": "- + - + -", "option_b": "- - + + +", "option_c": "+ - + - +", "option_d": "+ + - - -", "answer": 1},
    {"question": "Se non è vera la proposizione \"A tutti i gatti neri piace il pesce\", si può affermare che:", "option_a": "Esiste almeno un gatto nero a cui non piace il pesce", "option_b": "A tutti i gatti neri non piace la carne", "option_c": "Il pesce piace a tutti i gatti", "option_d": "Il pesce piace anche a qualche gatto che non è nero", "answer": 0},
]

_FEWSHOT_PROFESSIONAL_CERTIFICATION_RAW = [
    {"question": "A 48 ore dall'esordio di una torsione del testicolo quale tra le seguenti è la diagnosi differenziale più difficile:", "option_a": "Idrocele", "option_b": "Seminoma", "option_c": "Orchiepididimite acuta aspecifica", "option_d": "Epididimite tubercolare", "answer": 2},
    {"question": "Conclusione, interpretazione e adempimento del contratto - Adempimento del contratto   Può il creditore rifiutare l'adempimento parziale di una prestazione pecuniaria divisibile?", "option_a": "No, a meno che la prestazione principale sia eseguita con gli interessi e la rivalutazione monetaria", "option_b": "No, se si oppone il coniuge del creditore in regime di comunione legale", "option_c": "Sì, salvo che la legge e gli usi dispongano diversamente", "option_d": "No, mai", "answer": 2},
    {"question": "L'esame diagnostico piu' importante in un paziente con il forte sospetto di carcinoma esofageo e':", "option_a": "una TC del torace", "option_b": "una esofagoscopia", "option_c": "un esame Rx con bario", "option_d": "una TC dell'addome superiore", "answer": 1},
    {"question": "Ai sensi del Testo Unico della Finanza (TUF), per il perseguimento degli obiettivi della vigilanza, la Commissione nazionale per le società e la borsa (Consob) è competente per quanto riguarda...", "option_a": "La trasparenza e correttezza dei comportamenti", "option_b": "I controlli antitrust sugli intermediari creditizi", "option_c": "I controlli di stabilità sugli intermediari mobiliari", "option_d": "La stabilità dei tassi di interesse del mercato monetario", "answer": 0},
    {"question": "Con riferimento agli Accordi di Basilea, che cosa si intende per prociclicità?", "option_a": "L'accentuazione delle fluttuazioni del ciclo economico a causa dell'aumento dei requisiti patrimoniali durante le fasi recessive del ciclo economico", "option_b": "La possibilità di incrementare l'erogazione di finanziamenti al peggiorare del ciclo economico", "option_c": "Nessuna delle precedenti", "option_d": "La diluizione delle fluttuazioni del ciclo economico, poiché i requisiti patrimoniali tendono ad aumentare durante le fasi recessive del ciclo economico", "answer": 0},
]

_FEWSHOT_STEM_RAW = [
    {"question": "-46) Quali tra i seguenti composti possono essere generati dall'idrolisi di una glicoproteina?", "option_a": "amminoacidi e monosaccaridi", "option_b": "amminoacidi e acidi grassi", "option_c": "amminoacidi e nucleotidi", "option_d": "C- amminoacidi e glicogeno", "answer": 0},
    {"question": "Cosa usano le stampanti a matrice per stampare un testo?", "option_a": "cartuccia con inchiostro", "option_b": "nastro inchiostrato", "option_c": "cartuccia con tonner", "option_d": "cartuccia con polvere d'inchiostro", "answer": 1},
    {"question": "-31) II DNA è formato da:", "option_a": "due filamenti polinucleotidici avvolti ad elica", "option_b": "un filamento polinucleotidico", "option_c": "due filamenti di amminoacidi avvolti ad elica", "option_d": "una sequenza semplice di aminoacidi", "answer": 0},
    {"question": "186. Una cellula umana che contiene 22 autosomi e un cromosoma Y è:", "option_a": "una cellula uovo", "option_b": "uno spermatozoo", "option_c": "uno zigote", "option_d": "una cellula somatica di un individuo di sesso maschile", "answer": 1},
    {"question": "Considerati quattro condensatori C1, C2 rispettivamente di 8 mF e 12 mF in serie tra loro ed in parallelo con C3 di 20 mF e C4 di 5 mF, qual è la capacità equivalente del sistema?", "option_a": "29,8 mF", "option_b": "8,8 mF", "option_c": "24,8 mF", "option_d": "45 mF", "answer": 0},
]

_FEWSHOT_SOCIAL_SCIENCE_RAW = [
    {"question": "Anche se i genitori hanno educato i loro due figli allo stesso modo, uno reagisce in modo molto esplosivo, mentre l'altro è contenuto e calmo nelle sue reazioni. A quale caratteristica possono venir riferite, nella maggior parte dei casi, le differenze nelle loro reazioni?", "option_a": "al temperamento", "option_b": "all'intelligenza", "option_c": "al carattere", "option_d": "alle abitudini", "answer": 0},
    {"question": "Fra le caratteristiche elencate di seguito, qual è la caratteristica più frequente nella vecchiaia?", "option_a": "diminuzione della quantità di ruoli sociali", "option_b": "l'apice dell'intelligenza fluida", "option_c": "diminuzione del tempo libero", "option_d": "l'apice della capacità lavorativa", "answer": 0},
    {"question": "Quale delle seguenti frasi contiene un predicato nominale?", "option_a": "Maria è a casa da sola questa settimana", "option_b": "Giorgio è stato a letto tutto il giorno", "option_c": "Il viaggio è stato lungo e faticoso", "option_d": "C'è una vastissima scelta in quel supermercato", "answer": 2},
    {"question": "Quale fra le seguenti emozioni verrà espressa dalla persona cieca con la stessa intensità di espressione del volto di una persona vedente?", "option_a": "disprezzo", "option_b": "invidia", "option_c": "paura", "option_d": "gelosia", "answer": 2},
    {"question": "Anche se i genitori hanno educato i loro due figli allo stesso modo, uno reagisce in modo molto esplosivo, mentre l'altro è contenuto e calmo nelle sue reazioni. A quale caratteristica possono venir riferite, nella maggior parte dei casi, le differenze nelle loro reazioni?", "option_a": "al temperamento", "option_b": "all'intelligenza", "option_c": "al carattere", "option_d": "alle abitudini", "answer": 0},
]


def list_fewshot_applied_science():
    return [_raw_to_fewshot(r) for r in _FEWSHOT_APPLIED_SCIENCE_RAW]


def list_fewshot_arts_humanities():
    return [_raw_to_fewshot(r) for r in _FEWSHOT_ARTS_HUMANITIES_RAW]


def list_fewshot_health_education():
    return [_raw_to_fewshot(r) for r in _FEWSHOT_HEALTH_EDUCATION_RAW]


def list_fewshot_professional_certification():
    return [_raw_to_fewshot(r) for r in _FEWSHOT_PROFESSIONAL_CERTIFICATION_RAW]


def list_fewshot_stem():
    return [_raw_to_fewshot(r) for r in _FEWSHOT_STEM_RAW]


def list_fewshot_social_science():
    return [_raw_to_fewshot(r) for r in _FEWSHOT_SOCIAL_SCIENCE_RAW]


# ---------------------------------------------------------------------------
# MC variant — letter-labeled choices (A/B/C/D) in prompt
# ---------------------------------------------------------------------------

def doc_to_text_mc(doc):
    options = doc["shuffled_options"]
    choices_text = "\n".join(
        f" {LABELS[i]}. {options[i]}" for i in range(len(options))
    )
    return f"Domanda: {doc['question'].strip()}\n{choices_text}\nRisposta:"


def doc_to_choice_mc(doc):
    return list(LABELS[:len(doc["shuffled_options"])])


def doc_to_target_mc(doc):
    return doc["shuffled_answer"]


# ---------------------------------------------------------------------------
# RC variant — question only, score full answer texts
# ---------------------------------------------------------------------------

def doc_to_text_rc(doc):
    return f"Domanda: {doc['question'].strip()}\nRisposta:"


def doc_to_choice_rc(doc):
    return doc["shuffled_options"]


# ---------------------------------------------------------------------------
# BPB variant — score only the gold answer, normalize by byte length
# ---------------------------------------------------------------------------

def doc_to_target_bpb(doc):
    return f" {doc['shuffled_options'][doc['shuffled_answer']]}"


def process_results_bpb(doc, results):
    """BPB = -log2 P(answer | context) / bytes(answer)."""
    ll, _is_greedy = results[0]

    gold_text = doc["shuffled_options"][doc["shuffled_answer"]]
    gold_bytes = len((" " + gold_text).encode("utf-8"))
    bpb = -ll / (math.log(2) * max(gold_bytes, 1))

    return {"bits_per_byte": bpb}
