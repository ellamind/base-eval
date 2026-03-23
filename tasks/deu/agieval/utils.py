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
    """5-shot curated examples for AQUA-RAT (German)."""
    return [
        {
            "question": "Ein Zug fährt mit 60 km/h und legt eine Strecke von 240 km zurück. Wie lange dauert die Fahrt?",
            "correct_answer": "4 Stunden",
            "choices": ["2 Stunden", "4 Stunden", "6 Stunden", "3 Stunden", "5 Stunden"],
            "answer_idx": 1,
        },
        {
            "question": "Wenn 6 Arbeiter 8 Tage brauchen, um eine Mauer zu bauen, wie viele Tage brauchen 12 Arbeiter für dieselbe Mauer?",
            "correct_answer": "4 Tage",
            "choices": ["4 Tage", "6 Tage", "8 Tage", "2 Tage", "10 Tage"],
            "answer_idx": 0,
        },
        {
            "question": "Ein Händler kauft einen Artikel für 80 € und verkauft ihn mit 25 % Gewinn. Wie hoch ist der Verkaufspreis?",
            "correct_answer": "100 €",
            "choices": ["90 €", "95 €", "100 €", "105 €", "110 €"],
            "answer_idx": 2,
        },
        {
            "question": "Das Verhältnis von Jungen zu Mädchen in einer Klasse beträgt 3:5. Wenn es 24 Jungen gibt, wie viele Mädchen gibt es?",
            "correct_answer": "40",
            "choices": ["30", "35", "40", "45", "50"],
            "answer_idx": 2,
        },
        {
            "question": "Wie hoch ist der einfache Zins auf 5000 € bei 4 % Zinssatz für 3 Jahre?",
            "correct_answer": "600 €",
            "choices": ["400 €", "500 €", "600 €", "700 €", "800 €"],
            "answer_idx": 2,
        },
    ]


def list_fewshot_gaokao_english():
    """3-shot curated examples for Gaokao English (German)."""
    return [
        {
            "question": "UHRENSTEUERUNG\nDies ist eine Uhr, die James Bond mit Stolz tragen würde! Deine elektronische PENGO UHRENSTEUERUNG dient als Fernbedienung für Fernseher und Videos.\nMit Hilfe eines Mr. H kannst du ___.",
            "correct_answer": "deine Hausaufgaben pünktlich erledigen.",
            "choices": [
                "aufhören, Batterien zu verwenden.",
                "deine Hausaufgaben pünktlich erledigen.",
                "dir die Anweisungen deines Lehrers merken.",
                "dein Zimmer auf dem Heimweg aufräumen lassen.",
            ],
            "answer_idx": 1,
        },
        {
            "question": "Ein Schild an der Tür eines Geschäfts lautet: 'Wir sind an 7 Tagen pro Woche und 365 Tagen im Jahr für Sie da.' Was ist der Hauptzweck des Schilds?",
            "correct_answer": "Die Öffnungszeiten des Geschäfts mitzuteilen.",
            "choices": [
                "Die Öffnungszeiten des Geschäfts mitzuteilen.",
                "Neue Mitarbeiter anzuwerben.",
                "Für ein neues Produkt zu werben.",
                "Auf eine Preisänderung hinzuweisen.",
            ],
            "answer_idx": 0,
        },
        {
            "question": "Laut dem Text ist der Hauptgrund, warum Menschen ehrenamtlich arbeiten, ___.",
            "correct_answer": "dass sie anderen helfen und etwas zurückgeben möchten",
            "choices": [
                "dass sie Geld verdienen wollen",
                "dass sie anderen helfen und etwas zurückgeben möchten",
                "dass sie neue Fähigkeiten erlernen möchten",
                "dass sie berufliche Kontakte knüpfen wollen",
            ],
            "answer_idx": 1,
        },
    ]


def list_fewshot_logiqa_en():
    """3-shot curated examples for LogiQA (German)."""
    return [
        {
            "question": "In einem Büro sitzen vier Personen: A, B, C und D. A sitzt B gegenüber. C sitzt rechts von A. Wer sitzt links von B?",
            "correct_answer": "C",
            "choices": ["A", "C", "D", "Keiner"],
            "answer_idx": 1,
        },
        {
            "question": "Alle Philosophen sind Denker. Einige Denker sind Schriftsteller. Welche Schlussfolgerung ist zwingend korrekt?",
            "correct_answer": "Einige Philosophen könnten Schriftsteller sein.",
            "choices": [
                "Alle Schriftsteller sind Philosophen.",
                "Kein Philosoph ist Schriftsteller.",
                "Einige Philosophen könnten Schriftsteller sein.",
                "Alle Denker sind Philosophen.",
            ],
            "answer_idx": 2,
        },
        {
            "question": "Wenn es regnet, wird die Straße nass. Die Straße ist nass. Welche Schlussfolgerung ist korrekt?",
            "correct_answer": "Man kann nicht sicher sagen, ob es geregnet hat.",
            "choices": [
                "Es hat geregnet.",
                "Es hat nicht geregnet.",
                "Man kann nicht sicher sagen, ob es geregnet hat.",
                "Die Straße wurde gereinigt.",
            ],
            "answer_idx": 2,
        },
    ]


def list_fewshot_lsat_ar():
    """3-shot curated examples for LSAT-AR (German)."""
    return [
        {
            "question": "Fünf Vorträge – F, G, H, J und K – werden an einem Tag nacheinander gehalten. G wird vor H gehalten. J wird direkt nach F gehalten. Welche Reihenfolge ist möglich?",
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
            "question": "Ein Blumengeschäft ordnet sieben Sträuße – S, T, U, V, W, X und Y – in einer Reihe. V steht an dritter Stelle. T steht direkt links von U. Welcher Strauß könnte an erster Stelle stehen?",
            "correct_answer": "T",
            "choices": ["U", "V", "T", "Y"],
            "answer_idx": 2,
        },
        {
            "question": "Drei Teams – Rot, Blau und Grün – spielen jeweils zwei Spiele. Rot spielt vor Blau. Grün spielt nicht als Erstes. Welche Reihenfolge der ersten Spiele ist möglich?",
            "correct_answer": "Rot, Grün, Blau",
            "choices": [
                "Blau, Rot, Grün",
                "Grün, Rot, Blau",
                "Rot, Grün, Blau",
                "Rot, Blau, Grün",
            ],
            "answer_idx": 2,
        },
    ]


def list_fewshot_lsat_lr():
    """3-shot curated examples for LSAT-LR (German)."""
    return [
        {
            "question": "Leitartikel: Da die Bevölkerung altert, steigen die Gesundheitskosten. Daher muss die Regierung mehr in Prävention investieren. Welche Annahme liegt diesem Argument zugrunde?",
            "correct_answer": "Präventionsmaßnahmen können die Gesundheitskosten einer alternden Bevölkerung senken.",
            "choices": [
                "Die Bevölkerung wird in Zukunft nicht mehr altern.",
                "Präventionsmaßnahmen können die Gesundheitskosten einer alternden Bevölkerung senken.",
                "Die Regierung gibt derzeit nichts für Prävention aus.",
                "Steigende Gesundheitskosten sind unvermeidlich.",
            ],
            "answer_idx": 1,
        },
        {
            "question": "Kritiker: Dieses Museum zeigt nur Werke bekannter Künstler. Daher versäumt es, aufstrebende Talente zu fördern. Welche Aussage schwächt dieses Argument am meisten?",
            "correct_answer": "Das Museum hat einen eigenen Ausstellungsbereich für neue Künstler.",
            "choices": [
                "Bekannte Künstler ziehen mehr Besucher an.",
                "Das Museum hat einen eigenen Ausstellungsbereich für neue Künstler.",
                "Andere Museen zeigen ebenfalls nur bekannte Künstler.",
                "Aufstrebende Künstler bevorzugen kleinere Galerien.",
            ],
            "answer_idx": 1,
        },
        {
            "question": "Wenn alle Mitarbeiter einer Firma pünktlich sind und Stefan ist ein Mitarbeiter dieser Firma, dann muss Stefan pünktlich sein. Stefan kommt häufig zu spät. Was folgt daraus?",
            "correct_answer": "Nicht alle Mitarbeiter dieser Firma sind pünktlich.",
            "choices": [
                "Stefan ist kein Mitarbeiter der Firma.",
                "Nicht alle Mitarbeiter dieser Firma sind pünktlich.",
                "Stefan ist immer pünktlich.",
                "Die Firma hat keine Regeln zur Pünktlichkeit.",
            ],
            "answer_idx": 1,
        },
    ]


def list_fewshot_lsat_rc():
    """3-shot curated examples for LSAT-RC (German)."""
    return [
        {
            "question": "Rechtsanwälte haben die Pflicht, ihre Mandanten bestmöglich zu verteidigen. Gleichzeitig haben sie eine Verantwortung gegenüber der Gesellschaft. Was beschreibt die Hauptaussage des Textes am besten?",
            "correct_answer": "Anwälte müssen sowohl die Interessen ihrer Mandanten als auch die der Gesellschaft berücksichtigen.",
            "choices": [
                "Anwälte sollten nur die Interessen ihrer Mandanten vertreten.",
                "Anwälte müssen sowohl die Interessen ihrer Mandanten als auch die der Gesellschaft berücksichtigen.",
                "Die Gesellschaft sollte die Arbeit von Anwälten stärker kontrollieren.",
                "Mandanten sollten ihre Anwälte selbst wählen dürfen.",
            ],
            "answer_idx": 1,
        },
        {
            "question": "Der Autor diskutiert verschiedene Ansätze zur Urheberrechtsreform. Welche Position vertritt der Autor hauptsächlich?",
            "correct_answer": "Ein ausgewogener Ansatz, der sowohl Urheber als auch die Öffentlichkeit schützt, ist notwendig.",
            "choices": [
                "Das Urheberrecht sollte vollständig abgeschafft werden.",
                "Ein ausgewogener Ansatz, der sowohl Urheber als auch die Öffentlichkeit schützt, ist notwendig.",
                "Nur Unternehmen sollten Urheberrechte besitzen können.",
                "Das aktuelle System funktioniert einwandfrei.",
            ],
            "answer_idx": 1,
        },
        {
            "question": "Im Text wird die Entwicklung der Umweltgesetzgebung beschrieben. Was war laut dem Text der Hauptgrund für die Verschärfung der Gesetze?",
            "correct_answer": "Zunehmende wissenschaftliche Erkenntnisse über Umweltschäden.",
            "choices": [
                "Zunehmende wissenschaftliche Erkenntnisse über Umweltschäden.",
                "Wirtschaftliche Interessen der Industrie.",
                "Internationale politische Abkommen.",
                "Proteste einzelner Bürger.",
            ],
            "answer_idx": 0,
        },
    ]


def list_fewshot_sat_en():
    """3-shot curated examples for SAT-EN (German)."""
    return [
        {
            "question": "Akira kam direkt und brach mit jeder Tradition. Er pochte an die Tür an einem Winterabend. 'Ich möchte Ihre Tochter Naomi heiraten', sagte er. Welche Aussage beschreibt am besten, was im Text geschieht?",
            "correct_answer": "Eine Figur erhält eine überraschende Bitte von einer anderen Figur.",
            "choices": [
                "Eine Figur streitet sich mit einer anderen Figur.",
                "Eine Figur erhält eine überraschende Bitte von einer anderen Figur.",
                "Eine Figur denkt an vergangene Entscheidungen zurück.",
                "Eine Figur kritisiert eine andere für ihr unerwartetes Verhalten.",
            ],
            "answer_idx": 1,
        },
        {
            "question": "Der Erzähler beschreibt eine Reise durch eine fremde Stadt. Die Straßen waren eng und die Gebäude alt. Was ist die Hauptstimmung des Textes?",
            "correct_answer": "Neugier gemischt mit Unsicherheit.",
            "choices": [
                "Freude und Begeisterung.",
                "Neugier gemischt mit Unsicherheit.",
                "Tiefe Traurigkeit.",
                "Wut und Frustration.",
            ],
            "answer_idx": 1,
        },
        {
            "question": "Im Text vergleicht der Autor zwei wissenschaftliche Theorien. Was ist der Hauptzweck dieses Vergleichs?",
            "correct_answer": "Die Stärken und Schwächen beider Ansätze aufzuzeigen.",
            "choices": [
                "Zu beweisen, dass eine Theorie falsch ist.",
                "Die Stärken und Schwächen beider Ansätze aufzuzeigen.",
                "Eine völlig neue Theorie vorzuschlagen.",
                "Die Geschichte der Wissenschaft zusammenzufassen.",
            ],
            "answer_idx": 1,
        },
    ]


def list_fewshot_sat_en_without_passage():
    """3-shot curated examples for SAT-EN without passage (German)."""
    return [
        {
            "question": "Welche Aussage beschreibt am besten, was im Text geschieht?",
            "correct_answer": "Eine Figur erhält eine überraschende Bitte von einer anderen Figur.",
            "choices": [
                "Eine Figur streitet sich mit einer anderen Figur.",
                "Eine Figur erhält eine überraschende Bitte von einer anderen Figur.",
                "Eine Figur denkt an vergangene Entscheidungen zurück.",
                "Eine Figur kritisiert eine andere für ihr unerwartetes Verhalten.",
            ],
            "answer_idx": 1,
        },
        {
            "question": "Welchen Zweck erfüllt der dritte Absatz im Gesamtzusammenhang des Textes?",
            "correct_answer": "Er liefert ein konkretes Beispiel für die zuvor aufgestellte These.",
            "choices": [
                "Er widerlegt das Hauptargument.",
                "Er liefert ein konkretes Beispiel für die zuvor aufgestellte These.",
                "Er führt ein völlig neues Thema ein.",
                "Er fasst den gesamten Text zusammen.",
            ],
            "answer_idx": 1,
        },
        {
            "question": "Welches Wort beschreibt am besten den Ton des Autors?",
            "correct_answer": "sachlich",
            "choices": ["begeistert", "sachlich", "sarkastisch", "gleichgültig"],
            "answer_idx": 1,
        },
    ]


def list_fewshot_sat_math():
    """5-shot curated examples for SAT-Math (German)."""
    return [
        {
            "question": "Wenn $\\frac{x-1}{3}=k$ und $k=3$ gilt, was ist der Wert von $x$?",
            "correct_answer": "10",
            "choices": ["2", "4", "9", "10"],
            "answer_idx": 3,
        },
        {
            "question": "Wenn $3x + 2 = 14$, was ist der Wert von $x$?",
            "correct_answer": "4",
            "choices": ["2", "3", "4", "6"],
            "answer_idx": 2,
        },
        {
            "question": "Eine Funktion ist definiert als $f(x) = 2x^2 - 3x + 1$. Was ist $f(2)$?",
            "correct_answer": "3",
            "choices": ["1", "3", "5", "7"],
            "answer_idx": 1,
        },
        {
            "question": "Der Umfang eines Kreises beträgt $10\\pi$. Wie groß ist der Radius?",
            "correct_answer": "5",
            "choices": ["3", "5", "10", "20"],
            "answer_idx": 1,
        },
        {
            "question": "Wenn $y = 3x - 7$ und $y = 5$, was ist der Wert von $x$?",
            "correct_answer": "4",
            "choices": ["2", "3", "4", "5"],
            "answer_idx": 2,
        },
    ]
