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
    """5-shot curated examples for AQUA-RAT (Spanish)."""
    return [
        {
            "question": "Un tren viaja a 60 km/h y recorre una distancia de 240 km. \u00bfCu\u00e1nto dura el viaje?",
            "correct_answer": "4 horas",
            "choices": ["2 horas", "4 horas", "6 horas", "3 horas", "5 horas"],
            "answer_idx": 1,
        },
        {
            "question": "Si 6 trabajadores necesitan 8 d\u00edas para construir un muro, \u00bfcu\u00e1ntos d\u00edas necesitan 12 trabajadores para el mismo muro?",
            "correct_answer": "4 d\u00edas",
            "choices": ["4 d\u00edas", "6 d\u00edas", "8 d\u00edas", "2 d\u00edas", "10 d\u00edas"],
            "answer_idx": 0,
        },
        {
            "question": "Un comerciante compra un art\u00edculo por 80 \u20ac y lo vende con un 25 % de ganancia. \u00bfCu\u00e1l es el precio de venta?",
            "correct_answer": "100 \u20ac",
            "choices": ["90 \u20ac", "95 \u20ac", "100 \u20ac", "105 \u20ac", "110 \u20ac"],
            "answer_idx": 2,
        },
        {
            "question": "La proporci\u00f3n de chicos a chicas en una clase es de 3:5. Si hay 24 chicos, \u00bfcu\u00e1ntas chicas hay?",
            "correct_answer": "40",
            "choices": ["30", "35", "40", "45", "50"],
            "answer_idx": 2,
        },
        {
            "question": "\u00bfCu\u00e1l es el inter\u00e9s simple sobre 5000 \u20ac a una tasa del 4 % durante 3 a\u00f1os?",
            "correct_answer": "600 \u20ac",
            "choices": ["400 \u20ac", "500 \u20ac", "600 \u20ac", "700 \u20ac", "800 \u20ac"],
            "answer_idx": 2,
        },
    ]


def list_fewshot_gaokao_english():
    """3-shot curated examples for Gaokao English (Spanish)."""
    return [
        {
            "question": "CONTROL DE RELOJ\nEste es un reloj que James Bond llevar\u00eda con orgullo. Tu CONTROL DE RELOJ electr\u00f3nico PENGO sirve como mando a distancia para televisores y v\u00eddeos.\nCon la ayuda de un Mr. H puedes ___.",
            "correct_answer": "terminar tus deberes a tiempo.",
            "choices": [
                "dejar de usar pilas.",
                "terminar tus deberes a tiempo.",
                "recordar las instrucciones de tu profesor.",
                "hacer que limpien tu habitaci\u00f3n de camino a casa.",
            ],
            "answer_idx": 1,
        },
        {
            "question": "Un cartel en la puerta de una tienda dice: 'Estamos abiertos para usted los 7 d\u00edas de la semana y los 365 d\u00edas del a\u00f1o.' \u00bfCu\u00e1l es el prop\u00f3sito principal del cartel?",
            "correct_answer": "Informar sobre el horario de apertura de la tienda.",
            "choices": [
                "Informar sobre el horario de apertura de la tienda.",
                "Reclutar nuevos empleados.",
                "Promocionar un nuevo producto.",
                "Indicar un cambio de precios.",
            ],
            "answer_idx": 0,
        },
        {
            "question": "Seg\u00fan el texto, la raz\u00f3n principal por la que las personas hacen voluntariado es ___.",
            "correct_answer": "que quieren ayudar a los dem\u00e1s y contribuir a la sociedad",
            "choices": [
                "que quieren ganar dinero",
                "que quieren ayudar a los dem\u00e1s y contribuir a la sociedad",
                "que quieren aprender nuevas habilidades",
                "que quieren establecer contactos profesionales",
            ],
            "answer_idx": 1,
        },
    ]


def list_fewshot_logiqa_en():
    """3-shot curated examples for LogiQA (Spanish)."""
    return [
        {
            "question": "En una oficina est\u00e1n sentadas cuatro personas: A, B, C y D. A est\u00e1 frente a B. C est\u00e1 a la derecha de A. \u00bfQui\u00e9n est\u00e1 a la izquierda de B?",
            "correct_answer": "C",
            "choices": ["A", "C", "D", "Ninguno"],
            "answer_idx": 1,
        },
        {
            "question": "Todos los fil\u00f3sofos son pensadores. Algunos pensadores son escritores. \u00bfQu\u00e9 conclusi\u00f3n es necesariamente correcta?",
            "correct_answer": "Algunos fil\u00f3sofos podr\u00edan ser escritores.",
            "choices": [
                "Todos los escritores son fil\u00f3sofos.",
                "Ning\u00fan fil\u00f3sofo es escritor.",
                "Algunos fil\u00f3sofos podr\u00edan ser escritores.",
                "Todos los pensadores son fil\u00f3sofos.",
            ],
            "answer_idx": 2,
        },
        {
            "question": "Si llueve, la calle se moja. La calle est\u00e1 mojada. \u00bfQu\u00e9 conclusi\u00f3n es correcta?",
            "correct_answer": "No se puede afirmar con seguridad si ha llovido.",
            "choices": [
                "Ha llovido.",
                "No ha llovido.",
                "No se puede afirmar con seguridad si ha llovido.",
                "La calle fue limpiada.",
            ],
            "answer_idx": 2,
        },
    ]


def list_fewshot_lsat_ar():
    """3-shot curated examples for LSAT-AR (Spanish)."""
    return [
        {
            "question": "Cinco conferencias \u2013 F, G, H, J y K \u2013 se imparten consecutivamente en un d\u00eda. G se imparte antes que H. J se imparte justo despu\u00e9s de F. \u00bfQu\u00e9 orden es posible?",
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
            "question": "Una florist\u00eda ordena siete ramos \u2013 S, T, U, V, W, X e Y \u2013 en una fila. V est\u00e1 en el tercer lugar. T est\u00e1 justo a la izquierda de U. \u00bfQu\u00e9 ramo podr\u00eda estar en el primer lugar?",
            "correct_answer": "T",
            "choices": ["U", "V", "T", "Y"],
            "answer_idx": 2,
        },
        {
            "question": "Tres equipos \u2013 Rojo, Azul y Verde \u2013 juegan dos partidos cada uno. Rojo juega antes que Azul. Verde no juega primero. \u00bfQu\u00e9 orden de los primeros partidos es posible?",
            "correct_answer": "Rojo, Verde, Azul",
            "choices": [
                "Azul, Rojo, Verde",
                "Verde, Rojo, Azul",
                "Rojo, Verde, Azul",
                "Rojo, Azul, Verde",
            ],
            "answer_idx": 2,
        },
    ]


def list_fewshot_lsat_lr():
    """3-shot curated examples for LSAT-LR (Spanish)."""
    return [
        {
            "question": "Editorial: A medida que la poblaci\u00f3n envejece, los costos sanitarios aumentan. Por lo tanto, el gobierno debe invertir m\u00e1s en prevenci\u00f3n. \u00bfQu\u00e9 supuesto subyace a este argumento?",
            "correct_answer": "Las medidas preventivas pueden reducir los costos sanitarios de una poblaci\u00f3n que envejece.",
            "choices": [
                "La poblaci\u00f3n no seguir\u00e1 envejeciendo en el futuro.",
                "Las medidas preventivas pueden reducir los costos sanitarios de una poblaci\u00f3n que envejece.",
                "El gobierno actualmente no gasta nada en prevenci\u00f3n.",
                "El aumento de los costos sanitarios es inevitable.",
            ],
            "answer_idx": 1,
        },
        {
            "question": "Cr\u00edtico: Este museo solo exhibe obras de artistas conocidos. Por lo tanto, no fomenta el talento emergente. \u00bfQu\u00e9 afirmaci\u00f3n debilita m\u00e1s este argumento?",
            "correct_answer": "El museo tiene una sala de exposiciones dedicada a nuevos artistas.",
            "choices": [
                "Los artistas conocidos atraen a m\u00e1s visitantes.",
                "El museo tiene una sala de exposiciones dedicada a nuevos artistas.",
                "Otros museos tambi\u00e9n exhiben solo artistas conocidos.",
                "Los artistas emergentes prefieren galer\u00edas m\u00e1s peque\u00f1as.",
            ],
            "answer_idx": 1,
        },
        {
            "question": "Si todos los empleados de una empresa son puntuales y Stefan es un empleado de esa empresa, entonces Stefan debe ser puntual. Stefan llega tarde con frecuencia. \u00bfQu\u00e9 se deduce de esto?",
            "correct_answer": "No todos los empleados de esa empresa son puntuales.",
            "choices": [
                "Stefan no es un empleado de la empresa.",
                "No todos los empleados de esa empresa son puntuales.",
                "Stefan siempre es puntual.",
                "La empresa no tiene normas de puntualidad.",
            ],
            "answer_idx": 1,
        },
    ]


def list_fewshot_lsat_rc():
    """3-shot curated examples for LSAT-RC (Spanish)."""
    return [
        {
            "question": "Los abogados tienen el deber de defender a sus clientes de la mejor manera posible. Al mismo tiempo, tienen una responsabilidad ante la sociedad. \u00bfQu\u00e9 describe mejor la idea principal del texto?",
            "correct_answer": "Los abogados deben considerar tanto los intereses de sus clientes como los de la sociedad.",
            "choices": [
                "Los abogados solo deben representar los intereses de sus clientes.",
                "Los abogados deben considerar tanto los intereses de sus clientes como los de la sociedad.",
                "La sociedad deber\u00eda controlar m\u00e1s el trabajo de los abogados.",
                "Los clientes deber\u00edan poder elegir libremente a sus abogados.",
            ],
            "answer_idx": 1,
        },
        {
            "question": "El autor analiza distintos enfoques para la reforma de los derechos de autor. \u00bfCu\u00e1l es la posici\u00f3n principal del autor?",
            "correct_answer": "Es necesario un enfoque equilibrado que proteja tanto a los autores como al p\u00fablico.",
            "choices": [
                "Los derechos de autor deber\u00edan abolirse por completo.",
                "Es necesario un enfoque equilibrado que proteja tanto a los autores como al p\u00fablico.",
                "Solo las empresas deber\u00edan poder poseer derechos de autor.",
                "El sistema actual funciona perfectamente.",
            ],
            "answer_idx": 1,
        },
        {
            "question": "En el texto se describe la evoluci\u00f3n de la legislaci\u00f3n medioambiental. Seg\u00fan el texto, \u00bfcu\u00e1l fue la raz\u00f3n principal del endurecimiento de las leyes?",
            "correct_answer": "El aumento de las evidencias cient\u00edficas sobre los da\u00f1os medioambientales.",
            "choices": [
                "El aumento de las evidencias cient\u00edficas sobre los da\u00f1os medioambientales.",
                "Los intereses econ\u00f3micos de la industria.",
                "Acuerdos pol\u00edticos internacionales.",
                "Protestas de ciudadanos individuales.",
            ],
            "answer_idx": 0,
        },
    ]


def list_fewshot_sat_en():
    """3-shot curated examples for SAT-EN (Spanish)."""
    return [
        {
            "question": "Akira lleg\u00f3 directamente y rompi\u00f3 con toda tradici\u00f3n. Llam\u00f3 a la puerta una noche de invierno. 'Quiero casarme con su hija Naomi', dijo. \u00bfQu\u00e9 afirmaci\u00f3n describe mejor lo que ocurre en el texto?",
            "correct_answer": "Un personaje recibe una petici\u00f3n sorprendente de otro personaje.",
            "choices": [
                "Un personaje discute con otro personaje.",
                "Un personaje recibe una petici\u00f3n sorprendente de otro personaje.",
                "Un personaje reflexiona sobre decisiones pasadas.",
                "Un personaje critica a otro por su comportamiento inesperado.",
            ],
            "answer_idx": 1,
        },
        {
            "question": "El narrador describe un viaje por una ciudad desconocida. Las calles eran estrechas y los edificios antiguos. \u00bfCu\u00e1l es el tono principal del texto?",
            "correct_answer": "Curiosidad mezclada con incertidumbre.",
            "choices": [
                "Alegr\u00eda y entusiasmo.",
                "Curiosidad mezclada con incertidumbre.",
                "Profunda tristeza.",
                "Ira y frustraci\u00f3n.",
            ],
            "answer_idx": 1,
        },
        {
            "question": "En el texto, el autor compara dos teor\u00edas cient\u00edficas. \u00bfCu\u00e1l es el prop\u00f3sito principal de esta comparaci\u00f3n?",
            "correct_answer": "Mostrar las fortalezas y debilidades de ambos enfoques.",
            "choices": [
                "Demostrar que una teor\u00eda es incorrecta.",
                "Mostrar las fortalezas y debilidades de ambos enfoques.",
                "Proponer una teor\u00eda completamente nueva.",
                "Resumir la historia de la ciencia.",
            ],
            "answer_idx": 1,
        },
    ]


def list_fewshot_sat_en_without_passage():
    """3-shot curated examples for SAT-EN without passage (Spanish)."""
    return [
        {
            "question": "\u00bfQu\u00e9 afirmaci\u00f3n describe mejor lo que ocurre en el texto?",
            "correct_answer": "Un personaje recibe una petici\u00f3n sorprendente de otro personaje.",
            "choices": [
                "Un personaje discute con otro personaje.",
                "Un personaje recibe una petici\u00f3n sorprendente de otro personaje.",
                "Un personaje reflexiona sobre decisiones pasadas.",
                "Un personaje critica a otro por su comportamiento inesperado.",
            ],
            "answer_idx": 1,
        },
        {
            "question": "\u00bfQu\u00e9 funci\u00f3n cumple el tercer p\u00e1rrafo en el contexto general del texto?",
            "correct_answer": "Proporciona un ejemplo concreto de la tesis planteada anteriormente.",
            "choices": [
                "Refuta el argumento principal.",
                "Proporciona un ejemplo concreto de la tesis planteada anteriormente.",
                "Introduce un tema completamente nuevo.",
                "Resume todo el texto.",
            ],
            "answer_idx": 1,
        },
        {
            "question": "\u00bfQu\u00e9 palabra describe mejor el tono del autor?",
            "correct_answer": "objetivo",
            "choices": ["entusiasta", "objetivo", "sarc\u00e1stico", "indiferente"],
            "answer_idx": 1,
        },
    ]


def list_fewshot_sat_math():
    """5-shot curated examples for SAT-Math (Spanish)."""
    return [
        {
            "question": "Si $\\frac{x-1}{3}=k$ y $k=3$, \u00bfcu\u00e1l es el valor de $x$?",
            "correct_answer": "10",
            "choices": ["2", "4", "9", "10"],
            "answer_idx": 3,
        },
        {
            "question": "Si $3x + 2 = 14$, \u00bfcu\u00e1l es el valor de $x$?",
            "correct_answer": "4",
            "choices": ["2", "3", "4", "6"],
            "answer_idx": 2,
        },
        {
            "question": "Una funci\u00f3n se define como $f(x) = 2x^2 - 3x + 1$. \u00bfCu\u00e1nto es $f(2)$?",
            "correct_answer": "3",
            "choices": ["1", "3", "5", "7"],
            "answer_idx": 1,
        },
        {
            "question": "La circunferencia de un c\u00edrculo mide $10\\pi$. \u00bfCu\u00e1l es el radio?",
            "correct_answer": "5",
            "choices": ["3", "5", "10", "20"],
            "answer_idx": 1,
        },
        {
            "question": "Si $y = 3x - 7$ e $y = 5$, \u00bfcu\u00e1l es el valor de $x$?",
            "correct_answer": "4",
            "choices": ["2", "3", "4", "5"],
            "answer_idx": 2,
        },
    ]
