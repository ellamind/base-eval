import math
import re


# ---------------------------------------------------------------------------
# Dataset names (hails/agieval-* HuggingFace datasets)
# ---------------------------------------------------------------------------

SUBSETS = [
    "aqua-rat",
    "gaokao-english",
    "logiqa-en",
    "lsat-ar",
    "lsat-lr",
    "lsat-rc",
    "sat-en",
    "sat-en-without-passage",
    "sat-math",
]


# ---------------------------------------------------------------------------
# Choice / query cleaning  (following OLMES _strip_choice_prefix + _clean_text)
# ---------------------------------------------------------------------------


def _strip_choice_prefix(choice):
    """Strip '(A)', ' A.' etc. prefixes from a choice string."""
    return re.sub(r"^\s*\([A-E]\)\s*|^\s*[A-E][.?]?\s*", "", choice)


def _clean_query(query):
    """Strip 'Q: ', 'Answer Choices: (A)...' and 'A: Among...' boilerplate from query."""
    # Remove leading "Q: "
    q = re.sub(r"^Q:\s*", "", query)
    # Remove trailing "Answer Choices: (A)... (B)..." block
    q = re.sub(r"\s*Answer Choices:\s*\(A\).*$", "", q, flags=re.DOTALL)
    # Remove trailing "A: Among A through ..., the answer is"
    q = re.sub(r"\s*A:\s*Among.*$", "", q, flags=re.DOTALL)
    return q.strip()


def prepare(dataset):
    """Clean choices and query for all rows."""
    return dataset.map(_prepare_doc)


def _prepare_doc(doc):
    doc["clean_query"] = _clean_query(doc["query"])
    doc["clean_choices"] = [_strip_choice_prefix(c) for c in doc["choices"]]
    # gold is a list of valid indices; take the first for single-answer evaluation
    doc["answer_idx"] = doc["gold"][0]
    doc["correct_answer"] = doc["clean_choices"][doc["answer_idx"]]
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
# 3-shot default, 5-shot for aqua-rat and sat-math (matching OLMES).
# ---------------------------------------------------------------------------


def list_fewshot_aqua_rat():
    """5-shot curated examples for AQUA-RAT."""
    return [
        {
            "clean_query": "A train travels at 60 km/h and covers a distance of 240 km. How long does the journey take?",
            "correct_answer": "4 hours",
            "clean_choices": ["2 hours", "4 hours", "6 hours", "3 hours", "5 hours"],
            "answer_idx": 1,
        },
        {
            "clean_query": "If 6 workers take 8 days to build a wall, how many days would 12 workers take for the same wall?",
            "correct_answer": "4 days",
            "clean_choices": ["4 days", "6 days", "8 days", "2 days", "10 days"],
            "answer_idx": 0,
        },
        {
            "clean_query": "A merchant buys an article for $80 and sells it at 25% profit. What is the selling price?",
            "correct_answer": "$100",
            "clean_choices": ["$90", "$95", "$100", "$105", "$110"],
            "answer_idx": 2,
        },
        {
            "clean_query": "The ratio of boys to girls in a class is 3:5. If there are 24 boys, how many girls are there?",
            "correct_answer": "40",
            "clean_choices": ["30", "35", "40", "45", "50"],
            "answer_idx": 2,
        },
        {
            "clean_query": "What is the simple interest on $5000 at 4% rate for 3 years?",
            "correct_answer": "$600",
            "clean_choices": ["$400", "$500", "$600", "$700", "$800"],
            "answer_idx": 2,
        },
    ]


def list_fewshot_gaokao_english():
    """3-shot curated examples for Gaokao English."""
    return [
        {
            "clean_query": "WATCH CONTROL\nThis is a watch that James Bond would be proud to wear! Your electronic PENGO WATCH CONTROL acts as a remote control for TVs and videos.\nWith help from a Mr. H, you can ___.",
            "correct_answer": "finish your homework on time.",
            "clean_choices": [
                "stop using batteries.",
                "finish your homework on time.",
                "remember your teacher's instructions.",
                "get your room tidied on your way home.",
            ],
            "answer_idx": 1,
        },
        {
            "clean_query": "A sign on the door of a shop reads: 'We are open for you 7 days a week and 365 days a year.' What is the main purpose of the sign?",
            "correct_answer": "To inform customers of the shop's opening hours.",
            "clean_choices": [
                "To inform customers of the shop's opening hours.",
                "To recruit new employees.",
                "To advertise a new product.",
                "To announce a price change.",
            ],
            "answer_idx": 0,
        },
        {
            "clean_query": "According to the passage, the main reason people volunteer is ___.",
            "correct_answer": "that they want to help others and give back",
            "clean_choices": [
                "that they want to earn money",
                "that they want to help others and give back",
                "that they want to learn new skills",
                "that they want to make professional contacts",
            ],
            "answer_idx": 1,
        },
    ]


def list_fewshot_logiqa_en():
    """3-shot curated examples for LogiQA."""
    return [
        {
            "clean_query": "In an office, four people sit: A, B, C, and D. A sits opposite B. C sits to the right of A. Who sits to the left of B?",
            "correct_answer": "C",
            "clean_choices": ["A", "C", "D", "None of them"],
            "answer_idx": 1,
        },
        {
            "clean_query": "All philosophers are thinkers. Some thinkers are writers. Which conclusion necessarily follows?",
            "correct_answer": "Some philosophers might be writers.",
            "clean_choices": [
                "All writers are philosophers.",
                "No philosopher is a writer.",
                "Some philosophers might be writers.",
                "All thinkers are philosophers.",
            ],
            "answer_idx": 2,
        },
        {
            "clean_query": "If it rains, the road gets wet. The road is wet. Which conclusion is correct?",
            "correct_answer": "One cannot be certain whether it rained.",
            "clean_choices": [
                "It rained.",
                "It did not rain.",
                "One cannot be certain whether it rained.",
                "The road was cleaned.",
            ],
            "answer_idx": 2,
        },
    ]


def list_fewshot_lsat_ar():
    """3-shot curated examples for LSAT-AR."""
    return [
        {
            "clean_query": "Five lectures—F, G, H, J, and K—are given consecutively in one day. G is given before H. J is given immediately after F. Which order is possible?",
            "correct_answer": "G, H, F, J, K",
            "clean_choices": [
                "F, J, G, K, H",
                "G, H, F, J, K",
                "H, G, F, J, K",
                "J, F, G, H, K",
            ],
            "answer_idx": 1,
        },
        {
            "clean_query": "A florist arranges seven bouquets—S, T, U, V, W, X, and Y—in a row. V is in third place. T is immediately to the left of U. Which bouquet could be in first place?",
            "correct_answer": "T",
            "clean_choices": ["U", "V", "T", "Y"],
            "answer_idx": 2,
        },
        {
            "clean_query": "Three teams—Red, Blue, and Green—each play two games. Red plays before Blue. Green does not play first. Which order of first games is possible?",
            "correct_answer": "Red, Green, Blue",
            "clean_choices": [
                "Blue, Red, Green",
                "Green, Red, Blue",
                "Red, Green, Blue",
                "Red, Blue, Green",
            ],
            "answer_idx": 2,
        },
    ]


def list_fewshot_lsat_lr():
    """3-shot curated examples for LSAT-LR."""
    return [
        {
            "clean_query": "Editorial: As the population ages, healthcare costs rise. Therefore, the government must invest more in prevention. Which assumption underlies this argument?",
            "correct_answer": "Prevention measures can reduce healthcare costs of an aging population.",
            "clean_choices": [
                "The population will stop aging in the future.",
                "Prevention measures can reduce healthcare costs of an aging population.",
                "The government currently spends nothing on prevention.",
                "Rising healthcare costs are unavoidable.",
            ],
            "answer_idx": 1,
        },
        {
            "clean_query": "Critic: This museum only shows works by well-known artists. Therefore, it fails to promote emerging talent. Which statement most weakens this argument?",
            "correct_answer": "The museum has a dedicated exhibition space for new artists.",
            "clean_choices": [
                "Well-known artists attract more visitors.",
                "The museum has a dedicated exhibition space for new artists.",
                "Other museums also only show well-known artists.",
                "Emerging artists prefer smaller galleries.",
            ],
            "answer_idx": 1,
        },
        {
            "clean_query": "If all employees of a company are punctual and Stefan is an employee of this company, then Stefan must be punctual. Stefan is frequently late. What follows?",
            "correct_answer": "Not all employees of this company are punctual.",
            "clean_choices": [
                "Stefan is not an employee of the company.",
                "Not all employees of this company are punctual.",
                "Stefan is always punctual.",
                "The company has no rules about punctuality.",
            ],
            "answer_idx": 1,
        },
    ]


def list_fewshot_lsat_rc():
    """3-shot curated examples for LSAT-RC."""
    return [
        {
            "clean_query": "Defense lawyers have a duty to provide the best possible defense for their clients. At the same time, they have a responsibility to society. Which best describes the main point of the passage?",
            "correct_answer": "Lawyers must consider both their clients' interests and those of society.",
            "clean_choices": [
                "Lawyers should only represent their clients' interests.",
                "Lawyers must consider both their clients' interests and those of society.",
                "Society should more tightly control the work of lawyers.",
                "Clients should be free to choose their own lawyers.",
            ],
            "answer_idx": 1,
        },
        {
            "clean_query": "The author discusses various approaches to copyright reform. What position does the author primarily advocate?",
            "correct_answer": "A balanced approach protecting both creators and the public is necessary.",
            "clean_choices": [
                "Copyright should be completely abolished.",
                "A balanced approach protecting both creators and the public is necessary.",
                "Only corporations should be able to hold copyrights.",
                "The current system works perfectly.",
            ],
            "answer_idx": 1,
        },
        {
            "clean_query": "The passage describes the development of environmental legislation. According to the passage, what was the main reason for stricter laws?",
            "correct_answer": "Increasing scientific evidence of environmental damage.",
            "clean_choices": [
                "Increasing scientific evidence of environmental damage.",
                "Economic interests of industry.",
                "International political agreements.",
                "Protests by individual citizens.",
            ],
            "answer_idx": 0,
        },
    ]


def list_fewshot_sat_en():
    """3-shot curated examples for SAT-EN."""
    return [
        {
            "clean_query": "Akira came directly, breaking all tradition. He knocked on the door on a winter evening. 'I want to marry your daughter Naomi,' he said. Which choice best describes what happens in the passage?",
            "correct_answer": "One character receives a surprising request from another character.",
            "clean_choices": [
                "One character argues with another character.",
                "One character receives a surprising request from another character.",
                "One character reminisces about past decisions.",
                "One character criticizes another for unexpected behavior.",
            ],
            "answer_idx": 1,
        },
        {
            "clean_query": "The narrator describes a journey through a foreign city. The streets were narrow and the buildings old. What is the main mood of the passage?",
            "correct_answer": "Curiosity mixed with uncertainty.",
            "clean_choices": [
                "Joy and excitement.",
                "Curiosity mixed with uncertainty.",
                "Deep sadness.",
                "Anger and frustration.",
            ],
            "answer_idx": 1,
        },
        {
            "clean_query": "In the passage, the author compares two scientific theories. What is the main purpose of this comparison?",
            "correct_answer": "To show the strengths and weaknesses of both approaches.",
            "clean_choices": [
                "To prove that one theory is wrong.",
                "To show the strengths and weaknesses of both approaches.",
                "To propose an entirely new theory.",
                "To summarize the history of science.",
            ],
            "answer_idx": 1,
        },
    ]


def list_fewshot_sat_en_without_passage():
    """3-shot curated examples for SAT-EN without passage."""
    return [
        {
            "clean_query": "Which choice best describes what happens in the passage?",
            "correct_answer": "One character receives a surprising request from another character.",
            "clean_choices": [
                "One character argues with another character who intrudes on her home.",
                "One character receives a surprising request from another character.",
                "One character reminisces about choices she has made over the years.",
                "One character criticizes another character for pursuing an unexpected course of action.",
            ],
            "answer_idx": 1,
        },
        {
            "clean_query": "What purpose does the third paragraph serve in the overall context of the passage?",
            "correct_answer": "It provides a concrete example of the previously stated thesis.",
            "clean_choices": [
                "It refutes the main argument.",
                "It provides a concrete example of the previously stated thesis.",
                "It introduces an entirely new topic.",
                "It summarizes the entire passage.",
            ],
            "answer_idx": 1,
        },
        {
            "clean_query": "Which word best describes the author's tone?",
            "correct_answer": "objective",
            "clean_choices": ["enthusiastic", "objective", "sarcastic", "indifferent"],
            "answer_idx": 1,
        },
    ]


def list_fewshot_sat_math():
    """5-shot curated examples for SAT-Math."""
    return [
        {
            "clean_query": "If $\\frac{x-1}{3}=k$ and $k=3$, what is the value of $x$?",
            "correct_answer": "10",
            "clean_choices": ["2", "4", "9", "10"],
            "answer_idx": 3,
        },
        {
            "clean_query": "If $3x + 2 = 14$, what is the value of $x$?",
            "correct_answer": "4",
            "clean_choices": ["2", "3", "4", "6"],
            "answer_idx": 2,
        },
        {
            "clean_query": "A function is defined as $f(x) = 2x^2 - 3x + 1$. What is $f(2)$?",
            "correct_answer": "3",
            "clean_choices": ["1", "3", "5", "7"],
            "answer_idx": 1,
        },
        {
            "clean_query": "The circumference of a circle is $10\\pi$. What is the radius?",
            "correct_answer": "5",
            "clean_choices": ["3", "5", "10", "20"],
            "answer_idx": 1,
        },
        {
            "clean_query": "If $y = 3x - 7$ and $y = 5$, what is the value of $x$?",
            "correct_answer": "4",
            "clean_choices": ["2", "3", "4", "5"],
            "answer_idx": 2,
        },
    ]
