def list_fewshot_samples():
    """5-shot curated examples for WinoGrande (German).

    Hand-written examples designed to sound natural in German rather than
    translated from English.
    """
    return [
        {
            "sentence": "Der Koffer passte nicht in den Kofferraum, weil _ zu groß war.",
            "option1": "der Koffer",
            "option2": "der Kofferraum",
            "answer": "1",
        },
        {
            "sentence": "Anna half Sophie beim Umzug, weil _ eine neue Wohnung gefunden hatte.",
            "option1": "Anna",
            "option2": "Sophie",
            "answer": "2",
        },
        {
            "sentence": "Markus lud Tim zum Abendessen ein, weil _ ihm beim Umzug geholfen hatte.",
            "option1": "Markus",
            "option2": "Tim",
            "answer": "2",
        },
        {
            "sentence": "Lisa gab Katharina ihr altes Fahrrad, weil _ sich ein neues gekauft hatte.",
            "option1": "Lisa",
            "option2": "Katharina",
            "answer": "1",
        },
        {
            "sentence": "Stefan bot seinem Bruder Geld an, weil _ in finanziellen Schwierigkeiten steckte.",
            "option1": "Stefan",
            "option2": "sein Bruder",
            "answer": "2",
        },
    ]


def doc_to_text(doc):
    """Return the correct answer index.
    
    This is a hack used by lm-eval-harness winogrande - when doc_to_target
    returns text (the continuation), this integer is used as the label.
    """
    return {"1": 0, "2": 1}[doc["answer"]]


def doc_to_target(doc):
    """Return the continuation (text after the blank) with leading space.

    The leading space ensures correct word boundaries when appended to
    partial contexts. Requires target_delimiter: "" in the YAML configs.
    """
    idx = doc["sentence"].index("_") + 1
    return " " + doc["sentence"][idx:].strip()


def doc_to_choice(doc):
    """Return the partial contexts (sentence up to blank + option).
    
    lm-eval computes P(doc_to_target | choice) for each choice.
    """
    idx = doc["sentence"].index("_")
    prefix = doc["sentence"][:idx]
    return [prefix + doc["option1"], prefix + doc["option2"]]


import math


def doc_to_text_bpb(doc):
    """Return the prefix + correct option as context for BPB evaluation."""
    answer_idx = {"1": 0, "2": 1}[doc["answer"]]
    option = doc["option1"] if answer_idx == 0 else doc["option2"]
    idx = doc["sentence"].index("_")
    return doc["sentence"][:idx] + option


def doc_to_target_bpb(doc):
    """Return the full sentence with the correct answer for BPB evaluation."""
    answer_idx = {"1": 0, "2": 1}[doc["answer"]]
    option = doc["option1"] if answer_idx == 0 else doc["option2"]
    return doc["sentence"].replace("_", option)


def process_results_bpb(doc, results):
    """Compute answer-only BPB (OLMES-style) for Winogrande."""
    ll, _ = results[0]
    # Target is the continuation after the blank
    idx = doc["sentence"].index("_") + 1
    gold_text = doc["sentence"][idx:].strip()
    # Include leading space to match default target_delimiter " "
    gold_bytes = len((" " + gold_text).encode("utf-8"))
    return {"bits_per_byte": -ll / (math.log(2) * max(gold_bytes, 1))}
