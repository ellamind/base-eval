
def doc_to_text(doc):
    """Return the correct answer index.
    
    This is a hack used by lm-eval-harness winogrande - when doc_to_target
    returns text (the continuation), this integer is used as the label.
    """
    return {"1": 0, "2": 1}[doc["answer"]]


def doc_to_target(doc):
    """Return the continuation (text after the blank) with leading space.

    The leading space ensures correct word boundaries when appended to
    partial contexts (e.g. "...so Maria" + " always got..." not "Mariaalways").
    Requires target_delimiter: "" in the YAML configs.
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


def get_winogrande_rc_fewshot():
    """10 fixed OLMES fewshot examples for WinoGrande RC.

    Source: OLMES:winogrande (fewshot_sources.py).
    With num_fewshot=5 and first_n sampler, only the first 5 are used.
    """
    return [
        {
            "sentence": "John moved the couch from the garage to the backyard to create space. The _ is small.",
            "option1": "garage",
            "option2": "backyard",
            "answer": "1",
        },
        {
            "sentence": "Dennis drew up a business proposal to present to Logan because _ wants his investment.",
            "option1": "Dennis",
            "option2": "Logan",
            "answer": "1",
        },
        {
            "sentence": "Felicia unexpectedly made fried eggs for breakfast in the morning for Katrina and now _ owes a favor.",
            "option1": "Felicia",
            "option2": "Katrina",
            "answer": "2",
        },
        {
            "sentence": "The circuit failed to power the television but kept the radio going, as the _ had a weak connection.",
            "option1": "television",
            "option2": "radio",
            "answer": "1",
        },
        {
            "sentence": "Neil told Craig that he has to take care of the child for the day because _ promised to do so.",
            "option1": "Neil",
            "option2": "Craig",
            "answer": "2",
        },
        {
            "sentence": "Emily was thrown out of the court for the lawsuit against Carrie because _ was the only one acting disruptively.",
            "option1": "Emily",
            "option2": "Carrie",
            "answer": "1",
        },
        {
            "sentence": "Rebecca was slimmer than Carrie, so _ started to worry about calories and a diet.",
            "option1": "Rebecca",
            "option2": "Carrie",
            "answer": "2",
        },
        {
            "sentence": "Amy was worried that her hair looked greasy, so she asked Monica, but _ regretted asking.",
            "option1": "Amy",
            "option2": "Monica",
            "answer": "1",
        },
        {
            "sentence": "The fish swam away from Neil and towards Robert because _ was pouring the food in the aquarium.",
            "option1": "Neil",
            "option2": "Robert",
            "answer": "2",
        },
        {
            "sentence": "James wanted to drink his cup of coffee but he could not until he added some water to cool it down. The _ is cold.",
            "option1": "water",
            "option2": "coffee",
            "answer": "1",
        },
    ]


def get_winogrande_mc_fewshot():
    """5 fixed OLMES fewshot examples for WinoGrande MC.

    Same data as RC fewshot, but with answer_idx for MC format.
    """
    examples = get_winogrande_rc_fewshot()[:5]
    for ex in examples:
        ex["answer_idx"] = int(ex["answer"]) - 1
    return examples


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
    idx = doc["sentence"].index("_") + 1
    gold_text = doc["sentence"][idx:].strip()
    # Leading space is part of the scored target (matches doc_to_target)
    gold_bytes = len((" " + gold_text).encode("utf-8"))
    return {"bits_per_byte": -ll / (math.log(2) * max(gold_bytes, 1))}
