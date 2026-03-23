"""CommonsenseQA utilities for English evaluation."""

import math
from typing import List, Dict


def get_csqa_fewshot() -> List[Dict]:
    """5-shot examples for CommonsenseQA.

    Taken from FEWSHOT_SOURCES["OLMES:commonsense_qa"] (first 5).
    """
    return [
        {
            "id": "61fe6e879ff18686d7552425a36344c8",
            "question": "Sammy wanted to go to where the people were.  Where might he go?",
            "question_concept": "people",
            "choices": {
                "label": ["A", "B", "C", "D", "E"],
                "text": ["race track", "populated areas", "the desert", "apartment", "roadblock"],
            },
            "answerKey": "B",
        },
        {
            "id": "02e821a3e53cb320790950aab4489e85",
            "question": "Google Maps and other highway and street GPS services have replaced what?",
            "question_concept": "highway",
            "choices": {
                "label": ["A", "B", "C", "D", "E"],
                "text": ["united states", "mexico", "countryside", "atlas", "oceans"],
            },
            "answerKey": "D",
        },
        {
            "id": "23505889b94e880c3e89cff4ba119860",
            "question": "The fox walked from the city into the forest, what was it looking for?",
            "question_concept": "fox",
            "choices": {
                "label": ["A", "B", "C", "D", "E"],
                "text": [
                    "pretty flowers.",
                    "hen house",
                    "natural habitat",
                    "storybook",
                    "dense forest",
                ],
            },
            "answerKey": "C",
        },
        {
            "id": "a76403b4921a9281b6ee2a7241a5ec9f",
            "question": "What is it called when you slowly cook using a grill?",
            "question_concept": "grill",
            "choices": {
                "label": ["A", "B", "C", "D", "E"],
                "text": ["backyard", "restaurant", "crockpot", "neighbor's house", "barbeque"],
            },
            "answerKey": "E",
        },
        {
            "id": "6dc921840aa1e5dda3333b79007f630b",
            "question": "What would you do if you want to be able to earn money?",
            "question_concept": "money",
            "choices": {
                "label": ["A", "B", "C", "D", "E"],
                "text": [
                    "apply for job",
                    "stand in line",
                    "take care of proposals",
                    "pass course",
                    "play the lottery",
                ],
            },
            "answerKey": "A",
        },
    ]


def process_results_bpb(doc, results):
    """Compute answer-only BPB (OLMES-style) for CommonsenseQA."""
    ll, _ = results[0]
    gold_idx = doc["choices"]["label"].index(doc["answerKey"])
    gold_text = doc["choices"]["text"][gold_idx]
    gold_bytes = len((" " + gold_text).encode("utf-8"))
    return {"bits_per_byte": -ll / (math.log(2) * max(gold_bytes, 1))}
