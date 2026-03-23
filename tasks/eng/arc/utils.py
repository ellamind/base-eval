"""ARC utilities for English evaluation."""

import math
from typing import List, Dict


def get_arc_easy_fewshot() -> List[Dict]:
    """5-shot examples for ARC Easy."""
    return [
        {
            "id": "MCAS_2007_8_5189",
            "question": "Lichens are symbiotic organisms made of green algae and fungi. What do the green algae supply to the fungi in this symbiotic relationship?",
            "choices": {
                "text": ["carbon dioxide", "food", "protection", "water"],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "B",
        },
        {
            "id": "Mercury_SC_401169",
            "question": "When a switch is used in an electrical circuit, the switch can",
            "choices": {
                "text": [
                    "cause the charge to build.",
                    "increase and decrease the voltage.",
                    "cause the current to change direction.",
                    "stop and start the flow of current.",
                ],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "D",
        },
        {
            "id": "MCAS_2004_8_27",
            "question": "Which of the following is an example of an assistive device?",
            "choices": {
                "text": ["contact lens", "motorcycle", "raincoat", "coffee pot"],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "A",
        },
        {
            "id": "NYSEDREGENTS_2006_8_10",
            "question": "Rocks are classified as igneous, metamorphic, or sedimentary according to",
            "choices": {
                "text": ["their color", "their shape", "how they formed", "the minerals they contain"],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "C",
        },
        {
            "id": "Mercury_7013388",
            "question": "A chewable calcium carbonate tablet is a common treatment for stomach discomfort. Calcium carbonate is most likely used as this type of medicine because calcium carbonate",
            "choices": {
                "text": [
                    "has a pleasant flavor.",
                    "is inexpensive to produce.",
                    "neutralizes digestive acid.",
                    "occurs naturally in the body.",
                ],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "C",
        },
    ]


def get_arc_challenge_fewshot() -> List[Dict]:
    """5-shot examples for ARC Challenge."""
    return [
        {
            "id": "Mercury_SC_415702",
            "question": "George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat?",
            "choices": {
                "text": [
                    "dry palms",
                    "wet palms",
                    "palms covered with oil",
                    "palms covered with lotion",
                ],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "A",
        },
        {
            "id": "Mercury_7234078",
            "question": "A boat is acted on by a river current flowing north and by wind blowing on its sails. The boat travels northeast. In which direction is the wind most likely applying force to the sails of the boat?",
            "choices": {
                "text": ["west", "east", "north", "south"],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "B",
        },
        {
            "id": "Mercury_7220990",
            "question": "Which land form is the result of the constructive force of a glacier?",
            "choices": {"text": ["valleys carved by a moving glacier", "piles of rocks deposited by a melting glacier", "grooves created in a rock by a glacier", "bedrock hills roughened by a glacier"], "label": ["A", "B", "C", "D"]},
            "answerKey": "B",
        },
        {
            "id": "MCAS_2016_5_9",
            "question": "Which of the following is an example of a reflex that occurs only in humans?",
            "choices": {
                "text": ["sneezing", "flinching", "crying", "blinking"],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "C",
        },
        {
            "id": "Mercury_7026593",
            "question": "Using nonrenewable resources for energy produces waste products that can have long-term, negative effects on Earth's subsystems. Which energy source produces waste products that can have these effects for the longest amount of time?",
            "choices": {
                "text": ["natural gas", "uranium", "crude oil", "coal"],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "B",
        },
    ]

def get_arc_easy_mc_fewshot() -> List[Dict]:
    """5-shot examples for ARC Easy."""
    return [
        {
            "id": "MCAS_2007_8_5189",
            "question": "Lichens are symbiotic organisms made of green algae and fungi. What do the green algae supply to the fungi in this symbiotic relationship?",
            "choices": {
                "text": ["carbon dioxide", "food", "protection", "water"],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "B",
        },
        {
            "id": "Mercury_SC_401169",
            "question": "When a switch is used in an electrical circuit, the switch can",
            "choices": {
                "text": [
                    "cause the charge to build.",
                    "increase and decrease the voltage.",
                    "cause the current to change direction.",
                    "stop and start the flow of current.",
                ],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "D",
        },
        {
            "id": "MCAS_2004_8_27",
            "question": "Which of the following is an example of an assistive device?",
            "choices": {
                "text": ["contact lens", "motorcycle", "raincoat", "coffee pot"],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "A",
        },
        {
            "id": "NYSEDREGENTS_2006_8_10",
            "question": "Rocks are classified as igneous, metamorphic, or sedimentary according to",
            "choices": {
                "text": ["their color", "their shape", "how they formed", "the minerals they contain"],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "C",
        },
        {
            "id": "Mercury_7013388",
            "question": "A chewable calcium carbonate tablet is a common treatment for stomach discomfort. Calcium carbonate is most likely used as this type of medicine because calcium carbonate",
            "choices": {
                "text": [
                    "has a pleasant flavor.",
                    "is inexpensive to produce.",
                    "neutralizes digestive acid.",
                    "occurs naturally in the body.",
                ],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "C",
        },
    ]


def get_arc_challenge_mc_fewshot() -> List[Dict]:
    """5-shot examples for ARC Challenge."""
    return [
        {
        "id": "Mercury_SC_415702",
        "question": "George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat?",
        "choices": {
            "text": [
                "dry palms",
                "wet palms",
                "palms covered with oil",
                "palms covered with lotion",
            ],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "A",
        },
        {
            "id": "MCAS_2009_5_6516",
            "question": "Which of the following statements best explains why magnets usually stick to a refrigerator door?",
            "choices": {
                "text": [
                    "The refrigerator door is smooth.",
                    "The refrigerator door contains iron.",
                    "The refrigerator door is a good conductor.",
                    "The refrigerator door has electric wires in it.",
                ],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "B",
        },
        {
            "id": "Mercury_7233695",
            "question": "A fold observed in layers of sedimentary rock most likely resulted from the",
            "choices": {
                "text": [
                    "cooling of flowing magma.",
                    "converging of crustal plates.",
                    "deposition of river sediments.",
                    "solution of carbonate minerals.",
                ],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "B",
        },
        {
            "id": "Mercury_7041615",
            "question": "Which of these do scientists offer as the most recent explanation as to why many plants and animals died out at the end of the Mesozoic era?",
            "choices": {
                "text": [
                    "worldwide disease",
                    "global mountain building",
                    "rise of mammals that preyed upon plants and animals",
                    "impact of an asteroid created dust that blocked the sunlight",
                ],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "D",
        },
        {
            "id": "MCAS_1998_4_3",
            "question": "Which of the following is a trait that a dog does NOT inherit from its parents?",
            "choices": {
                "text": [
                    "the length of its fur",
                    "the shape of its nose",
                    "the size of its appetite",
                    "the color of its fur",
                ],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "C",
        },
    ]


def process_results_bpb(doc, results):
    """Compute answer-only BPB (OLMES-style) for ARC."""
    ll, _ = results[0]
    gold_idx = doc["choices"]["label"].index(doc["answerKey"])
    gold_text = doc["choices"]["text"][gold_idx]
    gold_bytes = len((" " + gold_text).encode("utf-8"))
    return {"bits_per_byte": -ll / (math.log(2) * max(gold_bytes, 1))}
