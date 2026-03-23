"""SocialIQA utilities for English evaluation."""

import math
from typing import List, Dict


def socialiqa_doc_to_target(doc: dict) -> int:
    """Convert SocialIQA label to 0-indexed target."""
    return int(doc["label"]) - 1


def get_socialiqa_mc_fewshot() -> List[Dict]:
    """Few-shots from FEWSHOT_SOURCES["OLMES:social_i_qa"]"""
    return [
    {
        "context": "Cameron decided to have a barbecue and gathered her friends together.",
        "question": "How would Others feel as a result?",
        "answerA": "like attending",
        "answerB": "like staying home",
        "answerC": "a good friend to have",
        "label": "1",
    },
    {
        "context": "Quinn wanted to help me clean my room up because it was so messy.",
        "question": "What will Quinn want to do next?",
        "answerA": "Eat messy snacks",
        "answerB": "help out a friend",
        "answerC": "Pick up the dirty clothes",
        "label": "3",
    },
    {
        "context": "Jan needed to give out jobs for an upcoming project at work.",
        "question": "What will Others want to do next?",
        "answerA": "disagree with Jan",
        "answerB": "get to work",
        "answerC": "argue with the assignments",
        "label": "2",
    },
    {
        "context": "Their cat kept trying to escape out of the window, so Jan placed an obstacle in the way.",
        "question": "How would Jan feel afterwards?",
        "answerA": "scared of losing the cat",
        "answerB": "normal",
        "answerC": "relieved for fixing the problem",
        "label": "3",
    },
    {
        "context": "Remy was an expert fisherman and was on the water with Kai. Remy baited Kai's hook.",
        "question": "What will Remy want to do next?",
        "answerA": "cast the line",
        "answerB": "put the boat in the water",
        "answerC": "invite Kai out on the boat",
        "label": "1",
    },
    {
        "context": "Kendall worked the weekend at the steakhouse and made bank on tips.",
        "question": "What will Kendall want to do next?",
        "answerA": "Save the money",
        "answerB": "get hired at the steakhouse",
        "answerC": "Quit her job",
        "label": "1",
    },
    {
        "context": "Bailey relieved every one of her friends when she announced her plans to stay.",
        "question": "Why did Bailey do this?",
        "answerA": "wanted to live by herself",
        "answerB": "wanted to show her dedication to her friends",
        "answerC": "wanted to forget about her friends",
        "label": "2",
    },
    {
        "context": "Kendall ran back and thanked Lee for helping her find the dog.",
        "question": "How would you describe Kendall?",
        "answerA": "grateful",
        "answerB": "super",
        "answerC": "amazing",
        "label": "1",
    },
]

def get_socialiqa_fewshot() -> List[Dict]:
    """5-shot examples for SocialIQA."""
    return [
        {
            "context": "Tracy didn't go home that evening and resisted Riley's attacks.",
            "question": "What does Tracy need to do before this?",
            "answerA": "make a new plan",
            "answerB": "Go home and see Riley",
            "answerC": "Find somewhere to go",
            "label": "2",
        },
        {
            "context": "Cameron decided to have a barbecue and gathered her friends together.",
            "question": "How would you describe Cameron?",
            "answerA": "like attending",
            "answerB": "sociable",
            "answerC": "like staying home",
            "label": "1",
        },
        {
            "context": "Austin made Jan's enemies for life after he testified at her trial.",
            "question": "What will happen to Austin?",
            "answerA": "hate him",
            "answerB": "get back at Austin",
            "answerC": "like him",
            "label": "1",
        },
        {
            "context": "Addison worked hard and at the end of each month, gave the landlord their money for rent.",
            "question": "How would you describe Addison?",
            "answerA": "happy",
            "answerB": "responsible",
            "answerC": "a bad tenant",
            "label": "1",
        },
        {
            "context": "Skylar brought Bailey's girlfriend to the party because the girl didn't know anyone there.",
            "question": "Why did Skylar do this?",
            "answerA": "sneak out of the party",
            "answerB": "leave her alone at the party",
            "answerC": "be a good friend",
            "label": "2",
        },
    ]


def process_results_bpb(doc, results):
    """Compute answer-only BPB (OLMES-style) for SocialIQA."""
    ll, _ = results[0]
    gold_text = [doc["answerA"], doc["answerB"], doc["answerC"]][int(doc["label"]) - 1]
    gold_bytes = len((" " + gold_text).encode("utf-8"))
    return {"bits_per_byte": -ll / (math.log(2) * max(gold_bytes, 1))}
