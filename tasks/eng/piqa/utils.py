"""PiQA utilities for English evaluation."""

import math
from typing import List, Dict



def get_piqa_fewshot() -> List[Dict]:
    """5-shot examples for PiQA."""
    return [
        {
            "goal": "When boiling butter, when it's ready, you can",
            "sol1": "Pour it onto a plate",
            "sol2": "Pour it into a jar",
            "label": 1,
        },
        {
            "goal": "To separate egg whites from the yolk using a water bottle, you should",
            "sol1": "Squeeze the water bottle and press it against the yolk. Release, which creates suction and lifts the yolk.",
            "sol2": "Place the water bottle and press it against the yolk. Keep pushing, which creates suction and lifts the yolk.",
            "label": 0,
        },
        {
            "goal": "how do you shake something?",
            "sol1": "stir it around.",
            "sol2": "move it up and down and side to side quickly.",
            "label": 1,
        },
        {
            "goal": "how to clean a dishwasher?",
            "sol1": "put a cup of vinegar in the top rack, and run the hottest cycle on your machine.",
            "sol2": "put a cup of vinegar in the top rack, and run the coldest cycle on your machine.",
            "label": 0,
        },
        {
            "goal": "how do you drizzle something?",
            "sol1": "pour it in a very small amount.",
            "sol2": "dump it on whatever you're putting it on.",
            "label": 0,
        },
    ]


def get_piqa_mc_fewshot() -> List[Dict]:
    """5-shot examples for PiQA.
    Examples now from FEWSHOT_SOURCES["OLMES:piqa"] """
    return [
    {
        "goal": "how do you stab something?",
        "sol1": "stick a sharp object through it.",
        "sol2": "pin it with a sharp object.",
        "label": 0,
    },
    {
        "goal": "how do you shake something?",
        "sol1": "move it up and down and side to side quickly.",
        "sol2": "stir it very quickly.",
        "label": 0,
    },
    {
        "goal": "Clean tires",
        "sol1": "Pour water, cape off caked on dirt. Use  speed wool to clean out crevices and sparrow spaces.",
        "sol2": "Pour water, scrape off caked on dirt. Use a steel wool to clean out crevices and narrow spaces.",
        "label": 1,
    },
    {
        "goal": "how do you taste something?",
        "sol1": "smell it enough to taste it.",
        "sol2": "place it in your mouth to taste.",
        "label": 1,
    },
    {
        "goal": "To create a makeshift ice pack,",
        "sol1": "take a sponge and soak it in oil. Put the sponge in a refrigerator and let it freeze. Once frozen, take it out and put it in a ziploc bag. You can now use it as an ice pack.",
        "sol2": "take a sponge and soak it in water. Put the sponge in a refrigerator and let it freeze. Once frozen, take it out and put it in a ziploc bag. You can now use it as an ice pack.",
        "label": 1,
    },
    {
        "goal": "What should I use as a stain on a wooden bowl I've just made.",
        "sol1": "You should coat the wooden bowl with a butcher block oil & finish per manufacturer directions.",
        "sol2": "You should coat the wooden bowl with a butcher knife oil & finish per manufacturer directions.",
        "label": 0,
    },
    {
        "goal": "How to boil eggs.",
        "sol1": "Place your eggs in a pot and cover with no water by 1 inch, bring to a boil over medium-high heat, then cover, remove from the heat and set aside 8 to 10 minutes.",
        "sol2": "Place your eggs in a pot and cover with cold water by 1 inch, bring to a boil over medium-high heat, then cover, remove from the heat and set aside 8 to 10 minutes.",
        "label": 1,
    },
]


def process_results_bpb(doc, results):
    """Compute answer-only BPB (OLMES-style) for PiQA."""
    ll, _ = results[0]
    gold_text = [doc["sol1"], doc["sol2"]][doc["label"]]
    gold_bytes = len((" " + gold_text).encode("utf-8"))
    return {"bits_per_byte": -ll / (math.log(2) * max(gold_bytes, 1))}
