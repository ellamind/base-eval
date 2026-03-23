"""DROP utilities for English evaluation."""

import math
import re
import string
from typing import List, Dict




def drop_normalize_answer(answer: str) -> str:
    """Normalize DROP answer for comparison."""
    answer = answer.lower()
    answer = re.sub(r'\b(a|an|the)\b', ' ', answer)
    answer = re.sub(r'[^\w\s]', '', answer)
    answer = ' '.join(answer.split())
    return answer.strip()


# =============================================================================
# Token-level F1 metrics (DROP-style, based on lm_eval implementation)
# =============================================================================

def _normalize_answer(text: str) -> str:
    """Normalize answer for comparison."""
    text = text.lower()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = ''.join(ch for ch in text if ch not in string.punctuation)
    text = ' '.join(text.split())
    return text.strip()


def _get_tokens(text: str) -> List[str]:
    """Tokenize normalized text."""
    return _normalize_answer(text).split()


def _compute_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1."""
    pred_tokens = _get_tokens(prediction)
    ref_tokens = _get_tokens(reference)

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = set(pred_tokens) & set(ref_tokens)
    num_common = sum(min(pred_tokens.count(t), ref_tokens.count(t)) for t in common)

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return (2 * precision * recall) / (precision + recall)


def _compute_exact_match(prediction: str, reference: str) -> float:
    """Compute exact match."""
    return 1.0 if _normalize_answer(prediction) == _normalize_answer(reference) else 0.0


def _extract_drop_answers(answers_spans: dict) -> List[str]:
    """Extract answer strings from DROP answers_spans format."""
    answers = []
    spans = answers_spans.get("spans", [])
    if spans:
        answers.extend(spans)
    number = answers_spans.get("number", "")
    if number:
        answers.append(str(number))
    date = answers_spans.get("date", {})
    if date:
        date_parts = [date.get("day", ""), date.get("month", ""), date.get("year", "")]
        date_str = " ".join(p for p in date_parts if p).strip()
        if date_str:
            answers.append(date_str)
    return answers


def process_results_drop(doc: dict, results: List[str]) -> dict:
    """Process results for DROP generative task."""
    prediction = results[0] if results else ""

    answers_spans = doc.get("answers_spans", {})
    references = _extract_drop_answers(answers_spans)

    if not references:
        return {"em": 0.0, "f1": 0.0}

    max_em = max(_compute_exact_match(prediction, ref) for ref in references)
    max_f1 = max(_compute_f1(prediction, ref) for ref in references)

    return {"em": max_em, "f1": max_f1}


def get_drop_fewshot() -> List[Dict]:
    """5-shot curated examples for DROP BPB (from OLMES:drop).

    Uses EleutherAI/drop schema (passage, question, answer dict).
    Source: oe_eval/tasks/fewshot_sources.py OLMES:drop (first 5).
    """
    return [
        {
            "section_id": "nfl_2201",
            "passage": "To start the season, the Lions traveled south to Tampa, Florida to take on the Tampa Bay Buccaneers. The Lions scored first in the first quarter with a 23-yard field goal by Jason Hanson. The Buccaneers tied it up with a 38-yard field goal by Connor Barth, then took the lead when Aqib Talib intercepted a pass from Matthew Stafford and ran it in 28 yards. The Lions responded with a 28-yard field goal. In the second quarter, Detroit took the lead with a 36-yard touchdown catch by Calvin Johnson, and later added more points when Tony Scheffler caught an 11-yard TD pass. Tampa Bay responded with a 31-yard field goal just before halftime. The second half was relatively quiet, with each team only scoring one touchdown. First, Detroit's Calvin Johnson caught a 1-yard pass in the third quarter. The game's final points came when Mike Williams of Tampa Bay caught a 5-yard pass.  The Lions won their regular season opener for the first time since 2007",
            "question": "How many field goals did the Lions score?",
            "query_id": "c9582e03-b01b-42ed-83e0-b90a5334aefa",
            "answer": {"number": "2", "date": {"day": "", "month": "", "year": ""}, "spans": [], "worker_id": "", "hit_id": ""},
            "validated_answers": {"number": [""], "date": [{"day": "", "month": "", "year": ""}], "spans": [[]], "worker_id": [""], "hit_id": [""]},
        },
        {
            "section_id": "nfl_1491",
            "passage": "Coming off their road win over the Redskins, the Chiefs went home, donned their Dallas Texans throwbacks, and played a Week 7 AFL Legacy game with the San Diego Chargers. Kansas City would find themselves trailing in the first quarter as Chargers quarterback Philip Rivers completed a 3-yard touchdown pass to wide receiver Malcom Floyd, followed by a 10-yard touchdown pass to wide receiver Vincent Jackson. San Diego would add onto their lead in the second quarter with a 20-yard and a 39-yard field goal from kicker Nate Kaeding. The Chiefs would get onto the board in the third quarter with quarterback Matt Cassel completing a 7-yard touchdown pass to wide receiver Dwayne Bowe, but the Chargers kept their momentum going with Rivers finding running back Darren Sproles on a 58-yard touchdown pass. In the fourth quarter, San Diego sealed the win with Kaeding's 19-yard field goal and fullback Jacob Hester recovering a blocked punt in the end zone for a touchdown. With the loss, Kansas City went into their bye week at 1-6. Larry Johnson was suspended for two weeks after he made offensive comments about Todd Haley and made offensive comments about homosexuals on Twitter and in public.",
            "question": "which player scored the longest field goal?",
            "query_id": "a849cb71-b6da-4c1a-9791-b962c9ff2d65",
            "answer": {"number": "", "date": {"day": "", "month": "", "year": ""}, "spans": ["Nate Kaeding"], "worker_id": "", "hit_id": ""},
            "validated_answers": {"number": [""], "date": [{"day": "", "month": "", "year": ""}], "spans": [[]], "worker_id": [""], "hit_id": [""]},
        },
        {
            "section_id": "history_269",
            "passage": "In 1344, Momchil, the independent Bulgarian ruler of the Rhodope and Aegean regions, whose army grew to 2,000 men, took an important role in the Byzantine civil war. While at first he supported John Kantakouzenos, from the spring of 1344 Momchil reneged, provoked by the aggression of the Ottoman allies. In June he defeated the Ottoman fleet near the Portogalos bay. According to sources, at night the Bulgarian ruler sent boats to burn the anchored Ottoman ships and soon after he defeated the army of Kantakouzenos at Mosynopolis. Probably the first local ruler to become aware of the impending Ottoman threat, Momchil unsuccessfully pleaded with the emperors of Bulgaria and Byzantium for help. Even though his troops continued the resistance in the Eastern Rhodopes, in May 1345 the Turks led by Umur Beg marched from Asia Minor and devastated Bulgarian territories driving away people and livestock. Soon after, on 7 July 1345, Ottoman forces under Umur Beg defeated Momchil's army in the battle of Peritor near his capital Xanthi. Sources attest that the independent ruler perished in the battle without leaving asuccessor, and with little political will or leadership left to counter the Ottoman invasion.",
            "question": "When did Momchil defeat the Ottoman fleet near the Portogalos bay?",
            "query_id": "3637ce9d-203d-49e0-a454-b8a49e376148",
            "answer": {"number": "", "date": {"day": "", "month": "June", "year": "1344"}, "spans": [], "worker_id": "", "hit_id": ""},
            "validated_answers": {"number": [""], "date": [{"day": "", "month": "", "year": ""}], "spans": [[]], "worker_id": [""], "hit_id": [""]},
        },
        {
            "section_id": "history_690",
            "passage": "The French king, John II, had been held captive in England. The Treaty of Br\u00e9tigny set his ransom at 3\u00a0million\u00a0crowns and allowed for hostages to be held in lieu of John. The hostages included two of his sons, several princes and nobles, four inhabitants of Paris, and two citizens from each of the nineteen principal towns of France. While these hostages were held, John returned to France to try and raise funds to pay the ransom. In 1362 John's son Louis of Anjou, a hostage in English-held Calais, escaped captivity. So, with his stand-in hostage gone, John felt honor-bound to return to captivity in England. The French crown had been at odds with Navarre  since 1354, and in 1363 the Navarrese used the captivity of John II in London and the political weakness of the Dauphin to try to seize power. Although there was no formal treaty, Edward III supported the Navarrese moves, particularly as there was a prospect that he might gain control over the northern and western provinces as a consequence. With this in mind, Edward deliberately slowed the peace negotiations. In 1364, John II died in London, while still in honourable captivity. Charles V succeeded him as king of France. On 7 May 1364, one month after the dauphin's accession and three days before his coronation as Charles V, the Navarrese suffered a crushing defeat at the Battle of Cocherel.",
            "question": "What did one of John II's replacements do in captivity?",
            "query_id": "c420c10d-1384-47f1-9850-ec9269570ca8",
            "answer": {"number": "", "date": {"day": "", "month": "", "year": ""}, "spans": ["escaped captivity"], "worker_id": "", "hit_id": ""},
            "validated_answers": {"number": [""], "date": [{"day": "", "month": "", "year": ""}], "spans": [[]], "worker_id": [""], "hit_id": [""]},
        },
        {
            "section_id": "history_2184",
            "passage": "As of the census of 2000, there were 218,590 people, 79,667 households, and 60,387 families residing in the county.  The population density was 496 people per square mile (192/km\u00b2). There were 83,146 housing units at an average density of 189 per square mile (73/km\u00b2). The racial makeup of the county was 86.77% Race (United States Census), 9.27% Race (United States Census), 0.23% Race (United States Census), 1.52% Race (United States Census), 0.06% Race (United States Census), 0.69% from Race (United States Census), and 1.47% from two or more races.  1.91% of the population were Race (United States Census) or Race (United States Census) of any race. 22.5% were of German people, 13.1% Irish people, 9.8% Italian people, 9.2% English, 8.1% \"American\" and 6.0% Polish ancestry.",
            "question": "How many more people were there than families?",
            "query_id": "c312a5d0-5318-4bbb-806e-22333f00e990",
            "answer": {"number": "158203", "date": {"day": "", "month": "", "year": ""}, "spans": [], "worker_id": "", "hit_id": ""},
            "validated_answers": {"number": [""], "date": [{"day": "", "month": "", "year": ""}], "spans": [[]], "worker_id": [""], "hit_id": [""]},
        },
    ]


def get_drop_gen_fewshot() -> List[Dict]:
    """5-shot curated examples for DROP generative (from OLMES:drop).

    Adapts OLMES examples to ucinlp/drop schema (answers_spans).
    Source: oe_eval/tasks/fewshot_sources.py OLMES:drop (first 5).
    """
    return [
        {
            "passage": "To start the season, the Lions traveled south to Tampa, Florida to take on the Tampa Bay Buccaneers. The Lions scored first in the first quarter with a 23-yard field goal by Jason Hanson. The Buccaneers tied it up with a 38-yard field goal by Connor Barth, then took the lead when Aqib Talib intercepted a pass from Matthew Stafford and ran it in 28 yards. The Lions responded with a 28-yard field goal. In the second quarter, Detroit took the lead with a 36-yard touchdown catch by Calvin Johnson, and later added more points when Tony Scheffler caught an 11-yard TD pass. Tampa Bay responded with a 31-yard field goal just before halftime. The second half was relatively quiet, with each team only scoring one touchdown. First, Detroit's Calvin Johnson caught a 1-yard pass in the third quarter. The game's final points came when Mike Williams of Tampa Bay caught a 5-yard pass.  The Lions won their regular season opener for the first time since 2007",
            "question": "How many field goals did the Lions score?",
            "answers_spans": {"spans": [], "number": "2", "date": {"day": "", "month": "", "year": ""}},
        },
        {
            "passage": "Coming off their road win over the Redskins, the Chiefs went home, donned their Dallas Texans throwbacks, and played a Week 7 AFL Legacy game with the San Diego Chargers. Kansas City would find themselves trailing in the first quarter as Chargers quarterback Philip Rivers completed a 3-yard touchdown pass to wide receiver Malcom Floyd, followed by a 10-yard touchdown pass to wide receiver Vincent Jackson. San Diego would add onto their lead in the second quarter with a 20-yard and a 39-yard field goal from kicker Nate Kaeding. The Chiefs would get onto the board in the third quarter with quarterback Matt Cassel completing a 7-yard touchdown pass to wide receiver Dwayne Bowe, but the Chargers kept their momentum going with Rivers finding running back Darren Sproles on a 58-yard touchdown pass. In the fourth quarter, San Diego sealed the win with Kaeding's 19-yard field goal and fullback Jacob Hester recovering a blocked punt in the end zone for a touchdown. With the loss, Kansas City went into their bye week at 1-6. Larry Johnson was suspended for two weeks after he made offensive comments about Todd Haley and made offensive comments about homosexuals on Twitter and in public.",
            "question": "which player scored the longest field goal?",
            "answers_spans": {"spans": ["Nate Kaeding"], "number": "", "date": {"day": "", "month": "", "year": ""}},
        },
        {
            "passage": "In 1344, Momchil, the independent Bulgarian ruler of the Rhodope and Aegean regions, whose army grew to 2,000 men, took an important role in the Byzantine civil war. While at first he supported John Kantakouzenos, from the spring of 1344 Momchil reneged, provoked by the aggression of the Ottoman allies. In June he defeated the Ottoman fleet near the Portogalos bay. According to sources, at night the Bulgarian ruler sent boats to burn the anchored Ottoman ships and soon after he defeated the army of Kantakouzenos at Mosynopolis. Probably the first local ruler to become aware of the impending Ottoman threat, Momchil unsuccessfully pleaded with the emperors of Bulgaria and Byzantium for help. Even though his troops continued the resistance in the Eastern Rhodopes, in May 1345 the Turks led by Umur Beg marched from Asia Minor and devastated Bulgarian territories driving away people and livestock. Soon after, on 7 July 1345, Ottoman forces under Umur Beg defeated Momchil's army in the battle of Peritor near his capital Xanthi. Sources attest that the independent ruler perished in the battle without leaving asuccessor, and with little political will or leadership left to counter the Ottoman invasion.",
            "question": "When did Momchil defeat the Ottoman fleet near the Portogalos bay?",
            "answers_spans": {"spans": [], "number": "", "date": {"day": "", "month": "June", "year": "1344"}},
        },
        {
            "passage": "The French king, John II, had been held captive in England. The Treaty of Br\u00e9tigny set his ransom at 3\u00a0million\u00a0crowns and allowed for hostages to be held in lieu of John. The hostages included two of his sons, several princes and nobles, four inhabitants of Paris, and two citizens from each of the nineteen principal towns of France. While these hostages were held, John returned to France to try and raise funds to pay the ransom. In 1362 John's son Louis of Anjou, a hostage in English-held Calais, escaped captivity. So, with his stand-in hostage gone, John felt honor-bound to return to captivity in England. The French crown had been at odds with Navarre  since 1354, and in 1363 the Navarrese used the captivity of John II in London and the political weakness of the Dauphin to try to seize power. Although there was no formal treaty, Edward III supported the Navarrese moves, particularly as there was a prospect that he might gain control over the northern and western provinces as a consequence. With this in mind, Edward deliberately slowed the peace negotiations. In 1364, John II died in London, while still in honourable captivity. Charles V succeeded him as king of France. On 7 May 1364, one month after the dauphin's accession and three days before his coronation as Charles V, the Navarrese suffered a crushing defeat at the Battle of Cocherel.",
            "question": "What did one of John II's replacements do in captivity?",
            "answers_spans": {"spans": ["escaped captivity"], "number": "", "date": {"day": "", "month": "", "year": ""}},
        },
        {
            "passage": "As of the census of 2000, there were 218,590 people, 79,667 households, and 60,387 families residing in the county.  The population density was 496 people per square mile (192/km\u00b2). There were 83,146 housing units at an average density of 189 per square mile (73/km\u00b2). The racial makeup of the county was 86.77% Race (United States Census), 9.27% Race (United States Census), 0.23% Race (United States Census), 1.52% Race (United States Census), 0.06% Race (United States Census), 0.69% from Race (United States Census), and 1.47% from two or more races.  1.91% of the population were Race (United States Census) or Race (United States Census) of any race. 22.5% were of German people, 13.1% Irish people, 9.8% Italian people, 9.2% English, 8.1% \"American\" and 6.0% Polish ancestry.",
            "question": "How many more people were there than families?",
            "answers_spans": {"spans": [], "number": "158203", "date": {"day": "", "month": "", "year": ""}},
        },
    ]


def get_drop_rc_fewshot() -> List[Dict]:
    """5-shot curated examples for DROP RC (from OLMES:drop_mc).

    Uses allenai/drop-gen2mc schema (passage_original, question_original, choices, answerKey).
    Source: oe_eval/tasks/fewshot_sources.py OLMES:drop_mc (first 5).
    """
    return [
        {
            "choices": {"text": ["1", "2", "3", "4"], "label": ["A", "B", "C", "D"]},
            "answerKey": "B",
            "passage_original": "To start the season, the Lions traveled south to Tampa, Florida to take on the Tampa Bay Buccaneers. The Lions scored first in the first quarter with a 23-yard field goal by Jason Hanson. The Buccaneers tied it up with a 38-yard field goal by Connor Barth, then took the lead when Aqib Talib intercepted a pass from Matthew Stafford and ran it in 28 yards. The Lions responded with a 28-yard field goal. In the second quarter, Detroit took the lead with a 36-yard touchdown catch by Calvin Johnson, and later added more points when Tony Scheffler caught an 11-yard TD pass. Tampa Bay responded with a 31-yard field goal just before halftime. The second half was relatively quiet, with each team only scoring one touchdown. First, Detroit's Calvin Johnson caught a 1-yard pass in the third quarter. The game's final points came when Mike Williams of Tampa Bay caught a 5-yard pass.  The Lions won their regular season opener for the first time since 2007",
            "question_original": "How many field goals did the Lions score?",
        },
        {
            "choices": {"text": ["Nate Kaeding", "Dwayne Bowe", "Jacob Hester", "Philip Rivers"], "label": ["A", "B", "C", "D"]},
            "answerKey": "A",
            "passage_original": "Coming off their road win over the Redskins, the Chiefs went home, donned their Dallas Texans throwbacks, and played a Week 7 AFL Legacy game with the San Diego Chargers. Kansas City would find themselves trailing in the first quarter as Chargers quarterback Philip Rivers completed a 3-yard touchdown pass to wide receiver Malcom Floyd, followed by a 10-yard touchdown pass to wide receiver Vincent Jackson. San Diego would add onto their lead in the second quarter with a 20-yard and a 39-yard field goal from kicker Nate Kaeding. The Chiefs would get onto the board in the third quarter with quarterback Matt Cassel completing a 7-yard touchdown pass to wide receiver Dwayne Bowe, but the Chargers kept their momentum going with Rivers finding running back Darren Sproles on a 58-yard touchdown pass. In the fourth quarter, San Diego sealed the win with Kaeding's 19-yard field goal and fullback Jacob Hester recovering a blocked punt in the end zone for a touchdown. With the loss, Kansas City went into their bye week at 1-6. Larry Johnson was suspended for two weeks after he made offensive comments about Todd Haley and made offensive comments about homosexuals on Twitter and in public.",
            "question_original": "which player scored the longest field goal?",
        },
        {
            "choices": {"text": ["May 1345", "July 1345", "April 1344", "June 1344"], "label": ["A", "B", "C", "D"]},
            "answerKey": "D",
            "passage_original": "In 1344, Momchil, the independent Bulgarian ruler of the Rhodope and Aegean regions, whose army grew to 2,000 men, took an important role in the Byzantine civil war. While at first he supported John Kantakouzenos, from the spring of 1344 Momchil reneged, provoked by the aggression of the Ottoman allies. In June he defeated the Ottoman fleet near the Portogalos bay. According to sources, at night the Bulgarian ruler sent boats to burn the anchored Ottoman ships and soon after he defeated the army of Kantakouzenos at Mosynopolis. Probably the first local ruler to become aware of the impending Ottoman threat, Momchil unsuccessfully pleaded with the emperors of Bulgaria and Byzantium for help. Even though his troops continued the resistance in the Eastern Rhodopes, in May 1345 the Turks led by Umur Beg marched from Asia Minor and devastated Bulgarian territories driving away people and livestock. Soon after, on 7 July 1345, Ottoman forces under Umur Beg defeated Momchil's army in the battle of Peritor near his capital Xanthi. Sources attest that the independent ruler perished in the battle without leaving asuccessor, and with little political will or leadership left to counter the Ottoman invasion.",
            "question_original": "When did Momchil defeat the Ottoman fleet near the Portogalos bay?",
        },
        {
            "choices": {"text": ["Led a rebellion", "Became king of France", "Escaped captivity", "Negotiated a peace treaty"], "label": ["A", "B", "C", "D"]},
            "answerKey": "C",
            "passage_original": "The French king, John II, had been held captive in England. The Treaty of Br\u00e9tigny set his ransom at 3\u00a0million\u00a0crowns and allowed for hostages to be held in lieu of John. The hostages included two of his sons, several princes and nobles, four inhabitants of Paris, and two citizens from each of the nineteen principal towns of France. While these hostages were held, John returned to France to try and raise funds to pay the ransom. In 1362 John's son Louis of Anjou, a hostage in English-held Calais, escaped captivity. So, with his stand-in hostage gone, John felt honor-bound to return to captivity in England. The French crown had been at odds with Navarre  since 1354, and in 1363 the Navarrese used the captivity of John II in London and the political weakness of the Dauphin to try to seize power. Although there was no formal treaty, Edward III supported the Navarrese moves, particularly as there was a prospect that he might gain control over the northern and western provinces as a consequence. With this in mind, Edward deliberately slowed the peace negotiations. In 1364, John II died in London, while still in honourable captivity. Charles V succeeded him as king of France. On 7 May 1364, one month after the dauphin's accession and three days before his coronation as Charles V, the Navarrese suffered a crushing defeat at the Battle of Cocherel.",
            "question_original": "What did one of John II's replacements do in captivity?",
        },
        {
            "choices": {"text": ["30", "12", "25", "18"], "label": ["A", "B", "C", "D"]},
            "answerKey": "D",
            "passage_original": "Hoping to rebound from their tough overtime road loss to the Raiders, the Jets went home for a Week 8 duel with the Kansas City Chiefs.  In the first quarter, New York took flight as QB Brett Favre completed an 18-yard TD pass to RB Leon Washington.  In the second quarter, the Chiefs tied the game as QB Tyler Thigpen completed a 19-yard TD pass to TE Tony Gonzalez.  The Jets would answer with Washington getting a 60-yard TD run.  Kansas City closed out the half as Thigpen completed an 11-yard TD pass to WR Mark Bradley. In the third quarter, the Chiefs took the lead as kicker Connor Barth nailed a 30-yard field goal, yet New York replied with RB Thomas Jones getting a 1-yard TD run.  In the fourth quarter, Kansas City got the lead again as CB Brandon Flowers returned an interception 91 yards for a touchdown.  Fortunately, the Jets pulled out the win with Favre completing the game-winning 15-yard TD pass to WR Laveranues Coles. During halftime, the Jets celebrated the 40th anniversary of their Super Bowl III championship team.",
            "question_original": "How many yards was the first TD pass?",
        },
    ]


def get_drop_rc_fewshot() -> List[Dict]:
    """10 fixed OLMES fewshot examples for DROP RC (gen2mc format).

    Source: OLMES:drop_mc (fewshot_sources.py).
    With num_fewshot=5 and first_n sampler, only the first 5 are used.
    """
    return [
        {
            "passage_original": "To start the season, the Lions traveled south to Tampa, Florida to take on the Tampa Bay Buccaneers. The Lions scored first in the first quarter with a 23-yard field goal by Jason Hanson. The Buccaneers tied it up with a 38-yard field goal by Connor Barth, then took the lead when Aqib Talib intercepted a pass from Matthew Stafford and ran it in 28 yards. The Lions responded with a 28-yard field goal. In the second quarter, Detroit took the lead with a 36-yard touchdown catch by Calvin Johnson, and later added more points when Tony Scheffler caught an 11-yard TD pass. Tampa Bay responded with a 31-yard field goal just before halftime. The second half was relatively quiet, with each team only scoring one touchdown. First, Detroit's Calvin Johnson caught a 1-yard pass in the third quarter. The game's final points came when Mike Williams of Tampa Bay caught a 5-yard pass.  The Lions won their regular season opener for the first time since 2007",
            "question_original": "How many field goals did the Lions score?",
            "choices": {"text": ["1", "2", "3", "4"], "label": ["A", "B", "C", "D"]},
            "answerKey": "B",
        },
        {
            "passage_original": "Coming off their road win over the Redskins, the Chiefs went home, donned their Dallas Texans throwbacks, and played a Week 7 AFL Legacy game with the San Diego Chargers. Kansas City would find themselves trailing in the first quarter as Chargers quarterback Philip Rivers completed a 3-yard touchdown pass to wide receiver Malcom Floyd, followed by a 10-yard touchdown pass to wide receiver Vincent Jackson. San Diego would add onto their lead in the second quarter with a 20-yard and a 39-yard field goal from kicker Nate Kaeding. The Chiefs would get onto the board in the third quarter with quarterback Matt Cassel completing a 7-yard touchdown pass to wide receiver Dwayne Bowe, but the Chargers kept their momentum going with Rivers finding running back Darren Sproles on a 58-yard touchdown pass. In the fourth quarter, San Diego sealed the win with Kaeding's 19-yard field goal and fullback Jacob Hester recovering a blocked punt in the end zone for a touchdown. With the loss, Kansas City went into their bye week at 1-6. Larry Johnson was suspended for two weeks after he made offensive comments about Todd Haley and made offensive comments about homosexuals on Twitter and in public.",
            "question_original": "which player scored the longest field goal?",
            "choices": {"text": ["Nate Kaeding", "Dwayne Bowe", "Jacob Hester", "Philip Rivers"], "label": ["A", "B", "C", "D"]},
            "answerKey": "A",
        },
        {
            "passage_original": "In 1344, Momchil, the independent Bulgarian ruler of the Rhodope and Aegean regions, whose army grew to 2,000 men, took an important role in the Byzantine civil war. While at first he supported John Kantakouzenos, from the spring of 1344 Momchil reneged, provoked by the aggression of the Ottoman allies. In June he defeated the Ottoman fleet near the Portogalos bay. According to sources, at night the Bulgarian ruler sent boats to burn the anchored Ottoman ships and soon after he defeated the army of Kantakouzenos at Mosynopolis. Probably the first local ruler to become aware of the impending Ottoman threat, Momchil unsuccessfully pleaded with the emperors of Bulgaria and Byzantium for help. Even though his troops continued the resistance in the Eastern Rhodopes, in May 1345 the Turks led by Umur Beg marched from Asia Minor and devastated Bulgarian territories driving away people and livestock. Soon after, on 7 July 1345, Ottoman forces under Umur Beg defeated Momchil's army in the battle of Peritor near his capital Xanthi. Sources attest that the independent ruler perished in the battle without leaving asuccessor, and with little political will or leadership left to counter the Ottoman invasion.",
            "question_original": "When did Momchil defeat the Ottoman fleet near the Portogalos bay?",
            "choices": {"text": ["May 1345", "July 1345", "April 1344", "June 1344"], "label": ["A", "B", "C", "D"]},
            "answerKey": "D",
        },
        {
            "passage_original": "The French king, John II, had been held captive in England. The Treaty of Br\u00e9tigny set his ransom at 3\u00a0million\u00a0crowns and allowed for hostages to be held in lieu of John. The hostages included two of his sons, several princes and nobles, four inhabitants of Paris, and two citizens from each of the nineteen principal towns of France. While these hostages were held, John returned to France to try and raise funds to pay the ransom. In 1362 John's son Louis of Anjou, a hostage in English-held Calais, escaped captivity. So, with his stand-in hostage gone, John felt honor-bound to return to captivity in England. The French crown had been at odds with Navarre  since 1354, and in 1363 the Navarrese used the captivity of John II in London and the political weakness of the Dauphin to try to seize power. Although there was no formal treaty, Edward III supported the Navarrese moves, particularly as there was a prospect that he might gain control over the northern and western provinces as a consequence. With this in mind, Edward deliberately slowed the peace negotiations. In 1364, John II died in London, while still in honourable captivity. Charles V succeeded him as king of France. On 7 May 1364, one month after the dauphin's accession and three days before his coronation as Charles V, the Navarrese suffered a crushing defeat at the Battle of Cocherel.",
            "question_original": "What did one of John II's replacements do in captivity?",
            "choices": {"text": ["Led a rebellion", "Became king of France", "Escaped captivity", "Negotiated a peace treaty"], "label": ["A", "B", "C", "D"]},
            "answerKey": "C",
        },
        {
            "passage_original": "Hoping to rebound from their tough overtime road loss to the Raiders, the Jets went home for a Week 8 duel with the Kansas City Chiefs.  In the first quarter, New York took flight as QB Brett Favre completed an 18-yard TD pass to RB Leon Washington.  In the second quarter, the Chiefs tied the game as QB Tyler Thigpen completed a 19-yard TD pass to TE Tony Gonzalez.  The Jets would answer with Washington getting a 60-yard TD run.  Kansas City closed out the half as Thigpen completed an 11-yard TD pass to WR Mark Bradley. In the third quarter, the Chiefs took the lead as kicker Connor Barth nailed a 30-yard field goal, yet New York replied with RB Thomas Jones getting a 1-yard TD run.  In the fourth quarter, Kansas City got the lead again as CB Brandon Flowers returned an interception 91 yards for a touchdown.  Fortunately, the Jets pulled out the win with Favre completing the game-winning 15-yard TD pass to WR Laveranues Coles. During halftime, the Jets celebrated the 40th anniversary of their Super Bowl III championship team.",
            "question_original": "How many yards was the first TD pass?",
            "choices": {"text": ["30", "12", "25", "18"], "label": ["A", "B", "C", "D"]},
            "answerKey": "D",
        },
        {
            "passage_original": "In South America , the Portuguese conquered from Spain most of the Rio Negro valley, and repelled a Spanish attack on Mato Grosso . Between September 1762 and April 1763, Spanish forces led by don Pedro Antonio de Cevallos, Governor of Buenos Aires  undertook a campaign against the Portuguese in Uruguay and South Brazil. The Spaniards conquered the Portuguese territories of Colonia do Sacramento and Rio Grande de S\u00e3o Pedro and forced the Portuguese to surrender and retreat. Under the Treaty of Paris , Spain had to return to Portugal the colony of Sacramento, while the vast and rich territory of the so-called \"Continent of S. Peter\"  would be retaken from the Spanish army during the undeclared Hispano-Portuguese war of 1763-1777. As consequence of the war the Valdivian Fort System, a Spanish defensive complex in southern Chile, was updated and reinforced from 1764 onwards. Other vulnerable localities of colonial Chile such as Chilo\u00e9 Archipelago, Concepci\u00f3n, Juan Fern\u00e1ndez Islands and Valpara\u00edso were also made ready for an eventual English attack.",
            "question_original": "What was the latter month that Spanish forces led a campaign against the Portuguese in Uruguay and South Brazil?",
            "choices": {"text": ["November", "June", "April", "September"], "label": ["A", "B", "C", "D"]},
            "answerKey": "C",
        },
        {
            "passage_original": "Trying to snap a two-game skid, the Bills flew to Gillette Stadium for a Week 3 divisional fight with the New England Patriots.  In the first quarter, QB J. P. Losman was immediately injured on the first offensive play of the game.  He would finish the series, but ended up on the bench for the rest of the game.  After New England took the lead with kicker Stephen Gostkowski's 24-yard field goal, rookie QB Trent Edwards played the rest of the game for Buffalo.  The Bills would get their only score of the game as RB Marshawn Lynch got an 8-yard TD run, and a Rian Lindell extra point put the Bills ahead surprisingly 7-3.  However, in the second quarter, the Patriots were able to open up their running game when Bills rookie standout Paul Posluszny was lost due to a broken arm. This left passing lanes open, and for the rest of the game, the Patriots dominated. QB Tom Brady's 8-yard TD pass to TE Benjamin Watson and a 3-yard TD pass to WR Randy Moss made it 17-7 at the half.  In the third quarter, New England continued its conquest with Brady's 4-yard TD pass to WR Jabar Gaffney and RB Sammy Morris' 4-yard TD run.  In the fourth quarter, the Patriots ended the day with Brady and Moss hooking up with each other again on a 45-yard TD pass.",
            "question_original": "Which team scored first?",
            "choices": {"text": ["New England", "Miami", "New York", "Buffalo"], "label": ["A", "B", "C", "D"]},
            "answerKey": "A",
        },
        {
            "passage_original": "The French conquest of Morocco took place in 1911 in the aftermath of the Agadir Crisis, when Moroccan forces besieged the French-occupied city of Fez. On 30 March 1912, Sultan Abdelhafid signed the Treaty of Fez, formally ceding Moroccan sovereignty to France, transforming Morocco into a protectorate of France. However, many regions remained in revolt until 1934, when Morocco was declared to be pacified, but in several regions French authority was maintained by cooperation with local chiefs and not military strength. On 17 April 1912, Moroccan infantrymen mutinied in the French garrison in Fez. The Moroccans were unable to take the city and were defeated by a French relief force. In late May 1912, Moroccan forces unsuccessfully attacked the enhanced French garrison at Fez. The last aftermath of the conquest of Morocco occurred in 1933-34, the pacification of Morocco took over 22 years.",
            "question_original": "In what year did the last aftermath of the conquest of Morocco occur?",
            "choices": {"text": ["1869", "1934", "1947", "1898"], "label": ["A", "B", "C", "D"]},
            "answerKey": "B",
        },
    ]


def _extract_drop_answers_gen(answer: dict) -> List[str]:
    """Extract answer strings from EleutherAI/drop answer format."""
    answers = []
    spans = answer.get("spans", [])
    if spans:
        answers.extend(spans)
    number = answer.get("number", "")
    if number:
        answers.append(str(number))
    date = answer.get("date", {})
    if date:
        date_parts = [date.get("day", ""), date.get("month", ""), date.get("year", "")]
        date_str = " ".join(p for p in date_parts if p).strip()
        if date_str:
            answers.append(date_str)
    return answers


def process_results_drop_gen(doc: dict, results: List[str]) -> dict:
    """Process results for DROP generative task (EleutherAI/drop schema)."""
    prediction = results[0] if results else ""

    answer = doc.get("answer", {})
    references = _extract_drop_answers_gen(answer)

    if not references:
        return {"em": 0.0, "f1": 0.0}

    max_em = max(_compute_exact_match(prediction, ref) for ref in references)
    max_f1 = max(_compute_f1(prediction, ref) for ref in references)

    return {"em": max_em, "f1": max_f1}


def get_drop_gen_fewshot_gen() -> List[Dict]:
    """5-shot curated examples for DROP generative (OLMES-aligned).

    Uses EleutherAI/drop schema (passage, question, answer dict).
    Source: oe_eval/tasks/fewshot_sources.py OLMES:drop (first 5).
    """
    return [
        {
            "section_id": "nfl_2201",
            "passage": "To start the season, the Lions traveled south to Tampa, Florida to take on the Tampa Bay Buccaneers. The Lions scored first in the first quarter with a 23-yard field goal by Jason Hanson. The Buccaneers tied it up with a 38-yard field goal by Connor Barth, then took the lead when Aqib Talib intercepted a pass from Matthew Stafford and ran it in 28 yards. The Lions responded with a 28-yard field goal. In the second quarter, Detroit took the lead with a 36-yard touchdown catch by Calvin Johnson, and later added more points when Tony Scheffler caught an 11-yard TD pass. Tampa Bay responded with a 31-yard field goal just before halftime. The second half was relatively quiet, with each team only scoring one touchdown. First, Detroit's Calvin Johnson caught a 1-yard pass in the third quarter. The game's final points came when Mike Williams of Tampa Bay caught a 5-yard pass.  The Lions won their regular season opener for the first time since 2007",
            "question": "How many field goals did the Lions score?",
            "query_id": "c9582e03-b01b-42ed-83e0-b90a5334aefa",
            "answer": {"number": "2", "date": {"day": "", "month": "", "year": ""}, "spans": [], "worker_id": "", "hit_id": ""},
            "validated_answers": {"number": [""], "date": [{"day": "", "month": "", "year": ""}], "spans": [[]], "worker_id": [""], "hit_id": [""]},
        },
        {
            "section_id": "nfl_1491",
            "passage": "Coming off their road win over the Redskins, the Chiefs went home, donned their Dallas Texans throwbacks, and played a Week 7 AFL Legacy game with the San Diego Chargers. Kansas City would find themselves trailing in the first quarter as Chargers quarterback Philip Rivers completed a 3-yard touchdown pass to wide receiver Malcom Floyd, followed by a 10-yard touchdown pass to wide receiver Vincent Jackson. San Diego would add onto their lead in the second quarter with a 20-yard and a 39-yard field goal from kicker Nate Kaeding. The Chiefs would get onto the board in the third quarter with quarterback Matt Cassel completing a 7-yard touchdown pass to wide receiver Dwayne Bowe, but the Chargers kept their momentum going with Rivers finding running back Darren Sproles on a 58-yard touchdown pass. In the fourth quarter, San Diego sealed the win with Kaeding's 19-yard field goal and fullback Jacob Hester recovering a blocked punt in the end zone for a touchdown. With the loss, Kansas City went into their bye week at 1-6. Larry Johnson was suspended for two weeks after he made offensive comments about Todd Haley and made offensive comments about homosexuals on Twitter and in public.",
            "question": "which player scored the longest field goal?",
            "query_id": "a849cb71-b6da-4c1a-9791-b962c9ff2d65",
            "answer": {"number": "", "date": {"day": "", "month": "", "year": ""}, "spans": ["Nate Kaeding"], "worker_id": "", "hit_id": ""},
            "validated_answers": {"number": [""], "date": [{"day": "", "month": "", "year": ""}], "spans": [[]], "worker_id": [""], "hit_id": [""]},
        },
        {
            "section_id": "history_269",
            "passage": "In 1344, Momchil, the independent Bulgarian ruler of the Rhodope and Aegean regions, whose army grew to 2,000 men, took an important role in the Byzantine civil war. While at first he supported John Kantakouzenos, from the spring of 1344 Momchil reneged, provoked by the aggression of the Ottoman allies. In June he defeated the Ottoman fleet near the Portogalos bay. According to sources, at night the Bulgarian ruler sent boats to burn the anchored Ottoman ships and soon after he defeated the army of Kantakouzenos at Mosynopolis. Probably the first local ruler to become aware of the impending Ottoman threat, Momchil unsuccessfully pleaded with the emperors of Bulgaria and Byzantium for help. Even though his troops continued the resistance in the Eastern Rhodopes, in May 1345 the Turks led by Umur Beg marched from Asia Minor and devastated Bulgarian territories driving away people and livestock. Soon after, on 7 July 1345, Ottoman forces under Umur Beg defeated Momchil's army in the battle of Peritor near his capital Xanthi. Sources attest that the independent ruler perished in the battle without leaving asuccessor, and with little political will or leadership left to counter the Ottoman invasion.",
            "question": "When did Momchil defeat the Ottoman fleet near the Portogalos bay?",
            "query_id": "3637ce9d-203d-49e0-a454-b8a49e376148",
            "answer": {"number": "", "date": {"day": "", "month": "June", "year": "1344"}, "spans": [], "worker_id": "", "hit_id": ""},
            "validated_answers": {"number": [""], "date": [{"day": "", "month": "", "year": ""}], "spans": [[]], "worker_id": [""], "hit_id": [""]},
        },
        {
            "section_id": "history_690",
            "passage": "The French king, John II, had been held captive in England. The Treaty of Br\u00e9tigny set his ransom at 3\u00a0million\u00a0crowns and allowed for hostages to be held in lieu of John. The hostages included two of his sons, several princes and nobles, four inhabitants of Paris, and two citizens from each of the nineteen principal towns of France. While these hostages were held, John returned to France to try and raise funds to pay the ransom. In 1362 John's son Louis of Anjou, a hostage in English-held Calais, escaped captivity. So, with his stand-in hostage gone, John felt honor-bound to return to captivity in England. The French crown had been at odds with Navarre  since 1354, and in 1363 the Navarrese used the captivity of John II in London and the political weakness of the Dauphin to try to seize power. Although there was no formal treaty, Edward III supported the Navarrese moves, particularly as there was a prospect that he might gain control over the northern and western provinces as a consequence. With this in mind, Edward deliberately slowed the peace negotiations. In 1364, John II died in London, while still in honourable captivity. Charles V succeeded him as king of France. On 7 May 1364, one month after the dauphin's accession and three days before his coronation as Charles V, the Navarrese suffered a crushing defeat at the Battle of Cocherel.",
            "question": "What did one of John II's replacements do in captivity?",
            "query_id": "c420c10d-1384-47f1-9850-ec9269570ca8",
            "answer": {"number": "", "date": {"day": "", "month": "", "year": ""}, "spans": ["escaped captivity"], "worker_id": "", "hit_id": ""},
            "validated_answers": {"number": [""], "date": [{"day": "", "month": "", "year": ""}], "spans": [[]], "worker_id": [""], "hit_id": [""]},
        },
        {
            "section_id": "history_2184",
            "passage": "As of the census of 2000, there were 218,590 people, 79,667 households, and 60,387 families residing in the county.  The population density was 496 people per square mile (192/km\u00b2). There were 83,146 housing units at an average density of 189 per square mile (73/km\u00b2). The racial makeup of the county was 86.77% Race (United States Census), 9.27% Race (United States Census), 0.23% Race (United States Census), 1.52% Race (United States Census), 0.06% Race (United States Census), 0.69% from Race (United States Census), and 1.47% from two or more races.  1.91% of the population were Race (United States Census) or Race (United States Census) of any race. 22.5% were of German people, 13.1% Irish people, 9.8% Italian people, 9.2% English, 8.1% \"American\" and 6.0% Polish ancestry.",
            "question": "How many more people were there than families?",
            "query_id": "c312a5d0-5318-4bbb-806e-22333f00e990",
            "answer": {"number": "158203", "date": {"day": "", "month": "", "year": ""}, "spans": [], "worker_id": "", "hit_id": ""},
            "validated_answers": {"number": [""], "date": [{"day": "", "month": "", "year": ""}], "spans": [[]], "worker_id": [""], "hit_id": [""]},
        },
    ]


def process_results_bpb(doc, results):
    """Compute answer-only BPB (OLMES-style) for DROP."""
    ll, _ = results[0]
    answer = doc["answer"]
    if answer.get("spans"):
        gold_text = answer["spans"][0]
    elif answer.get("number"):
        gold_text = str(answer["number"])
    else:
        d = answer.get("date", {})
        gold_text = f"{d.get('day', '')} {d.get('month', '')} {d.get('year', '')}".strip()
    gold_bytes = len((" " + gold_text).encode("utf-8"))
    return {"bits_per_byte": -ll / (math.log(2) * max(gold_bytes, 1))}
