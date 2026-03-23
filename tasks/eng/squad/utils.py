"""SQuAD utilities for English evaluation."""

import math
import re
import string
from typing import List, Dict



# =============================================================================
# Token-level F1 metrics (SQuAD-style)
# Based on OLMES SQuADF1EMRecallMetric and lm_eval DROP implementation
# =============================================================================

def _normalize_answer(text: str) -> str:
    """Normalize answer for comparison: lowercase, remove articles/punctuation, fix whitespace."""
    # Lowercase
    text = text.lower()
    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    # Remove punctuation
    text = ''.join(ch for ch in text if ch not in string.punctuation)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text.strip()


def _get_tokens(text: str) -> List[str]:
    """Tokenize normalized text."""
    return _normalize_answer(text).split()


def _compute_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1 between prediction and reference."""
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
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def _compute_exact_match(prediction: str, reference: str) -> float:
    """Compute exact match between prediction and reference."""
    return 1.0 if _normalize_answer(prediction) == _normalize_answer(reference) else 0.0


def _max_over_references(metric_fn, prediction: str, references: List[str]) -> float:
    """Return max metric score over all reference answers."""
    if not references:
        return 0.0
    return max(metric_fn(prediction, ref) for ref in references)


def process_results_squad(doc: dict, results: List[str]) -> dict:
    """
    Process results for SQuAD generative task.
    Returns em (exact match) and f1 (token F1) metrics.
    """
    prediction = results[0] if results else ""

    # Get reference answers
    answers = doc.get("answers", {})
    references = answers.get("text", [])

    if not references:
        return {"em": 0.0, "f1": 0.0}

    em = _max_over_references(_compute_exact_match, prediction, references)
    f1 = _max_over_references(_compute_f1, prediction, references)

    return {"em": em, "f1": f1}


def squad_doc_to_target(doc: dict) -> str:
    """Extract first answer from SQuAD document."""
    answers = doc.get("answers", {})
    texts = answers.get("text", [])
    if texts:
        return texts[0]
    return ""


def get_squad_fewshot() -> List[Dict]:
    """5-shot examples for SQuAD (MC)."""
    return [
        {
            "id": "squad_mc_format_fewshot_0",
            "choices": {
                "text": [
                    "Saint Thomas Aquinas",
                    "Saint Bernadette Soubirous",
                    "Saint Francis of Assisi",
                    "Saint Joan of Arc",
                ],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "B",
            "title_original": "University_of_Notre_Dame",
            "context": 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.',
            "question": "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?",
            "answers": {"text": ["Saint Bernadette Soubirous"], "answer_start": [515]},
        },
        {
            "id": "squad_mc_format_fewshot_1",
            "choices": {"text": ["2003", "1995", "1999", "2010"], "label": ["A", "B", "C", "D"]},
            "answerKey": "A",
            "title_original": "Beyoncé",
            "context": 'Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny\'s Child. Managed by her father, Mathew Knowles, the group became one of the world\'s best-selling girl groups of all time. Their hiatus saw the release of Beyoncé\'s debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles "Crazy in Love" and "Baby Boy".',
            "question": "When did Beyonce leave Destiny's Child and become a solo singer?",
            "answers": {"text": ["2003"], "answer_start": [526]},
        },
        {
            "id": "squad_mc_format_fewshot_2",
            "choices": {"text": ["10th", "25th", "50th", "4th"], "label": ["A", "B", "C", "D"]},
            "answerKey": "D",
            "title_original": "Montana",
            "context": 'Montana i/mɒnˈtænə/ is a state in the Western region of the United States. The state\'s name is derived from the Spanish word montaña (mountain). Montana has several nicknames, although none official, including "Big Sky Country" and "The Treasure State", and slogans that include "Land of the Shining Mountains" and more recently "The Last Best Place". Montana is ranked 4th in size, but 44th in population and 48th in population density of the 50 United States. The western third of Montana contains numerous mountain ranges. Smaller island ranges are found throughout the state. In total, 77 named ranges are part of the Rocky Mountains.',
            "question": "What is the states rank in size?",
            "answers": {"text": ["4th"], "answer_start": [370]},
        },
        {
            "id": "squad_mc_format_fewshot_3",
            "choices": {
                "text": [
                    "That political destruction was sufficient",
                    "That economic destruction was necessary",
                    "That biological-physical destruction was necessary",
                    "That cultural destruction was sufficient",
                ],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "C",
            "title_original": "Genocide",
            "context": 'In the same judgement the ECHR reviewed the judgements of several international and municipal courts judgements. It noted that International Criminal Tribunal for the Former Yugoslavia and the International Court of Justice had agreed with the narrow interpretation, that biological-physical destruction was necessary for an act to qualify as genocide. The ECHR also noted that at the time of its judgement, apart from courts in Germany which had taken a broad view, that there had been few cases of genocide under other Convention States municipal laws and that "There are no reported cases in which the courts of these States have defined the type of group destruction the perpetrator must have intended in order to be found guilty of genocide".',
            "question": "Two bodies of the United Nations agreed with what restricted provision in defining genocide?",
            "answers": {"text": ["that biological-physical destruction was necessary"], "answer_start": [267]},
        },
        {
            "id": "squad_mc_format_fewshot_4",
            "choices": {
                "text": [
                    "Ciprofloxacin and vancomycin",
                    "Penicillin and erythromycin",
                    "Azithromycin and doxycycline",
                    "Amoxicillin and tetracycline",
                ],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "B",
            "title_original": "Antibiotics",
            "context": "The emergence of resistance of bacteria to antibiotics is a common phenomenon. Emergence of resistance often reflects evolutionary processes that take place during antibiotic therapy. The antibiotic treatment may select for bacterial strains with physiologically or genetically enhanced capacity to survive high doses of antibiotics. Under certain conditions, it may result in preferential growth of resistant bacteria, while growth of susceptible bacteria is inhibited by the drug. For example, antibacterial selection for strains having previously acquired antibacterial-resistance genes was demonstrated in 1943 by the Luria–Delbrück experiment. Antibiotics such as penicillin and erythromycin, which used to have a high efficacy against many bacterial species and strains, have become less effective, due to the increased resistance of many bacterial strains.",
            "question": "Which two antibiotics that have high efficacy are much less useful now?",
            "answers": {"text": ["penicillin and erythromycin"], "answer_start": [669]},
        },
    ]


def get_squad_gen_fewshot() -> List[Dict]:
    """5-shot examples for SQuAD (generative). Format matches rajpurkar/squad dataset."""
    return [
        {
            "context": "Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend \"Venite Ad Me Omnes\". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.",
            "question": "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?",
            "answers": {"text": ["Saint Bernadette Soubirous"], "answer_start": [515]},
        },
        {
            "context": "Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny's Child. Managed by her father, Mathew Knowles, the group became one of the world's best-selling girl groups of all time. Their hiatus saw the release of Beyoncé's debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles \"Crazy in Love\" and \"Baby Boy\".",
            "question": "When did Beyonce leave Destiny's Child and become a solo singer?",
            "answers": {"text": ["2003"], "answer_start": [526]},
        },
        {
            "context": "Montana i/mɒnˈtænə/ is a state in the Western region of the United States. The state's name is derived from the Spanish word montaña (mountain). Montana has several nicknames, although none official, including \"Big Sky Country\" and \"The Treasure State\", and slogans that include \"Land of the Shining Mountains\" and more recently \"The Last Best Place\". Montana is ranked 4th in size, but 44th in population and 48th in population density of the 50 United States. The western third of Montana contains numerous mountain ranges. Smaller island ranges are found throughout the state. In total, 77 named ranges are part of the Rocky Mountains.",
            "question": "What is the state's rank in size?",
            "answers": {"text": ["4th"], "answer_start": [370]},
        },
        {
            "context": "The emergence of resistance of bacteria to antibiotics is a common phenomenon. Emergence of resistance often reflects evolutionary processes that take place during antibiotic therapy. The antibiotic treatment may select for bacterial strains with physiologically or genetically enhanced capacity to survive high doses of antibiotics. Under certain conditions, it may result in preferential growth of resistant bacteria, while growth of susceptible bacteria is inhibited by the drug. For example, antibacterial selection for strains having previously acquired antibacterial-resistance genes was demonstrated in 1943 by the Luria–Delbrück experiment. Antibiotics such as penicillin and erythromycin, which used to have a high efficacy against many bacterial species and strains, have become less effective, due to the increased resistance of many bacterial strains.",
            "question": "Which two antibiotics that have high efficacy are much less useful now?",
            "answers": {"text": ["penicillin and erythromycin"], "answer_start": [669]},
        },
        {
            "context": "Frédéric François Chopin (/ˈʃoʊpæn/; French pronunciation: ​[fʁe.de.ʁik fʁɑ̃.swa ʃɔ.pɛ̃]; 22 February or 1 March 1810 – 17 October 1849), born Fryderyk Franciszek Chopin,[n 1] was a Polish and French (by citizenship and birth of father) composer and a virtuoso pianist of the Romantic era, who wrote primarily for the solo piano. He gained and has maintained renown worldwide as one of the leading musicians of his era, whose \"poetic genius was based on a professional technique that was without equal in his generation.\" Chopin was born in what was then the Duchy of Warsaw, and grew up in Warsaw, which after 1815 became part of Congress Poland. A child prodigy, he completed his musical education and composed his earlier works in Warsaw before leaving Poland at the age of 20, less than a month before the outbreak of the November 1830 Uprising.",
            "question": "Where did Chopin grow up?",
            "answers": {"text": ["Warsaw"], "answer_start": [550]},
        },
    ]


def get_squad_rc_fewshot() -> List[Dict]:
    """10 fixed OLMES fewshot examples for SQuAD RC (gen2mc format).

    Source: OLMES:squad_mc (fewshot_sources.py).
    """
    return [
        {
            "title_original": "University_of_Notre_Dame",
            "context_original": 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.',
            "question_original": "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?",
            "choices": {"text": ["Saint Thomas Aquinas", "Saint Bernadette Soubirous", "Saint Francis of Assisi", "Saint Joan of Arc"], "label": ["A", "B", "C", "D"]},
            "answerKey": "B",
        },
        {
            "title_original": "Beyonc\u00e9",
            "context_original": 'Beyonc\u00e9 Giselle Knowles-Carter (/bi\u02d0\u02c8j\u0252nse\u026a/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny\'s Child. Managed by her father, Mathew Knowles, the group became one of the world\'s best-selling girl groups of all time. Their hiatus saw the release of Beyonc\u00e9\'s debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles "Crazy in Love" and "Baby Boy".',
            "question_original": "When did Beyonce leave Destiny's Child and become a solo singer?",
            "choices": {"text": ["2003", "1995", "1999", "2010"], "label": ["A", "B", "C", "D"]},
            "answerKey": "A",
        },
        {
            "title_original": "Montana",
            "context_original": 'Montana i/m\u0252n\u02c8t\u00e6n\u0259/ is a state in the Western region of the United States. The state\'s name is derived from the Spanish word monta\u00f1a (mountain). Montana has several nicknames, although none official, including "Big Sky Country" and "The Treasure State", and slogans that include "Land of the Shining Mountains" and more recently "The Last Best Place". Montana is ranked 4th in size, but 44th in population and 48th in population density of the 50 United States. The western third of Montana contains numerous mountain ranges. Smaller island ranges are found throughout the state. In total, 77 named ranges are part of the Rocky Mountains.',
            "question_original": "What is the states rank in size?",
            "choices": {"text": ["10th", "25th", "50th", "4th"], "label": ["A", "B", "C", "D"]},
            "answerKey": "D",
        },
        {
            "title_original": "Genocide",
            "context_original": 'In the same judgement the ECHR reviewed the judgements of several international and municipal courts judgements. It noted that International Criminal Tribunal for the Former Yugoslavia and the International Court of Justice had agreed with the narrow interpretation, that biological-physical destruction was necessary for an act to qualify as genocide. The ECHR also noted that at the time of its judgement, apart from courts in Germany which had taken a broad view, that there had been few cases of genocide under other Convention States municipal laws and that "There are no reported cases in which the courts of these States have defined the type of group destruction the perpetrator must have intended in order to be found guilty of genocide".',
            "question_original": "Two bodies of the United Nations agreed with what restricted provision in defining genocide?",
            "choices": {"text": ["That political destruction was sufficient", "That economic destruction was necessary", "That biological-physical destruction was necessary", "That cultural destruction was sufficient"], "label": ["A", "B", "C", "D"]},
            "answerKey": "C",
        },
        {
            "title_original": "Antibiotics",
            "context_original": "The emergence of resistance of bacteria to antibiotics is a common phenomenon. Emergence of resistance often reflects evolutionary processes that take place during antibiotic therapy. The antibiotic treatment may select for bacterial strains with physiologically or genetically enhanced capacity to survive high doses of antibiotics. Under certain conditions, it may result in preferential growth of resistant bacteria, while growth of susceptible bacteria is inhibited by the drug. For example, antibacterial selection for strains having previously acquired antibacterial-resistance genes was demonstrated in 1943 by the Luria\u2013Delbr\u00fcck experiment. Antibiotics such as penicillin and erythromycin, which used to have a high efficacy against many bacterial species and strains, have become less effective, due to the increased resistance of many bacterial strains.",
            "question_original": "Which two antibiotics that have high efficacy are much less useful now?",
            "choices": {"text": ["Ciprofloxacin and vancomycin", "Penicillin and erythromycin", "Azithromycin and doxycycline", "Amoxicillin and tetracycline"], "label": ["A", "B", "C", "D"]},
            "answerKey": "B",
        },
        {
            "title_original": "Fr\u00e9d\u00e9ric_Chopin",
            "context_original": 'Fr\u00e9d\u00e9ric Fran\u00e7ois Chopin (/\u02c8\u0283o\u028ap\u00e6n/; French pronunciation: \u200b[f\u0281e.de.\u0281ik f\u0281\u0251\u0303.swa \u0283\u0254.p\u025b\u0303]; 22 February or 1 March 1810 \u2013 17 October 1849), born Fryderyk Franciszek Chopin,[n 1] was a Polish and French (by citizenship and birth of father) composer and a virtuoso pianist of the Romantic era, who wrote primarily for the solo piano. He gained and has maintained renown worldwide as one of the leading musicians of his era, whose "poetic genius was based on a professional technique that was without equal in his generation." Chopin was born in what was then the Duchy of Warsaw, and grew up in Warsaw, which after 1815 became part of Congress Poland. A child prodigy, he completed his musical education and composed his earlier works in Warsaw before leaving Poland at the age of 20, less than a month before the outbreak of the November 1830 Uprising.',
            "question_original": "Where did Chopin grow up?",
            "choices": {"text": ["Warsaw", "Paris", "Krak\u00f3w", "Vienna"], "label": ["A", "B", "C", "D"]},
            "answerKey": "A",
        },
        {
            "title_original": "Sino-Tibetan_relations_during_the_Ming_dynasty",
            "context_original": "The exact nature of relations between Tibet and the Ming dynasty of China (1368\u20131644) is unclear. Analysis of the relationship is further complicated by modern political conflicts and the application of Westphalian sovereignty to a time when the concept did not exist. Some Mainland Chinese scholars, such as Wang Jiawei and Nyima Gyaincain, assert that the Ming dynasty had unquestioned sovereignty over Tibet, pointing to the Ming court's issuing of various titles to Tibetan leaders, Tibetans' full acceptance of these titles, and a renewal process for successors of these titles that involved traveling to the Ming capital. Scholars within China also argue that Tibet has been an integral part of China since the 13th century and that it was thus a part of the Ming Empire. But most scholars outside China, such as Turrell V. Wylie, Melvin C. Goldstein, and Helmut Hoffman, say that the relationship was one of suzerainty, that Ming titles were only nominal, that Tibet remained an independent region outside Ming control, and that it simply paid tribute until the Jiajing Emperor (1521\u20131566), who ceased relations with Tibet.",
            "question_original": "Who were Wang Jiawei and Nyima Gyaincain?",
            "choices": {"text": ["European historians", "Ming dynasty emperors", "Tibetan monks", "Mainland Chinese scholars"], "label": ["A", "B", "C", "D"]},
            "answerKey": "D",
        },
        {
            "title_original": "IPod",
            "context_original": "The iPod is a line of portable media players and multi-purpose pocket computers designed and marketed by Apple Inc. The first line was released on October 23, 2001, about 8\u00bd months after iTunes (Macintosh version) was released. The most recent iPod redesigns were announced on July 15, 2015. There are three current versions of the iPod: the ultra-compact iPod Shuffle, the compact iPod Nano and the touchscreen iPod Touch.",
            "question_original": "Which company produces the iPod?",
            "choices": {"text": ["Sony", "Microsoft", "Apple", "Samsung"], "label": ["A", "B", "C", "D"]},
            "answerKey": "C",
        },
        {
            "title_original": "The_Legend_of_Zelda:_Twilight_Princess",
            "context_original": "The Legend of Zelda: Twilight Princess (Japanese: \u30bc\u30eb\u30c0\u306e\u4f1d\u8aac \u30c8\u30ef\u30a4\u30e9\u30a4\u30c8\u30d7\u30ea\u30f3\u30bb\u30b9, Hepburn: Zeruda no Densetsu: Towairaito Purinsesu?) is an action-adventure game developed and published by Nintendo for the GameCube and Wii home video game consoles. It is the thirteenth installment in the The Legend of Zelda series. Originally planned for release on the GameCube in November 2005, Twilight Princess was delayed by Nintendo to allow its developers to refine the game, add more content, and port it to the Wii. The Wii version was released alongside the console in North America in November 2006, and in Japan, Europe, and Australia the following month. The GameCube version was released worldwide in December 2006.",
            "question_original": "When was Twilight Princess launched in North America?",
            "choices": {"text": ["November 2006", "November 2005", "October 2006", "December 2006"], "label": ["A", "B", "C", "D"]},
            "answerKey": "A",
        },
        {
            "title_original": "Spectre_(2015_film)",
            "context_original": "Spectre (2015) is the twenty-fourth James Bond film produced by Eon Productions. It features Daniel Craig in his fourth performance as James Bond, and Christoph Waltz as Ernst Stavro Blofeld, with the film marking the character's re-introduction into the series. It was directed by Sam Mendes as his second James Bond film following Skyfall, and was written by John Logan, Neal Purvis, Robert Wade and Jez Butterworth. It is distributed by Metro-Goldwyn-Mayer and Columbia Pictures. With a budget around $245 million, it is the most expensive Bond film and one of the most expensive films ever made.",
            "question_original": "How many James Bond films has Eon Productions produced?",
            "choices": {"text": ["Thirty", "Twenty-four", "Fifteen", "Twenty"], "label": ["A", "B", "C", "D"]},
            "answerKey": "B",
        },
    ]

def get_squad_olmes_gen_fewshot() -> List[Dict]:
    """taken from FEWSHOT_SOURCES["OLMES:squad"] """
    return [
    {
        "id": "5733be284776f41900661182",
        "title": "University_of_Notre_Dame",
        "context": 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.',
        "question": "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?",
        "answers": {"text": ["Saint Bernadette Soubirous"], "answer_start": [515]},
    },
    {
        "id": "56be85543aeaaa14008c9066",
        "title": "Beyoncé",
        "context": 'Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny\'s Child. Managed by her father, Mathew Knowles, the group became one of the world\'s best-selling girl groups of all time. Their hiatus saw the release of Beyoncé\'s debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles "Crazy in Love" and "Baby Boy".',
        "question": "When did Beyonce leave Destiny's Child and become a solo singer?",
        "answers": {"text": ["2003"], "answer_start": [526]},
    },
    {
        "id": "5733bd9bd058e614000b619a",
        "title": "Montana",
        "context": 'Montana i/mɒnˈtænə/ is a state in the Western region of the United States. The state\'s name is derived from the Spanish word montaña (mountain). Montana has several nicknames, although none official, including "Big Sky Country" and "The Treasure State", and slogans that include "Land of the Shining Mountains" and more recently "The Last Best Place". Montana is ranked 4th in size, but 44th in population and 48th in population density of the 50 United States. The western third of Montana contains numerous mountain ranges. Smaller island ranges are found throughout the state. In total, 77 named ranges are part of the Rocky Mountains.',
        "question": "What is the states rank in size?",
        "answers": {"text": ["4th"], "answer_start": [370]},
    },
    {
        "id": "5733ba844776f41900661146",
        "title": "Genocide",
        "context": 'In the same judgement the ECHR reviewed the judgements of several international and municipal courts judgements. It noted that International Criminal Tribunal for the Former Yugoslavia and the International Court of Justice had agreed with the narrow interpretation, that biological-physical destruction was necessary for an act to qualify as genocide. The ECHR also noted that at the time of its judgement, apart from courts in Germany which had taken a broad view, that there had been few cases of genocide under other Convention States municipal laws and that "There are no reported cases in which the courts of these States have defined the type of group destruction the perpetrator must have intended in order to be found guilty of genocide".',
        "question": "Two bodies of the United Nations agreed with what restricted provision in defining genocide?",
        "answers": {
            "text": ["that biological-physical destruction was necessary"],
            "answer_start": [267],
        },
    },
    {
        "id": "5733bc38d058e614000b618a",
        "title": "Antibiotics",
        "context": "The emergence of resistance of bacteria to antibiotics is a common phenomenon. Emergence of resistance often reflects evolutionary processes that take place during antibiotic therapy. The antibiotic treatment may select for bacterial strains with physiologically or genetically enhanced capacity to survive high doses of antibiotics. Under certain conditions, it may result in preferential growth of resistant bacteria, while growth of susceptible bacteria is inhibited by the drug. For example, antibacterial selection for strains having previously acquired antibacterial-resistance genes was demonstrated in 1943 by the Luria–Delbrück experiment. Antibiotics such as penicillin and erythromycin, which used to have a high efficacy against many bacterial species and strains, have become less effective, due to the increased resistance of many bacterial strains.",
        "question": "Which two antibiotics that have high efficacy are much less useful now?",
        "answers": {"text": ["penicillin and erythromycin"], "answer_start": [669]},
    },
    {
        "id": "56ce0a3762d2951400fa69d8",
        "title": "Frédéric_Chopin",
        "context": 'Frédéric François Chopin (/ˈʃoʊpæn/; French pronunciation: \u200b[fʁe.de.ʁik fʁɑ̃.swa ʃɔ.pɛ̃]; 22 February or 1 March 1810 – 17 October 1849), born Fryderyk Franciszek Chopin,[n 1] was a Polish and French (by citizenship and birth of father) composer and a virtuoso pianist of the Romantic era, who wrote primarily for the solo piano. He gained and has maintained renown worldwide as one of the leading musicians of his era, whose "poetic genius was based on a professional technique that was without equal in his generation." Chopin was born in what was then the Duchy of Warsaw, and grew up in Warsaw, which after 1815 became part of Congress Poland. A child prodigy, he completed his musical education and composed his earlier works in Warsaw before leaving Poland at the age of 20, less than a month before the outbreak of the November 1830 Uprising.',
        "question": "Where did Chopin grow up?",
        "answers": {"text": ["Warsaw"], "answer_start": [568]},
    },
    {
        "id": "56cc239e6d243a140015eeb7",
        "title": "Sino-Tibetan_relations_during_the_Ming_dynasty",
        "context": "The exact nature of relations between Tibet and the Ming dynasty of China (1368–1644) is unclear. Analysis of the relationship is further complicated by modern political conflicts and the application of Westphalian sovereignty to a time when the concept did not exist. Some Mainland Chinese scholars, such as Wang Jiawei and Nyima Gyaincain, assert that the Ming dynasty had unquestioned sovereignty over Tibet, pointing to the Ming court's issuing of various titles to Tibetan leaders, Tibetans' full acceptance of these titles, and a renewal process for successors of these titles that involved traveling to the Ming capital. Scholars within China also argue that Tibet has been an integral part of China since the 13th century and that it was thus a part of the Ming Empire. But most scholars outside China, such as Turrell V. Wylie, Melvin C. Goldstein, and Helmut Hoffman, say that the relationship was one of suzerainty, that Ming titles were only nominal, that Tibet remained an independent region outside Ming control, and that it simply paid tribute until the Jiajing Emperor (1521–1566), who ceased relations with Tibet.",
        "question": "Who were Wang Jiawei and Nyima Gyaincain?",
        "answers": {"text": ["Mainland Chinese scholars"], "answer_start": [274]},
    },
    {
        "id": "56cc55856d243a140015ef0a",
        "title": "IPod",
        "context": "The iPod is a line of portable media players and multi-purpose pocket computers designed and marketed by Apple Inc. The first line was released on October 23, 2001, about 8½ months after iTunes (Macintosh version) was released. The most recent iPod redesigns were announced on July 15, 2015. There are three current versions of the iPod: the ultra-compact iPod Shuffle, the compact iPod Nano and the touchscreen iPod Touch.",
        "question": "Which company produces the iPod?",
        "answers": {"text": ["Apple"], "answer_start": [105]},
    },
    {
        "id": "56cd8a5f62d2951400fa6691",
        "title": "The_Legend_of_Zelda:_Twilight_Princess",
        "context": "The Legend of Zelda: Twilight Princess (Japanese: ゼルダの伝説 トワイライトプリンセス, Hepburn: Zeruda no Densetsu: Towairaito Purinsesu?) is an action-adventure game developed and published by Nintendo for the GameCube and Wii home video game consoles. It is the thirteenth installment in the The Legend of Zelda series. Originally planned for release on the GameCube in November 2005, Twilight Princess was delayed by Nintendo to allow its developers to refine the game, add more content, and port it to the Wii. The Wii version was released alongside the console in North America in November 2006, and in Japan, Europe, and Australia the following month. The GameCube version was released worldwide in December 2006.[b]",
        "question": "When was Twilight Princess launched in North America?",
        "answers": {"text": ["November 2006"], "answer_start": [569]},
    },
    {
        "id": "56cf2e15aab44d1400b88dc9",
        "title": "Spectre_(2015_film)",
        "context": "Spectre (2015) is the twenty-fourth James Bond film produced by Eon Productions. It features Daniel Craig in his fourth performance as James Bond, and Christoph Waltz as Ernst Stavro Blofeld, with the film marking the character's re-introduction into the series. It was directed by Sam Mendes as his second James Bond film following Skyfall, and was written by John Logan, Neal Purvis, Robert Wade and Jez Butterworth. It is distributed by Metro-Goldwyn-Mayer and Columbia Pictures. With a budget around $245 million, it is the most expensive Bond film and one of the most expensive films ever made.",
        "question": "How many James Bond films has Eon Productions produced?",
        "answers": {"text": ["twenty-four"], "answer_start": [22]},
    },
]


def get_squad_olmes_mc_fewshot() -> List[Dict]:
    """Taken from FEWSHOT_SOURCES["OLMES:squad_mc"]"""
    return [
    {
        "id": "squad_mc_format_fewshot_0",
        "choices": {
            "text": [
                "Saint Thomas Aquinas",
                "Saint Bernadette Soubirous",
                "Saint Francis of Assisi",
                "Saint Joan of Arc",
            ],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "B",
        "title_original": "University_of_Notre_Dame",
        "context_original": 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.',
        "question_original": "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?",
        "answers_original": {"text": ["Saint Bernadette Soubirous"], "answer_start": [515]},
    },
    {
        "id": "squad_mc_format_fewshot_1",
        "choices": {"text": ["2003", "1995", "1999", "2010"], "label": ["A", "B", "C", "D"]},
        "answerKey": "A",
        "title_original": "Beyonc\u00e9",
        "context_original": 'Beyonc\u00e9 Giselle Knowles-Carter (/bi\u02d0\u02c8j\u0252nse\u026a/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny\'s Child. Managed by her father, Mathew Knowles, the group became one of the world\'s best-selling girl groups of all time. Their hiatus saw the release of Beyonc\u00e9\'s debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles "Crazy in Love" and "Baby Boy".',
        "question_original": "When did Beyonce leave Destiny's Child and become a solo singer?",
        "answers_original": {"text": ["2003"], "answer_start": [526]},
    },
    {
        "id": "squad_mc_format_fewshot_2",
        "choices": {"text": ["10th", "25th", "50th", "4th"], "label": ["A", "B", "C", "D"]},
        "answerKey": "D",
        "title_original": "Montana",
        "context_original": 'Montana i/m\u0252n\u02c8t\u00e6n\u0259/ is a state in the Western region of the United States. The state\'s name is derived from the Spanish word monta\u00f1a (mountain). Montana has several nicknames, although none official, including "Big Sky Country" and "The Treasure State", and slogans that include "Land of the Shining Mountains" and more recently "The Last Best Place". Montana is ranked 4th in size, but 44th in population and 48th in population density of the 50 United States. The western third of Montana contains numerous mountain ranges. Smaller island ranges are found throughout the state. In total, 77 named ranges are part of the Rocky Mountains.',
        "question_original": "What is the states rank in size?",
        "answers_original": {"text": ["4th"], "answer_start": [370]},
    },
    {
        "id": "squad_mc_format_fewshot_3",
        "choices": {
            "text": [
                "That political destruction was sufficient",
                "That economic destruction was necessary",
                "That biological-physical destruction was necessary",
                "That cultural destruction was sufficient",
            ],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "C",
        "title_original": "Genocide",
        "context_original": 'In the same judgement the ECHR reviewed the judgements of several international and municipal courts judgements. It noted that International Criminal Tribunal for the Former Yugoslavia and the International Court of Justice had agreed with the narrow interpretation, that biological-physical destruction was necessary for an act to qualify as genocide. The ECHR also noted that at the time of its judgement, apart from courts in Germany which had taken a broad view, that there had been few cases of genocide under other Convention States municipal laws and that "There are no reported cases in which the courts of these States have defined the type of group destruction the perpetrator must have intended in order to be found guilty of genocide".',
        "question_original": "Two bodies of the United Nations agreed with what restricted provision in defining genocide?",
        "answers_original": {
            "text": ["that biological-physical destruction was necessary"],
            "answer_start": [267],
        },
    },
    {
        "id": "squad_mc_format_fewshot_4",
        "choices": {
            "text": [
                "Ciprofloxacin and vancomycin",
                "Penicillin and erythromycin",
                "Azithromycin and doxycycline",
                "Amoxicillin and tetracycline",
            ],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "B",
        "title_original": "Antibiotics",
        "context_original": "The emergence of resistance of bacteria to antibiotics is a common phenomenon. Emergence of resistance often reflects evolutionary processes that take place during antibiotic therapy. The antibiotic treatment may select for bacterial strains with physiologically or genetically enhanced capacity to survive high doses of antibiotics. Under certain conditions, it may result in preferential growth of resistant bacteria, while growth of susceptible bacteria is inhibited by the drug. For example, antibacterial selection for strains having previously acquired antibacterial-resistance genes was demonstrated in 1943 by the Luria\u2013Delbr\u00fcck experiment. Antibiotics such as penicillin and erythromycin, which used to have a high efficacy against many bacterial species and strains, have become less effective, due to the increased resistance of many bacterial strains.",
        "question_original": "Which two antibiotics that have high efficacy are much less useful now?",
        "answers_original": {"text": ["penicillin and erythromycin"], "answer_start": [669]},
    },
    {
        "id": "squad_mc_format_fewshot_5",
        "choices": {
            "text": ["Warsaw", "Paris", "Krak\u00f3w", "Vienna"],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "A",
        "title_original": "Fr\u00e9d\u00e9ric_Chopin",
        "context_original": 'Fr\u00e9d\u00e9ric Fran\u00e7ois Chopin (/\u02c8\u0283o\u028ap\u00e6n/; French pronunciation: \u200b[f\u0281e.de.\u0281ik f\u0281\u0251\u0303.swa \u0283\u0254.p\u025b\u0303]; 22 February or 1 March 1810 \u2013 17 October 1849), born Fryderyk Franciszek Chopin,[n 1] was a Polish and French (by citizenship and birth of father) composer and a virtuoso pianist of the Romantic era, who wrote primarily for the solo piano. He gained and has maintained renown worldwide as one of the leading musicians of his era, whose "poetic genius was based on a professional technique that was without equal in his generation." Chopin was born in what was then the Duchy of Warsaw, and grew up in Warsaw, which after 1815 became part of Congress Poland. A child prodigy, he completed his musical education and composed his earlier works in Warsaw before leaving Poland at the age of 20, less than a month before the outbreak of the November 1830 Uprising.',
        "question_original": "Where did Chopin grow up?",
        "answers_original": {"text": ["Warsaw"], "answer_start": [568]},
    },
    {
        "id": "squad_mc_format_fewshot_6",
        "choices": {
            "text": [
                "European historians",
                "Ming dynasty emperors",
                "Tibetan monks",
                "Mainland Chinese scholars",
            ],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "D",
        "title_original": "Sino-Tibetan_relations_during_the_Ming_dynasty",
        "context_original": "The exact nature of relations between Tibet and the Ming dynasty of China (1368\u20131644) is unclear. Analysis of the relationship is further complicated by modern political conflicts and the application of Westphalian sovereignty to a time when the concept did not exist. Some Mainland Chinese scholars, such as Wang Jiawei and Nyima Gyaincain, assert that the Ming dynasty had unquestioned sovereignty over Tibet, pointing to the Ming court's issuing of various titles to Tibetan leaders, Tibetans' full acceptance of these titles, and a renewal process for successors of these titles that involved traveling to the Ming capital. Scholars within China also argue that Tibet has been an integral part of China since the 13th century and that it was thus a part of the Ming Empire. But most scholars outside China, such as Turrell V. Wylie, Melvin C. Goldstein, and Helmut Hoffman, say that the relationship was one of suzerainty, that Ming titles were only nominal, that Tibet remained an independent region outside Ming control, and that it simply paid tribute until the Jiajing Emperor (1521\u20131566), who ceased relations with Tibet.",
        "question_original": "Who were Wang Jiawei and Nyima Gyaincain?",
        "answers_original": {"text": ["Mainland Chinese scholars"], "answer_start": [274]},
    },
    {
        "id": "squad_mc_format_fewshot_7",
        "choices": {
            "text": ["Sony", "Microsoft", "Apple", "Samsung"],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "C",
        "title_original": "IPod",
        "context_original": "The iPod is a line of portable media players and multi-purpose pocket computers designed and marketed by Apple Inc. The first line was released on October 23, 2001, about 8\u00bd months after iTunes (Macintosh version) was released. The most recent iPod redesigns were announced on July 15, 2015. There are three current versions of the iPod: the ultra-compact iPod Shuffle, the compact iPod Nano and the touchscreen iPod Touch.",
        "question_original": "Which company produces the iPod?",
        "answers_original": {"text": ["Apple"], "answer_start": [105]},
    },
    {
        "id": "squad_mc_format_fewshot_8",
        "choices": {
            "text": ["November 2006", "November 2005", "October 2006", "December 2006"],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "A",
        "title_original": "The_Legend_of_Zelda:_Twilight_Princess",
        "context_original": "The Legend of Zelda: Twilight Princess (Japanese: \u30bc\u30eb\u30c0\u306e\u4f1d\u8aac \u30c8\u30ef\u30a4\u30e9\u30a4\u30c8\u30d7\u30ea\u30f3\u30bb\u30b9, Hepburn: Zeruda no Densetsu: Towairaito Purinsesu?) is an action-adventure game developed and published by Nintendo for the GameCube and Wii home video game consoles. It is the thirteenth installment in the The Legend of Zelda series. Originally planned for release on the GameCube in November 2005, Twilight Princess was delayed by Nintendo to allow its developers to refine the game, add more content, and port it to the Wii. The Wii version was released alongside the console in North America in November 2006, and in Japan, Europe, and Australia the following month. The GameCube version was released worldwide in December 2006.[b]",
        "question_original": "When was Twilight Princess launched in North America?",
        "answers_original": {"text": ["November 2006"], "answer_start": [569]},
    },
    {
        "id": "squad_mc_format_fewshot_9",
        "choices": {
            "text": ["Thirty", "Twenty-four", "Fifteen", "Twenty"],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "B",
        "title_original": "Spectre_(2015_film)",
        "context_original": "Spectre (2015) is the twenty-fourth James Bond film produced by Eon Productions. It features Daniel Craig in his fourth performance as James Bond, and Christoph Waltz as Ernst Stavro Blofeld, with the film marking the character's re-introduction into the series. It was directed by Sam Mendes as his second James Bond film following Skyfall, and was written by John Logan, Neal Purvis, Robert Wade and Jez Butterworth. It is distributed by Metro-Goldwyn-Mayer and Columbia Pictures. With a budget around $245 million, it is the most expensive Bond film and one of the most expensive films ever made.",
        "question_original": "How many James Bond films has Eon Productions produced?",
        "answers_original": {"text": ["twenty-four"], "answer_start": [22]},
    },
]


def process_results_bpb(doc, results):
    """Compute answer-only BPB (OLMES-style) for SQuAD."""
    ll, _ = results[0]
    gold_text = doc["answers"]["text"][0]
    gold_bytes = len((" " + gold_text).encode("utf-8"))
    return {"bits_per_byte": -ll / (math.log(2) * max(gold_bytes, 1))}
