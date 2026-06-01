"""Utility functions for Italian GSM8K Platinum COT task.

Dataset: ellamind/gsm8k-platinum-multilingual (ita subset)
- 1,209 grade-school math word problems (cleaned & verified)
- Fields: question, solution, final_answer
- 8-shot chain-of-thought with static Italian fewshot examples

Fewshot examples are Italian translations of the 8 canonical GSM8K
examples used by both lm-eval-harness and OLMES.
"""


def process_docs(dataset):
    """Remove rows flagged for review (translation quality issues)."""
    return dataset.filter(lambda x: not x.get("flag_for_review", False))


def doc_to_text(doc):
    return f"Domanda: {doc['question']}\nRisposta:"


def list_fewshot_samples():
    """8 canonical GSM8K fewshot examples, translated to Italian.

    Source: lm-eval-harness gsm8k-cot.yaml / OLMES STD:GSM8k.
    Reasoning in Italian, ending with 'La risposta è X.'
    """
    return [
        {
            "question": (
                "In un boschetto ci sono 15 alberi. Oggi gli operai "
                "pianteranno altri alberi nel boschetto. Alla fine ce "
                "ne saranno 21. Quanti alberi hanno piantato oggi "
                "gli operai?"
            ),
            "target": (
                "In origine ci sono 15 alberi. Poi ce ne erano 21 "
                "dopo averne piantati altri. Quindi devono essere stati "
                "21 - 15 = 6. La risposta è 6."
            ),
        },
        {
            "question": (
                "Se ci sono 3 auto nel parcheggio e ne arrivano "
                "altre 2, quante auto ci sono nel parcheggio?"
            ),
            "target": (
                "In origine ci sono 3 auto. Ne arrivano altre 2. "
                "3 + 2 = 5. La risposta è 5."
            ),
        },
        {
            "question": (
                "Lea aveva 32 cioccolatini e sua sorella ne aveva 42. "
                "Se ne hanno mangiati 35, quanti ne restano in totale?"
            ),
            "target": (
                "In origine Lea aveva 32 cioccolatini. Sua sorella "
                "ne aveva 42. In totale avevano 32 + 42 = 74. "
                "Dopo averne mangiati 35, ne restavano 74 - 35 = 39. "
                "La risposta è 39."
            ),
        },
        {
            "question": (
                "Jan aveva 20 lecca-lecca. Ne ha dati alcuni a Daniel. "
                "Ora Jan ha 12 lecca-lecca. Quanti lecca-lecca ha dato "
                "Jan a Daniel?"
            ),
            "target": (
                "Jan aveva inizialmente 20 lecca-lecca. Poi ne aveva 12 "
                "dopo averne dati alcuni a Daniel. Quindi ha dato a "
                "Daniel 20 - 12 = 8. La risposta è 8."
            ),
        },
        {
            "question": (
                "Luca ha cinque giocattoli. A Natale ha ricevuto due "
                "giocattoli dalla mamma e due dal papà. Quanti "
                "giocattoli ha adesso?"
            ),
            "target": (
                "Luca aveva inizialmente 5 giocattoli. Se ha ricevuto "
                "2 dalla mamma e 2 dal papà, sono 4 giocattoli in più. "
                "5 + 4 = 9. La risposta è 9."
            ),
        },
        {
            "question": (
                "Nella sala server c'erano nove computer. Da lunedì a "
                "giovedì sono stati installati cinque computer in più "
                "ogni giorno. Quanti computer ci sono ora nella sala "
                "server?"
            ),
            "target": (
                "In origine c'erano 9 computer. In ciascuno dei "
                "4 giorni sono stati aggiunti 5 computer. Quindi sono "
                "stati aggiunti 5 * 4 = 20 computer. 9 + 20 fa 29. "
                "La risposta è 29."
            ),
        },
        {
            "question": (
                "Michele aveva 58 palline da golf. Martedì ne ha perse "
                "23. Mercoledì ne ha perse altre 2. Quante palline da "
                "golf aveva alla fine del mercoledì?"
            ),
            "target": (
                "Michele aveva inizialmente 58 palline da golf. Dopo "
                "averne perse 23 martedì, ne aveva 58 - 23 = 35. "
                "Dopo averne perse altre 2, ne aveva 35 - 2 = 33. "
                "La risposta è 33."
            ),
        },
        {
            "question": (
                "Olivia ha 23 €. Ha comprato cinque panini a 3 € "
                "ciascuno. Quanti soldi le restano?"
            ),
            "target": (
                "Olivia aveva 23 euro. 5 panini a 3 euro ciascuno "
                "costano 5 x 3 = 15 euro. Le restano 23 - 15 euro. "
                "23 - 15 fa 8. La risposta è 8."
            ),
        },
    ]
