"""Utility functions for French GSM8K Platinum COT task.

Dataset: ellamind/gsm8k-platinum-multilingual (fra subset)
- 1,209 grade-school math word problems (cleaned & verified)
- Fields: question, solution, final_answer
- 8-shot chain-of-thought with static French fewshot examples

Fewshot examples are French translations of the 8 canonical GSM8K
examples used by both lm-eval-harness and OLMES.
"""


def process_docs(dataset):
    """Remove rows flagged for review (translation quality issues)."""
    return dataset.filter(lambda x: not x.get("flag_for_review", False))


def doc_to_text(doc):
    return f"Question : {doc['question']}\nR\u00e9ponse :"


def list_fewshot_samples():
    """8 canonical GSM8K fewshot examples, translated to French.

    Source: lm-eval-harness gsm8k-cot.yaml / OLMES STD:GSM8k.
    Reasoning in French, ending with 'La r\u00e9ponse est X.'
    """
    return [
        {
            "question": (
                "Dans un bosquet, il y a 15 arbres. Aujourd'hui, les "
                "ouvriers vont planter d'autres arbres dans le bosquet. "
                "\u00c0 la fin, il y en aura 21. Combien d'arbres les "
                "ouvriers ont-ils plant\u00e9s aujourd'hui ?"
            ),
            "target": (
                "\u00c0 l'origine, il y a 15 arbres. Ensuite, il y en avait "
                "21 apr\u00e8s en avoir plant\u00e9 d'autres. Il devait donc y en "
                "avoir 21 - 15 = 6. La r\u00e9ponse est 6."
            ),
        },
        {
            "question": (
                "S'il y a 3 voitures sur le parking et que 2 autres "
                "arrivent, combien y a-t-il de voitures sur le parking ?"
            ),
            "target": (
                "\u00c0 l'origine, il y a 3 voitures. 2 autres arrivent. "
                "3 + 2 = 5. La r\u00e9ponse est 5."
            ),
        },
        {
            "question": (
                "L\u00e9a avait 32 chocolats et sa s\u0153ur en avait 42. "
                "S'ils en ont mang\u00e9 35, combien leur en reste-t-il "
                "au total ?"
            ),
            "target": (
                "\u00c0 l'origine, L\u00e9a avait 32 chocolats. Sa s\u0153ur en "
                "avait 42. Au total, elles en avaient 32 + 42 = 74. "
                "Apr\u00e8s en avoir mang\u00e9 35, il leur restait 74 - 35 = 39. "
                "La r\u00e9ponse est 39."
            ),
        },
        {
            "question": (
                "Jan avait 20 sucettes. Il en a donn\u00e9 quelques-unes "
                "\u00e0 Daniel. Maintenant, Jan a 12 sucettes. Combien "
                "de sucettes Jan a-t-il donn\u00e9es \u00e0 Daniel ?"
            ),
            "target": (
                "Jan avait initialement 20 sucettes. Ensuite, il en "
                "avait 12 apr\u00e8s en avoir donn\u00e9 quelques-unes \u00e0 Daniel. "
                "Il a donc donn\u00e9 \u00e0 Daniel 20 - 12 = 8. "
                "La r\u00e9ponse est 8."
            ),
        },
        {
            "question": (
                "Lucas a cinq jouets. Pour No\u00ebl, il a re\u00e7u deux "
                "jouets de sa m\u00e8re et deux de son p\u00e8re. Combien "
                "de jouets a-t-il maintenant ?"
            ),
            "target": (
                "Lucas avait initialement 5 jouets. S'il a re\u00e7u 2 "
                "de sa m\u00e8re et 2 de son p\u00e8re, cela fait 4 jouets "
                "de plus. 5 + 4 = 9. La r\u00e9ponse est 9."
            ),
        },
        {
            "question": (
                "Dans la salle des serveurs, il y avait neuf "
                "ordinateurs. Du lundi au jeudi, cinq ordinateurs "
                "suppl\u00e9mentaires ont \u00e9t\u00e9 install\u00e9s chaque jour. "
                "Combien y a-t-il d'ordinateurs maintenant dans "
                "la salle des serveurs ?"
            ),
            "target": (
                "\u00c0 l'origine, il y avait 9 ordinateurs. Chacun des "
                "4 jours, 5 ordinateurs suppl\u00e9mentaires ont \u00e9t\u00e9 "
                "ajout\u00e9s. Donc 5 * 4 = 20 ordinateurs ont \u00e9t\u00e9 "
                "ajout\u00e9s. 9 + 20 font 29. La r\u00e9ponse est 29."
            ),
        },
        {
            "question": (
                "Michel avait 58 balles de golf. Mardi, il en a perdu "
                "23. Mercredi, il en a perdu 2 de plus. Combien de "
                "balles de golf avait-il \u00e0 la fin du mercredi ?"
            ),
            "target": (
                "Michel avait initialement 58 balles de golf. Apr\u00e8s en "
                "avoir perdu 23 mardi, il en avait 58 - 23 = 35. "
                "Apr\u00e8s en avoir perdu 2 de plus, il en avait "
                "35 - 2 = 33 balles de golf. La r\u00e9ponse est 33."
            ),
        },
        {
            "question": (
                "Olivia a 23 \u20ac. Elle a achet\u00e9 cinq petits pains \u00e0 "
                "3 \u20ac chacun. Combien d'argent lui reste-t-il ?"
            ),
            "target": (
                "Olivia avait 23 euros. 5 petits pains \u00e0 3 euros "
                "chacun co\u00fbtent 5 x 3 = 15 euros. Il lui reste "
                "23 - 15 euros. 23 - 15 font 8. La r\u00e9ponse est 8."
            ),
        },
    ]
