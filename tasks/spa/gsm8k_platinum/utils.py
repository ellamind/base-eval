"""Utility functions for Spanish GSM8K Platinum COT task.

Dataset: ellamind/gsm8k-platinum-multilingual (spa subset)
- 1,209 grade-school math word problems (cleaned & verified)
- Fields: question, solution, final_answer
- 8-shot chain-of-thought with static Spanish fewshot examples

Fewshot examples are Spanish translations of the 8 canonical GSM8K
examples used by both lm-eval-harness and OLMES.
"""


def process_docs(dataset):
    """Remove rows flagged for review (translation quality issues)."""
    return dataset.filter(lambda x: not x.get("flag_for_review", False))


def doc_to_text(doc):
    return f"Pregunta: {doc['question']}\nRespuesta:"


def list_fewshot_samples():
    """8 canonical GSM8K fewshot examples, translated to Spanish.

    Source: lm-eval-harness gsm8k-cot.yaml / OLMES STD:GSM8k.
    Reasoning in Spanish, ending with 'La respuesta es X.'
    """
    return [
        {
            "question": (
                "En un bosque hay 15 árboles. Hoy los trabajadores "
                "plantarán más árboles en el bosque. Cuando terminen, "
                "habrá 21 árboles. ¿Cuántos árboles han plantado hoy "
                "los trabajadores?"
            ),
            "target": (
                "Originalmente hay 15 árboles. Luego hubo 21 árboles "
                "después de plantar más. Así que deben haber sido "
                "21 - 15 = 6. La respuesta es 6."
            ),
        },
        {
            "question": (
                "Si hay 3 coches en el aparcamiento y llegan 2 más, "
                "¿cuántos coches hay en el aparcamiento?"
            ),
            "target": (
                "Originalmente hay 3 coches. Llegan 2 más. "
                "3 + 2 = 5. La respuesta es 5."
            ),
        },
        {
            "question": (
                "Lea tenía 32 bombones y su hermana tenía 42. "
                "Si se comieron 35, ¿cuántos les quedan en total?"
            ),
            "target": (
                "Originalmente Lea tenía 32 bombones. Su hermana "
                "tenía 42. En total tenían 32 + 42 = 74. "
                "Después de comerse 35, les quedaron 74 - 35 = 39. "
                "La respuesta es 39."
            ),
        },
        {
            "question": (
                "Jan tenía 20 piruletas. Le dio algunas a Daniel. "
                "Ahora Jan tiene 12 piruletas. ¿Cuántas piruletas "
                "le dio Jan a Daniel?"
            ),
            "target": (
                "Jan tenía inicialmente 20 piruletas. Después tenía 12 "
                "tras darle algunas a Daniel. Así que le dio a Daniel "
                "20 - 12 = 8. La respuesta es 8."
            ),
        },
        {
            "question": (
                "Lucas tiene cinco juguetes. En Navidad recibió dos "
                "juguetes de su madre y dos de su padre. ¿Cuántos "
                "juguetes tiene ahora?"
            ),
            "target": (
                "Lucas tenía inicialmente 5 juguetes. Si recibió 2 de "
                "su madre y 2 de su padre, son 4 juguetes más. "
                "5 + 4 = 9. La respuesta es 9."
            ),
        },
        {
            "question": (
                "En la sala de servidores había nueve ordenadores. "
                "De lunes a jueves se instalaron cinco ordenadores "
                "más cada día. ¿Cuántos ordenadores hay ahora en "
                "la sala de servidores?"
            ),
            "target": (
                "Originalmente había 9 ordenadores. En cada uno de "
                "los 4 días se añadieron 5 ordenadores más. Así que "
                "se añadieron 5 * 4 = 20 ordenadores. 9 + 20 es 29. "
                "La respuesta es 29."
            ),
        },
        {
            "question": (
                "Miguel tenía 58 pelotas de golf. El martes perdió 23. "
                "El miércoles perdió 2 más. ¿Cuántas pelotas de golf "
                "tenía al final del miércoles?"
            ),
            "target": (
                "Miguel tenía inicialmente 58 pelotas de golf. Después "
                "de perder 23 el martes, tenía 58 - 23 = 35. "
                "Después de perder 2 más, tenía 35 - 2 = 33 pelotas "
                "de golf. La respuesta es 33."
            ),
        },
        {
            "question": (
                "Olivia tiene 23 €. Compró cinco panecillos a 3 € "
                "cada uno. ¿Cuánto dinero le queda?"
            ),
            "target": (
                "Olivia tenía 23 euros. 5 panecillos a 3 euros cada uno "
                "cuestan 5 x 3 = 15 euros. Le quedan 23 - 15 euros. "
                "23 - 15 es 8. La respuesta es 8."
            ),
        },
    ]
