"""Utility functions for German GSM8K Platinum COT task.

Dataset: ellamind/gsm8k-platinum-multilingual (deu subset)
- 1,209 grade-school math word problems (cleaned & verified)
- Fields: question, solution, final_answer
- 8-shot chain-of-thought with static German fewshot examples

Fewshot examples are German translations of the 8 canonical GSM8K
examples used by both lm-eval-harness and OLMES.
"""


def process_docs(dataset):
    """Remove rows flagged for review (translation quality issues)."""
    return dataset.filter(lambda x: not x.get("flag_for_review", False))


def doc_to_text(doc):
    return f"Frage: {doc['question']}\nAntwort:"


def list_fewshot_samples():
    """8 canonical GSM8K fewshot examples, translated to German.

    Source: lm-eval-harness gsm8k-cot.yaml / OLMES STD:GSM8k.
    Reasoning in German, ending with 'Die Antwort ist X.'
    """
    return [
        {
            "question": (
                "In einem Hain stehen 15 Bäume. Heute werden die Arbeiter "
                "weitere Bäume im Hain pflanzen. Danach werden es 21 Bäume "
                "sein. Wie viele Bäume haben die Arbeiter heute gepflanzt?"
            ),
            "target": (
                "Ursprünglich stehen 15 Bäume dort. Danach waren es 21 Bäume, "
                "nachdem weitere gepflanzt wurden. Es müssen also "
                "21 - 15 = 6 gewesen sein. Die Antwort ist 6."
            ),
        },
        {
            "question": (
                "Wenn 3 Autos auf dem Parkplatz stehen und 2 weitere "
                "ankommen, wie viele Autos sind dann auf dem Parkplatz?"
            ),
            "target": (
                "Ursprünglich stehen 3 Autos dort. 2 weitere kommen hinzu. "
                "3 + 2 = 5. Die Antwort ist 5."
            ),
        },
        {
            "question": (
                "Lea hatte 32 Pralinen und ihre Schwester hatte 42. "
                "Wenn sie 35 davon gegessen haben, wie viele Stück "
                "haben sie dann insgesamt noch übrig?"
            ),
            "target": (
                "Ursprünglich hatte Lea 32 Pralinen. Ihre Schwester "
                "hatte 42. Insgesamt hatten sie also 32 + 42 = 74. "
                "Nachdem sie 35 gegessen haben, hatten sie 74 - 35 = 39. "
                "Die Antwort ist 39."
            ),
        },
        {
            "question": (
                "Jan hatte 20 Lutscher. Er hat Daniel einige Lutscher "
                "gegeben. Jetzt hat Jan 12 Lutscher. Wie viele Lutscher "
                "hat Jan an Daniel gegeben?"
            ),
            "target": (
                "Jan hatte anfangs 20 Lutscher. Danach hatte er 12, "
                "nachdem er einige an Daniel gegeben hat. Er hat also "
                "Daniel 20 - 12 = 8 gegeben. Die Antwort ist 8."
            ),
        },
        {
            "question": (
                "Lukas hat fünf Spielzeuge. Zu Weihnachten hat er von "
                "seiner Mutter und seinem Vater jeweils zwei Spielzeuge "
                "bekommen. Wie viele Spielzeuge hat er jetzt?"
            ),
            "target": (
                "Lukas hatte anfangs 5 Spielzeuge. Wenn er von seiner "
                "Mutter und seinem Vater jeweils 2 bekommen hat, sind "
                "das 4 weitere Spielzeuge. 5 + 4 = 9. Die Antwort ist 9."
            ),
        },
        {
            "question": (
                "Im Serverraum standen neun Computer. Von Montag bis "
                "Donnerstag wurden jeden Tag fünf weitere Computer "
                "installiert. Wie viele Computer sind jetzt im Serverraum?"
            ),
            "target": (
                "Ursprünglich waren 9 Computer vorhanden. An jedem der "
                "4 Tage wurden 5 weitere Computer hinzugefügt. Also "
                "wurden 5 * 4 = 20 Computer hinzugefügt. 9 + 20 ist 29. "
                "Die Antwort ist 29."
            ),
        },
        {
            "question": (
                "Michael hatte 58 Golfbälle. Am Dienstag hat er 23 "
                "Golfbälle verloren. Am Mittwoch hat er 2 weitere "
                "verloren. Wie viele Golfbälle hatte er am Ende des "
                "Mittwochs?"
            ),
            "target": (
                "Michael hatte anfangs 58 Golfbälle. Nachdem er am "
                "Dienstag 23 verloren hat, hatte er 58 - 23 = 35. "
                "Nachdem er 2 weitere verloren hat, hatte er "
                "35 - 2 = 33 Golfbälle. Die Antwort ist 33."
            ),
        },
        {
            "question": (
                "Olivia hat 23 €. Sie hat fünf Brötchen für je 3 € "
                "gekauft. Wie viel Geld hat sie noch übrig?"
            ),
            "target": (
                "Olivia hatte 23 Euro. 5 Brötchen zu je 3 Euro kosten "
                "5 x 3 = 15 Euro. Sie hat also 23 - 15 Euro übrig. "
                "23 - 15 ist 8. Die Antwort ist 8."
            ),
        },
    ]
