"""CoQA utilities for English evaluation."""

import math
import re
import string
from typing import List, Dict
from datasets import Dataset


def _process_doc_to_multi(doc):
    """Explode each CoQA document into per-turn instances.
 
    EleutherAI/coqa schema:
      - questions.input_text: list[str]
      - answers.input_text: list[str]
      - additional_answers: dict of annotator sets, each with input_text list
 
    Each turn becomes a separate document with:
      - query: the full prompt (Passage + Preceding questions + Final question)
      - answers: list of reference answers (primary + additional annotators)
    """
    new_docs=[]
    core_id = doc["id"]
    source = doc["source"]
    story = doc["story"]
    questions = doc["questions"]["input_text"]
    all_answers = doc["answers"]["input_text"]
    additional_answers = [x["input_text"] for x in doc["additional_answers"].values()]
    previous_qa: list = []
    for turn_idx, question in enumerate(questions):
        question = questions[turn_idx]
        answers = [all_answers[turn_idx]]
        for answer_list in additional_answers:
            if len(answer_list) > turn_idx and answer_list[turn_idx]:
                answers.append(answer_list[turn_idx])
        query = f"Passage: {story}"
        if previous_qa:
            query += "\n\nPreceding questions:"
            for prev in previous_qa:
                query += f"\n\nQuestion: {prev['question']}\nAnswer: {prev['answer']}"
        query += "\n\nFinal question:"
        query += f"\n\nQuestion: {question}\nAnswer:"
        new_doc = {
            "id": f"{core_id}_turn{turn_idx}",
            "source": source,
            "story": story,
            "query": query,
            "question": question,
            "answers": answers,
            "choices": [answers[0]],
            "previous_qa": previous_qa,
        }
        previous_qa.append({"question": question, "answer": answers[0]})
        new_docs.append(new_doc)
    return new_docs


def process_docs_coqa_gen(dataset):
    new_docs = []
    for doc in dataset:
        new_docs.extend(_process_doc_to_multi(doc))
    return Dataset.from_list(new_docs)

def make_mcq_prompt(
    question,
    choices,
    question_prefix="Question: ",
    choices_prefix="",
    answer_prefix="Answer:",
    label_prefix=" ",
    label_format=None,
):
    choice_labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    label_format = label_format or label_prefix + "A."
    if "A" not in label_format:
        raise ValueError(
            "Label format must contain 'A' to indicate where to place the answer label"
        )
    # Use prefix space before each answer label to avoid ambiguity in tokenization
    choices_text = "\n".join(
        [
            f"{label_format.replace('A', label)} {text}"
            for label, text in zip(choice_labels, choices)
        ]
    )
    prompt = f"{question_prefix}{question}\n{choices_prefix}{choices_text}\n{answer_prefix}"
    return prompt


def process_docs_coqa_mc(dataset):
    new_docs=[]
    for doc in dataset:
        query = doc["query_original"].rsplit("Question:", 1)[0] + make_mcq_prompt(
            doc["question_original"], doc["choices"]["text"]
        )
        out_doc = {
            **doc,
            "query_processed": query,
        }
        new_docs.append(out_doc)
    return Dataset.from_list(new_docs)

# =============================================================================
# Token-level F1 metrics (SQuAD-style)
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


def process_results_coqa(doc: dict, results: List[str]) -> dict:
    """Process results for CoQA generative task.
 
    Computes max F1/EM across all reference answers (primary + additional annotators),
    matching OLMES SQuADF1EMRecallMetric behavior.
    """
    prediction = results[0] if results else ""
    references = doc.get("answers", [])
    if not references:
        return {"em": 0.0, "f1": 0.0}
 
    #em = _compute_exact_match(prediction, reference)
    #f1 = _compute_f1(prediction, reference)
    best_f1 = max(_compute_f1(prediction, ref) for ref in references)
    best_em = max(_compute_exact_match(prediction, ref) for ref in references)
 
    #return {"em": em, "f1": f1}
    return {"em": best_em, "f1": best_f1}


def process_results_bpb(doc, results):
    """Compute answer-only BPB (OLMES-style) for CoQA."""
    ll, _ = results[0]
    gold_text = doc["answers"]["input_text"][0]
    gold_bytes = len((" " + gold_text).encode("utf-8"))
    return {"bits_per_byte": -ll / (math.log(2) * max(gold_bytes, 1))}
