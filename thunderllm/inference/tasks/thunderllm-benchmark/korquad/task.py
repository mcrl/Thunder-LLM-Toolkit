from functools import partial
from math import exp

import datasets
from packaging import version

from thunderllm.inference.api.instance import Instance
from thunderllm.inference.api.task import ConfigurableTask
from thunderllm.inference.api.metrics import mean

from collections import Counter
import string
import re
import argparse
import json
import sys
import os
from bs4 import BeautifulSoup

_CITATION = """
@inproceedings{kim2019korquad,
  title={KorQuAD 2.0: Korean QA dataset for web document machine comprehension},
  author={Kim, Youngmin and Lim, Seungyoung and Lee, Hyunjeong and Park, Soyoon and Kim, Myungji},
  booktitle={Annual Conference on Human and Language Technology},
  pages={97--102},
  year={2019},
  organization={Human and Language Technology}
}
"""


def normalize_answer(s):
    def tag_clean(t):
        return BeautifulSoup(t).get_text()

    def remove_(text):
        """불필요한 기호 제거"""
        text = re.sub("'", " ", text)
        text = re.sub('"', " ", text)
        text = re.sub("《", " ", text)
        text = re.sub("》", " ", text)
        text = re.sub("<", " ", text)
        text = re.sub(">", " ", text)
        text = re.sub("〈", " ", text)
        text = re.sub("〉", " ", text)
        text = re.sub("\(", " ", text)
        text = re.sub("\)", " ", text)
        text = re.sub("‘", " ", text)
        text = re.sub("’", " ", text)
        return text

    def white_space_fix(text):
        return (
            " ".join(text.split()).replace(
                "\n", "").replace("\t", "").replace(" ", "")
        )

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(remove_(tag_clean(s)))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    # F1 by character
    prediction_Char = []
    for tok in prediction_tokens:
        now = [a for a in tok]
        prediction_Char.extend(now)

    ground_truth_Char = []
    for tok in ground_truth_tokens:
        now = [a for a in tok]
        ground_truth_Char.extend(now)

    common = Counter(prediction_Char) & Counter(ground_truth_Char)
    num_same = sum(common.values())
    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(prediction_Char)
    recall = 1.0 * num_same / len(ground_truth_Char)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def _squad_agg(key, items):
    metric = exact_match_score if key == "exact" else f1_score

    score = total = 0
    for pred, ans in items:
        score += metric(pred["prediction_text"], ans["answer"]["text"])
        total += 1
    return score / total


def compute_score(prediction_text, answer):
    max_exact_score = exact_match_score(prediction_text, answer)
    max_f1_score = f1_score(prediction_text, answer)
    return max_exact_score, max_f1_score


class Korquad(ConfigurableTask):
    VERSION = 2
    # Reference: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/squad_completion/task.py#L10

    def __init__(self, config=None):
        self.DATASET_PATH = config["dataset_path"]
        super().__init__(
            config={
                "metadata": {"version": self.VERSION},
                "dataset_kwargs": {"trust_remote_code": True},
                "training_split": None,
                "validation_split": "validation",
            }
        )
    #preprocess input
    def clean_input(self,t):
        def tag_clean(t):
                return BeautifulSoup(t).get_text()

        def clean_wiki_text(text):

            # Remove excessive newlines (3+ -> 2)
            text = re.sub(r'\n{3,}', '\n\n', text)

            # Remove BOM characters and other control chars
            text = text.replace('\ufeff', '')

            # Remove bracketed edit tags like [편집]
            text = re.sub(r'\[편집\]', '', text)

            # Remove footnote markers like [1], [2]
            text = re.sub(r'\[\d+\]', '', text)

            # Normalize non-breaking space
            text = text.replace('\xa0', ' ')

            # Strip leading/trailing whitespace
            text = text.strip()

            return text
        def remove_footer_ui(text):
            # Heuristically chop off after "원본 주소" or "분류:"
            for marker in ['원본 주소', '분류:', '이 문서는']:
                if marker in text:
                    text = text.split(marker)[0]
            return text

        def clean_full(text):
            text= tag_clean(text)
            text = remove_footer_ui(text)
            return clean_wiki_text(text)
        
        return clean_full(t)



    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]

    def doc_to_text(self, doc):
        return (
            "주제: "
            + doc["title"]
            + "\n\n"
            + "배경지식: "
            + self.clean_input(doc["context"])
            + "\n\n"
            + "질문: "
            + doc["question"]
            + "\n\n"
            + "답: "
        )

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["context"]

    def doc_to_target(self, doc):
        answer_text = doc["answer"]["text"]
        return " " + answer_text

    def construct_requests(self, doc, ctx, **kwargs):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """

        return [
            Instance(
                request_type="generate_until",
                doc=doc,
                arguments=(ctx, {"until": ["\n"], "do_sample": False}),
                idx=0,
                **kwargs,
            )
        ]

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """

        continuation = results[0]

        answer = doc["answer"]["text"]
        exact_match, f1_score = compute_score(continuation, answer)
        return {"exact": exact_match, "f1": f1_score}

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metrics
        """
        return {"exact": mean, "f1": mean}

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        return {
            # Exact match (the normalized answer exactly match the gold answer)
            "exact": True,
            "f1": True,  # The F-score of predicted tokens versus the gold answer
        }
