from collections import Counter
import json
import os
import re
import string
import time
import evaluate
import openai
from openai import (
    RateLimitError,
    APIConnectionError,
    APIError,
)

from tqdm import tqdm
from instruct_qa.evaluation import Metric
from instruct_qa.evaluation.metrics import BERTScore, F1
import torch
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModelWithLMHead,
    AutoModelForQuestionAnswering,
)
import spacy
import pandas as pd

from allennlp.predictors.predictor import Predictor
import allennlp_models.pair_classification
from instruct_qa.prompt.templates import HistoryTemplate, PromptTemplate

INVALID_QUESTION = -1
NO_ANS = "[CLS]"
NO_VALID_QUESTIONS = "NO_Q"
NO_Q = -1
ENTAILMENT_SCORE = 1
CONTRADICTION_SCORE = 0
NEUTRAL_SCORE = 0.5


class FaithDialCritic(Metric):
    """
    FaithDialCritic is a metric that measures the faithfulness of a response to a given evidence.
    0 - faithfull
    1 - unfaithfull
    lower score is better
    """

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "McGill-NLP/roberta-large-faithcritic", return_tensors="pt"
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "McGill-NLP/roberta-large-faithcritic",
        ).cuda()

    def __call__(self, history_list, response_list, evidence_list, ids=None):
        """
        history_list: list of list of strings (won't be used)
        response_list: list of strings
        evidence_list: list of list passages from collection - text, title, sub_title
        """

        scores = []
        for i in tqdm(range(len(evidence_list))):
            evidence = evidence_list[i]
            response = response_list[i]
            evidence_string = " ".join([e for e in evidence])
            input = self.tokenizer(
                evidence_string, response, return_tensors="pt", truncation=True
            )
            input = {key: val.cuda() for key, val in input.items()}
            output_logits = self.model(**input).logits
            score = torch.softmax(output_logits, dim=1)[:, 1].item()

            scores.append(score)

        if self.store_individual_scores:
            self.save_individual_scores(
                ids, [{"faithcritic": score} for score in scores]
            )

        return {"faithcritic": np.mean(scores)}


class FaithDialCriticV2(Metric):
    """
    FaithDialCritic is a metric that measures the faithfulness of a response to a given evidence.
    0 - faithfull
    1 - unfaithfull
    lower score is better
    """

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "/home/toolkit/FaithDial/trained_models/best_model", return_tensors="pt"
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "/home/toolkit/FaithDial/trained_models/best_model",
        ).cuda()

    def __call__(self, history_list, response_list, evidence_list, ids=None):
        """
        history_list: list of list of strings (won't be used)
        response_list: list of strings
        evidence_list: list of list passages from collection - text, title, sub_title
        """

        scores = []
        for i in tqdm(range(len(evidence_list))):
            evidence = evidence_list[i]
            response = response_list[i]
            evidence_string = " ".join([e for e in evidence])
            input = self.tokenizer(
                evidence_string, response, return_tensors="pt", truncation=True
            )
            input = {key: val.cuda() for key, val in input.items()}
            output_logits = self.model(**input).logits
            score = torch.softmax(output_logits, dim=1)[:, 1].item()

            scores.append(score)

        if self.store_individual_scores:
            self.save_individual_scores(
                ids, [{"faithcritic_v2": score} for score in scores]
            )

        return {"faithcritic_v2": np.mean(scores)}


class QSquared(Metric):
    # Code taken from https://github.com/orhonovich/q-squared
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.qg_tokenizer = AutoTokenizer.from_pretrained(
            "mrm8488/t5-base-finetuned-question-generation-ap"
        )
        self.qg_model = AutoModelWithLMHead.from_pretrained(
            "mrm8488/t5-base-finetuned-question-generation-ap"
        ).cuda()
        self.qa_tokenizer = AutoTokenizer.from_pretrained(
            "ktrapeznikov/albert-xlarge-v2-squad-v2"
        )
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(
            "ktrapeznikov/albert-xlarge-v2-squad-v2",
        ).cuda()
        self.predictor = Predictor.from_path(
            "https://storage.googleapis.com/allennlp-public-models/snli_roberta-2020.06.09.tar.gz",
            predictor_name="textual_entailment",
        )

        self.nlp = spacy.load("en_core_web_sm")

    def __call__(self, history_list, response_list, evidence_list, ids=None):
        f1_scores = []
        nli_scores = []

        for idx in tqdm(range(len(evidence_list))):
            evidence = evidence_list[idx]
            evidence_string = " ".join([e for e in evidence])
            response = response_list[idx]
            (
                f1_score,
                res_questions,
                res_cands,
                res_answers,
                res_scores,
            ) = self.get_response_score(
                response,
                evidence_string,
                gen_method="beam",
                single=True,
                remove_personal=True,
            )

            if f1_score == INVALID_QUESTION:
                res_questions = [NO_VALID_QUESTIONS]
                res_cands = [NO_VALID_QUESTIONS]
                res_answers = [NO_VALID_QUESTIONS]
                res_scores = [INVALID_QUESTION]

            f1_scores_instance = []
            nli_scores_instance = []
            for i in range(len(res_questions)):
                f1_score_q2 = res_scores[i]
                evidence_answer = str(res_answers[i])

                nli_score_q2 = f1_score_q2

                if (
                    0 <= f1_score_q2 < 1
                    and NO_ANS not in evidence_answer
                    and evidence_answer != ""
                    and evidence_answer != "nan"
                ):
                    f1_scores_instance.append(f1_score_q2)

                    nli_label = self.get_nli_label(
                        str(res_questions[i]), str(res_cands[i]), evidence_answer
                    )

                    if nli_label == "entailment":
                        nli_score_q2 = ENTAILMENT_SCORE
                    elif nli_label == "contradiction":
                        nli_score_q2 = CONTRADICTION_SCORE

                elif f1_score_q2 == NO_Q:
                    nli_fallback = self.get_e2e_nli_score(
                        str(response), str(evidence_string).lower()
                    )
                    nli_score_q2 = nli_fallback
                    f1_scores_instance.append(nli_fallback)
                else:
                    f1_scores_instance.append(f1_score_q2)

                nli_scores_instance.append(nli_score_q2)

            f1_scores.append(np.mean(f1_scores_instance))
            nli_scores.append(np.mean(nli_scores_instance))

        if self.store_individual_scores:
            self.save_individual_scores(
                ids,
                [
                    {"f1": f1_score, "nli": nli_score}
                    for f1_score, nli_score in zip(f1_scores, nli_scores)
                ],
            )
        return {"f1": np.mean(f1_scores), "nli": np.mean(nli_scores)}

    def get_answer(
        self, question, text
    ):  # Code taken from https://huggingface.co/transformers/task_summary.html
        inputs = self.qa_tokenizer.encode_plus(
            question,
            text,
            add_special_tokens=True,
            return_tensors="pt",
            truncation=True,
        )
        inputs = {key: val.cuda() for key, val in inputs.items()}
        input_ids = inputs["input_ids"].tolist()[0]

        text_tokens = self.qa_tokenizer.convert_ids_to_tokens(input_ids)
        answer_start_scores, answer_end_scores = self.qa_model(
            **inputs, return_dict=False
        )

        answer_start = torch.argmax(
            answer_start_scores
        )  # Get the most likely beginning of answer with the argmax of the score
        answer_end = (
            torch.argmax(answer_end_scores) + 1
        )  # Get the most likely end of answer with the argmax of the score

        ans = self.qa_tokenizer.convert_tokens_to_string(
            self.qa_tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
        )
        return ans

    def get_answer_candidates(self, text):
        doc = self.nlp(text)
        candidates = [ent.text for ent in list(doc.ents)]
        noun_chunks = list(doc.noun_chunks)
        for chunk in noun_chunks:
            found = False
            for cand in candidates:
                if chunk.text.lower() == cand.lower():
                    found = True
            if not found:
                candidates.append(chunk.text)
        # candidates += [chunk.text for chunk in list(doc.noun_chunks) if chunk.text not in candidates]
        candidates = [cand for cand in candidates if cand.lower() != "i"]
        return candidates

    def get_question_greedy(self, answer, context, max_length=128):
        input_text = "answer: %s  context: %s </s>" % (answer, context)
        features = self.qg_tokenizer([input_text], return_tensors="pt", truncation=True)

        output = self.qg_model.generate(
            input_ids=features["input_ids"].cuda(),
            attention_mask=features["attention_mask"].cuda(),
            max_length=max_length,
        )

        question = self.qg_tokenizer.decode(output[0]).replace("question: ", "", 1)
        return question

    def get_questions_beam(
        self, answer, context, max_length=128, beam_size=5, num_return=5
    ):
        all_questions = []
        input_text = "answer: %s  context: %s </s>" % (answer, context)
        features = self.qg_tokenizer([input_text], return_tensors="pt", truncation=True)

        beam_outputs = self.qg_model.generate(
            input_ids=features["input_ids"].cuda(),
            attention_mask=features["attention_mask"].cuda(),
            max_length=max_length,
            num_beams=beam_size,
            no_repeat_ngram_size=3,
            num_return_sequences=num_return,
            early_stopping=True,
        )

        for beam_output in beam_outputs:
            all_questions.append(
                self.qg_tokenizer.decode(beam_output, skip_special_tokens=True).replace(
                    "question: ", "", 1
                )
            )

        return all_questions

    def get_questions_sample(
        self, answer, context, max_length=128, top_k=50, top_p=0.95, num_return=5
    ):
        all_questions = []
        input_text = "answer: %s  context: %s </s>" % (answer, context)
        features = self.qg_tokenizer([input_text], return_tensors="pt", truncation=True)

        sampled_outputs = self.qg_model.generate(
            input_ids=features["input_ids"],
            attention_mask=features["attention_mask"],
            max_length=max_length,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=num_return,
        )

        for sampled in sampled_outputs:
            all_questions.append(
                self.qg_tokenizer.decode(sampled, skip_special_tokens=True).replace(
                    "question: ", "", 1
                )
            )

        return all_questions

    def non_personal(self, question):
        question_tok = self.nlp(question)
        for tok in question_tok:
            if tok.dep_ == "nsubj":
                if tok.text.lower() == "i" or tok.text.lower() == "you":
                    return False
            elif tok.dep_ == "poss":
                if tok.text.lower() == "my" or tok.text.lower() == "your":
                    return False
        return True

    def clean_text(self, text):
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\b(a|an|the|in|our)\b", " ", text)
        return re.sub(" +", " ", text).strip()

    def filter_questions(self, exp_ans, pred_ans):
        if pred_ans == NO_ANS:
            return "NO MATCH"
        if self.clean_text(exp_ans) != self.clean_text(pred_ans):
            return "NO MATCH"
        return "VALID"

    def f1_score(self, a_gold, a_pred):
        if a_pred == "":
            return 0
        gold_toks = self.clean_text(a_gold).split()
        pred_toks = self.clean_text(a_pred).split()
        common = Counter(gold_toks) & Counter(pred_toks)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def single_question_score(self, question, cand, response, knowledge):
        pred_ans = self.get_answer(question, response)

        if self.filter_questions(cand, pred_ans) == "VALID":
            knowledge_ans = self.get_answer(question, knowledge)
            if knowledge_ans != NO_ANS:
                return self.f1_score(cand, knowledge_ans), knowledge_ans
            else:
                return 0, NO_ANS
        else:
            return INVALID_QUESTION, INVALID_QUESTION

    def get_response_score(
        self, response, knowledge, gen_method, single, remove_personal=True
    ):
        f1 = 0
        num_questions = 0

        valid_questions = []
        valid_cands = []
        knowledge_answers = []
        scores = []

        candidates = self.get_answer_candidates(response)
        for cand in candidates:
            if gen_method == "greedy":
                questions = [self.get_question_greedy(cand, response)]
            elif gen_method == "beam":
                questions = self.get_questions_beam(cand, response)
            else:
                questions = self.get_questions_sample(cand, response)

            for question in questions:
                if not remove_personal or self.non_personal(question):
                    question_score, knowledge_ans = self.single_question_score(
                        question, cand, response, knowledge
                    )
                    if question_score != INVALID_QUESTION:
                        num_questions += 1
                        f1 += question_score

                        valid_questions.append(question)
                        valid_cands.append(cand)
                        knowledge_answers.append(knowledge_ans)
                        scores.append(question_score)

                        if single:
                            break
        if num_questions:
            avg_f1 = f1 / num_questions
        else:
            avg_f1 = INVALID_QUESTION
        return avg_f1, valid_questions, valid_cands, knowledge_answers, scores

    def get_e2e_nli_score(self, response, knowledge):
        res = self.predictor.predict(premise=knowledge, hypothesis=response)

        nli_label = res["label"]

        if nli_label == "entailment":  # If entails, the score is 1
            return ENTAILMENT_SCORE
        elif nli_label == "contradiction":  # If contradicts, the score is 0
            return CONTRADICTION_SCORE
        else:
            return NEUTRAL_SCORE

    def get_nli_label(self, question, cand, evidence_ans):
        premise = question + " " + evidence_ans + "."
        hypothesis = question + " " + cand + "."

        res = self.predictor.predict(premise=premise, hypothesis=hypothesis)

        return res["label"]


class KBERTScore(Metric):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self._metric = evaluate.load("bertscore")

    def __call__(self, history_list, response_list, evidence_list, ids=None):
        evidence_strings = [
            " ".join([e for e in evidence]) for evidence in evidence_list
        ]
        scores = self._metric.compute(
            predictions=response_list, references=evidence_strings, lang="en"
        )
        if self.store_individual_scores:
            individual_scores = []
            for i in range(len(response_list)):
                individual_scores.append(
                    {
                        "precision": scores["precision"][i],
                        "recall": scores["recall"][i],
                        "f1": scores["f1"][i],
                    }
                )
            self.save_individual_scores(ids, individual_scores)
        return {
            "precision": np.mean(scores["precision"]),
            "recall": np.mean(scores["recall"]),
            "f1": np.mean(scores["f1"]),
        }


class KPrecision(Metric):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    def __call__(self, history_list, response_list, evidence_list, ids=None):
        evidence_strings = [
            [" ".join([e for e in evidence])] for evidence in evidence_list
        ]
        scores = [
            self._precision(prediction, reference)
            for prediction, reference in zip(response_list, evidence_strings)
        ]

        if self.store_individual_scores:
            self.save_individual_scores(
                ids, [{"kprecision": score} for score in scores]
            )
        return {"kprecision": np.mean(scores)}

    def _precision(self, prediction, references):
        precision_scores = [
            self._precision_score(prediction, reference) for reference in references
        ]
        return max(precision_scores)

    def _precision_score(self, prediction, reference):
        reference_tokens = self._get_tokens(reference)
        prediction_tokens = self._get_tokens(prediction)

        common_tokens = Counter(reference_tokens) & Counter(prediction_tokens)

        num_common = sum(common_tokens.values())

        if len(prediction_tokens) == 0:
            # if prediction is empty, precision is 0
            return 0

        if num_common == 0:
            return 0

        precision = 1.0 * num_common / len(prediction_tokens)

        return precision


class KPrecisionPlusPlus(Metric):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    def __call__(self, history_list, response_list, evidence_list, ids=None):
        evidence_strings = [
            [" ".join([e for e in evidence])] for evidence in evidence_list
        ]
        history_strings = [" ".join([e for e in history]) for history in history_list]
        scores = [
            self._precision_plusplus(prediction, reference, query)
            for prediction, reference, query in zip(
                response_list, evidence_strings, history_strings
            )
        ]

        if self.store_individual_scores:
            self.save_individual_scores(
                ids, [{"kprecisionplusplus": score} for score in scores]
            )
        return {"kprecisionplusplus": np.mean(scores)}

    def _precision_plusplus(self, prediction, references, query):
        precision_scores = [
            self._precision_plusplus_score(prediction, reference, query)
            for reference in references
        ]
        return max(precision_scores)

    def _precision_plusplus_score(self, prediction, reference, query):
        reference_tokens = self._get_tokens(reference)
        prediction_tokens = self._get_tokens(prediction)
        query_tokens = self._get_tokens(query)

        prediction_tokens = [
            token for token in prediction_tokens if token not in query_tokens
        ]

        if len(prediction_tokens) == 0:
            return 1.0

        common_tokens = Counter(reference_tokens) & Counter(prediction_tokens)

        num_common = sum(common_tokens.values())

        if len(prediction_tokens) == 0:
            # if prediction is empty, precision is 0
            return 0

        if num_common == 0:
            return 0

        precision = 1.0 * num_common / len(prediction_tokens)

        return precision


class KRecall(Metric):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    def __call__(self, history_list, response_list, evidence_list, ids=None):
        evidence_strings = [
            [" ".join([e for e in evidence])] for evidence in evidence_list
        ]
        scores = [
            self._recall(prediction, reference)
            for prediction, reference in zip(response_list, evidence_strings)
        ]

        if self.store_individual_scores:
            self.save_individual_scores(ids, [{"krecall": score} for score in scores])
        return {"krecall": np.mean(scores)}

    def _recall(self, prediction, references):
        recall_scores = [
            self._recall_score(prediction, reference) for reference in references
        ]
        return max(recall_scores)

    def _recall_score(self, prediction, reference):
        reference_tokens = self._get_tokens(reference)
        prediction_tokens = self._get_tokens(prediction)

        common_tokens = Counter(reference_tokens) & Counter(prediction_tokens)

        num_common = sum(common_tokens.values())

        if len(reference_tokens) == 0:
            # if prediction is empty, recall is one
            return 1

        if num_common == 0:
            return 0

        recall = 1.0 * num_common / len(reference_tokens)

        return recall


class KRecallPlusPlus(Metric):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    def __call__(self, history_list, response_list, evidence_list, ids=None):
        evidence_strings = [
            [" ".join([e for e in evidence])] for evidence in evidence_list
        ]
        history_strings = [" ".join([e for e in history]) for history in history_list]
        scores = [
            self._recall_plusplus(prediction, reference, query)
            for prediction, reference, query in zip(
                response_list, evidence_strings, history_strings
            )
        ]

        if self.store_individual_scores:
            self.save_individual_scores(
                ids, [{"krecallplusplus": score} for score in scores]
            )
        return {"krecallplusplus": np.mean(scores)}

    def _recall_plusplus(self, prediction, references, query):
        recall_scores = [
            self._recall_plusplus_score(prediction, reference, query)
            for reference in references
        ]
        return max(recall_scores)

    def _recall_plusplus_score(self, prediction, reference, query):
        reference_tokens = self._get_tokens(reference)
        prediction_tokens = self._get_tokens(prediction)
        query_tokens = self._get_tokens(query)

        prediction_tokens = [
            token for token in prediction_tokens if token not in query_tokens
        ]

        if len(prediction_tokens) == 0:
            return 1.0

        common_tokens = Counter(reference_tokens) & Counter(prediction_tokens)

        num_common = sum(common_tokens.values())

        if len(reference_tokens) == 0:
            # if prediction is empty, recall is one
            return 1

        if num_common == 0:
            return 0

        recall = 1.0 * num_common / len(reference_tokens)

        return recall


class KF1(Metric):
    """Computes average F1 score between a list of predictions and a list of
    list of references.

    Code taken from: https://github.com/McGill-NLP/topiocqa
    """

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    def __call__(self, history_list, response_list, evidence_list, ids=None):
        evidence_strings = [
            [" ".join([e for e in evidence])] for evidence in evidence_list
        ]
        scores = [
            self._f1(prediction, reference)
            for prediction, reference in zip(response_list, evidence_strings)
        ]

        if self.store_individual_scores:
            self.save_individual_scores(ids, [{"kf1": score} for score in scores])
        return {"kf1": np.mean(scores)}

    def _f1(self, prediction, references):
        """Computes F1 score between a prediction and a list of references.
        Take the max F1 score if there are multiple references.
        """

        f1_scores = [self._f1_score(prediction, reference) for reference in references]
        return max(f1_scores)

    def _f1_score(self, prediction, reference):
        reference_tokens = self._get_tokens(reference)
        prediction_tokens = self._get_tokens(prediction)

        common_tokens = Counter(reference_tokens) & Counter(prediction_tokens)

        num_common = sum(common_tokens.values())

        if len(reference_tokens) == 0 or len(prediction_tokens) == 0:
            # If either is empty, then F1 is 1 if they agree, 0 otherwise.
            return int(reference_tokens == prediction_tokens)

        if num_common == 0:
            return 0

        precision = 1.0 * num_common / len(prediction_tokens)
        recall = 1.0 * num_common / len(reference_tokens)
        f1 = (2 * precision * recall) / (precision + recall)

        return f1


class KF1PlusPlus(Metric):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    def __call__(self, history_list, response_list, evidence_list, ids=None):
        evidence_strings = [
            [" ".join([e for e in evidence])] for evidence in evidence_list
        ]
        history_strings = [" ".join([e for e in history]) for history in history_list]
        scores = [
            self._f1_plusplus(prediction, reference, query)
            for prediction, reference, query in zip(
                response_list, evidence_strings, history_strings
            )
        ]

        if self.store_individual_scores:
            self.save_individual_scores(
                ids, [{"kf1plusplus": score} for score in scores]
            )
        return {"kf1plusplus": np.mean(scores)}

    def _f1_plusplus(self, prediction, references, query):
        """Computes F1 score between a prediction and a list of references.
        Take the max F1 score if there are multiple references.
        """

        f1_scores = [
            self._f1_plusplus_score(prediction, reference, query)
            for reference in references
        ]
        return max(f1_scores)

    def _f1_plusplus_score(self, prediction, reference, query):
        reference_tokens = self._get_tokens(reference)
        prediction_tokens = self._get_tokens(prediction)
        query_tokens = self._get_tokens(query)

        prediction_tokens = [
            token for token in prediction_tokens if token not in query_tokens
        ]

        if len(prediction_tokens) == 0:
            return 1.0

        common_tokens = Counter(reference_tokens) & Counter(prediction_tokens)

        num_common = sum(common_tokens.values())

        if len(reference_tokens) == 0 or len(prediction_tokens) == 0:
            # If either is empty, then F1 is 1 if they agree, 0 otherwise.
            return int(reference_tokens == prediction_tokens)

        if num_common == 0:
            return 0

        precision = 1.0 * num_common / len(prediction_tokens)
        recall = 1.0 * num_common / len(reference_tokens)
        f1 = (2 * precision * recall) / (precision + recall)

        return f1


class KLLMEval(Metric):
    """
    Computes score using LLMs.
    """

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        args = kwargs.get("args", None)
        self.api_key = args.api_key
        self.model = args.model_name
        self.max_tokens = args.max_tokens
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.n = args.n
        self.stop = args.stop_seq
        self.presence_penalty = args.presence_penalty
        self.frequency_penalty = args.frequency_penalty
        self.wait = 10
        instruction = 'You are given a question, the corresponding evidence and a prediction from a model. Compare the "Prediction" and the "Evidence" to determine whether all the information of the prediction in present in the evidence or can be inferred from the evidence. You must answer "no" if there are any specific details in the prediction that are not mentioned in the evidence or cannot be inferred from the evidence.'
        prompt = (
            instruction
            + "\n\nQuestion: {question}\n\nPrediction: {prediction}\n\nEvidence: {evidence}\n\nCompareGPT response:"
        )
        self.prompt_template = PromptTemplate(
            variables=["question", "prediction", "evidence"],
            template=prompt,
        )
        self.system_prompt = "You are CompareGPT, a machine to verify the groudedness of predictions. Answer with only yes/no."
        openai.api_key = self.api_key
        self.individual_out_dir = self.individual_out_dir + f"/{self.model}"

    def __call__(self, history_list, response_list, evidence_list, ids=None):
        assert (
            self.store_individual_scores
        ), "LLM requires individual scores to be stored, to avoid unnecessary API calls"
        individual_scores = []
        if os.path.exists(os.path.join(self.individual_out_dir, self.file_name)):
            with open(os.path.join(self.individual_out_dir, self.file_name)) as f:
                individual_scores = f.readlines()
        num_already_calculated = len(individual_scores)

        history_list = history_list[num_already_calculated:]
        response_list = response_list[num_already_calculated:]
        evidence_list = evidence_list[num_already_calculated:]
        ids = ids[num_already_calculated:]

        if len(history_list) == 0:
            print("All scores already calculated")

        for i in tqdm(range(len(evidence_list))):
            # individual score handles differently to avoid repeated calls to the API
            self._llm_score(history_list[i], response_list[i], evidence_list[i], ids[i])

        with open(os.path.join(self.individual_out_dir, self.file_name)) as f:
            individual_scores = f.readlines()
        individual_scores = [json.loads(score) for score in individual_scores]

        scores = [score[self.name][self.name] for score in individual_scores]
        result = {"llm_eval": np.mean(scores)}
        return result

    def _llm_score(self, history, response, evidence, id_):
        assert len(history) == 1
        question = history[0]
        evidence_string = " ".join([e for e in evidence])
        prompt = self.prompt_template.format(
            {"question": question, "prediction": response, "evidence": evidence_string}
        )
        response = None
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                n=self.n,
                stop=self.stop,
                presence_penalty=self.presence_penalty,
                frequency_penalty=self.frequency_penalty,
            )
        # Except multiple errors in one except block
        except (
            RateLimitError,
            APIConnectionError,
            APIError,
        ) as e:
            print(f"Error: {e}. Waiting {self.wait} seconds before retrying.")
            time.sleep(self.wait)
            return self._llm_score(history, response, evidence, id_)

        response = (
            response["choices"][0]["message"]["content"].strip().strip(".").strip(",")
        )

        if response.lower() not in [
            "yes",
            "no",
        ]:
            print(
                f"Response {response} not in ['yes', 'no']\nSystem prompt: {self.system_prompt}\nPrompt: {prompt}"
            )
        if "yes" in response.lower():
            score = 1.0
        else:
            score = 0.0

        os.makedirs(self.individual_out_dir, exist_ok=True)
        with open(os.path.join(self.individual_out_dir, self.file_name), "a") as f:
            f.write(
                json.dumps(
                    {
                        "id_": id_,
                        "question": question,
                        "evidence": evidence_string,
                        "prediction": response,
                        self.name: {"kllm_eval": score},
                    }
                )
                + "\n"
            )


class KLLMEvalConv(KLLMEval):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        instruction = 'You are given a conversation, the corresponding evidence and a prediction from a model. Compare the "Prediction" and the "Evidence" to determine whether all the information of the prediction in present in the evidence or can be inferred from the evidence. You must answer "no" if there are any specific details in the prediction that are not mentioned in the evidence or cannot be inferred from the evidence.'
        prompt = (
            instruction
            + "\n\n{conversation_history}\n\nPrediction: {prediction}\n\nEvidence: {evidence}\n\nCompareGPT response:"
        )
        self.prompt_template = PromptTemplate(
            variables=["conversation_history", "prediction", "evidence"],
            template=prompt,
        )
        self.history_template = HistoryTemplate()

    def _llm_score(self, history, response, evidence, id_):
        utterances = []
        for i, utterance in enumerate(history):
            if i % 2 == 0:
                utterances.append({"speaker": "Human", "utterance": utterance})
            else:
                utterances.append({"speaker": "Assistant", "utterance": utterance})
        serialized_conv_history = self.history_template.serialize_history(utterances)

        evidence_string = " ".join([e for e in evidence])
        prompt = self.prompt_template.format(
            {
                "conversation_history": serialized_conv_history,
                "prediction": response,
                "evidence": evidence_string,
            }
        )
        response = None
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                n=self.n,
                stop=self.stop,
                presence_penalty=self.presence_penalty,
                frequency_penalty=self.frequency_penalty,
            )
        # Except multiple errors in one except block
        except (
            RateLimitError,
            APIConnectionError,
            ServiceUnavailableError,
            APIError,
        ) as e:
            print(f"Error: {e}. Waiting {self.wait} seconds before retrying.")
            time.sleep(self.wait)
            return self._llm_score(history, response, evidence, id_)

        response = (
            response["choices"][0]["message"]["content"].strip().strip(".").strip(",")
        )

        if response.lower() not in [
            "yes",
            "no",
        ]:
            print(
                f"Response {response} not in ['yes', 'no']\nSystem prompt: {self.system_prompt}\nPrompt: {prompt}"
            )
        if "yes" in response.lower():
            score = 1.0
        else:
            score = 0.0

        os.makedirs(self.individual_out_dir, exist_ok=True)
        with open(os.path.join(self.individual_out_dir, self.file_name), "a") as f:
            f.write(
                json.dumps(
                    {
                        "id_": id_,
                        "question": serialized_conv_history,
                        "evidence": evidence_string,
                        "prediction": response,
                        self.name: {"kllm_eval": score},
                    }
                )
                + "\n"
            )
