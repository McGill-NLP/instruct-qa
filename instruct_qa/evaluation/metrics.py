import json
import os
import re
import string
from collections import Counter
import time

import evaluate
import numpy as np
import openai
from openai.error import (
    RateLimitError,
    APIConnectionError,
    ServiceUnavailableError,
    APIError,
)
import torch
from tqdm import tqdm

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModelWithLMHead,
    AutoModelForQuestionAnswering,
)
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

import numpy as np
from scipy.special import softmax

from instruct_qa.prompt.templates import HistoryTemplate, PromptTemplate
from instruct_qa.evaluation import Metric

class Meteor(Metric):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self._metric = evaluate.load("meteor")

    def __call__(self, predictions, references, questions=None, ids=None):
        scores = []
        for i in tqdm(range(len(predictions))):
            scores.append(
                self._metric.compute(
                    predictions=[predictions[i]], references=[references[i]]
                )["meteor"]
            )

        if self.store_individual_scores:
            self.save_individual_scores(ids, [{"meteor": score} for score in scores])

        return {"meteor": np.mean(scores)}


class BERTScore(Metric):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self._metric = evaluate.load("bertscore")

    def __call__(self, predictions, references, questions=None, ids=None):
        scores = self._metric.compute(
            predictions=predictions, references=references, lang="en"
        )
        if self.store_individual_scores:
            individual_scores = []
            for i in range(len(predictions)):
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


class BEMScore(Metric):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        vocab_path = "gs://cloud-tpu-checkpoints/bert/keras_bert/uncased_L-12_H-768_A-12/vocab.txt"
        vocab_table = tf.lookup.StaticVocabularyTable(
            tf.lookup.TextFileInitializer(
                filename=vocab_path,
                key_dtype=tf.string,
                key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
                value_dtype=tf.int64,
                value_index=tf.lookup.TextFileIndex.LINE_NUMBER,
            ),
            num_oov_buckets=1,
        )
        self.cls_id, self.sep_id = vocab_table.lookup(
            tf.convert_to_tensor(["[CLS]", "[SEP]"])
        )
        self.tokenizer = text.BertTokenizer(
            vocab_lookup_table=vocab_table,
            token_out_type=tf.int64,
            preserve_unused_token=True,
            lower_case=True,
        )
        self.bem = hub.load("https://tfhub.dev/google/answer_equivalence/bem/1")

    def bertify_example(self, example):
        question = self.tokenizer.tokenize(example["question"]).merge_dims(1, 2)
        reference = self.tokenizer.tokenize(example["reference"]).merge_dims(1, 2)
        candidate = self.tokenizer.tokenize(example["candidate"]).merge_dims(1, 2)
        input_ids, segment_ids = text.combine_segments(
            (candidate, reference, question), self.cls_id, self.sep_id
        )
        return {"input_ids": input_ids.numpy(), "segment_ids": segment_ids.numpy()}

    def bertify_examples(self, examples):
        input_ids = []
        segment_ids = []
        for example in examples:
            example_inputs = self.bertify_example(example)
            input_ids.append(self.pad(example_inputs["input_ids"]))
            segment_ids.append(self.pad(example_inputs["segment_ids"]))

        return {"input_ids": np.stack(input_ids), "segment_ids": np.stack(segment_ids)}

    def pad(self, a, length=512):
        if a.shape[-1] >= length:
            return a[0][:length]
        else:
            return np.append(a, np.zeros(length - a.shape[-1], np.int32))

    def __call__(self, predictions, references, questions, ids=None):
        assert len(predictions) == len(references)

        scores = []
        for i in tqdm(range(len(predictions))):
            examples = [
                {
                    "question": questions[i],
                    "reference": reference,
                    "candidate": predictions[i],
                }
                for reference in references[i]
            ]
            inputs = self.bertify_examples(examples)
            raw_outputs = self.bem(inputs)
            score = float(max(softmax(raw_outputs, axis=1)[:, 1]))
            scores.append(score)

        if self.store_individual_scores:
            individual_scores = [{"bem": score} for score in scores]
            self.save_individual_scores(ids, individual_scores)

        return {"bem": np.mean(scores)}


class Rouge(Metric):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self._rouge_types = ["rouge1", "rouge2", "rougeL"]
        self._metric = evaluate.load("rouge", rouge_types=self._rouge_types)

    def __call__(self, predictions, references, questions=None, ids=None):
        scores = self._metric.compute(
            predictions=predictions, references=references, use_aggregator=False
        )
        result = {name: scores[name] for name in self._rouge_types}
        if self.store_individual_scores:
            individual_scores = []
            for i in range(len(predictions)):
                individual_scores.append(
                    {name: scores[name][i] for name in self._rouge_types}
                )
            self.save_individual_scores(ids, individual_scores)
        return {name: np.mean(result[name]) for name in self._rouge_types}


class Bleu(Metric):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self._metric = evaluate.load("bleu")

    def __call__(self, predictions, references, questions=None, ids=None):
        scores = []
        for i in tqdm(range(len(predictions))):
            if predictions[i] == "":
                scores.append(0.0)
                continue
            try:
                scores.append(
                    self._metric.compute(
                        predictions=[predictions[i]], references=[references[i]]
                    )["bleu"]
                )
            except ZeroDivisionError:
                print(f"ZeroDivisionError for {predictions[i]}, reference: {references[i]}")
                scores.append(0.0)

        if self.store_individual_scores:
            self.save_individual_scores(ids, [{"bleu": score} for score in scores])
        return {"bleu": np.mean(scores)}


class F1(Metric):
    """Computes average F1 score between a list of predictions and a list of
    list of references.

    Code taken from: https://github.com/McGill-NLP/topiocqa
    """

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    def __call__(self, predictions, references, questions=None, ids=None):
        scores = [
            self._f1(prediction, reference)
            for prediction, reference in zip(predictions, references)
        ]

        if self.store_individual_scores:
            self.save_individual_scores(ids, [{"f1": score} for score in scores])
        return {"f1": np.mean(scores)}

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


class EM(Metric):
    """Computes average exact match score between a list of predictions and a
    list of list of references.
    """

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    def __call__(self, predictions, references, questions=None, ids=None):
        scores = [
            self._exact_match(prediction, reference)
            for prediction, reference in zip(predictions, references)
        ]

        if self.store_individual_scores:
            self.save_individual_scores(ids, [{"em": score} for score in scores])

        return {"em": np.mean(scores)}

    def _exact_match(self, prediction, references):
        """Computes exact match score between a prediction and a list of
        references. Take the max EM score if there are multiple references.
        """

        em_scores = [
            self._exact_match_score(prediction, reference) for reference in references
        ]
        return max(em_scores)

    def _exact_match_score(self, prediction, reference):
        reference_tokens = self._get_tokens(reference)
        prediction_tokens = self._get_tokens(prediction)

        return int(reference_tokens == prediction_tokens)


class Recall(Metric):
    """
    Computes average recall score between a list of predictions and a list of
    list of references.

    Code taken from: https://github.com/McGill-NLP/topiocqa
    """

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    def __call__(self, predictions, references, questions=None, ids=None):
        scores = [
            self._recall(prediction, reference)
            for prediction, reference in zip(predictions, references)
        ]

        if self.store_individual_scores:
            self.save_individual_scores(ids, [{"recall": score} for score in scores])
        return {"recall": np.mean(scores)}

    def _recall(self, prediction, references):
        """
        Computes recall score between a prediction and a list of references.
        Take the max recall score if there are multiple references.
        """

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
            # If reference is empty than recall is one.
            return 1

        if num_common == 0:
            return 0

        recall = 1.0 * num_common / len(reference_tokens)

        return recall

class Precision(Metric):
    """
    Computes average precision score between a list of predictions and a list of
    list of references.

    Code taken from: https://github.com/McGill-NLP/topiocqa
    """

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    def __call__(self, predictions, references, questions=None, ids=None):
        scores = [
            self._precision(prediction, reference)
            for prediction, reference in zip(predictions, references)
        ]

        if self.store_individual_scores:
            self.save_individual_scores(ids, [{"precision": score} for score in scores])
        return {"precision": np.mean(scores)}

    def _precision(self, prediction, references):
        """
        Computes precision score between a prediction and a list of references.
        Take the max precision score if there are multiple references.
        """

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
            # If prediction is empty than precision is 0.
            return 0

        if num_common == 0:
            return 0

        precision = 1.0 * num_common / len(prediction_tokens)

        return precision

class RecallEM(Metric):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    def __call__(self, predictions, references, questions=None, ids=None):
        scores = [
            self._recallem(prediction, reference)
            for prediction, reference in zip(predictions, references)
        ]

        if self.store_individual_scores:
            self.save_individual_scores(ids, [{"recallem": score} for score in scores])
        return {"recallem": np.mean(scores)}

    def _recallem(self, prediction, references):
        """
        Computes recall score between a prediction and a list of references.
        Take the max recall score if there are multiple references.
        """

        recallem_scores = [
            self._recallem_score(prediction, reference) for reference in references
        ]
        return max(recallem_scores)

    def _recallem_score(self, prediction, reference):
        reference = self._normalize_text(reference)
        prediction = self._normalize_text(prediction)

        if reference in prediction:
            return 1.0
        else:
            return 0.0


class LLMEval(Metric):
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
        instruction = 'You are given a question, the corresponding ground-truth answer and a prediction from a model. Compare the "Ground-truth answer" and the "Prediction" to determine whether the prediction correctly answers the question. All information in the ground-truth answer must be present in the prediction, including numbers and dates. You must answer "no" if there are any specific details in the ground-truth answer that are not mentioned in the prediction. There should be no contradicting statements in the prediction. The prediction may contain extra information. If the prediction states something as a possibility, treat it as a definitive answer.'
        prompt = (
            instruction
            + "\n\nQuestion: {question}\nGround-truth answer: {gt_answer}\nPrediction: {prediction}\n\nCompareGPT response:"
        )
        self.prompt_template = PromptTemplate(
            variables=["question", "gt_answer", "prediction"],
            template=prompt,
        )
        self.system_prompt = "You are CompareGPT, a machine to verify the correctness of predictions. Answer with only yes/no."
        openai.api_key = self.api_key
        self.individual_out_dir = self.individual_out_dir + f"/{self.model}"

    def __call__(self, predictions, references, questions, ids=None):
        assert (
            self.store_individual_scores
        ), "LLM requires individual scores to be stored, to avoid unnecessary API calls"
        individual_scores = []
        if os.path.exists(os.path.join(self.individual_out_dir, self.file_name)):
            with open(os.path.join(self.individual_out_dir, self.file_name)) as f:
                individual_scores = f.readlines()
        num_already_calculated = len(individual_scores)

        predictions = predictions[num_already_calculated:]
        references = references[num_already_calculated:]
        questions = questions[num_already_calculated:]
        ids = ids[num_already_calculated:]

        if len(predictions) == 0:
            print("All scores already calculated")

        for i in tqdm(range(len(predictions))):
            # individual score handles differently to avoid repeated calls to the API
            self._llm_score(predictions[i], references[i], questions[i], ids[i])

        with open(os.path.join(self.individual_out_dir, self.file_name)) as f:
            individual_scores = f.readlines()
        individual_scores = [json.loads(score) for score in individual_scores]

        scores = [score[self.name][self.name] for score in individual_scores]
        result = {"llm_eval": np.mean(scores)}
        return result

    def _llm_score(self, prediction, references, question, id_):
        llm_scores = [
            self._llm_score_single(prediction, reference, question)
            for reference in references
        ]
        score = max(llm_scores)
        os.makedirs(self.individual_out_dir, exist_ok=True)
        with open(os.path.join(self.individual_out_dir, self.file_name), "a") as f:
            f.write(
                json.dumps(
                    {
                        "id_": id_,
                        "question": question,
                        "references": references,
                        "prediction": prediction,
                        self.name: {"llm_eval": score},
                    }
                )
                + "\n"
            )

    def _llm_score_single(self, prediction, reference, question):
        prompt = self.prompt_template.format(
            {"question": question, "gt_answer": reference, "prediction": prediction}
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
            return self._llm_score_single(prediction, reference, question)

        response = response["choices"][0]["message"]["content"].strip().strip('.').strip(',')

        if response.lower() not in [
            "yes",
            "no",
        ]:
            print(f"Response {response} not in ['yes', 'no']\nSystem prompt: {self.system_prompt}\nPrompt: {prompt}")
        if "yes" in response.lower():
            return 1.0
        else:
            return 0.0

class LLMEvalConv(LLMEval):

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        instruction = "You are given a conversation, the corresponding ground-truth answer and a prediction from a model. Compare the \"Ground-truth answer\" and the \"Prediction\" to determine whether the prediction correctly answers the last question of the conversation. All information in the ground-truth answer must be present in the prediction, including numbers and dates. You must answer \"no\" if there are any specific details in the ground-truth answer that are not mentioned in the prediction. There should be no contradicting statements in the prediction. The prediction may contain extra information. If the prediction states something as a possibility, treat it as a definitive answer."
        prompt = (
            instruction
            + "\n\n{conversation_history}\n\nGround-truth answer: {gt_answer}\nPrediction: {prediction}\n\nCompareGPT response:"
        )
        self.prompt_template = PromptTemplate(
            variables=["conversation_history", "gt_answer", "prediction"],
            template=prompt,
        )
        self.history_template = HistoryTemplate()
    
    def _llm_score_single(self, prediction, reference, conv_history):

        # serialize conversation history
        serialized_conv_history = self.history_template.serialize_history(conv_history)

        prompt = self.prompt_template.format(
            {"conversation_history": serialized_conv_history, "gt_answer": reference, "prediction": prediction}
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
            return self._llm_score_single(prediction, reference, conv_history)

        response = response["choices"][0]["message"]["content"].strip().strip('.').strip(',')

        if response.lower() not in [
            "yes",
            "no",
        ]:
            print(f"Response {response} not in ['yes', 'no']\nSystem prompt: {self.system_prompt}\nPrompt: {prompt}")
        if "yes" in response.lower():
            return 1.0
        else:
            return 0.0


class FaithDialCriticInverse(Metric):

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "McGill-NLP/roberta-large-faithcritic", return_tensors="pt"
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "McGill-NLP/roberta-large-faithcritic",
        ).cuda()

    def __call__(self, predictions, references, questions, ids=None):
        scores = [
            self._faith_critic_inv(prediction, reference)
            for prediction, reference in zip(predictions, references)
        ]

        if self.store_individual_scores:
            self.save_individual_scores(ids, [{"faithcritic_inverse": score} for score in scores])
        return {"faithcritic_inverse": np.mean(scores)}

    def _faith_critic_inv(self, prediction, references):

        faith_critic_inv_scores = [
            self._faith_critic_inv_score(prediction, reference) for reference in references
        ]
        return min(faith_critic_inv_scores)

    def _faith_critic_inv_score(self, prediction, reference):

        input = self.tokenizer(
            prediction,
            reference,
            return_tensors="pt",
            truncation=True,
        )
        input = {key: val.cuda() for key, val in input.items()}
        output = torch.argmax(self.model(**input).logits)
        return float(output.item())
