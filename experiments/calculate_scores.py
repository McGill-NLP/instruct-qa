import argparse
import glob
import json
import os
from os import path

import jsonlines
from rag_eval.dataset.utils import load_dataset
from rag_eval.evaluation.utils import load_metric


def calculate_score_for_single_file(file_name, metrics=None, args=None):
    aggregate_scores_calculated = False
    if os.path.exists(path.join(args.score_dir, file_name)):
        with open(path.join(args.score_dir, file_name)) as f:
            scores = json.load(f)
        metrics_calculated = list(scores.keys())
        if all([m in metrics_calculated for m in metrics]):
            aggregate_scores_calculated = True
    if aggregate_scores_calculated and not args.store_individual_scores:
        print(
            "Scores for file {} already exist for all mentioned metrics - {}. Skipping...".format(
                file_name, " ".join(metrics)
            )
        )
        return

    ids, predictions, references, questions = [], [], [], []
    with jsonlines.open(path.join(args.response_dir, file_name)) as responses:
        for resp in responses:
            ids.append(resp["id_"])
            references.append(resp["answer"])
            predictions.append(resp["response"])
            questions.append(resp["question"])

    individual_scores_calculated = True
    for metric_name in metrics:
        individual_file_path = path.join(args.score_dir, metric_name, file_name)
        if not os.path.exists(individual_file_path):
            individual_scores_calculated = False
            break
        if os.path.exists(individual_file_path):
            with open(individual_file_path) as f:
                individual_scores = f.readlines()
                if len(individual_scores) < len(predictions):
                    print(
                        "Individual scores for file {} do not exist for all responses for metric {}. Running evaluation...".format(
                            file_name, metric_name
                        )
                    )
                    individual_scores_calculated = False
                    break

    if (
        args.store_individual_scores
        and individual_scores_calculated
        and aggregate_scores_calculated
    ):
        print(
            "Scores for file {} already exist for all mentioned metrics along with individual responses - {}. Skipping...".format(
                file_name, " ".join(metrics)
            )
        )
        return

    scores = {}
    for metric_name in metrics:
        print("Calculating {} for file {}".format(metric_name, file_name))
        metric = load_metric(metric_name, file_name=file_name, args=args)
        if metric_name == "llm_eval_conv":
            conv_history_list = []
            dataset = load_dataset(
                args.dataset_name,
                split=args.dataset_split,
                name=args.dataset_config_name,
                file_path=args.dataset_file_path,
            )
            for id in ids:
                sample = dataset[id]
                history = [x for x in sample.context] + [{"speaker": "Human", "utterance": sample.question}]
                conv_history_list.append(history)
            scores[metric_name] = metric(
                predictions,
                references,
                questions=conv_history_list,
                ids=ids,
            )
        else:
            scores[metric_name] = metric(
                predictions, references, questions=questions, ids=ids
            )

    # write scores to file if all metrics are calculated
    if args.all_metrics:
        os.makedirs(args.score_dir, exist_ok=True)
        with jsonlines.open(path.join(args.score_dir, file_name), "w") as writer:
            writer.write(scores)

    return scores


parser = argparse.ArgumentParser()
parser.add_argument(
    "--response_dir",
    action="store",
    type=str,
)
parser.add_argument(
    "--score_dir",
    action="store",
    type=str,
)
parser.add_argument(
    "--response_file_name",
    action="store",
    type=str,
    default="all",
    help="A specific response's file name, or simply 'all' to calculate scores for all the files.",
)
parser.add_argument(
    "--store_individual_scores",
    action="store_true",
)
parser.add_argument(
    "--meteor",
    action="store_true",
)
parser.add_argument(
    "--bertscore",
    action="store_true",
)
parser.add_argument(
    "--bem",
    action="store_true",
)
parser.add_argument(
    "--rouge",
    action="store_true",
)
parser.add_argument(
    "--bleu",
    action="store_true",
)
parser.add_argument(
    "--f1",
    action="store_true",
)
parser.add_argument(
    "--em",
    action="store_true",
)
parser.add_argument(
    "--recall",
    action="store_true",
)
parser.add_argument(
    "--recallem",
    action="store_true",
)
parser.add_argument(
    "--precision",
    action="store_true",
)
parser.add_argument(
    "--llm_eval",
    action="store_true",
)
parser.add_argument(
    "--llm_eval_conv",
    action="store_true",
)
parser.add_argument(
    "--faithcritic_inverse",
    action="store_true",
)
parser.add_argument(
    "--api_key",
    action="store",
    type=str,
    default=None,
    help="API key if generating from OpenAI model.",
)
parser.add_argument(
    "--model_name",
    action="store",
    type=str,
    default=None,
    help="The OpenAI model to be used for LLMEval",
)
parser.add_argument(
    "--max_tokens",
    action="store",
    type=int,
    default=2,
    help="The maximum number of tokens to generate.",
)
parser.add_argument(
    "--temperature",
    action="store",
    type=float,
    default=0.0,
    help="The temperature to use during generation.",
)
parser.add_argument(
    "--top_p",
    action="store",
    type=float,
    default=1.0,
    help="Sampling temperature used during generation.",
)
parser.add_argument(
    "--n",
    action="store",
    type=int,
    default=1,
    help="Number of completions to generate for each prompt",
)
parser.add_argument(
    "--stop_seq",
    action="store",
    type=str,
    default="",
    help="When to stop generation",
)
parser.add_argument(
    "--presence_penalty",
    action="store",
    type=float,
    default=0.0,
    help="Positive values increases model's likelihood to talk about new topics",
)
parser.add_argument(
    "--frequency_penalty",
    action="store",
    type=float,
    default=0.0,
    help="Positive values decreases model's likelihood to repeat same line verbatim",
)
parser.add_argument(
    "--all_metrics",
    action="store_true",
)
parser.add_argument(
    "--dataset_name",
    action="store",
    type=str,
    default=None,
    help="The dataset to evaluate against.",
)
parser.add_argument(
    "--dataset_split",
    action="store",
    type=str,
    default="validation",
    help="The split of the dataset to use.",
)
parser.add_argument(
    "--dataset_config_name",
    action="store",
    type=str,
    default=None,
    help="The specific dataset configuration to use.",
)
parser.add_argument(
    "--dataset_file_path",
    action="store",
    type=str,
    default=None,
    help="The path to the dataset file.",
)

if __name__ == "__main__":
    args = parser.parse_args()

    if args.response_file_name == "all":
        paths = [
            path.basename(p) for p in glob.glob(path.join(args.response_dir, "*.jsonl"))
        ]
    else:
        response_path = path.join(args.response_dir, args.response_file_name)
        assert path.exists(response_path), "The path {} does not exist.".format(
            response_path
        )
        paths = [args.response_file_name]

    all_metrics = {
        "meteor": args.meteor,
        "rouge": args.rouge,
        "f1": args.f1,
        "bleu": args.bleu,
        "em": args.em,
        "recall": args.recall,
        "recallem": args.recallem,
        "precision": args.precision,
        "bertscore": args.bertscore,
        "bem": args.bem,
        "llm_eval": args.llm_eval,
        "llm_eval_conv": args.llm_eval_conv,
        "faithcritic_inverse": args.faithcritic_inverse,
    }
    if args.all_metrics:
        # LLM eval will be done on a subset, so we don't include it here to prevent accidental runs
        metrics = list(all_metrics.keys())
        metrics.remove("llm_eval")
        metrics.remove("llm_eval_conv")
    else:
        metrics = []
        for m, m_arg in all_metrics.items():
            if m_arg:
                metrics.append(m)
    # sanity check
    if 'llm_eval' in metrics or 'llm_eval_conv' in metrics:
        assert 'small' in args.response_file_name, "LLM eval can only be run on the small subset of responses, remove this check if you want to run it on the full set."

    print("Metrics calculated:", metrics)
    for file_name in paths:
        print("Calculating the scores for path", args.response_file_name)
        scores = calculate_score_for_single_file(
            file_name,
            metrics=metrics,
            args=args,
        )
        print("\tScores:", scores)

# python -m experiments.calculate_scores --all_metrics --response_dir [jsonl files response directory] --score_dir [the directory where you want the scores to be stored]
