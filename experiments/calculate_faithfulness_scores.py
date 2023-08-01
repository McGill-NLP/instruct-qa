import argparse
import glob
import json
import os
from os import path
import logging

import jsonlines
from rag_eval.evaluation.utils import load_metric
from rag_eval.dataset.utils import load_dataset
from rag_eval.collections.utils import load_collection
from rag_eval.experiment_utils import log_commandline_args


def calculate_score_for_single_file(
    file_name, dataset, collection, metrics=None, args=None
):
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

    ids, predictions, passages_indices = [], [], []
    with jsonlines.open(path.join(args.response_dir, file_name)) as responses:
        for resp in responses:
            ids.append(resp["id_"])
            predictions.append(resp["response"])
            # predictions.append(resp["answer"][0])
            passages_indices.append(resp["indices"])

    conv_history_list = []
    gold_passages_list = []
    for idx, id in enumerate(ids):
        sample = dataset[id]
        history = [x["utterance"] for x in sample.context] + [sample.question]
        conv_history_list.append(history)

        gold_passages = collection.get_passages_from_indices(passages_indices[idx])
        gold_passages_list.append(
            [collection.passage_to_string(p) for p in gold_passages]
        )

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
        scores[metric_name] = metric(
            conv_history_list, predictions, gold_passages_list, ids=ids
        )

    # write scores to file
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
    "--faithcritic",
    action="store_true",
)
parser.add_argument(
    "--faithcritic_v2",
    action="store_true",
)
parser.add_argument(
    "--q_squared",
    action="store_true",
)
parser.add_argument(
    "--kbertscore",
    action="store_true",
)
parser.add_argument(
    "--kprecision",
    action="store_true",
)
parser.add_argument(
    "--kprecision_plus_plus",
    action="store_true",
)
parser.add_argument(
    "--krecall",
    action="store_true",
)
parser.add_argument(
    "--krecall_plus_plus",
    action="store_true",
)
parser.add_argument(
    "--kllm_eval",
    action="store_true",
)
parser.add_argument(
    "--kllm_eval_conv",
    action="store_true",
)
parser.add_argument(
    "--kf1",
    action="store_true",
)
parser.add_argument(
    "--kf1_plus_plus",
    action="store_true",
)
parser.add_argument(
    "--all_metrics",
    action="store_true",
)
parser.add_argument(
    "--dataset_name",
    action="store",
    type=str,
    default="hotpot_qa",
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
parser.add_argument(
    "--document_collection_name",
    action="store",
    type=str,
    default="dpr_wiki_collection",
    help="Document collection to retrieve from.",
)
parser.add_argument(
    "--document_cache_dir",
    action="store",
    type=str,
    default=None,
    help="Directory that document collection is cached in.",
)
parser.add_argument(
    "--document_file_name",
    action="store",
    type=str,
    default=None,
    help="Basename of the path to the file containing the document collection.",
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

if __name__ == "__main__":
    args = parser.parse_args()

    # Create a logger that logs to the console.
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(os.path.basename(__file__))

    logger.info("Evaluating model:")
    log_commandline_args(args, logger.info)

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
        "faithcritic": args.faithcritic,
        "faithcritic_v2": args.faithcritic_v2,
        "kbertscore": args.kbertscore,
        "kprecision": args.kprecision,
        "kprecision++": args.kprecision_plus_plus,
        "krecall": args.krecall,
        "krecall++": args.krecall_plus_plus,
        "kf1": args.kf1,
        "kf1++": args.kf1_plus_plus,
        "kllm_eval": args.kllm_eval,
        "kllm_eval_conv": args.kllm_eval_conv,
        "q_squared": args.q_squared,
    }
    if args.all_metrics:
        metrics = list(all_metrics.keys())
        metrics.remove("kllm_eval")
        metrics.remove("kllm_eval_conv")
    else:
        metrics = []
        for m, m_arg in all_metrics.items():
            if m_arg:
                metrics.append(m)
    # sanity check
    if "llm_eval" in metrics or "llm_eval_conv" in metrics:
        assert (
            "small" in args.response_file_name
        ), "LLM eval can only be run on the small subset of responses, remove this check if you want to run it on the full set."

    dataset = load_dataset(
        args.dataset_name,
        split=args.dataset_split,
        name=args.dataset_config_name,
        file_path=args.dataset_file_path,
    )

    logger.info("Loading document collection...")
    document_collection = load_collection(
        args.document_collection_name,
        cache_dir=args.document_cache_dir,
        file_name=args.document_file_name,
    )

    print("Metrics calculated:", metrics)
    for file_name in paths:
        print("Calculating the scores for path", args.response_file_name)
        scores = calculate_score_for_single_file(
            file_name,
            dataset,
            document_collection,
            metrics=metrics,
            args=args,
        )
        print("\tScores:", scores)

# python -m experiments.calculate_scores --all_metrics --response_dir [jsonl files response directory] --score_dir [the directory where you want the scores to be stored]
