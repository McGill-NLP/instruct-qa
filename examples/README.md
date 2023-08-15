# Examples
Here we present examples of how to use the library. Our library supports using and evaluating retrieval-augmented generation in several settings.

- [Running on a set of custom queries](#running-on-a-set-of-custom-queries)
- [Running on pre-defined datasets](#running-on-pre-defined-datasets)
- [Evaluating model responses using a suite of evaluation metrics](#evaluation)

## Running on a set of custom queries

An example of running the retrieval-augmented instruction-following models on custom queries is present in [get_started.py](get_started.py) file.

## Running on pre-defined datasets

Our library supports both question answering (QA) and conversational question answering (ConvQA) datasets. The following datasets are currently incorporated in the library
- [Natural Questions (Open-domain)](https://huggingface.co/datasets/nq_open)
- [HotpotQA](https://huggingface.co/datasets/hotpot_qa)
- [TopiOCQA](https://huggingface.co/datasets/McGill-NLP/TopiOCQA)

Here is an example to generate responses for Natural Questions using DPR retriever and Flan-T5 generator.
```bash
python experiments/question_answering.py \
--prompt_type qa \
--dataset_name natural_questions \
--document_collection_name dpr_wiki_collection \
--index_name dpr-nq-multi-hnsw \
--retriever_name facebook-dpr-question_encoder-multiset-base \
--batch_size 1 \
--model_name flan-t5-xxl \
--k 8
```

Here `k` denotes the number of retrieved passages to use for generation.

Some model choices may require additional arguments. For example, running OpenAI models requires additional specifying API key. Here is an example on how to use `gpt-3.5-turbo` as a generator.
```bash
python experiments/question_answering.py \
--prompt_type qa \
--dataset_name natural_questions \
--document_collection_name dpr_wiki_collection \
--index_name dpr-nq-multi-hnsw \
--retriever_name facebook-dpr-question_encoder-multiset-base \
--batch_size 1 \
--model_name gpt-3.5-turbo \
--api_key <API_KEY> \
--k 8
```

Currently we support Alpaca and Llama-2 when the path to the model weights is provided. Here is an example on how to use Llama-2 (7B chat model) as a generator.
```bash
python experiments/question_answering.py \
--prompt_type qa \
--dataset_name natural_questions \
--document_collection_name dpr_wiki_collection \
--index_name dpr-nq-multi-hnsw \
--retriever_name facebook-dpr-question_encoder-multiset-base \
--batch_size 1 \
--model_name llama-2-7b-chat \
--weights_path <PATH_TO_WEIGHTS> \
--k 8
```
Alpaca can be used by replacing `llama-2-7b-chat` with `alpaca-7b` in the above command, along with the corresponding path to the weights.

### Extending the library to new datasets
Coming soon

## Evaluation
Coming soon
