# Evaluating Correctness and Faithfulness of Instruction-Following Models for Question Answering

[![arXiv](https://img.shields.io/badge/arXiv-2307.16877-b31b1b.svg)](https://arxiv.org/abs/2307.16877)
[![License](https://img.shields.io/badge/License-Apache_2.0-yellowgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPi](https://img.shields.io/pypi/v/instruct-qa)](https://pypi.org/project/instruct-qa/)

## Quick Start
### Installation

Make sure you have Python 3.7+ installed. It is also a good idea to use a virtual environment.

<details>
<summary>Show instructions for creating a Virtual Environment</summary>

<div>

```bash
python3 -m venv instruct-qa-venv
source instruct-qa-venv/bin/activate
```
    
</div>

</details>


You can install the library via `pip`:

```bash
# Install the latest release
pip3 install instruct-qa

# Install the latest version from GitHub
pip3 install git+https://github.com/McGill-NLP/instruct-qa
```

For development, you can install it in editable mode with:
```
git clone https://github.com/McGill-NLP/instruct-qa
cd instruct-qa/
pip3 install -e .
```

### Usage
Here is a simple example to get started. Using this library, use can easily leverage retrieval-augmented instruction-following models for question-answering in ~25 lines of code. The source file for this example is [examples/get_started.py](examples/get_started.py).

```python
from instruct_qa.collections.utils import load_collection
from instruct_qa.retrieval.utils import load_retriever, load_index
from instruct_qa.prompt.utils import load_template
from instruct_qa.generation.utils import load_model
from instruct_qa.response_runner import ResponseRunner

collection = load_collection("dpr_wiki_collection")
index = load_index("dpr-nq-multi-hnsw")
retriever = load_retriever("facebook-dpr-question_encoder-multiset-base", index)
model = load_model("flan-t5-xxl")
prompt_template = load_template("qa")

queries = ["what is haleys comet"]

runner = ResponseRunner(
    model=model,
    retriever=retriever,
    document_collection=collection,
    prompt_template=prompt_template,
    queries=queries,
)

responses = runner()
print(responses[0]["response"])
# Halley's Comet Halley's Comet or Comet Halley, officially designated 1P/Halley, is a short-period comet visible from Earth every 75–76 years. Halley is the only known short-period comet that is regularly visible to the naked eye from Earth, and the only naked-eye comet that might appear twice in a human lifetime. Halley last appeared...
```
You can also check the input prompt given to the instruction-sollowing model that contains the instruction and the retrieved passages.
```python
print(responses[0]["prompt"])
"""
Please answer the following question given the following passages:
- Title: Bill Haley
then known as Bill Haley's Saddlemen...

- Title: C/2016 R2 (PANSTARRS)
(CO) with a blue coma. The blue color...

...

Question: what is haleys comet
Answer:
"""

```
Detailed documentation of different modules of the library can be found [here](instruct_qa/README.md)

## Generating responses for entire datasets
Our library supports both question answering (QA) and conversational question answering (ConvQA) datasets. The following datasets are currently incorporated in the library
- [Natural Questions (Open-domain)](https://huggingface.co/datasets/nq_open)
- [HotpotQA](https://huggingface.co/datasets/hotpot_qa)
- [TopiOCQA](https://huggingface.co/datasets/McGill-NLP/TopiOCQA)

<!-- It is easy to add any HuggingFace dataset to the library by providing a mapping, as demonstrated [here](). -->

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
--k 8 \
--max_new_tokens 500 \
--post_process_response
```

By default, a `results` directory is created within the repository that stores the model responses. The default directory location can be overidden by providing an additional command line argument `--persistent_dir <OUTPUT_DIR>` More examples are present in the [examples](examples) directory.

## Download model responses and human evaluation data
We release the model responses generated using the above commands for all three datasets. The scores reported in the paper are based on these responses. The responses can be downloaded with the following command:
```bash
python download_data.py --resource results
```
The responses are automatically unzipped and stored as JSON lines in the following directory structure:
```
results
├── {dataset_name}
│   ├── response
│   │   ├── {dataset}_{split}_c-{collection}_m-{model}_r-{retriever}_prompt-{prompt}_p-{top_p}_t-{temperature}_s-{seed}.jsonl
```

Currently, the following models are included:
- `fid` (Fusion-in-Decoder, separately fine-tuned on each dataset)
- `gpt-3.5-turbo` (GPT-3.5)
- `alpaca-7b` (Alpaca)
- `llama-2-7b-chat` (Llama-2)
- `flan-t5-xxl` (Flan-T5)

We also release the human annotations for correctness and faithfulness on a subset of responses for all datasets. The annotations can be downloaded with the following command:
```bash
python download_data.py --resource human_eval_annotations
```

The responses will be automatically unzipped in the following directory structure:
```
human_eval_annotations
├── correctness
│   ├── {dataset_name}
│   │   ├── {model}_human_eval_results.json
|
├── faithfulness
│   ├── {dataset_name}
│   │   ├── {model}_human_eval_results.json
```

## LLM-based evaluation
The following prompt templates and instructions were used for LLM-based evaluation.

### Correctness
```
System prompt: You are CompareGPT, a machine to verify the correctness of predictions. Answer with only yes/no.

You are given a question, the corresponding ground-truth answer and a prediction from a model. Compare the "Ground-truth answer" and the "Prediction" to determine whether the prediction correctly answers the question. All information in the ground-truth answer must be present in the prediction, including numbers and dates. You must answer "no" if there are any specific details in the ground-truth answer that are not mentioned in the prediction. There should be no contradicting statements in the prediction. The prediction may contain extra information. If the prediction states something as a possibility, treat it as a definitive answer.

Question: {Question}
Ground-truth answer: {Reference answer}
Prediction:  {{Model response}

CompareGPT response:
```

### Faithfulness
```
System prompt: You are CompareGPT, a machine to verify the groundedness of predictions. Answer with only yes/no.

You are given a question, the corresponding evidence and a prediction from a model. Compare the "Prediction" and the "Evidence" to determine whether all the information of the prediction in present in the evidence or can be inferred from the evidence. You must answer "no" if there are any specific details in the prediction that are not mentioned in the evidence or cannot be inferred from the evidence.

Question: {Question}
Prediction:  {Model response}
Evidence: {Reference passage}

CompareGPT response:
```

## License

This work is licensed under the Apache 2 license. See [LICENSE](LICENSE) for details.

## Citation


To cite this work, please use the following citation:
```
@article{adlakha2023evaluating,
    author = {Adlakha, Vaibhav and BehnamGhader, Parishad and Lu, Xing Han and Meade, Nicholas and Reddy, Siva},
    title = "{Evaluating Correctness and Faithfulness of Instruction-Following Models for Question Answering}",
    journal = {Transactions of the Association for Computational Linguistics},
    volume = {12},
    pages = {681-699},
    year = {2024},
    month = {05},
    issn = {2307-387X},
    doi = {10.1162/tacl_a_00667},
    url = {https://doi.org/10.1162/tacl\_a\_00667},
    eprint = {https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl\_a\_00667/2374800/tacl\_a\_00667.pdf},
}
```

## Contact

For queries and clarifications please contact **vaibhav.adlakha (at) mila (dot) quebec**
