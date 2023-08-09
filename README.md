# Evaluating Correctness and Faithfulness of Instruction-Following Models for Question Answering

[![arXiv](https://img.shields.io/badge/arXiv-2307.16877-b31b1b.svg)](https://arxiv.org/abs/2307.16877)
[![License](https://img.shields.io/badge/License-Apache_2.0-yellowgreen.svg)](https://opensource.org/licenses/Apache-2.0)  
[![PyPi](https://img.shields.io/pypi/v/instruct-qa)](https://pypi.org/project/instruct-qa/)

## Quick Start
### Installation

Make sure you have Python 3.7+ installed. It is also a good idea to use a virtual environment.
<details>
<summary>Show instructions for Virtual Environments</summary>
<br>
```bash
python3 -m venv instruct-qa-venv
source instruct-qa-venv/bin/activate
```
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


## Data and Resources (Coming soon!)
We plan to release data and resources soon! Stay tuned!

## License

This work is licensed under the Apache 2 license. See [LICENSE](LICENSE) for details.

## Citation
To cite this work, please use the following citation:
```
@article{adlakha2023evaluating,
      title={Evaluating Correctness and Faithfulness of Instruction-Following Models for Question Answering}, 
      author={Vaibhav Adlakha and Parishad BehnamGhader and Xing Han Lu and Nicholas Meade and Siva Reddy},
      year={2023},
      journal={arXiv:2307.16877},
}
```

## Contact

For queries and clarifications please contact **vaibhav.adlakha (at) mila (dot) quebec**