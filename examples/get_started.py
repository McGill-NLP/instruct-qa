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
"""
Halley's Comet Halley's Comet or Comet Halley, officially designated 1P/Halley...
"""
