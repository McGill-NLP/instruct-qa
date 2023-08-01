from setuptools import setup

setup(
    name="rag-eval",
    description="Empirical evaluation of retrieval-augmented instruction-following models.",
    version="0.0.1",
    url="https://github.com/McGill-NLP/instruct-qa",
    python_requires=">=3.7",
    packages=["rag_eval"],
    install_requires=[
        "torch",
        "transformers",
        "datasets",
        "evaluate",
        "wget",
        "nltk",
        "rouge_score",
        "faiss-cpu",
        "sentence-transformers",
        "pyserini",
        "flask",
        "gunicorn",
        "openai",
        "nltk",
        "accelerate",
        "jsonlines",
        "protobuf==3.20.*",
    ],
    extras_require={
        "evaluation": ["tensorflow-text", "bert_score", "allennlp", "allennlp-models"],
    },
    include_package_data=True,
    zip_safe=False,
)
