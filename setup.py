from setuptools import setup, find_packages


version = {}
with open("instruct_qa/version.py") as fp:
    exec(fp.read(), version)

with open("README.md") as fp:
    long_description = fp.read()

setup(
    name="instruct-qa",
    description="Empirical evaluation of retrieval-augmented instruction-following models.",
    version=version["__version__"],
    author="McGill NLP",
    url="https://github.com/McGill-NLP/instruct-qa",
    python_requires=">=3.7",
    packages=find_packages(include=['instruct_qa*']),
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
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        # Apache license
        "License :: OSI Approved :: Apache Software License",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    zip_safe=False,
)
