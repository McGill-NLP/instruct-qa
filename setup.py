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
        "transformers>=4.29.1",
        "datasets>=2.13.1",
        "evaluate>=0.4.0",
        "wget>=3.2",
        "rouge_score>=0.1.2",
        "faiss-cpu>=1.7.4",
        "sentence-transformers>=2.2.2",
        "pyserini>=0.19.0",
        "flask>=2.2.5",
        "gunicorn",
        "openai>=0.27.8",
        "nltk",
        "accelerate>=0.20.3",
        "jsonlines",
        "protobuf>=3.20.0",
    ],
    extras_require={
        "evaluation": ["tensorflow>=2.9.0", "tensorflow-text>=2.9.0", "bert_score>=0.3.13", "allennlp>=2.10.1", "allennlp-models>=2.10.1"],
        "evaluation-reproduce": ["tensorflow==2.9.0", "tensorflow-text==2.9.0", "bert_score==0.3.13", "allennlp==2.10.1", "allennlp-models==2.10.1"],
        "reproduce": [
            "torch",
            "transformers==4.29.1",
            "datasets==2.13.1",
            "evaluate==0.4.0",
            "wget==3.2",
            "rouge_score==0.1.2",
            "faiss-cpu==1.7.4",
            "sentence-transformers==2.2.2",
            "pyserini==0.19.0",
            "flask==2.2.5",
            "gunicorn",
            "openai==0.27.8",
            "nltk",
            "accelerate==0.20.3",
            "jsonlines",
            "protobuf==3.20.*",
        ],
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
