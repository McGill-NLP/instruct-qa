import json
from datasets import load_dataset

from rag_eval.dataset import DataSample, Dataset


class TopiOCQADataset(Dataset):
    def __init__(
        self,
        dataset_name: str = "topiocqa",
        split: str = "validation",
        name: str = None,
        file_path: str = None,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.name = name
        self.data = []
        self.load_data(file_path)

    def __len__(self):
        return len(self.data)

    def load_data(self, file_path: str = None):
        if file_path:
            assert (
                self.split == "test"
            ), "Only test split can be loaded from file_path for TopiOCQA"
            with open(file_path, "r") as f:
                hf_dataset = json.load(f)
        else:
            hf_dataset = load_dataset("McGill-NLP/TopiOCQA", split=self.split)

        for id_, sample in enumerate(hf_dataset):
            # Validation examples have additional answers.
            answers = [sample["Answer"]]
            if self.split in ["validation", "test"]:
                if type(sample["Additional_answers"]) == dict:
                    for answer in sample["Additional_answers"]["Answer"]:
                        answers.append(answer)
                elif type(sample["Additional_answers"]) == list:
                    for answer in sample["Additional_answers"]:
                        answers.append(answer["Answer"])

            # We assume utterances at even positions in the conversation are from the
            # human and odd positions are from the assistant.
            context = [
                {
                    "utterance": utterance,
                    "speaker": "Human" if i % 2 == 0 else "Assistant",
                }
                for i, utterance in enumerate(sample["Context"])
            ]

            self.data.append(
                DataSample(
                    id_=id_,
                    question=sample["Question"],
                    answer=answers,
                    context=context,
                    metadata={},
                )
            )

    def __getitem__(self, index) -> DataSample:
        return self.data[index]

    def get_queries(self, batch):
        queries = []
        for example in batch:
            conv_history = [x["utterance"] for x in example.context] + [
                example.question
            ]
            conv_history = [
                x.replace("’", "'").lower().strip("?").strip() for x in conv_history
            ]
            conv_history = " [SEP] ".join(conv_history)
            queries.append(conv_history)
        return queries
