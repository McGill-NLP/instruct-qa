from datasets import load_dataset

from rag_eval.dataset import DataSample, Dataset


class FaithDialDataset(Dataset):
    def __init__(
        self,
        dataset_name: str = "faithdial",
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
            raise NotImplementedError(
                "FaithDialDataset does not support loading from file_path"
            )
        hf_dataset = load_dataset(
            "McGill-NLP/FaithDial", split=self.split, name=self.name
        )

        for id_, sample in enumerate(hf_dataset):
            # We assume utterances at even positions in the conversation are from the
            # human and odd positions are from the assistant.
            context = [
                {
                    "utterance": utterance,
                    "speaker": "Human" if i % 2 == 0 else "Assistant",
                }
                for i, utterance in enumerate(sample["history"][:-1])
            ]

            self.data.append(
                DataSample(
                    id_=id_,
                    question=sample["history"][-1],
                    answer=[sample["response"]],
                    context=context,
                    metadata={},
                )
            )

    def __getitem__(self, index) -> DataSample:
        return self.data[index]

    def get_queries(self, batch):
        return [example.id_ for example in batch]
