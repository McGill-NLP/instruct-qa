from datasets import load_dataset

from instruct_qa.dataset import Dataset, DataSample


class HotpotQADataset(Dataset):
    """Loads HotpotQA dataset. For more information, see: https://hotpotqa.github.io/"""

    def __init__(
        self,
        dataset_name: str = "hotpot_qa",
        split: str = "validation",
        name: str = "distractor",
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
            raise NotImplementedError("HotpotQADataset does not support loading from file_path")
        hf_dataset = load_dataset("hotpot_qa", name=self.name, split=self.split)

        for id_, sample in enumerate(hf_dataset):
            self.data.append(
                DataSample(
                    id_=id_,
                    question=sample["question"],
                    answer=[sample["answer"]],
                    context=[],
                    metadata={},
                )
            )

    def __getitem__(self, index) -> DataSample:
        return self.data[index]

    def get_queries(self, batch):
        return [example.question for example in batch]