from datasets import load_dataset

from rag_eval.dataset import Dataset, DataSample


class NaturalQuestionsDataset(Dataset):
    """Loads the Open Natural Questions dataset. For more information, see:
    https://aclanthology.org/P19-1612/
    """

    def __init__(
        self,
        dataset_name: str = "natural_questions",
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
            raise NotImplementedError("NaturalQuestionsDataset does not support loading from file_path")
        hf_dataset = load_dataset("nq_open", split=self.split, name=self.name)

        for id_, sample in enumerate(hf_dataset):
            self.data.append(
                DataSample(
                    id_=id_,
                    question=sample["question"],
                    answer=sample["answer"],
                    context=[],
                    metadata={},
                )
            )

    def __getitem__(self, index) -> DataSample:
        return self.data[index]

    def get_queries(self, batch):
        return [example.question for example in batch]