from instruct_qa.dataset import Dataset, DataSample


class GenericQADataset(Dataset):
    def __init__(
        self,
        queries,
        dataset_name: str = "generic_qa",
        split: str = "validation",
        name: str = None,
        file_path: str = None,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.name = name
        self.data = []
        self.load_data(queries)

    def __len__(self):
        return len(self.data)

    def load_data(self, queries):
        for id_, query in enumerate(queries):
            self.data.append(
                DataSample(
                    id_=id_,
                    question=query,
                    answer="",
                    context=[],
                    metadata={},
                )
            )

    def __getitem__(self, index) -> DataSample:
        return self.data[index]

    def get_queries(self, batch):
        return [example.question for example in batch]
