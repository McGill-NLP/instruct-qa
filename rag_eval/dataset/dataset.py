from dataclasses import dataclass
from typing import List, Dict, Any

import torch


@dataclass
class DataSample:
    id_: int
    question: str
    answer: List[str]
    context: List[str]
    metadata: Dict[str, Any]


class Dataset(torch.utils.data.Dataset):
    def load_data(self, file_path: str = None):
        raise NotImplementedError()

    def __getitem__(self, index) -> DataSample:
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def get_queries(self, batch):
        raise NotImplementedError()
