from dataclasses import dataclass
from typing import List, Dict

class PassageCollection(object):
    def __init__(self):
        self.passages = []

    def load_data(self, path_to_file: str):
        raise NotImplementedError

    def get_passage_from_id(self, id: str) -> Dict[str, str]:
        raise NotImplementedError

    def get_passages_from_indices(self, indices: List[int]) -> List[Dict[str, str]]:
        return [self.passages[i] for i in indices]

    def get_all_passages(self) -> List[Dict[str, str]]:
        return self.passages

    def get_indices_from_ids(self, ids: List[str]) -> List[int]:
        raise NotImplementedError

    def passage_to_string(self, passage: Dict[str, str]) -> str:
        return passage["text"]
