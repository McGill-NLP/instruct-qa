from datasets import load_dataset
from typing import List, Dict

from . import PassageCollection


class FaithDialCollection(PassageCollection):
    def __init__(
        self,
        file_name: str = None,
        cachedir: str = None,
    ):
        super().__init__()
        self._id_to_index = {}
        self.load_data()

    def load_data(self):
        hf_dataset = load_dataset(
            "McGill-NLP/FaithDial",
            split="validation",
        )

        for id_, sample in enumerate(hf_dataset):
            index = len(self.passages)
            self.passages.append(
                {
                    "id": id_,
                    "text": sample["knowledge"],
                    "title": "",
                    "sub_title": "",
                    "index": index,
                }
            )
            self._id_to_index[id_] = index

    def get_passage_from_id(self, id: str) -> Dict[str, str]:
        passage = self.passages[self._id_to_index[id]]
        assert passage["index"] == self._id_to_index[id]
        return passage

    def get_indices_from_ids(self, ids: List[str]) -> List[int]:
        return [self._id_to_index[id] for id in ids]
