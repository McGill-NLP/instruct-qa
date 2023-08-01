import csv
import json
import os
from typing import List, Dict
from tqdm import tqdm

from . import PassageCollection, utils


class HotpotWikiCollection(PassageCollection):
    def __init__(
        self,
        file_name: str = "wiki_id2doc.json",
        cachedir: str = "data",
    ):
        super().__init__()
        self._id_to_index = {}
        self.title_to_id = {}
        self.load_data(os.path.join(cachedir, file_name))

    def load_data(self, path_to_file: str):
        assert os.path.exists(path_to_file), f"File does not exist: {path_to_file}"

        with open(path_to_file) as ifile:
            data = json.load(ifile)
        for id, doc in tqdm(data.items()):
            index = len(self.passages)
            self.passages.append(
                {
                    "id": id,
                    "text": doc["text"],
                    "title": doc["title"],
                    "sub_title": "",
                    "index": index,
                }
            )
            self._id_to_index[id] = index
            assert doc["title"] not in self.title_to_id
            self.title_to_id[doc["title"]] = id

    def get_passage_from_id(self, id: str) -> Dict[str, str]:
        passage = self.passages[self._id_to_index[id]]
        assert passage["index"] == self._id_to_index[id]
        return passage

    def get_indices_from_ids(self, ids: List[str]) -> List[int]:
        return [self._id_to_index[id] for id in ids]
