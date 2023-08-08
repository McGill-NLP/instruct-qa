import json
import os
import re
import string

class Metric:
    def __init__(self, name, **kwargs):
        self.name = name
        args = kwargs.get("args", None)
        self.out_dir = args.score_dir
        self.individual_out_dir = os.path.join(self.out_dir, self.name)
        self.store_individual_scores = args.store_individual_scores
        self.file_name = kwargs.get("file_name", None)

    def __call__(self, predictions, references, questions=None, ids=None):
        raise NotImplementedError()

    @classmethod
    def _normalize_text(cls, text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        text = text.lower()
        text = "".join(char for char in text if char not in set(string.punctuation))
        text = re.sub(regex, " ", text)
        text = " ".join(text.split())
        return text

    def _get_tokens(self, text):
        if not text:
            return []
        return self._normalize_text(text).split()

    def save_individual_scores(self, ids, scores):
        assert len(ids) == len(scores), f"ids: {len(ids)}, scores: {len(scores)}"
        os.makedirs(self.individual_out_dir, exist_ok=True)
        with open(os.path.join(self.individual_out_dir, self.file_name), "w") as f:
            f.writelines(
                json.dumps({"id_": id_, self.name: score}) + "\n"
                for id_, score in zip(ids, scores)
            )
