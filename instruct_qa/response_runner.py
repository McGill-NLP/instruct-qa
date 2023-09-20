import json
import os
from pathlib import Path
import numpy as np
import requests
import re

from instruct_qa.retrieval.utils import dict_values_list_to_numpy
from instruct_qa.dataset.qa import GenericQADataset
from tqdm import tqdm


class ResponseRunner:
    def __init__(
        self,
        model,
        retriever,
        document_collection,
        prompt_template,
        dataset=None,
        queries=None,
        output_path=None,
        k=10,
        batch_size=1,
        logging_interval=256,
        use_hosted_retriever=False,
        hosted_retriever_url="http://10.140.16.91:42010/search",
        use_cached_retrieved_results=False,
        post_process_response=False,
    ):
        self._model = model
        self._retriever = retriever
        self._document_collection = document_collection
        self._prompt_template = prompt_template

        # either dataset or queries should be specified, but not both
        assert (dataset is None) != (queries is None), "Either dataset or queries should be specified, but not both"
        if queries:
            dataset = GenericQADataset(queries)
        self._dataset = dataset
        self._output_path = output_path
        self._k = k
        self._batch_size = batch_size
        self._logging_interval = logging_interval
        self._use_hosted_retriever = use_hosted_retriever
        self._hosted_retriever_url = hosted_retriever_url
        self._use_cached_retrieved_results = use_cached_retrieved_results
        self._collection_name = document_collection.get_name()
        self._post_process_response = post_process_response

    def post_process_response(self, response):
        return self._model.post_process_response(response)

    def __call__(self):
        if self._output_path and os.path.exists(self._output_path):
            with open(self._output_path, "r") as f:
                existing_results = [json.loads(line) for line in f.readlines()]
            num_done = len(existing_results)
            if num_done >= len(self._dataset):
                print(f"Already done with {num_done} examples.")
                return
            if num_done > 0:
                print(f"Skipping {num_done} examples that are already done.")
                self._dataset.data = self._dataset.data[num_done:]
        batches = [
            self._dataset[i : i + self._batch_size]
            for i in range(0, len(self._dataset), self._batch_size)
        ]

        results = []
        for i, batch in enumerate(
            tqdm(batches, desc="Collecting responses", leave=False)
        ):
            queries = self._dataset.get_queries(batch)

            if self._use_hosted_retriever:
                post_results = requests.post(
                    url=self._hosted_retriever_url,
                    json={
                        "queries": queries,
                        "k": self._k,
                        "dataset": self._collection_name,
                    },
                )
                r_dict = dict_values_list_to_numpy(post_results.json())
                retrieved_indices = r_dict["indices"]
            elif self._use_cached_retrieved_results:
                retrieved_ctx_ids = self._retriever.retrieve(queries, k=self._k)
                retrieved_indices = [
                    self._document_collection.get_indices_from_ids(x)
                    for x in retrieved_ctx_ids
                ]
            else:
                r_dict = self._retriever.retrieve(queries, k=self._k)
                retrieved_indices = r_dict["indices"]

            # Get the document texts.
            passages = [
                self._document_collection.get_passages_from_indices(indices)
                for indices in retrieved_indices
            ]

            prompts = [
                self._prompt_template(
                    sample=sample,
                    passages=p,
                )
                for sample, p in zip(batch, passages)
            ]

            responses = self._model(prompts)

            if self._post_process_response:
                responses = [self.post_process_response(response) for response in responses]

            results.extend(
                {
                    "id_": example.id_,
                    "question": example.question,
                    "response": response,
                    "answer": example.answer,
                    "prompt": prompt,
                    "indices": indices.tolist()
                    if type(indices) == np.ndarray
                    else indices,
                }
                for example, response, prompt, indices in zip(
                    batch, responses, prompts, retrieved_indices
                )
            )

            if self._output_path and (i + 1) % self._logging_interval == 0:
                self._write_results_to_file(results)
                results = []
        if self._output_path is not None:
            self._write_results_to_file(results)
        
        return results

    def _write_results_to_file(self, results):
        # Use pathlib to create a folder of the output path if it is not created
        # already.
        Path(self._output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self._output_path, "a") as f:
            f.writelines(json.dumps(result) + "\n" for result in results)
