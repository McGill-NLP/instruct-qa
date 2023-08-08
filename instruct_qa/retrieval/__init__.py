import json
import numpy as np

from instruct_qa.retrieval.index import IndexBase, IndexTorchFlat, IndexFaissFlatIP


class RetrieverBase:
    def __init__(self, index: IndexBase):
        raise NotImplementedError()

    def encode_queries(self, queries):
        raise NotImplementedError()

    def encode_documents(self, documents):
        raise NotImplementedError()

    def build_index(self, documents):
        raise NotImplementedError()

    def retrieve(self, queries, k=10):
        raise NotImplementedError()


class RetrieverFromFile(RetrieverBase):
    def __init__(self, index: IndexBase, filename: str):
        self.filename = filename
        with open(self.filename, "r") as f:
            retrieved_results = json.load(f)
        self._retrieved_results = {}
        for item in retrieved_results:
            self._retrieved_results[item["question"]] = item["ctxs"]

    def encode_queries(self, queries):
        pass

    def encode_documents(self, documents):
        pass

    def build_index(self, documents):
        pass

    def retrieve(self, queries, k=10):
        results = []
        for query in queries:
            ctxs = self._retrieved_results[query][:k]
            ctx_ids = [ctx["id"] for ctx in ctxs]
            results.append(ctx_ids)
        return results


class SentenceTransformerRetriever(RetrieverBase):
    def __init__(self, query_model, doc_model=None, index: IndexBase = None):
        """
        Parameters
        ----------
        query_model: sentence_transformers.SentenceTransformer
            The model used to encode queries.

        doc_model: sentence_transformers.SentenceTransformer
            The model used to encode documents. If None, the query_model is used.

        index: instruct_qa.retrieval.index.IndexBase
            The index used to retrieve documents. If you don't provide one, you
            must create one with `build_index`.
        """
        self.query_model = query_model

        if doc_model is None:
            self.doc_model = self.query_model

        self.index = index

    def encode_queries(self, queries, **kwargs):
        """
        Parameters
        ----------
        queries: list of strings
            The list of text queries to encode into embeddings.
        **kwargs: dict
            Additional keyword arguments to pass to the query encoder.
        """
        return self.query_model.encode(queries, **kwargs)

    def encode_documents(self, documents, **kwargs):
        """
        Parameters
        ----------
        documents: list of strings
            The list of text documents to encode into embeddings. This method
            is only useful if you don't provide an index to the constructor.
        **kwargs: dict
            Additional keyword arguments to pass to the document encoder.
        """
        return self.doc_model.encode(documents, **kwargs)

    def build_index(self, documents, index_cls: IndexBase):
        self.index = index_cls(self.encode_documents(documents))
        return self.index

    def retrieve(self, queries, k=10, **kwargs):
        """
        Retrieve documents for a given query.

        Parameters
        ----------
        queries: numpy.ndarray, torch.Tensor, list of strings, or string
            The queries to search for. If a single string is provided, it will be
            converted to a list of length 1. If a list of strings is provided,
            each string will be encoded and the resulting embeddings will be
            stacked into a single numpy array, before being passed to the index.

        k: int
            The number of documents to retrieve for each query.

        **kwargs: dict
            Additional keyword arguments to pass to the query encoder.

        Returns
        -------
        dict
            A dictionary containing the scores and indices of the retrieved documents.

            - scores: numpy.ndarray of shape (n_queries, k)
            - indices: numpy.ndarray of shape (n_queries, k)
        """
        error_msg = f"queries must be a numpy array, a torch tensor, a list of strings, or a string. Got {type(queries)} instead."

        if isinstance(queries, str):
            queries = [queries]

        if isinstance(queries, list):
            if all(isinstance(q, str) for q in queries):
                queries = self.encode_queries(queries, **kwargs)
            else:
                raise ValueError(error_msg)

        if not isinstance(queries, np.ndarray):
            raise ValueError(error_msg)

        if self.index is None:
            raise ValueError("You must create an index first. Use `build_index`.")

        return self.index.search(queries, k=k)


class BM25Retriever(RetrieverBase):
    def __init__(self, index: index.IndexBase = None):
        self.index = index

    def build_index(
        self,
        documents,
        directory="",
        index_subdir="bm25_pyserini",
        index_cls=index.IndexPyseriniBM25,
        **kwargs,
    ):
        """
        This is a convenience method that builds an index and saves it to disk,
        then loads it into memory. This is useful if you want to build an index
        once and then use it multiple times. Behind the scene, this uses the
        `IndexPyseriniBM25` class. Specifically, it calls the `build_index` and
        `load` methods of that class in succession. See the documentation for
        those methods for more details.

        Parameters
        ----------
        documents: list of strings
            The list of text documents to encode into embeddings. This method
            is only useful if you don't provide an index to the constructor.

        directory: str
            This is the directory where the index will be saved. If the directory
            does not exist, it will be created. If the directory already exists,
            it will be overwritten.

        index_subdir: str
            This is the name of the subdirectory where the index will be saved.

        index_cls: instruct_qa.retrieval.index.IndexBase
            The index class to use.

        **kwargs: dict
            Additional keyword arguments to pass to the `build_index` method of
            the index class.
        """
        records = [{"index": i, "text": doc} for i, doc in enumerate(documents)]

        index_cls.build_index(records, directory, index_subdir=index_subdir, **kwargs)
        index = index_cls.load(directory, index_subdir=index_subdir)
        self.index = index

        return self.index

    def retrieve(self, queries, k=10):
        if isinstance(queries, str):
            queries = [queries]

        if self.index is None:
            raise ValueError("You must create an index first. Use `build_index`.")

        return self.index.search(queries, k=k)
