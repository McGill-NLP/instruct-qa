import abc
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np


def _to_np(tensor):
    import torch

    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().detach().numpy()
    else:
        return np.array(tensor)


class IndexBase(metaclass=abc.ABCMeta):
    not_implemented_error = "This method is not implemented for this index type."

    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def __len__(self):
        pass

    @abc.abstractmethod
    def save(self, directory: str, filename: str):
        pass

    @abc.abstractmethod
    def load(cls, directory: str, filename: str):
        """
        Load an index from a directory.

        Parameters
        ----------
        directory: str
            The directory to load the index from.

        filename: str
            The name of the file to load the index from.
        """
        pass

    def get_embeddings(self, start_ix=0, end_ix=-1):
        """
        Get the embeddings of a range of documents in the index.

        Parameters
        ----------
        start_ix: int
            The index of the first document to retrieve.

        end_ix: int
            The index of the last document to retrieve. If -1, all documents
            are retrieved.

        Returns
        -------
        numpy.ndarray
            The embeddings of the documents in the index.
            It will have shape (end_ix - start_ix, embedding_dim).

        Notes
        -----
        This method is not implemented for all index types. For example,
        it is not implemented for BM25.
        """
        raise NotImplementedError(self.not_implemented_error)

    @abc.abstractmethod
    def search(self, queries, k=10):
        """
        Search document embeddings for given queries embedding.

        Parameters
        ----------
        queries: numpy.ndarray, torch.Tensor
            The embedded queries that we will use to search the index.

        k: int
            The number of documents to retrieve for each query.

        Returns
        -------
        results: dict
            A dictionary containing the scores and indices of the retrieved documents.

            - scores: numpy.ndarray of shape (n_queries, k)
            - indices: numpy.ndarray of shape (n_queries, k)
        """
        pass


class IndexTorchFlat(IndexBase):
    def __init__(self, embeddings, sim_func="dot", device="auto"):
        """
        Parameters
        ----------
        embeddings: numpy.ndarray or torch.Tensor
            The embeddings of the documents in the index. If a numpy array is
            provided, it will be converted to a torch tensor.

        sim_func: str or callable
            The similarity function to use when searching the index. If a string
            is provided, it must be one of "cosine" or "dot". If a callable is
            provided, it must take two arguments (the query and document embeddings)
            and return a similarity score.

        device: str
            The device to use when converting the embeddings to a torch tensor.
            If "auto", the device will be determined automatically. If None, the
            embeddings will not be moved to a device.
        """
        import torch

        error_message = f'Unknown similarity function {sim_func}. Use "cosine", "dot", or provide a function.'

        if isinstance(sim_func, str):
            if sim_func == "cosine":

                def cosine(x, y):
                    denom = (
                        torch.norm(x, dim=1, keepdim=True)
                        * torch.norm(y, dim=1, keepdim=True).t()
                    )
                    return torch.mm(x, y.t()) / denom

                sim_func = cosine

            elif sim_func in ["dot", "dot_product"]:

                def dot(x, y):
                    return torch.mm(x, y.t())

                sim_func = dot

            else:
                raise ValueError(error_message)

        if not callable(sim_func):
            raise ValueError(error_message)

        self.sim_func = sim_func

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if isinstance(embeddings, torch.Tensor):
            self.index = embeddings
        else:
            self.index = torch.tensor(embeddings)

        if device is not None:
            self.index = self.index.to(device)

    def __len__(self):
        return self.index.shape[0]

    def save(self, directory="index", filename="flat.index.pt"):
        import torch

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.index, directory / filename)

    @classmethod
    def load(cls, directory="index", filename="flat.index.pt", device="auto"):
        import torch

        directory = Path(directory)
        return cls(torch.load(directory / filename), device=device)

    def get_embeddings(self, start_ix=0, end_ix=-1):
        if end_ix == -1:
            end_ix = self.index.shape[0]
        end_ix = min(end_ix, self.index.shape[0])

        return _to_np(self.index[start_ix:end_ix])

    def search(self, queries, k=10):
        import torch

        if isinstance(queries, torch.Tensor):
            queries = queries
        else:
            queries = torch.tensor(queries)

        scores, indices = torch.topk(
            self.sim_func(queries, self.index), k=k, dim=1, largest=True, sorted=True
        )
        return {"scores": _to_np(scores), "indices": _to_np(indices)}


class IndexFaissFlatIP(IndexBase):
    def __init__(self, embeddings):
        import faiss

        # If we are given a faiss index, use it directly
        if isinstance(embeddings, faiss.IndexFlat):
            self.index = embeddings
        # Otherwise, create a new index (i.e. if we are given a numpy array or torch tensor)
        else:
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
            self.index.add(_to_np(embeddings))

    def __len__(self):
        return self.index.ntotal

    def get_embeddings(self, start_ix=0, end_ix=-1):
        if end_ix == -1:
            end_ix = self.index.ntotal
        end_ix = min(end_ix, self.index.ntotal)

        return self.index.reconstruct_n(start_ix, end_ix)

    def save(self, directory="index", filename="flat.index.faiss"):
        import faiss

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / filename))

    @classmethod
    def load(cls, directory="index", filename="flat.index.faiss"):
        import faiss

        directory = Path(directory)
        index = faiss.read_index(str(directory / filename))
        return cls(index)

    def search(self, queries, k=10):
        if not isinstance(queries, np.ndarray):
            queries = np.array(queries)

        scores, indices = self.index.search(queries, k=k)

        return {"scores": scores, "indices": indices}


class IndexFaissHNSW(IndexFaissFlatIP):
    def __init__(self, embeddings, store_n=512, ef_search=128, ef_construction=200):
        import faiss

        if isinstance(embeddings, faiss.IndexHNSWFlat):
            self.index = embeddings
        else:
            index = faiss.IndexHNSWFlat(embeddings.shape[1] + 1, store_n)
            index.hnsw.efSearch = ef_search
            index.hnsw.efConstruction = ef_construction
            self.index = index
            # TODO: load index from embeddings

    @classmethod
    def load(cls, directory="index", filename="hnsw.index.faiss"):
        import faiss

        directory = Path(directory)
        index = faiss.read_index(str(directory / filename))
        return cls(index)

    def search(self, queries, k=10):
        if not isinstance(queries, np.ndarray):
            queries = np.array(queries)

        aux_dim = np.zeros(len(queries), dtype="float32")
        queries_nhsw_vectors = np.hstack((queries, aux_dim.reshape(-1, 1)))

        scores, indices = self.index.search(queries_nhsw_vectors, k=k)

        return {"scores": scores, "indices": indices}


class IndexPyseriniBM25(IndexBase):
    def __init__(self, searcher):
        """
        Parameters
        ----------
        searcher: pyserini.search.LuceneSearcher
            The Pyserini Lucene searcher object. This is the object returned by the
            build_index method.

        Notes
        -----
        This class is only available if Pyserini is installed. Moreover, the __init__ method
        does not actually build the index. Instead, you must call the `build_index` method to
        build the index, then `load` the index from disk. This is because the index building
        process is slow and happens on disk, so we don't want to do it every time we initialize
        the class. Instead, we build the index once and save it to disk, then load it from disk
        every time we initialize the class. This is the same pattern used by Pyserini.

        Examples
        --------
        >>> from rag_eval.retrieval.index import IndexPyseriniBM25
        >>> records = [{"index": 0, "text": "My name is Nick"}, {"index": 1, "text": "I was born in 1974"}]
        >>> IndexPyseriniBM25.build_index(records, "/tmp/pyserini", index_subdir="bm25_pyserini")
        >>> index = IndexPyseriniBM25.load("/tmp/pyserini", index_subdir="bm25_pyserini")
        >>> index.search(["What is my name?", "When were you born?"], k=1)
        """
        self.index = searcher

    @staticmethod
    def build_index(
        records: List[Dict[str, str]],
        directory: str,
        index_subdir: str = "bm25_pyserini",
        overwrite: bool = False,
        n_jobs: int = -1,
        python_str: str = None,
        verbose: int = 0,
    ):
        """
        Build a Pyserini index from a list of records.

        Parameters
        ----------
        records: list of dicts
            This is a list of dictionaries, each one representing a document. This follows the
            format defined in this library. You can use the convert_records_to_texts function
            to convert a list of records to a list of texts.
        directory: str
            The directory to save the artifacts created by Pyserini. Two things will be saved here:
            - A JSON file containing the documents, which will be used to build the index. It will
              be saved to `<directory>/documents.json`.
            - The index itself (a subdirectory). It will be saved to `<directory>/<index_dir_name>`.
        index_subdir: str
            The name of the subdirectory to save the index to. This will be created inside the
            directory. If you want to build multiple indices, you can change this.
        overwrite: bool
            Whether to overwrite the documents.json file if it already exists. Defaults to False.
        n_jobs: int
            The number of threads to use when building the index. Defaults to -1, which means use
            all available threads.
        python_str: str
            The path to the python executable to use when building the index. Defaults to None,
            which means use the current python executable. If you are using a virtual environment,
            you may need to set this to the path to the python executable in the virtual environment.
        verbose: int
            The verbosity level. Defaults to 0, which means no output. Set to 1 to print the
            output of the Pyserini build_index script.
        """
        from . import pyserini_utils

        # First, create the documents, which will be saved to the index_path
        pyserini_utils.create_pyserini_json(
            records=records,
            directory=directory,
            verbose=verbose,
            overwrite=overwrite,
            n_jobs=n_jobs,
        )

        # Now, build the index, which will be saved to the index_path
        input_dir = Path(directory)
        index_dir = input_dir / index_subdir
        pyserini_utils.build_pyserini_index(
            input_dir=input_dir,
            index_dir=index_dir,
            n_jobs=n_jobs,
            python_str=python_str,
            verbose=verbose,
        )

    def __len__(self):
        return self.index.num_docs

    def get_embeddings(self, start_ix=0, end_ix=-1):
        raise NotImplementedError(
            "Pyserini BM25 does not support retrieving embeddings."
        )

    def save(self, directory, filename):
        raise NotImplementedError(
            "Pyserini BM25 does not support saving indices, because they are created directly on disks. "
            "Use build_index instead."
        )

    @classmethod
    def load(cls, directory, index_subdir="bm25_pyserini"):
        """
        Load a Pyserini index from disk.

        Parameters
        ----------
        directory: str
            The directory where the index is saved. This is the same directory that was passed
            to the build_index method.
        index_subdir: str
            The name of the subdirectory where the index is saved. This is the same directory
            that was passed to the build_index method.

        Returns
        -------
        IndexPyseriniBM25
            The index object. You can use this to search the index.
        """
        from . import pyserini_utils

        index_path = str(Path(directory) / index_subdir)
        return cls(pyserini_utils.LuceneSearcher(index_path))

    def search(self, queries, k=10):
        indices = []
        scores = []

        if isinstance(queries, str):
            queries = [queries]

        for q in queries:
            hits = self.index.search(q, k=k)
            indices.append([int(h.docid) for h in hits])
            scores.append([h.score for h in hits])

        results = dict(
            scores=np.array(scores),
            indices=np.array(indices),
        )

        return results


if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Create a small corpus
    documents = [
        "The cat sat on the mat.",
        "The dog is in the fog.",
        "My name is John.",
        "This guy's name is Jeremy.",
        "I like to eat pizza.",
        "Nick was born in 1974.",
        "Nick's age is 46.",
    ]

    # Encode the documents
    embeddings = model.encode(documents)

    embeddings = model.encode(documents)
    pt_index = IndexTorchFlat(embeddings)
    faiss_ip_index = IndexFaissFlatIP(embeddings)

    # Search for the top 5 most similar documents
    query_text = ["What is your name?", "What age is Nick?"]

    query_embeddings = model.encode(query_text)

    results = {}
    # Use the torch index
    results["pt"] = pt_index.search(query_embeddings, k=5)
    # Use the faiss index
    results["faiss"] = faiss_ip_index.search(query_embeddings, k=5)

    # Print the results
    for q in range(len(query_text)):
        print("\n" + "=" * 80)
        print(f"Query: {query_text[q]}")
        for index_obj in results:
            print(f"Top 5 most similar sentences in corpus ({index_obj}):")
            for i in range(5):
                text = documents[results[index_obj]["indices"][q][i]]
                score = results[index_obj]["scores"][q][i]
                print(f"\t{text} \t\tScore: {score:.4f}")
