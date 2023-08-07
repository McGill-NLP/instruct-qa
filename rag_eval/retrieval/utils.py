import os
from typing import Dict, List

import rag_eval.experiment_utils as utils
from rag_eval.retrieval import RetrieverFromFile, SentenceTransformerRetriever
from rag_eval.retrieval.index import IndexFaissFlatIP, IndexFaissHNSW

INDEX_NAME_TO_PATH_URL = {
    "dpr-nq-multi-hnsw": {
        "url": "https://rag-eval.s3.us-east-2.amazonaws.com/indexes/dpr/nq/multi/hnsw/index.dpr",
        "path": "data/nq/index/hnsw/index.dpr",
    },
}


def convert_dict_to_text(
    d,
    sep=" ",
    key_order=("title", "subtitle", "text"),
    space_around_sep=True,
):
    """
    Convert a dictionary to a string by concatenating the values in the given order,
    separated by the given separator.

    Parameters
    ----------
    d: dict
        The dictionary to convert.

    key_order: list of strings
        The order in which to concatenate the values. If a key is not present in
        the dictionary, it is ignored.

    sep: string
        The separator to use when concatenating the values. For example, if sep is
        " ", then the values will be concatenated with a space between them.

    space_around_sep: bool
        Whether to add spaces around the separator. For example, if sep is "[SEP]", and
        space_around_sep is True, then the values will be concatenated with a space
        before and after the separator. If space_around_sep is False, then the values
        will be concatenated with no spaces before or after the separator. If sep is
        " ", then this parameter has no effect.

    Returns
    -------
    string
        The converted text. Each text is the concatenation of the values in the
        given order, separated by the given separator.
    """
    if space_around_sep and sep != " ":
        sep = f" {sep} "

    return sep.join(d[k] for k in key_order if k in d)


def convert_records_to_texts(
    records: List[Dict[str, str]],
    sep=" ",
    key_order=("title", "subtitle", "text"),
    space_around_sep=True,
    n_jobs=-1,
    chunk_size=1000,
) -> List[str]:
    """
    Convert records (list of dictionaries, each one representing a document) to a list of texts (the
    text representation of each document). Behind the scenes, this calls convert_dict_to_text on each
    record. This function uses multiprocessing to speed up the process.

    Parameters
    ----------
    records: list of dicts
        The records to convert. Each record is a dictionary with keys "title",
        "subtitle", and "text".

    sep: string
        The separator to use when concatenating the values. For example, if sep is
        " ", then the values will be concatenated with a space between them.

    key_order: list of strings
        The order in which to concatenate the values. If a key is not present in
        the dictionary, it is ignored.

    space_around_sep: bool
        Whether to add a space before and after the separator when concatenating the values.

    n_jobs: int, default=-1
        The number of processes to use. If -1, use all available processes.

    chunk_size: int, default=1000
        The number of records to process in each chunk. This is used to control the
        memory usage of the process pool.

    Returns
    -------
    list of strings
        The converted texts. Each text is the concatenation of the values in the
        given order, separated by the given separator.
    """
    import multiprocessing as mp

    n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count()

    def _process(d: dict):
        return convert_dict_to_text(
            d, sep=sep, key_order=key_order, space_around_sep=space_around_sep
        )

    with mp.pool.ThreadPool(n_jobs) as pool:
        texts = pool.map(_process, records, chunksize=chunk_size)

    return texts


def change_pooling_method(model, pooling_mode):
    """
    Change the pooling method of a sentence-transformers model.

    Parameters
    ----------
    model: sentence_transformers.SentenceTransformer
        The model to change the pooling method of.

    pooling_mode: str
        The pooling method to use. Either 'max', 'mean', 'cls'.  See the documentation for
        sentence_transformers.models.Pooling  for more information.

    Returns
    -------
    sentence_transformers.SentenceTransformer
        The model with the new pooling method. Note that this is the same object as the
        input model, so you don't need to do anything with the return value.
    """
    from sentence_transformers.models import Pooling

    dim = model[0].get_word_embedding_dimension()
    model[1] = Pooling(dim, pooling_mode=pooling_mode)

    return model


def dict_values_list_to_numpy(d: Dict, recursive=False) -> Dict:
    """
    Convert the values of a dictionary from lists to numpy arrays.

    Parameters
    ----------
    d: dict
        The dictionary to convert. Any value that is a list will be converted to a numpy array.
        If a value is a dictionary, this function will be called recursively on it if recursive is True.

    recursive: bool
        Whether to call this function recursively on any dictionary values.

    Returns
    -------
    dict
        A new dictionary with the same keys as the input dictionary, but with the values converted
        to numpy arrays.

    """
    # Use deepcopy to avoid modifying the input dictionary
    import copy

    import numpy as np

    d = copy.deepcopy(d)

    for k, v in d.items():
        if isinstance(v, list):
            d[k] = np.array(v)
        elif recursive and isinstance(v, dict):
            d[k] = dict_values_list_to_numpy(v, recursive=recursive)

    return d


def dict_values_numpy_to_list(d: Dict, recursive=False) -> Dict:
    """
    Convert the values of a dictionary from numpy arrays to lists.

    Parameters
    ----------
    d: dict
        The dictionary to convert. Any value that is a numpy array will be converted to a list.
        If a value is a dictionary, this function will be called recursively on it if recursive is True.

    recursive: bool
        Whether to call this function recursively on any dictionary values.

    Returns
    -------
    dict
        A new dictionary with the same keys as the input dictionary, but with the values converted
        to lists.

    """
    import copy

    import numpy as np

    # Use deepcopy to avoid modifying the input dictionary
    d = copy.deepcopy(d)

    for k, v in d.items():
        if isinstance(v, np.ndarray):
            d[k] = v.tolist()
        elif recursive and isinstance(v, dict):
            d[k] = dict_values_numpy_to_list(v, recursive=recursive)

    return d


def load_index(index_name, **kwargs):
    """
    Load an index by name.

    Parameters
    ----------
    index_name (str): Name of index to load.
    kwargs: Additional parameters for the index (e.g., index_path).

    Returns
    -------
    rag_eval.retrieval.index.IndexBase
        The loaded index.
    """
    index_path = kwargs.get("index_path", None)
    if index_path is None:
        index_path = INDEX_NAME_TO_PATH_URL[index_name]["path"]
        if not os.path.exists(index_path):
            utils.wget(
                INDEX_NAME_TO_PATH_URL[index_name]["url"],
                index_path,
            )

    print("Loading index...")
    if "hnsw" in index_name:
        return IndexFaissHNSW.load(
            directory=os.path.dirname(index_path),
            filename=os.path.basename(index_path),
        )
    else:
        return IndexFaissFlatIP.load(
            directory=os.path.dirname(index_path),
            filename=os.path.basename(index_path),
        )


def load_retriever(model_name, index, retriever_cached_results_fp=None):
    """
    Loads retriever by name.

    Args:
        model_name (str): Name of query model to load from sentence_transformers
        kwargs: Additional parameters for the retriever (e.g., index_path).

    Returns:
        BaseRetriever: Retriever object.
    """
    if retriever_cached_results_fp is not None:
        return RetrieverFromFile(index, filename=retriever_cached_results_fp)

    from sentence_transformers import SentenceTransformer

    query_model = SentenceTransformer(model_name)
    return SentenceTransformerRetriever(query_model, index=index)
