from functools import partial
import subprocess
import json
from pathlib import Path
import sys
from typing import List, Dict
import multiprocessing as mp

from tqdm.auto import tqdm

from .utils import convert_dict_to_text

try:
    from pyserini.search import LuceneSearcher
except ImportError as e:
    raise ImportError(
        f"""
        You must install pyserini in order to use this module. You can learn more by
        referring to the pyserini docs. Note that Java is required to use pyserini.
        """
    )


def convert_to_pyserini_records(
    records: List[Dict[str, str]],
    dict_to_text_fn=partial(
        convert_dict_to_text, key_order=("title", "sub_title", "text")
    ),
    n_jobs=-1,
    chunk_size=1000,
):
    """
    Convert records (list of dictionaries, each one representing a document) to a list of Pyserini
    records (list of dictionaries, each one representing a document in Pyserini's JSON format). This
    function uses multiprocessing to speed up the process.

    Parameters
    ----------
    records: list of dicts
        This is a list of dictionaries, each one representing a document. This follows the format defined
        in this library. You can use the convert_records_to_texts function to convert a list of records
        to a list of texts.

    dict_to_text_fn: function
        This is a function that converts a dictionary to a string. This is used to convert each record
        to a string. The default function converts the record to a string by concatenating the values
        in the given order, separated by the given separator (see the convert_dict_to_text function for).
        You can use this parameter to customize the conversion process.

    n_jobs: int, default=-1
        Number of processes to use for converting the records. If -1, the number of processes will be
        set to the number of CPU cores.

    chunk_size: int, default=1000
        The number of records to process in each chunk. This is used to control the memory usage of the
        process pool.

    Returns
    -------
    list of dicts
        This is a list of dictionaries, each one representing a document in Pyserini's JSON format.
        For example, a dictionary in this list may look like:
        - Original: {"id": "doc1", "title": "My Title", "text": "This is the text.", "sub_title": "Some subtitle"}
        - Pyserini: {"id": "doc1", "contents": "My Title Some subtitle This is the text."}

    Notes
    -----
    The docid field in the Pyserini record is set to the index of the record in the list of records.
    Note that index is different from the id field in the original record. For example, if the original
    record has id "doc1", and is the first record in the `records` list, the Pyserini record will have
    id "0", not "doc1". The original document id is not preserved in the Pyserini record.
    """
    n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs

    def _process_single_record(rec):
        contents = dict_to_text_fn(rec)
        return {"id": rec["index"], "contents": contents}

    with mp.pool.ThreadPool(n_jobs) as pool:
        new_records = pool.map(_process_single_record, records, chunksize=chunk_size)

    return new_records


def create_pyserini_json(
    records,
    directory="",
    filename="documents.json",
    overwrite=False,
    verbose=1,
    n_jobs=-1,
):
    """
    Create a Pyserini JSON file from a list of records, and save it to the given directory.
    This is a thin wrapper around the convert_to_pyserini_records function.

    Parameters
    ----------
    records: list of dicts
        This is the same as the records parameter in the convert_to_pyserini_records function.

    directory: str, default=""
        Path to the directory where the JSON file will be saved.

    filename: str, default="documents.json"
        Name of the JSON file.

    overwrite: bool, default=False
        If True, the JSON file will be overwritten if it already exists. If False, the JSON file will
        not be overwritten if it already exists.

    verbose: int, default=1
        If 1, print progress messages. If 0, don't print progress messages.

    n_jobs: int, default=-1
        Number of processes to use for converting the records. If -1, the number of processes will be
        set to the number of CPU cores.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename

    if path.is_file() and not overwrite:
        if verbose > 0:
            print(
                f"Pyserini JSON file already exists at '{path}', and overwrite=False. Skipping."
            )
    else:
        if verbose > 0:
            print(f"Converting dataset to Pyserini's JSON format...")
        # Convert the dataset to Pyserini's JSON (not JSONL) format
        new_records = convert_to_pyserini_records(records, n_jobs=n_jobs)

        with open(path, "w") as f:
            json.dump(new_records, f)

        if verbose > 0:
            print(f"Pyserini JSON file saved to {path}.")


def build_pyserini_index(
    input_dir, index_dir=None, n_jobs=-1, python_str=None, verbose=1
):
    """
    Build a Pyserini index from a directory of JSON files. The JSON files should have the following
    format:

    {
        "id": "doc1",
        "contents": "This is the text of the document."
    }

    Parameters
    ----------
    input_dir: str
        Path to the directory containing the JSON files. All files in this directory will be indexed.
    index_dir: str
        Path to the directory where the index will be stored. If None, the index will be stored in
        the same directory as the JSON files, in a subdirectory called "index".
    n_jobs: int
        Number of threads to use for indexing. Defaults to 1. If -1, the number of threads will be
        set to the number of CPU cores.
    python_str: str
        String of Python executable to use for indexing. Defaults to "python". If you are using a
        virtual environment, you may need to specify the full path to the Python executable, e.g.
        "/path/to/venv/bin/python".
    verbose: int
        Verbosity level. Defaults to 1. If 0, no output will be printed. If 1, the command that is
        run will be printed.

    Returns
    -------
    subprocess.CompletedProcess
        The result of the subprocess run. See the Python documentation for more details.
    """
    if n_jobs < 1:
        import multiprocessing

        n_jobs = multiprocessing.cpu_count()

    if python_str is None:
        python_str = sys.executable

    input_dir = Path(input_dir)

    if index_dir is None:
        index_dir = input_dir / "index"

    command = f"""
    {python_str} -m pyserini.index.lucene \\
    --collection JsonCollection \\
    --input {input_dir} \\
    --index {index_dir} \\
    --generator DefaultLuceneDocumentGenerator \\
    --threads {n_jobs} \\
    --storePositions \\
    --storeDocvectors \\
    --storeRaw
    """
    if verbose > 0:
        print("Running command:")
        print(command)
        print("Output:")
        out = subprocess.run(command, shell=True)
        print("Done.")
    else:
        out = subprocess.run(
            command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
        )

    return out
