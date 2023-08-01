from pathlib import Path
import shutil
from tqdm.auto import tqdm
from tqdm.utils import CallbackIOWrapper
import urllib.request
import gzip

from rag_eval.collections.dpr_wiki_collection import DPRWikiCollection
from rag_eval.collections.hotpot_wiki_collection import HotpotWikiCollection
from rag_eval.collections.topiocqa_wiki_collection import TopiocqaWikiCollection
from rag_eval.collections.faithdial_collection import FaithDialCollection


def load_collection(document_collection_name, cache_dir, file_name):
    """
    Loads a document collection.

    Args:
        document_collection_name (str): The name of the document collection to load.
        cache_dir (str): The directory the document collection is stored in.
        file_name (str): The basename of the path to the document collection.

    Returns:
        PassageCollection: The loaded document collection.
    """
    document_collection_mapping = {
        "dpr_wiki_collection": DPRWikiCollection,
        "topiocqa_wiki_collection": TopiocqaWikiCollection,
        "hotpot_wiki_collection": HotpotWikiCollection,
        "faithdial_collection": FaithDialCollection,
    }

    if document_collection_name not in document_collection_mapping:
        raise NotImplementedError(
            f"Document collection {document_collection_name} not supported."
        )

    return document_collection_mapping[document_collection_name](file_name, cache_dir)


def wget(url, path, progress=True, overwrite=False, create_dir=True, compressed=False):
    """
    Download a file from a URL to a given path.

    Parameters
    ----------
    url : str
        The URL to download from.
    path : str
        The path to save the downloaded file to.
    progress : bool, optional
        Whether to display a progress bar, by default True
    overwrite : bool, optional
        Whether to overwrite the file if it already exists, by default False
    create_dir : bool, optional
        Whether to create the directory if it doesn't exist, by default True
    compressed : bool, optional
        Whether the downloaded file is compressed, by default False
        Only works for .gz files.
    """
    if not overwrite and Path(path).exists():
        return None

    path = Path(path)

    if create_dir:
        path.parent.mkdir(parents=True, exist_ok=True)

    # Give a nice description for the download progress bar
    if compressed:
        download_path = Path(path.as_posix() + ".gz")
    else:
        download_path = path

    if not download_path.exists():
        if progress:
            print(f"Downloading '{download_path}' from {url}")

        # Get content length of file to download
        with urllib.request.urlopen(url) as u:
            meta = u.info()
            file_size = int(meta["Content-Length"])

        # Use tqdm to display download progress, urlretrieve to download
        with tqdm(
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            total=file_size,
            desc=download_path.name,
            disable=not progress,
        ) as t:
            out = urllib.request.urlretrieve(
                url,
                filename=download_path,
                reporthook=lambda b, bsize, tsize: t.update(bsize),
            )

    # Decompress the file if necessary
    if compressed:
        print(f"Decompressing '{download_path}' to '{path}'")
        with gzip.open(download_path, "rb") as f_in:
            with open(path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Remove the compressed file
        download_path.unlink()
