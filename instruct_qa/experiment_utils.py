import gzip
import re
import shutil
import urllib.request
from pathlib import Path

from tqdm.auto import tqdm
from tqdm.utils import CallbackIOWrapper


def generate_experiment_id(name, split, collection_name, model_name, retriever_name, prompt_type, top_p, temperature, seed):
    """
    Generates a unique experiment identifier.

    Args:
        name (str): Name of the experiment. This is typically the name of the
            dataset.
        collection_name (str): Name of the collection.
        model_name (str): Name of the model evaluated.
        retriever_name (str): Name of the retriever evaluated.
        prompt_type (str): Type of prompt used.
        top_p (float): Parameter used for Nucleus Sampling.
        temperature (float): Temperate used for generation.
        seed (int): Seed for RNG.

    Returns:
        str: Unique experiment identifier.
    """
    experiment_id = name + "_" + split

    for arg in [collection_name, model_name, retriever_name, top_p, temperature, seed]:
        if isinstance(arg, str):
            arg = arg.replace("/", "_")

    if isinstance(collection_name, str):
        experiment_id += f"_c-{collection_name}"
    if isinstance(model_name, str):
        experiment_id += f"_m-{model_name}"
    if isinstance(retriever_name, str):
        experiment_id += f"_r-{retriever_name}"
    if isinstance(prompt_type, str):
        experiment_id += f"_prompt-{prompt_type}"
    if isinstance(top_p, float):
        experiment_id += f"_p-{top_p}"
    if isinstance(temperature, float):
        experiment_id += f"_t-{temperature}"
    if isinstance(seed, int):
        experiment_id += f"_s-{seed}"

    return experiment_id


def parse_experiment_id(experiment_id):
    """
    Parses experiment identifier into key-value pairs.

    Args:
        experiment_id (str): Unique experiment identifier to parse.

    Returns:
        dict: Dictionary containing the parsed key-value pairs.
    """
    regex = "([A-Za-z0-9-_.]+)"
    regex += "_c-([A-Z-a-z0-9-_.]+)"
    regex += "_m-([A-Z-a-z0-9-_.]+)"
    regex += "_r-([A-Z-a-z0-9-_.]+)"
    regex += "_prompt-([A-Z-a-z0-9-_.]+)"
    regex += "_p-(\d+\.\d+)"
    regex += "_t-(\d+\.\d+)"
    regex += "_s-(\d+)"

    parts = re.match(regex, experiment_id).groups()

    result = {
        "name": parts[0],
        "collection_name": parts[1],
        "model_name_or_path": parts[2],
        "retriever_name": parts[3],
        "prompt_type": parts[4],
        "top_p": float(parts[5]),
        "temperature": float(parts[6]),
        "seed": int(parts[7]),
    }

    return result


def log_commandline_args(args, logger=print):
    for arg in vars(args):
        logger(f" - {arg}: {getattr(args, arg)}")

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
