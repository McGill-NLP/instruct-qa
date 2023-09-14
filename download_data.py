#!/usr/bin/env python3

# This script is adapted from DPR CC-BY-NC 4.0 licensed repo (https://github.com/facebookresearch/DPR/blob/main/dpr/data/download_data.py)

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import gzip
import tarfile
import logging
import os
import pathlib
import wget

from typing import Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

RESOURCES_MAP = {
    "results": {
        "url": "https://instruct-qa.s3.us-east-2.amazonaws.com/results.tar.gz",
        "desc": "Response files from all instruction-following and models",
        "original_ext": None,
        "compressed": True,
    },
    "human_eval_annotations": {
        "url": "https://instruct-qa.s3.us-east-2.amazonaws.com/human_eval_annotations.tar.gz",
        "desc": "Human evaluation annotations",
        "compressed": True,
        "original_ext": None,
    },
}

def unpack_tar(tar_file: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    logger.info("Uncompressing %s", tar_file)
    tar = tarfile.open(tar_file)
    tar.extractall(out_dir)
    tar.close()
    logger.info(" Saved to %s", out_dir)

def unpack(gzip_file: str, out_file: str):
    logger.info("Uncompressing %s", gzip_file)
    input = gzip.GzipFile(gzip_file, "rb")
    s = input.read()
    input.close()
    output = open(out_file, "wb")
    output.write(s)
    output.close()
    logger.info(" Saved to %s", out_file)


def download_resource(
    url: str, original_ext: str, compressed: bool, resource_key: str, out_dir: str
) -> Tuple[str, str]:
    logger.info("Requested resource from %s", url)
    path_names = resource_key.split(".")

    if out_dir:
        root_dir = out_dir
    else:
        root_dir = os.path.abspath("./")
        if "/outputs/" in root_dir:
            root_dir = root_dir[: root_dir.index("/outputs/")]

    logger.info("Download root_dir %s", root_dir)

    save_root = os.path.join(root_dir, *path_names[:-1]) 

    pathlib.Path(save_root).mkdir(parents=True, exist_ok=True)

    if original_ext is None:
        local_file_uncompressed = os.path.abspath(os.path.join(save_root, path_names[-1]))
    else:
        local_file_uncompressed = os.path.abspath(os.path.join(save_root, path_names[-1] + original_ext))
    logger.info("File to be downloaded as %s", local_file_uncompressed)

    if os.path.exists(local_file_uncompressed):
        logger.info("File (or folder) already exist %s", local_file_uncompressed)
        return save_root, local_file_uncompressed

    if original_ext is None:
        local_file = os.path.abspath(os.path.join(save_root, path_names[-1]) + '.tar.gz')
    else:
        local_file = os.path.abspath(os.path.join(save_root, path_names[-1] +  original_ext))

    wget.download(url, out=local_file)

    logger.info("Downloaded to %s", local_file)

    if compressed:
        if original_ext is None:
            uncompressed_file = os.path.join(save_root, path_names[-1])
            unpack_tar(local_file, save_root)
        else:
            uncompressed_file = os.path.join(save_root, path_names[-1] + original_ext)
            unpack(local_file, uncompressed_file)
        os.remove(local_file)
        local_file = uncompressed_file

    return save_root, local_file


def download_file(url: str, out_dir: str, file_name: str):
    logger.info("Loading from %s", url)
    local_file = os.path.join(out_dir, file_name)

    if os.path.exists(local_file):
        logger.info("File already exist %s", local_file)
        return

    wget.download(url, out=local_file)
    logger.info("Downloaded to %s", local_file)


def download(resource_key: str, out_dir: str = None):
    if resource_key not in RESOURCES_MAP:
        # match by prefix
        resources = [k for k in RESOURCES_MAP.keys() if k.startswith(resource_key)]
        if resources:
            for key in resources:
                download(key, out_dir)
        else:
            logger.info("no resources found for specified key")
        return []
    download_info = RESOURCES_MAP[resource_key]

    url = download_info["url"]

    save_root_dir = None
    data_files = []
    if isinstance(url, list):
        for i, item_url in enumerate(url):
            save_root_dir, local_file = download_resource(
                item_url,
                download_info["original_ext"],
                download_info["compressed"],
                "{}_{}".format(resource_key, i),
                out_dir,
            )
            data_files.append(local_file)
    else:
        save_root_dir, local_file = download_resource(
            url,
            download_info["original_ext"],
            download_info["compressed"],
            resource_key,
            out_dir,
        )
        data_files.append(local_file)

    return data_files


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir",
        default="./",
        type=str,
        help="The output directory to download file",
    )
    parser.add_argument(
        "--resource",
        type=str,
        help="Resource name. See RESOURCES_MAP for all possible values",
        default="human_eval_annotations",
    )
    args = parser.parse_args()
    if args.resource:
        download(args.resource, args.output_dir)
    else:
        print("Please specify resource value. Possible options are:")
        for k, v in RESOURCES_MAP.items():
            print(f"Resource key={k}  :  {v['desc']}")


if __name__ == "__main__":
    main()