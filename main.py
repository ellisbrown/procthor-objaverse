import glob
import gzip
import multiprocessing as mp
import os
import pickle
import shutil
import warnings
from collections import defaultdict
from typing import Optional, Dict, Sequence, List, Any, Union

import prior
from tqdm import tqdm

try:
    from prior import LazyJsonDataset
except:
    raise ImportError("Please update the prior package (pip install --upgrade prior).")


def load_jsongz_as_str(subpath: Optional[str]) -> str:
    """Load the subpath file."""

    if subpath is None:
        # If the subpath is None, return a json representation of `None`
        return "null"

    with gzip.open(subpath, "r") as f:
        return f.read().strip()


def read_jsonlgz(path: str, max_lines: Optional[int] = None):
    with gzip.open(path, "r") as f:
        lines = []
        for line in tqdm(f, desc=f"Loading {path}"):
            lines.append(line)
            if max_lines is not None and len(lines) >= max_lines:
                break
    return lines


def read_jsongz_files(
    paths: Sequence[str], num_workers: int, max_files: Optional[int] = None
) -> List[str]:
    if len(paths) == 0:
        return []

    ind_to_path = {int(os.path.basename(p).split(".")[0]): p for p in paths}
    max_ind = max(ind_to_path.keys())

    missing_count = 0
    for i in range(max_ind + 1):
        if i not in ind_to_path:
            ind_to_path[i] = None
            missing_count += 1

    if missing_count > 0:
        warnings.warn(f"Missing {missing_count} files in {os.path.dirname(paths[0])}.")

    paths = [ind_to_path[i] for i in range(max_ind + 1)]

    if max_files is not None:
        paths = paths[:max_files]

    if len(paths) == 0:
        return []

    if num_workers > 0:
        with mp.Pool(num_workers) as p:  # Create a multiprocessing Pool
            file_data = p.map(load_jsongz_as_str, paths)
            return file_data
    else:
        return [load_jsongz_as_str(p) for p in tqdm(paths, desc=f"Loading {paths[0]}")]


def get_cache_file_path(path: str) -> str:
    """Get the cache file name for a split."""
    return os.path.join(path, f"dataset_cache.pkl.gz")


def compress_file_then_delete_old(
    input_file_path: str, output_file_path: str, compresslevel: int
):
    with open(input_file_path, "rb") as f_in:
        with gzip.open(output_file_path, "wb", compresslevel=compresslevel) as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(input_file_path)


def save_pickle_gzip(
    data: Any, save_path: str, compresslevel: int = 2, protocol: int = 4
) -> None:
    assert save_path.endswith(".pkl.gz")
    tmp_path = save_path.replace(".gz", "")
    assert not os.path.exists(tmp_path)
    print(f"Caching dataset to {save_path}, this may take a few minutes...")
    with open(tmp_path, "wb") as f:
        pickle.dump(
            obj=data,
            file=f,
            protocol=protocol,
        )
    compress_file_then_delete_old(tmp_path, save_path, compresslevel=compresslevel)


def load_pickle_gzip(path: str):
    assert path.endswith(".pkl.gz")
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def load_dataset(
    path_to_splits: Optional[str],
    split_to_path: Optional[Dict[str, str]] = None,
    num_workers: int = 0,
    use_cache: bool = False,  # TODO: Caching -> ~2x speed up when loading, should be faster to justify complexity
    max_houses_per_split: Optional[Union[int, Dict[str, int]]] = None,
) -> prior.DatasetDict:
    """Load the dataset from a path or a mapping from split to path.

    Arguments :
        path_to_splits (Optional[str]): Path to a directory containing train, val, and test splits.
        split_to_path (Optional[Dict[str, str]]): Mapping from split to path to a directory containing the split.
        num_workers (int): The number of worker processes to use. If 0, no parallelization is done. If <0, all
            available cores are used.
        use_cache (bool): Whether to use a cache file to speed up loading. If True, the cache file will be saved
            in the directory specified by `path_to_splits`.

    Returns:
        prior.DatasetDict: A dictionary of LazyJsonDataset objects for each split found in the input.
    """
    assert (path_to_splits is None) != (
        split_to_path is None
    ), "Exactly one of path or split_to_path must be provided."

    assert (not use_cache) or path_to_splits is not None, (
        "Must provide `path_to_splits` to splits to use cache as we"
        " will save the cache file into this directory."
    )
    assert (not use_cache) or (
        max_houses_per_split is None
    ), "Cannot use cache when `max_houses_per_split` is not None."

    if not isinstance(max_houses_per_split, Dict):
        max_houses_per_split = (lambda x: defaultdict(lambda: x))(max_houses_per_split)

    if use_cache:
        cache_file_path = get_cache_file_path(path_to_splits)
        if os.path.exists(cache_file_path):
            return load_pickle_gzip(cache_file_path)

    if num_workers < 0:
        num_workers = mp.cpu_count()

    if path_to_splits is not None:
        assert os.path.exists(path_to_splits), f"Paqth {path_to_splits} does not exist."
        split_to_path = {
            "train": os.path.join(path_to_splits, "train"),
            "val": os.path.join(path_to_splits, "val"),
            "test": os.path.join(path_to_splits, "test"),
        }

    for split, path in list(split_to_path.items()):
        if not os.path.exists(path):
            del split_to_path[split]
            warnings.warn(
                f"Split {split} does not exist at path {path}, won't be included."
            )

    if len(split_to_path) == 0:
        raise ValueError("No splits found.")

    split_to_house_strs = defaultdict(lambda: [])
    for split, path in split_to_path.items():
        if path.endswith(".jsonl.gz"):
            split_to_house_strs[split] = read_jsonlgz(
                path=path,
                max_lines=max_houses_per_split[split],
            )
        elif os.path.isdir(path):
            subpaths = glob.glob(os.path.join(path, "*.json.gz"))
            split_to_house_strs[split] = read_jsongz_files(
                paths=subpaths,
                num_workers=num_workers,
                max_files=max_houses_per_split[split],
            )
        else:
            raise NotImplementedError(f"Unknown path type: {path}")

    dd = prior.DatasetDict(
        **{
            split: LazyJsonDataset(data=houses, dataset="procthor-100k", split=split)
            for split, houses in split_to_house_strs.items()
        }
    )

    if use_cache:
        cache_file_path = get_cache_file_path(path_to_splits)
        if not os.path.exists(cache_file_path):
            save_pickle_gzip(data=dd, save_path=cache_file_path)

    return dd
