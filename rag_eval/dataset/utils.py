from rag_eval.dataset import (
    HotpotQADataset,
    NaturalQuestionsDataset,
    TopiOCQADataset,
    FaithDialDataset,
)


def load_dataset(dataset_name, split="validation", name=None, file_path=None):
    """
    Loads a dataset by name.

    Args:
        dataset_name (str): Name of the dataset to load.
        split (str): Split of the dataset to load.
        name (str, optional): The dataset configuration to load.
    """
    dataset_mapping = {
        "hotpot_qa": HotpotQADataset,
        "natural_questions": NaturalQuestionsDataset,
        "topiocqa": TopiOCQADataset,
        "faithdial": FaithDialDataset,
    }

    if dataset_name not in dataset_mapping:
        raise NotImplementedError(f"Dataset name {dataset_name} not supported.")

    if split not in ["train", "validation", "test"]:
        raise NotImplementedError(f"Split {split} not supported.")

    return dataset_mapping[dataset_name](split=split, name=name, file_path=file_path)
